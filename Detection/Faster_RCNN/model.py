import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import ops
from utils import *
from loss import *

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(pretrained=True)
        req_layers = list(model.children())[:8]
        self.backbone = nn.Sequential(*req_layers)

        for param in self.backbone.named_parameters():
            param[1].requires_grad = True

    def forward(self, x):
        return self.backbone(x)

class ProposalModule(nn.Module):
    def __init__(self, in_channels, mid_channels=512, n_anchors=9, dropout_p=0.3):
        super().__init__()
        self.n_anchors = n_anchors
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.dropout = nn.Dropout(dropout_p)
        self.cls_head = nn.Conv2d(mid_channels, n_anchors * 2, 1, 1, 0)
        self.reg_head = nn.Convd(mid_channels, n_anchors * 4, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature_map, pos_anchor_index=None, neg_anchor_index=None, pos_anchor_coords=None):
        if pos_anchor_index is None or neg_anchor_index is None or pos_anchor_coords is None:
            mode = 'eval'
        else:
            mode = 'train'

        out = self.conv1(feature_map)
        out = self.relu(self.dropout(out))

        reg_preds = self.reg_head(out) # B x 36 x hmap x wmap
        confidence_preds = self.cls_head(out) # B x 9 x hmap x wmap

        if mode == 'train':
            confidence_pos = confidence_preds.flatten()[pos_anchor_index]
            confidence_neg = confidence_preds.flatten()[neg_anchor_index]

            offsets_pos = reg_preds.contiguous().view(-1, 4)[pos_anchor_index]
            proposals = generate_proposals(pos_anchor_coords, offsets_pos)

            return confidence_pos, confidence_neg, offsets_pos, proposals

        else:
            return confidence_preds, reg_preds
        
class RPN(nn.Module):
    def __init__(self, img_size, out_size, out_channels):
        super().__init__()
        self.img_height, self.img_width = img_size
        self.out_h, self.out_w = out_size

        self.width_scale_factor = self.img_width // self.out_w
        self.height_scale_factor = self.img_height // self.out_h

        self.anchor_scales = [2, 4, 6]
        self.anchor_ratios = [0.5, 1, 1.5]
        self.n_anchor_boxes = len(self.anchor_scales) * len(self.anchor_ratios)

        self.pos_thresh = 0.7
        self.neg_thresh = 0.3

        self.weight_confidence = 1
        self.weight_regression = 5

        self.feature_extractor = FeatureExtractor()
        self.proposal_module = ProposalModule(out_channels)

    def forward(self, images, gt_boxes, gt_classes):
        batch_size = images.shape[0]
        feature_map = self.feature_extractor(images)

        anchor_point_x, anchor_point_y = get_anchor_center((self.out_h, self.out_w))
        anchor_base = gen_anchor_base(anchor_point_x, anchor_point_y, self.anchor_scales, self.anchor_ratios, (self.out_h, self.out_w))
        anchor_boxes = anchor_base.repeat(batch_size, 1, 1, 1, 1)

        gt_boxes_proj = project_gt_boxes(gt_boxes, self.width_scale_factor, self.height_scale_factor, mode='p2a')

        pos_anchor_index, neg_anchor_index, GT_confidence_score, GT_offsets, GT_class_pos, pos_anchor_coords, neg_anchor_coords, pos_anchor_index_seperate = get_req_anchors(anchor_boxes, gt_boxes_proj, gt_classes)

        confidence_score_pos, confidence_score_neg, offsets_pos, proposals = self.proposal_module(feature_map, pos_anchor_index, neg_anchor_index, pos_anchor_coords)
        
        cls_loss = calc_cls_loss(confidence_score_pos, confidence_score_neg, batch_size)
        reg_loss = calc_box_loss(GT_offsets, offsets_pos, batch_size)

        total_rpn_loss = self.weight_confidence * cls_loss + self.weight_regression * reg_loss

        return total_rpn_loss, feature_map, proposals, pos_anchor_index_seperate, GT_class_pos, self.width_scale_factor, self.height_scale_factor
    
    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        with torch.no_grad():
            batch_size = images.shape[0]
            feature_map = self.feature_extractor(images)

            anchor_point_x, anchor_point_y = get_anchor_center((self.out_h, self.out_w))
            anchor_base = gen_anchor_base(anchor_point_x, anchor_point_y, self.anchor_scales, self.anchor_ratios, (self.out_h, self.out_w))
            anchor_boxes = anchor_base.repeat(batch_size, 1, 1, 1)
            anchor_boxes_flat = anchor_boxes.reshape(batch_size, -1, 4)

            confidence_score_preds, offsets_preds = self.proposal_module(feature_map)
            confidence_score_preds = confidence_score_preds.reshape(batch_size, -1)
            offsets_preds = offsets_preds.reshape(batch_size, -1, 4)

            proposals_final = []
            confidence_score_final = []

            for i in range(batch_size):
                confidence_score = torch.sigmoid(confidence_score_preds[i])
                offsets = offsets_preds[i]
                anchor_b = anchor_boxes_flat[i]
                proposals = generate_proposals(anchor_b, offsets)
                # filter based on confidence threshold
                confidence_index = torch.where(confidence_score >= conf_thresh)[0]
                confidence_score_pos = confidence_score[confidence_index]
                proposals_pos = proposals[confidence_index]
                # filter based on nms threshold
                nms_index = ops.nms(proposals_pos, confidence_score_pos, nms_thresh)
                confidence_score_pos = confidence_score_pos[nms_index]
                proposals_pos = proposals_pos[nms_index]

                proposals_final.append(proposals_pos)
                confidence_score_final.append(confidence_score_pos)
                
        return proposals_final, confidence_score_final, feature_map
    
class Classification(nn.Module):
    def __init__(self, out_channels, n_classes, roi_size, hidden_dim=512, dropout_p=0.3):
        super().__init__()
        self.roi_size = roi_size
        self.avg_pool = nn.AvgPool2d(self.roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.relu = nn.ReLU(inplace=True)

        self.cls_head = nn.Linear(hidden_dim, n_classes)
        self.reg_head = nn.Linear(hidden_dim, 4)

    def forward(self, feature_map, proposal_list, width_scale_factor, height_scale_factor, gt_classes=None):
        if gt_classes is None:
            mode = 'eval'

        else:
            mode = 'train'

        roi_out = ops.roi_pool(feature_map, proposal_list, self.roi_size)
        roi_out = self.avg_pool(roi_out)

        roi_out = roi_out.squeeze(-1).squeeze(-1)

        out = self.fc(roi_out)
        out = self.relu(self.dropout(out))

        cls_scores = self.cls_head(out)
        reg_offsets = self.reg_head(out)

        # project proposal to image size
        proposal_list = project_gt_boxes(proposal_list, width_scale_factor, height_scale_factor, 'a2p')

        # find corresponding gt_boxes for each proposals

        if mode == 'eval':
            return cls_scores
        
        cls_loss = F.cross_entropy(cls_scores, gt_classes.long())

        return cls_loss
    
class FasterRCNN(nn.Module):
    def __init__(self, img_size, out_size, out_channels, n_classes, roi_size):
        super().__init__()
        self.rpn = RPN(img_size, out_size, out_channels)
        self.classifier = Classification(out_channels, n_classes, roi_size)

    def forward(self, images, gt_boxes, gt_classes):
        total_rpn_loss, feature_map, proposals, pos_anchor_index_seperate, GT_class_pos, width_scale_factor, height_scale_factor = self.rpn(images, gt_boxes, gt_classes)

        pos_proposal_list = []
        batch_size = images.shape[0]

        for i in range(batch_size):
            proposal_index = torch.where(pos_anchor_index_seperate == i)[0]
            proposal_seperate = proposals[proposal_index].detach().clone()
            pos_proposal_list.append(proposal_seperate)

        cls_loss = self.classifier(feature_map, pos_proposal_list, width_scale_factor, height_scale_factor, GT_class_pos)
        total_loss = cls_loss + total_rpn_loss

        return total_loss

    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        batch_size = images.shape[0]
        proposals_final, confidence_score_final, feature_map = self.rpn.inference(images, conf_thresh, nms_thresh)
        cls_scores = self.classifier(feature_map, proposals_final)

        cls_probs = F.softmax(cls_scores, dim=-1)
        classes = torch.argmax(cls_probs, dim=-1)

        classes_final = []
        index = 0

        for i in range(batch_size):
            n_proposals = len(proposals_final[i])
            classes_final.append(classes[index: index + n_proposals])
            index += n_proposals

        return proposals_final, confidence_score_final, classes_final
    
class MaskHead(nn.Module):
    def __init__(self, in_channels, mid_channels=256, out_channels=81):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature_map):
        out = self.relu(self.conv1(feature_map))
        out = self.conv2(out)
        return out
    
class MaskRCNN(nn.Module):
    def __init__(self, img_size, out_size, out_channels, n_classes, roi_size):
        super().__init__()
        self.rpn = RPN(img_size, out_size, out_channels)
        self.classifier = Classification(out_channels, n_classes, roi_size)
        self.mask_head = MaskHead(out_channels)
        self.bbox_head = nn.Linear(out_channels, 4)  # 4 coordinates for bounding box

    def forward(self, images, gt_boxes, gt_classes):
        total_rpn_loss, feature_map, proposals, pos_anchor_index_separate, GT_class_pos = self.rpn(images, gt_boxes,
                                                                                                    gt_classes)

        pos_proposal_list = []
        batch_size = images.shape[0]

        for i in range(batch_size):
            proposal_index = torch.where(pos_anchor_index_separate == i)[0]
            proposal_separate = proposals[proposal_index].detach().clone()
            pos_proposal_list.append(proposal_separate)

        cls_loss = self.classifier(feature_map, pos_proposal_list, GT_class_pos)
        
        # Bounding box regression
        bbox_losses = []
        for i in range(batch_size):
            pos_proposals_batch = pos_proposal_list[i]
            proposal_features = ops.roi_pool(feature_map[i].unsqueeze(0), pos_proposals_batch, self.roi_size)
            proposal_features = proposal_features.view(proposal_features.size(0), -1)
            bbox_preds = self.bbox_head(proposal_features)
            bbox_targets = gt_boxes[i][pos_anchor_index_separate == i]
            bbox_loss = F.smooth_l1_loss(bbox_preds, bbox_targets)
            bbox_losses.append(bbox_loss)
        
        total_bbox_loss = torch.stack(bbox_losses).mean()

        total_loss = cls_loss + total_rpn_loss + total_bbox_loss

        return total_loss

    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        batch_size = images.shape[0]
        proposals_final, confidence_score_final, feature_map = self.rpn.inference(images, conf_thresh, nms_thresh)
        cls_scores = self.classifier(feature_map, proposals_final)

        cls_probs = F.softmax(cls_scores, dim=-1)
        classes = torch.argmax(cls_probs, dim=-1)

        classes_final = []
        index = 0

        for i in range(batch_size):
            n_proposals = len(proposals_final[i])
            classes_final.append(classes[index: index + n_proposals])
            index += n_proposals

        masks = []
        bboxes = []
        for proposal, feature in zip(proposals_final, feature_map):
            roi_pooled_features = ops.roi_align(feature.unsqueeze(0), proposal, output_size=(self.mask_head.conv1.out_channels, self.mask_head.conv1.out_channels))
            mask_preds = self.mask_head(roi_pooled_features)
            masks.append(mask_preds)
            
            proposal_features = ops.roi_pool(feature.unsqueeze(0), proposal, self.roi_size)
            proposal_features = proposal_features.view(proposal_features.size(0), -1)
            bbox_preds = self.bbox_head(proposal_features)
            bboxes.append(bbox_preds)

        return proposals_final, confidence_score_final, classes_final, masks, bboxes