import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision
from torch.utils.data import Dataset, DataLoader

def get_anchor_centers(out_size):
    out_h, out_w = out_size
    center_point_x = torch.arange(0, out_w) + 0.5
    center_point_y = torch.arange(0, out_h) + 0.5

    return center_point_x, center_point_y

def get_anchor_base(center_point_x, center_point_y, anchor_scales, anchor_ratios, out_size):
    out_h, out_w = out_size
    num_boxes = len(anchor_scales) * len(anchor_ratios) * out_w * out_h
    anchor_base = torch.zeros(1, out_w, out_h, 9, 4)

    for ix, cx in enumerate(center_point_x):
        for iy, cy in enumerate(center_point_y):
            anchor_box = torch.zeros(9, 4)
            index = 0

            for scale in anchor_scales:
                for ratio in anchor_ratios:
                    w = scale * ratio
                    h = scale

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    anchor_box[index, :] = torch.tensor([x1, y1, x2, y2])
                    index += 1

            anchor_base[:, ix, iy, :, :] = ops.clip_boxes_to_image(anchor_box, out_size)

    return anchor_base

def get_iou_mat(batch_size, anchor_boxes, gt_boxes):
    # anchor_boxes: batch_size x w x h x 9 x 4
    # gt_boxes: batch_size x max_obj x 4
    total_boxes = anchor_boxes.shape[1] * anchor_boxes.shape[2] * anchor_boxes.shape[3]
    anchor_boxes = anchor_boxes.reshape(batch_size, -1, 4)
    iou_mat = torch.zeros(batch_size, total_boxes, gt_boxes.shape[1])

    for i in range(batch_size):
        anchor_box = anchor_boxes[i]
        gt_box = gt_boxes[i]
        iou_mat[i, :] = ops.box_iou(anchor_box, gt_box)

    return iou_mat

def project_gt_boxes(boxes, width_scale_factor, height_scale_factor, mode='p2a'):
    assert mode in ['p2a', 'a2p']
    proj_boxes = torch.zeros_like(boxes)
    invalid_box_mask = (proj_boxes == -1)

    if mode == 'a2p':
        proj_boxes[..., [0, 2]] *= width_scale_factor
        proj_boxes[..., [1, 3]] *= height_scale_factor

    else:
        proj_boxes[..., [0, 2]] /= width_scale_factor
        proj_boxes[..., [1, 3]] /= height_scale_factor

    proj_boxes.masked_fill(invalid_box_mask, -1)
    
    return proj_boxes

def calc_gt_offsets(pos_anchor_coords, gt_box_mapping):
    pos_anchor_coords = ops.box_convert(pos_anchor_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_box_mapping = ops.box_convert(gt_box_mapping, 'xyxy', 'cxcywh')

    gt_cx, gt_cy, gt_w, gt_h = gt_box_mapping[..., 0], gt_box_mapping[..., 1], gt_box_mapping[..., 2], gt_box_mapping[..., 3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anchor_coords[..., 0], pos_anchor_coords[..., 1], pos_anchor_coords[..., 2], pos_anchor_coords[..., 3]
    
    tx = (gt_cx - anc_cx) / anc_w
    ty = (gt_cy - anc_cy) / anc_h
    tw = torch.log(gt_w / anc_w)
    th = torch.log(gt_h / anc_h)

    offsets = torch.stack([tx, ty, tw, th], dim=-1)

    return offsets

def get_req_anchors(anchor_boxes, gt_boxes, gt_classes, pos_thresh=0.7, neg_thresh=0.3):
    # anchor_boxes: batch_size x w x h x 9 x 4
    # gt_boxes: batch_size x max_objs x 4
    # gt_classes: batch_size x max_objs
    B, w, h, n_anchors, _ = anchor_boxes.shape
    N = gt_boxes.shape[1]
    total_boxes = w * h * n_anchors

    iou_mat = get_iou_mat(B, anchor_boxes, gt_boxes)
    max_iou_per_gt_box, _ = iou_mat.max(dim=1, keepdim=True)

    pos_anchor_mask = torch.logical_and(iou_mat == max_iou_per_gt_box, max_iou_per_gt_box > 0) # 3 x 4
    pos_anchor_mask = torch.logical_or(pos_anchor_mask, max_iou_per_gt_box > pos_thresh)

    pos_anchor_index_sep = torch.where(pos_anchor_mask)[0]
    pos_anchor_mask = pos_anchor_mask.flatten(start_dim=0, end_dim=1) # 12 + n
    pos_anchor_index = torch.where(pos_anchor_mask)[0]

    max_iou_per_anchor, max_iou_per_anchor_index = iou_mat.max(dim=-1)
    max_iou_per_anchor = max_iou_per_anchor.flatten(start_dim=0, end_dim=1)

    GT_conf_scores = max_iou_per_anchor[pos_anchor_index]

    gt_class_expand = gt_classes.reshape(B, 1, N).expand(B, total_boxes, N)
    GT_class = torch.gather(gt_class_expand, -1, max_iou_per_anchor_index.unsqueeze(-1)).squeeze(-1)
    GT_class = GT_class.flatten(start_dim=0, end_dim=1)
    GT_class_pos = GT_class[pos_anchor_index]

    gt_boxes_expand = gt_boxes.reshape(B, 1, N, 4).expand(B, total_boxes, N, 4)
    GT_boxes = torch.gather(gt_boxes_expand, -2, max_iou_per_anchor_index.view(B, 1, N, 4))
    GT_boxes = GT_boxes.flatten(start_dim=0, end_dim=2)
    GT_boxes_pos = GT_boxes[pos_anchor_index]

    anchor_boxes_flat = anchor_boxes.flatten(start_dim=0, end_dim=-2)
    pos_anchor_coords = anchor_boxes_flat[pos_anchor_index]
    GT_offsets = calc_gt_offsets(pos_anchor_coords, GT_boxes_pos)

    neg_anchor_mask = (max_iou_per_anchor < neg_thresh)
    neg_anchor_index = torch.where(neg_anchor_mask)[0]
    neg_anchor_index = neg_anchor_index[torch.randint(0, neg_anchor_index.shape[0], pos_anchor_index.shape[0])]
    neg_anchor_coords = anchor_boxes_flat[neg_anchor_index]

    return pos_anchor_index, neg_anchor_index, GT_conf_scores, GT_offsets, GT_class_pos, pos_anchor_coords, neg_anchor_coords, pos_anchor_index_sep

def generate_proposals(anchors, offsets):
    anchors = ops.box_convert(anchors, 'xyxy', 'cxcywh')
    proposals_ = torch.zeros_like(anchors)

    proposals_[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
    proposals_[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
    proposals_[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
    proposals_[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])

    proposals_ = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

    return proposals_

def calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size):
    target_pos = torch.ones_like(conf_scores_pos)
    target_neg = torch.zeros_like(conf_scores_neg)
    
    target = torch.cat([target_pos, target_neg])
    input = torch.cat([conf_scores_pos, conf_scores_neg])

    loss = F.binary_cross_entropy_with_logits(input, target, reduction='sum') / batch_size

    return loss

def calc_reg_loss(gt_offsets, reg_offsets_pos, batch_size):
    assert gt_offsets.shape == reg_offsets_pos.shape
    loss = F.smooth_l1_loss(reg_offsets_pos, gt_offsets, reduction='sum') / batch_size

    return loss

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        req_layers = list(model.children())[:8]
        self.backbone = nn.Sequential(*req_layers)

    def forward(self, x):
        return self.backbone(x)
    
class ProposalModule(nn.Module):
    def __init__(self, in_channels, mid_channels=512, n_anchors=9, dropout_p=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.n_anchors = n_anchors
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.cls_head = nn.Conv2d(mid_channels, n_anchors, 1)
        self.reg_head = nn.Conv2d(mid_channels, n_anchors * 4, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature_map, pos_anchor_index=None, neg_anchor_index=None, pos_anchor_coords=None):
        if pos_anchor_index is None or neg_anchor_index is None or pos_anchor_coords is None:
            mode = 'eval'
        else:
            mode = 'train'

        out = self.conv1(feature_map)
        out = self.relu(self.dropout(out))

        conf_scores = self.cls_head(out) # batch_size 9 x w x h
        reg_offsets = self.reg_head(out) # batch_size x 36 x w x h

        if mode == 'train':
            conf_pos = conf_scores.flatten()[pos_anchor_index]
            conf_neg = conf_scores.flatten()[neg_anchor_index]

            offsets_pos = reg_offsets.contiguous().view(-1, 4)[pos_anchor_index]
            proposals = generate_proposals(pos_anchor_coords, offsets_pos)

            return conf_pos, conf_neg, offsets_pos, proposals
        
        return conf_scores, reg_offsets
    
class RPN(nn.Module):
    def __init__(self, img_size, out_size, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.out_h, self.out_w = out_size
        self.width, self.height = img_size

        self.width_scale_factor = self.width // self.out_w
        self.height_scale_factor = self.height // self.out_h

        self.anchor_scales = [2, 4, 6]
        self.anchor_ratios = [0.5, 1, 1.5]
        self.n_anchors = 9

        self.pos_thresh = 0.7
        self.neg_thresh = 0.3

        self.weight_conf = 1
        self.weight_reg = 5

        self.feature_extractor = FeatureExtractor()
        self.proposal_module = ProposalModule(out_channels)

    def forward(self, images, gt_boxes, gt_classes):
        batch_size = images.shape[0]
        feature_map = self.feature_extractor(images)

        center_point_x, center_point_y = get_anchor_centers((self.out_h, self.out_w))
        anchor_base = get_anchor_base(center_point_x, center_point_y, self.anchor_scales, self.anchor_ratios, (self.out_h, self.out_w))
        anchor_boxes = anchor_base.repeat(batch_size, 1, 1, 1, 1)
        
        gt_proj_boxes = project_gt_boxes(gt_boxes, self.width_scale_factor, self.height_scale_factor)

        pos_anchor_index, neg_anchor_index, GT_conf_scores, GT_offsets, \
        GT_class_pos, pos_anchor_coords, neg_anchor_coords, pos_anchor_index_sep = get_req_anchors(anchor_boxes, gt_proj_boxes, gt_classes, self.pos_thresh, self.neg_thresh)

        conf_pos, conf_neg, offsets_pos, proposals = self.proposal_module(feature_map, pos_anchor_index, neg_anchor_index, pos_anchor_coords)

        cls_loss = calc_cls_loss(conf_pos, conf_neg, batch_size)
        box_loss = calc_reg_loss(GT_offsets, offsets_pos, batch_size)

        total_rpn_loss = self.weight_conf * cls_loss + self.weight_reg * box_loss

        return total_rpn_loss, feature_map, proposals, pos_anchor_index_sep, GT_class_pos
    
    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        with torch.no_grad():
            batch_size = images.shape[0]
            feature_map = self.feature_extractor(images)

            center_point_x, center_point_y = get_anchor_centers((self.out_h, self.out_w))
            anchor_base = get_anchor_base(center_point_x, center_point_y, self.anchor_scales, self.anchor_ratios, (self.out_h, self.out_w))
            anchor_boxes = anchor_base.repeat(batch_size, 1, 1, 1, 1)
            anchor_boxes_flat = anchor_boxes.reshape(batch_size, -1, 4)

            conf_scores, reg_offsets = self.proposal_module(feature_map)
            conf_scores = conf_scores.reshape(batch_size, -1)
            reg_offsets = reg_offsets.reshape(batch_size, -1, 4)

            proposals_final = []
            conf_scores_final = []

            for i in range(batch_size):
                conf_score = torch.sigmoid(conf_scores[i])
                anchor_boxes = anchor_boxes_flat[i]
                offsets = reg_offsets[i]
                proposals = generate_proposals(anchor_boxes, offsets)
                
                keep = torch.where(conf_score > conf_thresh)[0]
                conf_scores = conf_scores[keep]
                proposals = proposals[keep]

                keep = ops.nms(proposals, conf_thresh, nms_thresh)
                conf_score = conf_score[keep]
                proposals = proposals[keep]

                proposals_final.append(proposals)
                conf_scores_final.append(conf_score)

        return proposals_final, conf_scores_final, feature_map
    
class Classification(nn.Module):
    def __init__(self, out_channels, n_classes, roi_size, mid_channels=512, dropout_p=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.roi_size = roi_size
        self.avg_pool = nn.AvgPool2d(roi_size)
        self.fc = nn.Linear(out_channels, mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.cls_head = nn.Linear(mid_channels, n_classes)
        self.reg_head = nn.Linear(mid_channels, 4)

    def forward(self, feature_map, proposal_list, batch_size, gt_classes=None, gt_offsets=None):
        if gt_classes is None:
            mode = 'eval'
        else:
            mode = 'train'

        roi_out = ops.roi_pool(feature_map, proposal_list, self.roi_size)
        roi_out = self.avg_pool(roi_out)

        roi_out = roi_out.unsqueeze(-1).unsqueeze(-1)

        out = self.fc(roi_out)
        out = self.relu(self.dropout(out))

        cls_preds = self.cls_head(out)
        reg_preds = self.reg_head(out)

        if mode == 'eval':
            return cls_preds, reg_preds
        
        cls_loss = F.cross_entropy(cls_preds, gt_classes)
        # reg_loss = calc_reg_loss(gt_offsets, reg_preds, batch_size)

        # return cls_loss + 3 * reg_loss

        return cls_loss
    
class FasterRCNN(nn.Module):
    def __init__(self, img_size, out_size, out_channels, n_classes, roi_size):
        super().__init__()
        self.rpn = RPN(img_size, out_size, out_channels)
        self.classifier = Classification(out_channels, n_classes, roi_size)

    def forward(self, images, gt_boxes, gt_classes):
        batch_size = images.shape[0]
        total_rpn_loss, feature_map, proposals, pos_anchor_index_sep, GT_class_pos = self.rpn(images, gt_boxes, gt_classes)

        pos_proposal_list = []
        
        for i in range(batch_size):
            proposal_index = torch.where(pos_anchor_index_sep == i)[0]
            proposal_sep = proposals[proposal_index].detach().clone()
            pos_proposal_list.append(proposal_sep)

        cls_loss = self.classifier.forward(feature_map, pos_proposal_list, batch_size, GT_class_pos)
        total_loss = cls_loss + total_rpn_loss

        return total_loss
    
    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        batch_size = images.shape[0]
        proposals_final, conf_scores_final, feature_map = self.rpn.inference(images, conf_thresh, nms_thresh)
        cls_preds, reg_preds = self.classifier(feature_map, proposals_final)

        cls_probs = F.softmax(cls_preds, dim=-1)
        classes = cls_probs.argmax(dim=-1)

        classes_final = []
        index = 0

        for i in range(batch_size):
            n_proposals = len(proposals_final[i])
            classes_final.append(classes[index:index + n_proposals])
            index += n_proposals

        return proposals_final, conf_scores_final, classes_final
    
class CamelDataset(Dataset):
    def __init__(self, annotation_path, img_dir, img_size, name2idx):
        super().__init__()
        self.annotation_path = annotation_path
        self.img_dir = img_dir
        self.img_size = img_size
        self.name2idx = name2idx

        self.img, self.gt_boxes, self.gt_classes = self.get_data()

    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, index):
        return self.img[index], self.gt_boxes[index], self.gt_classes[index]
    
    def get_data(self):
        img = []
        gt_idx = []

        