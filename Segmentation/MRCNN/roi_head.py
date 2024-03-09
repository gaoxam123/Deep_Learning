import torch
import torch.nn as nn
import torch.nn.functional as F
from roi_align import *
from utils import *
from box_ops import *

def fastrcnn_loss(class_logit, offsets, labels, regression_target):
    class_loss = F.cross_entropy(class_logit, labels)

    N, num_pos = class_logit.shape[0], regression_target.shape[0]
    offsets = offsets.reshape(N, -1, 4)
    offsets, labels = offsets[:num_pos], labels[:num_pos]
    box_idx = torch.arange(num_pos, device=labels.device)

    box_loss = F.smooth_l1_loss(offsets[box_idx, labels], regression_target, reduction='sum') / N

    return class_loss, box_loss

def maskrcnn(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat([matched_idx, proposal], dim=1)

    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1, M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)

    return mask_loss

class ROIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor, fg_iou_thresh, bg_iou_thresh, num_samples, pos_fraction, reg_weights, score_thresh, nms_thresh, num_detection):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor

        self.mask_roi_pool = None
        self.mask_predictor = None

        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, False)
        self.fg_bg_sampler = PosNegSampler(num_samples, pos_fraction)
        self.box_coder = BoxCoder(reg_weights)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detection = num_detection
        self.min_size = 1

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
    
    def select_training_samples(self, proposal, target):
        gt_box = target['boxes']
        gt_label = target['labels']
        proposal = torch.cat([proposal, gt_box])

        iou = box_iou(proposal, gt_box)
        pos_neg_label, matched_idx = self.proposal_matcher(iou)
        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
        idx = torch.cat([pos_idx, neg_idx])

        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])
        proposal = proposal[idx]
        matched_idx= matched_idx[idx]
        label = gt_label[matched_idx]
        num_pos = pos_idx.shape[0]
        label[num_pos:] = 0

        return proposal, matched_idx, label, regression_target
    
    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape):
        N, num_classes = class_logit.shape
        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=1)
        box_regression = box_regression.reshape(N, -1, 4)

        boxes = []
        labels = []
        scores = []
        for i in range(1, num_classes):
            score, offsets = pred_score[:, i], box_regression[:, i]

            keep = score >= self.score_thresh
            box, score, offsets = proposal[keep], score[keep], offsets[keep]
            box = self.box_coder.decode(box, offsets)

            box, score = process_box(box, score, image_shape, self.min_size)

            keep = nms(box, score, self.nms_thresh)[:self.num_detection]
            box, score = box[keep], score[keep]
            label = torch.full((len(keep),), i, dtype=keep.dtype, device=device)

            boxes.append(box)
            labels.append(label)
            scores.append(score)

        res = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores))

        return res
    
    def forward(self, feature, proposal, image_shape, target):
        if self.training:
            proposal, matched_idx, label, regression_target = self.select_training_samples(proposal, target)

        box_feature = self.box_roi_pool(feature, proposal, image_shape)
        class_logit, box_regression = self.box_predictor(box_feature)

        res, loss = {}, {}
        if self.training:
            cls_loss, box_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
            loss = dict(roi_cls_loss=cls_loss, roi_box_loss=box_loss)

        else:
            res = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape)

        if self.has_mask():
            if self.training:
                num_pos = regression_target.shape[0]
                mask_proposal = proposal[:num_pos]
                pos_matched_idx = matched_idx[:num_pos]
                mask_label = label[:num_pos]

                box_regression = box_regression[:num_pos].reshape(num_pos, -1, 4)
                idx = torch.arange(num_pos, device=mask_label.device)
                mask_proposal = self.box_coder.decode(box_regression[idx, mask_label], mask_proposal)

                if mask_proposal.shape[0] == 0:
                    loss.update(dict(roi_mask_loss=torch.tensor(0)))
                    return res, loss
            
            else:
                mask_proposal = res['boxes']

                if mask_proposal.shape[0] == 0:
                    res.update(dict(masks=torch.empty((0, 28, 28))))
                    return res, loss
                
            mask_feature = self.mask_roi_pool(feature, mask_proposal, image_shape)
            mask_logit = self.mask_predictor(mask_feature)

            if self.training:
                gt_mask = target['masks']
                mask_loss = maskrcnn(mask_logit, mask_proposal, matched_idx, mask_label, gt_mask)
                loss.update(dict(roi_mask_loss=mask_loss))

            else:
                label = res['label']
                idx = torch.arange(label.shape[0], device=label.device)
                mask_logit = mask_logit[idx, label]

                mask_prob = mask_logit.sigmoid()
                res.update(dict(masks=mask_prob))

        return res, loss