import torch
import torch.nn.functional as F
import torch.nn as nn

from box_ops import *
from utils import *

class RPNHead(nn.Module):
    def __init__(self, in_channels, n_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.cls_head = nn.Conv2d(in_channels, n_anchors, 1)
        self.reg_head = nn.Conv2d(in_channels, n_anchors * 4, 1)
        self.relu = nn.ReLU(inplace=True)

        for l in self.children():
            nn.init.normal_(l.weight, std=0.01)
            nn.init.zeros_(l.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv(x))
        conf_scores = self.cls_head(x)
        reg_offsets = self.reg_head(x)

        return conf_scores, reg_offsets
    
class RPN(nn.Module):
    def __init__(self, anchor_generator, head, fg_iou_thresh, bg_iou_thresh, num_samples, pos_fraction, reg_weights, pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head

        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, True)
        self.fg_bg_sampler = PosNegSampler(num_samples, pos_fraction)
        self.box_coder = BoxCoder(reg_weights)

        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1

    def create_proposal(self, anchor, conf_scores, offsets, image_shape):
        if self.training:
            pre_nms_top_n = self.pre_nms_top_n['training']
            post_nms_top_n = self.post_nms_top_n['training']
        else:
            pre_nms_top_n = self.pre_nms_top_n['testing']
            post_nms_top_n = self.post_nms_top_n['testing']

        pre_nms_top_n = min(conf_scores.shape[0], pre_nms_top_n)
        top_n_index = conf_scores.topk(pre_nms_top_n)[1]
        scores = conf_scores[top_n_index]
        proposal = self.box_coder.decode(anchor[top_n_index], offsets[top_n_index])

        proposal, scores = process_box(proposal, scores, image_shape, self.min_size)
        keep = nms(proposal, scores, self.nms_thresh)[:post_nms_top_n]
        proposal = proposal[keep]

        return proposal
    
    def compute_loss(self, conf_scores, offsets, gt_box, anchor):
        iou = box_iou(gt_box, anchor)
        labels, matched_index = self.proposal_matcher(iou)

        pos_idx, neg_idx = self.fg_bg_sampler(labels)
        idx = torch.cat([pos_idx, neg_idx])
        reg_target = self.box_coder.encode(gt_box[matched_index[pos_idx]], anchor[pos_idx])

        cls_loss = F.binary_cross_entropy_with_logits(conf_scores[idx], labels[idx])
        box_loss = F.l1_loss(offsets[pos_idx], reg_target, reduction='sum') / idx.numel()

        return cls_loss, box_loss
    
    def forward(self, feature, image_shape, target=None):
        if target is not None:
            gt_box = target['boxes']

        anchor = self.anchor_generator(feature, image_shape)

        conf_scores, offsets = self.head(feature)
        conf_scores = conf_scores.permute(0, 2, 3, 1).flatten()
        offsets = offsets.permute(0, 2, 3, 1).reshape(-1, 4)

        proposal = self.create_proposal(anchor, conf_scores.detach(), offsets.detach(), image_shape)

        if self.training:
            cls_loss, box_loss = self.compute_loss(conf_scores, offsets, gt_box, anchor)
            return proposal, dict(rpn_cls_loss=conf_scores, rpn_box_loss=box_loss)
        
        return proposal, {}