import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_cls_loss(confidence_score_pos, confidence_score_neg, batch_size):
    target_pos = torch.ones_like(confidence_score_pos)
    target_neg = torch.zeros_like(confidence_score_neg)

    target = torch.cat([target_pos, target_neg])
    inputs = torch.cat([confidence_score_pos, confidence_score_neg])

    loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='sum') * 1. / batch_size

    return loss

def calc_box_loss(gt_offsets, reg_offsets_pos, batch_size):
    assert gt_offsets.shape == reg_offsets_pos.shape
    loss = F.smooth_l1_loss(reg_offsets_pos, gt_offsets, reduction='sum') * 1. / batch_size

    return loss