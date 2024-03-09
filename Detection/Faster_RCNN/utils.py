import torch
import numpy as np
from torchvision import ops

def get_anchor_center(out_size):
    out_h, out_w = out_size

    anchor_point_x = torch.arange(0, out_w) + 0.5
    anchor_point_y = torch.arange(0, out_h) + 0.5

    return anchor_point_x, anchor_point_y

def gen_anchor_base(anchor_point_x, anchor_point_y, anchor_scales, anchor_ratios, out_size):
    n_anchor_boxes = len(anchor_scales) * len(anchor_scales)
    anchor_base = torch.zeros(1, anchor_point_x.shape[0], anchor_point_y.shape[0], n_anchor_boxes, 4)
    
    for ix, xc in enumerate(anchor_point_x):
        for jx, yc in enumerate(anchor_point_y):
            anchor_boxes = torch.zeros((n_anchor_boxes, 4))
            index = 0
            for i, scale in enumerate(anchor_scales):
                for j, ratio in enumerate(anchor_ratios):
                    w = scale * ratio
                    h = scale

                    x1 = xc - w / 2
                    y1 = yc - h / 2
                    x2 = xc + w / 2
                    y2 = yc + h / 2

                    anchor_boxes[index, :] = torch.tensor([x1, y1, x2, y2])
                    index += 1

            anchor_base[:, ix, jx, :] = ops.clip_boxes_to_image(anchor_boxes, size=out_size)

    return anchor_base

def get_iou_mat(batch_size, anchor_boxes, gt_boxes):
    # anchor_boxes = N x w x h x n_anchor_box x 4
    # gt_boxes = N x max_objs x 4
    anchor_boxes_flat = anchor_boxes.reshape(batch_size, -1, 4)
    total_anchor_boxes = anchor_boxes_flat.shape[1]

    iou_mat = torch.zeros(batch_size, total_anchor_boxes, gt_boxes.shape[1])

    for i in range(batch_size):
        gt_box = gt_boxes[i]
        anchor_box = anchor_boxes_flat[i]
        iou_mat[i, :] = ops.box_iou(anchor_box, gt_box)

    return iou_mat

def project_gt_boxes(boxes, width_scale_factor, height_scale_factor, mode='a2p'):
    assert mode in ['a2p', 'p2a']

    batch_size = boxes.shape[0]
    proj_boxes = boxes.clone().reshape(batch_size, -1, 4)
    invalid_box_mask = (proj_boxes == -1) # indicating padded bboxes

    if mode == 'a2p':
        proj_boxes[:, :, [0, 2]] *= width_scale_factor
        proj_boxes[:, :, [1, 3]] *= height_scale_factor

    else:
        proj_boxes[:, :, [0, 2]] /= width_scale_factor # snap to the nearest grid since we use integer dividing
        proj_boxes[:, :, [1, 3]] /= height_scale_factor

    proj_boxes.masked_fill_(invalid_box_mask, -1) # fill padded bboxes back with -1
    proj_boxes.resize_as_(boxes) 

    return proj_boxes

def cal_gt_offsets(pos_anchor_coords, gt_box_mapping):
    pos_anchor_coords = ops.box_convert(pos_anchor_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_box_mapping = ops.box_convert(gt_box_mapping, in_fmt='xyxy', out_fmt='cxcywh')

    gt_cx, gt_cy, gt_w, gt_h = gt_box_mapping[..., 0], gt_box_mapping[..., 1], gt_box_mapping[..., 2], gt_box_mapping[..., 3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anchor_coords[..., 0], pos_anchor_coords[..., 1], pos_anchor_coords[..., 2], pos_anchor_coords[..., 3]

    tx = (gt_cx - anc_cx) / anc_w
    ty = (gt_cy - anc_cy) / anc_h
    tw = torch.log(gt_w / anc_w)
    th = torch.log(gt_h / anc_h)

    return torch.stack([tx, ty, tw, th], dim=-1)

def get_req_anchors(anchor_boxes, gt_boxes, gt_classes, pos_thresh=0.7, neg_thresh=0.3):
    # anchor_boxes = N x w x h x n_anchor_box x 4
    # gt_boxes = N x max_objs x 4
    # gt_classes = N x max_objs x 1
    B, w_map, h_map, n_anchors, _ = anchor_boxes.shape
    N = gt_boxes.shape[1]
    total_boxes = n_anchors * w_map * h_map

    iou_mat = get_iou_mat(B, anchor_boxes, gt_boxes)
    max_iou_per_gt_box, _ = iou_mat.max(dim=1, keepdim=True) # max iou of each gt_box (each column)

    pos_anchor_mask = torch.logical_and(iou_mat == max_iou_per_gt_box, max_iou_per_gt_box > 0) # 1st condition
    pos_anchor_mask = torch.logical_or(pos_anchor_mask, iou_mat > pos_thresh) # 2nd condition

    pos_anchor_index_seperate = torch.where(pos_anchor_mask)[0] # in which batch?
    pos_anchor_mask = pos_anchor_mask.flatten(start_dim=0, end_dim=1) # merge all the batches -> more rows
    pos_anchor_index = torch.where(pos_anchor_mask)[0] # in which row?

    max_iou_per_anchor, max_iou_per_anchor_index = iou_mat.max(dim=-1) # max iou of each anchor with every gt_box
    max_iou_per_anchor = max_iou_per_anchor.flatten(start_dim=0, end_dim=1) # merge all batches

    GT_confidence_score = max_iou_per_anchor[pos_anchor_index] # max iou of that anchor with every anchor in the image

    # get gt class of the pos anchors

    gt_classes_expand = gt_classes.view(B, 1, N).expand(B, total_boxes, N) # expand gt classes to map against every anchor box
    # for every anchor box, consider only the class of the gt bbox it overlaps with the most
    GT_class = torch.gather(gt_classes_expand, dim=-1, index=max_iou_per_anchor_index.unsqueeze(-1)).squeeze(-1)
    GT_class = GT_class.flatten(start_dim=0, end_dim=1)
    GT_class_pos = GT_class[pos_anchor_index]

    # get gt bbox coordinates of the pos anchors

    gt_boxes_expand = gt_boxes.view(B, 1, N, 4).expand(B, total_boxes, N, 4)
    GT_boxes = torch.gather(gt_boxes_expand, dim=-2, index=max_iou_per_anchor_index.reshape(B, total_boxes, 1, 1).repeat(1, 1, 1, 4))
    GT_boxes = GT_boxes.flatten(start_dim=0, end_dim=2)
    GT_boxes_pos = GT_boxes[pos_anchor_index]

    # get coordinates of pos anchors

    anchor_boxes_flat = anchor_boxes.flatten(start_dim=0, end_dim=-2)
    pos_anchor_coords = anchor_boxes_flat[pos_anchor_index]

    GT_offsets = cal_gt_offsets(pos_anchor_coords, GT_boxes_pos)

    # get neg anchors

    neg_anchor_mask = (max_iou_per_anchor < neg_thresh)
    neg_anchor_index = torch.where(neg_anchor_mask)[0] # already merged the batches
    neg_anchor_index = neg_anchor_index[torch.randint(0, neg_anchor_index.shape[0], pos_anchor_index.shape[0])]
    neg_anchor_coords = anchor_boxes_flat[neg_anchor_index]

    return pos_anchor_index, neg_anchor_index, GT_confidence_score, GT_offsets, GT_class_pos, pos_anchor_coords, neg_anchor_coords, pos_anchor_index_seperate

def generate_proposals(anchors, offsets):
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

    proposals_ = torch.zeros_like(anchors)

    proposals_[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
    proposals_[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
    proposals_[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
    proposals_[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])

    proposals_ = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

    return proposals_