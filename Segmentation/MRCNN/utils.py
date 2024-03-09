import torch
import torchvision.ops as ops

class Matcher:
    def __init__(self, pos_thresh, neg_thresh, allow_low_quality_matches=False):
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou_mat):
        # iou_mat: total_anchor_boxes x num_gt_boxes
        max_iou_per_anchor_box, matched_index = iou_mat.max(dim=0) 
        labels = torch.full(iou_mat.shape[1], -1, dtype=torch.float, device=iou_mat.device)

        labels[max_iou_per_anchor_box >= self.pos_thresh] = 1
        labels[max_iou_per_anchor_box < self.neg_thresh] = 0

        if self.allow_low_quality_matches:
            highest_quality = iou_mat.max(dim=1)[0]
            gt_pred_pairs = torch.where(iou_mat == highest_quality)[1]
            labels[gt_pred_pairs] = 1

        return labels, matched_index
    
class PosNegSampler:
    def __init__(self, num_samples, pos_fraction):
        self.num_samples = num_samples
        self.pos_fraction = pos_fraction

    def __call__(self, labels):
        pos = torch.where(labels == 1)[0]
        neg = torch.where(labels == 0)[0]

        num_pos = int(self.num_samples * self.pos_fraction)
        num_pos = min(pos.numel(), num_pos)
        num_neg = self.num_samples - num_pos
        num_neg = min(neg.numel(), num_neg)

        pos_idx = torch.randperm(pos.numel(), device=pos.device)[:num_pos]
        neg_idx = torch.randperm(neg.numel(), device=neg.device)[:num_neg]

        pos = pos[pos_idx]
        neg = neg[neg_idx]

        return pos, neg
    
def roi_align(features, rois, spatial_scale, pooled_heigth, pooled_width, sampling_ratio):
    return ops.roi_align(features, rois, (pooled_heigth, pooled_width), spatial_scale, sampling_ratio)

class AnchorGenerator:
    def __init__(self, scales, ratios):
        self.scales = scales
        self.ratios = ratios

        self.cell_anchor = None
        self._cache = {}

    def set_cell_anchor(self, dtype, device):
        if self.cell_anchor is not None:
            return
        scales = torch.tensor(self.scales, dtype=dtype, device=device)
        ratios = torch.tensor(self.ratios, dtype, device)

        h = torch.sqrt(ratios)
        w = 1 / h

        hs = (scales[:, None] * h[None, :]).view(-1) # torch.matmul(scales, h.T)
        ws = (scales[:, None] * w[None, :]).view(-1)

        self.cell_anchor = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

    def grid_anchor(self, grid_size, stride):
        dtype, device = self.cell_anchor.dtype, self.cell_anchor.device
        shift_x = torch.arange(0, grid_size[1], dtype=dtype, device=device) * stride[1]
        shift_y = torch.arange(0, grid_size[0], dtype=dtype, device=device) * stride[0]

        y, x = torch.meshgrid(shift_y, shift_x)
        x = x.reshape(-1)
        y = y.reshape(-1)
        shift = torch.stack([x, y, x, y], dim=1).reshape(-1, 1, 4)

        anchor = (shift + self.cell_anchor).reshape(-1, 4)

        return anchor
    
    def cached_grid_anchor(self, grid_size, stride):
        key = grid_size + stride
        if key in self._cache:
            return self._cache[key]
        anchor = self.grid_anchor(grid_size, stride)

        if len(self._cache) >= 3:
            self._cache.clear()
        self._cache[key] = anchor

        return anchor
    
    def __call__(self, feature, image_size):
        dtype, device = feature.dtype, feature.device
        grid_size = tuple(feature.shape[-2:])
        stride = tuple(int(i / g) for i, g in zip(image_size, grid_size)) # ratio image / feature_map
        self.set_cell_anchor(dtype, device)
        anchor = self.cached_grid_anchor(grid_size, stride)

        return anchor