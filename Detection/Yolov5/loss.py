import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from box_utils import *
from plot_utils import *

class YoloLoss():
    def __init__(self, model):

        self.mse = nn.MSELoss()
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.CLS_PW))
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.OBJ_PW))
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 0.5 * (model.head.num_classes / 80 * 3 / model.head.num_layers)
        self.lambda_obj = ((config.IMAGE_SIZE / 640) ** 2 * 3 / model.head.num_layers)
        self.lambda_box = 0.05 * (3 / model.head.num_layers)

        self.num_classes = model.head.num_classes
        self.anchors_d = model.head.anchors.clone().detach()
        self.anchors = self.anchors_d.to('cpu')

        self.balance = [4.0, 1.0, 4.0]

        self.num_anchors = 9
        self.num_anchors_per_scale = 3
        self.stride = model.head.stride
        self.S = [640 / 8, 640 / 16, 640 / 32]
        self.ignore_iou_threshold = 0.5

    def __call__(self, preds, targets):
        targets = [self.build_targets(preds, boxes) for boxes in targets]

        t1 = torch.stack([target[0] for target in targets], dim=0).to(config.DEVICE, non_blocking=True)
        t2 = torch.stack([target[1] for target in targets], dim=0).to(config.DEVICE, non_blocking=True)
        t3 = torch.stack([target[2] for target in targets], dim=0).to(config.DEVICE, non_blocking=True)

        loss = self.compute_loss(preds[0], t1, anchors=self.anchors_d[0], balance=self.balance[0]) + \
                self.compute_loss(preds[1], t2, anchors=self.anchors_d[1], balance=self.balance[1]) + \
                self.compute_loss(preds[2], t3, anchors=self.anchors_d[2], balance=self.balance[2])
        
        return loss
    
    def build_targets(self, preds, boxes):
        targets = [torch.zeros(self.num_anchors_per_scale, preds[i].shape[2], preds[i].shape[3], 6) for i in range(len(self.S))]

        for box in boxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors_d)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchors = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors
                anchor_on_scale = anchor_idx % self.num_anchors
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchors[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = S * width, S * height
                    box_coords = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coords
                    targets[scale_idx][anchor_on_scale, i, j, 5] = class_label

                    has_anchors[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_threshold:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return tuple(targets)
    
    def compute_loss(self, preds, targets, anchors, balance):
        batch_size = preds.shape[0]
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        obj = targets[..., 0] == 1

        preds[..., 1:3] = torch.sigmoid(preds[..., 1:3]) * 2 - 0.5
        preds[..., 3:5] = anchors * (torch.sigmoid(preds[..., 3:5]) * 2) ** 2 
        gt_boxes = targets[..., 1:5][obj]
        pred_boxes = preds[..., 1:5][obj]

        iou = intersection_over_union(pred_boxes, gt_boxes, GIOU=True).squeeze()
        box_loss = (1.0 - iou).mean()

        iou = iou.detach().clamp(0)
        targets[..., 0][obj] *= iou

        # noobj and obj loss combined
        obj_loss = self.bce_obj(preds[..., 0], targets[..., 0]) * balance

        target_class = torch.zeros_like(preds[..., 5:][obj], device=config.DEVICE)
        target_class[torch.arange(target_class.shape[0]), targets[..., 5][obj].long()] = 1.0

        class_loss = self.bce_cls(preds[..., 5:][obj], target_class)

        return batch_size * (self.lambda_class * class_loss + self.lambda_box * box_loss + self.lambda_obj * obj_loss)