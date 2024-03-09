import torch
import math
import torchvision.ops as ops

class BoxCoder:
    def __init__(self, weights, box_from_clip=math.log(1000. / 16)):
        self.weights = weights
        self.box_from_clip = box_from_clip

    def encode(self, gt_boxes, proposals):
        # compute offsets
        width = proposals[:, 2] - proposals[:, 0]
        height = proposals[:, 3] - proposals[:, 1]
        x = proposals[:, 0] + width / 2
        y = proposals[:, 1] + height / 2

        gt_width = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_height = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_x = gt_boxes[:, 0] + gt_width / 2
        gt_y = gt_boxes[:, 1] + gt_height / 2

        tx = self.weights[0] * (gt_x - x) / width
        ty = self.weights[1] * (gt_y - y) / height
        tw = self.weights[2] * torch.log(gt_width / width)
        th = self.weights[3] * torch.log(gt_height / height)

        offsets = torch.stack([tx, ty, tw, th], dim=1)

        return offsets
    
    def decode(self, boxes, offsets):
        # compute correct boxes
        tx = offsets[:, 0] / self.weights[0]
        ty = offsets[:, 1] / self.weights[1]
        tw = offsets[:, 2] / self.weights[2]
        th = offsets[:, 3] / self.weights[3]

        tw = torch.clamp(tw, max=self.box_from_clip)
        th = torch.clamp(th, max=self.box_from_clip)

        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        x = boxes[:, 0] + width / 2
        y = boxes[:, 1] + height / 2

        x = tx * width + x
        y = ty * height + y
        width = torch.exp(tw) * width
        height = torch.exp(th) * height

        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        return boxes
    
def box_iou(box1, box2):
    return ops.box_iou(box1, box2)

def process_box(boxes, scores, image_size, min_size):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image_size[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image_size[0])

    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = torch.where((w >= min_size) & (h >= min_size))[0]
    boxes, scores = boxes[keep], scores[keep]

    return boxes, scores

def nms(boxes, scores, thresh):
    return ops.nms(boxes, scores, thresh)