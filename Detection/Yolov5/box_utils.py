import torch
import torchvision.ops as ops
import numpy as np

def iou_width_height(gt_boxes, anchors, stride_anchors=True, stride=[8, 16, 32]):
    # stride_anchors: if anchors are already divided by the stride in class Head
    
    # rescale anchors to relative size of the ori image
    anchors /= 640

    if stride_anchors:
        anchors = anchors.reshape(9, 2) * torch.tensor(stride).repeat(6, 1).T.reshape(9, 2)

    intersection = torch.min(gt_boxes[..., 0], anchors[..., 0]) * torch.min(gt_boxes[..., 1], anchors[..., 1])
    union = (gt_boxes[..., 0] * gt_boxes[..., 1] + anchors[..., 0] * anchors[..., 1] - intersection)

    return intersection / union

def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint', GIOU=False):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection + 1e-6

    iou = intersection / union

    if GIOU:
        cw = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
        ch = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        c_area = cw * ch + 1e-6
        
        return iou - (c_area - union) / c_area
    
    return iou

def non_max_surpression(boxes, iou_thresh, thresh, box_format='corners', max_detection=300):
    assert type(boxes) == list

    boxes = [box for box in boxes if box[1] > thresh]
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    if len(boxes) > max_detection:
        boxes = boxes[:max_detection]

    boxes_after_nms = []

    while boxes:
        chosen_box = boxes.pop()
        boxes_after_nms.append(chosen_box)

        boxes = [box for box in boxes if box[0] != chosen_box[0] or intersection_over_union(torch.tensor(box[2:]), torch.tensor(chosen_box[2:], box_format) < iou_thresh)]

    return boxes_after_nms    

def nms(batch_boxes, iou_thresh, thresh, max_detections=300, tolist=True):
    # batch_boxes: batch_size x num_boxes_per_image x 6
    boxes_after_nms = []
    
    for boxes in batch_boxes:
        boxes = torch.masked_select(boxes, boxes[..., 1:2] > thresh).reshape(-1, 6)

        # xyxy to cxcywh
        boxes[..., 2:3] = boxes[..., 2:3] - (boxes[..., 4:5] / 2)
        boxes[..., 3:4] = boxes[..., 3:4] - (boxes[..., 5:6] / 2)
        boxes[..., 5:6] = boxes[..., 5:6] + boxes[..., 3:4]
        boxes[..., 4:5] = boxes[..., 4:5] + boxes[..., 2:3]

        keep = ops.nms(boxes[..., 2:] + boxes[..., 0:1], boxes[..., 1:2], iou_thresh)
        boxes = boxes[keep]

        if boxes.shape[0] > max_detections:
            boxes = boxes[:max_detections, :]
        
        boxes_after_nms.append(boxes.tolist() if tolist else boxes)

    return boxes_after_nms if tolist else torch.cat(boxes_after_nms, dim=0)


# coco_labels_shape: left upper corners and width, height of the image -> mid point
def coco_to_yolo(box, image_w=640, image_h=640):
    x, y, w, h = box
    return [(2 * x + w) / (2 * image_w), (2 * y + h) / (2 * image_h), w / image_w, h / image_h]

def coco_to_yolo_tensors(box, image_w=640, image_h=640):
    x, y, w, h = np.split(box, 4, axis=1)
    return np.concatenate([(2 * x + w) / (2 * image_w), (2 * y + h) / (2 * image_h), w / image_w, h / image_h], axis=1)

def rescale_boxes(boxes, starting_size, ending_size):
    sw, sh = starting_size
    ew, eh = ending_size
    y = np.copy(boxes)

    y[:, 0:1] = np.floor(boxes[:, 0:1] * ew / sw * 100)/100
    y[:, 1:2] = np.floor(boxes[:, 1:2] * eh / sh * 100)/100
    y[:, 2:3] = np.floor(boxes[:, 2:3] * ew / sw * 100)/100
    y[:, 3:4] = np.floor(boxes[:, 3:4] * eh / sh * 100)/100

    return y

def clip_boxes(boxes, shape):
    # boxes: xyxy
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])