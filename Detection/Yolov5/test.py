import torch
import torch.nn as nn
import numpy as np
import torchvision.ops as ops
import os
from pathlib import Path
import albumentations as A
import torch.cuda
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import cv2
from collections import Counter
from tqdm import tqdm
import torch.optim as optim

class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__int__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                                  nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
                                  nn.SiLU(inplace=True))
        
    def forward(self, x):
        return self.conv(x)

class BottleNeck1(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiple=1):
        super().__init__()
        mid_channels = int(in_channels * width_multiple)
        self.conv1 = CBL(in_channels, mid_channels, 1, 1, 0)
        self.conv2 = CBL(mid_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x0 = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)

        return x + x0
    
class BottleNeck2(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiple=1):
        super().__init__()
        mid_channels = int(in_channels * width_multiple)
        self.conv1 = CBL(in_channels, mid_channels, 1, 1, 0)
        self.conv2 = CBL(mid_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x
    
class C3(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiple, num_layers, backbone=True):
        super().__init__()
        mid_channels = int(in_channels * width_multiple)

        self.skipped_connection = CBL(in_channels, mid_channels, 1, 1, 0)
        self.layers = nn.ModuleList()
        self.layers.append(CBL(in_channels, mid_channels, 1, 1, 0))

        for _ in range(num_layers):
            if backbone:
                self.layers.append(BottleNeck1(mid_channels, mid_channels, width_multiple=1))
            
            else:
                self.layers.append(BottleNeck2(mid_channels, mid_channels, width_multiple=1))

        self.layer_out = CBL(mid_channels * 2, out_channels, 1, 1, 0)

    def forward(self, x):
        x0 = self.skipped_connection(x)
        for layer in self.layers:
            x = layer(x)

        x = torch.cat([x, x0], dim=1)

        return self.layer_out(x)
    
class C3_Neck(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiple, num_layers):
        super().__init__()
        mid_channels = int(in_channels * width_multiple)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        self.skipped_connection = CBL(in_channels, mid_channels, 1, 1, 0)
        self.layer_out = CBL(mid_channels * 2, out_channels, 1, 1, 0)
        self.silu_block = self._make_silu_block(num_layers)

    def _make_silu_block(self, num_layers):
        layers = []

        for i in range(num_layers):
            if i == 0:
                layers.append(CBL(self.in_channels, self.mid_channels, 1, 1, 0))
            
            elif i == 1:
                layers.append(CBL(self.mid_channels, self.mid_channels, 3, 1, 1))
            
            else:
                layers.append(CBL(self.mid_channels, self.mid_channels, 1, 1, 0))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x0 = self.skipped_connection(x)
        x = self.silu_block(x)

        x = torch.cat([x, x0], dim=1)

        return self.layer_out(x)
    
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = CBL(in_channels, mid_channels, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv2 = CBL(mid_channels * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)
        x = torch.cat([x, pool1, pool2, pool3], dim=1)

        return self.conv2(x)
    
class Head(nn.Module):
    def __init__(self, num_classes, anchors=(), ch=()):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(anchors)
        self.num_anchors_per_scale = len(anchors[0])

        self.stride = [8, 16, 32]
        anchors_ = torch.tensor(anchors).float().view(self.num_layers, -1, 2) / torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
        self.register_buffer("anchors", anchors_)
        self.out = nn.ModuleList()

        for in_channels in ch:
            self.out.append(nn.Conv2d(in_channels, (5 + num_classes) * self.num_anchors_per_scale, 1, 1, 0))

    def forward(self, x):
        for i in range(self.num_layers):
            x[i] = self.out[i](x[i])
            x[i] = x[i].view(x[i].shape[0], self.num_anchors_per_scale, 5 + self.num_classes, x[i].shape[2], x[i].shape[3]).permute(0, 1, 3, 4, 2).contiguous()

        return x
    
class Yolov5(nn.Module):
    def __init__(self, first_out, num_classes=80, anchors=(), ch=(), inference=False):
        super().__init__()
        self.inference = inference
        self.backbone = nn.ModuleList()
        self.backbone += [CBL(in_channels=3, out_channels=first_out, kernel_size=6, stride=2, padding=2),
                        CBL(in_channels=first_out, out_channels=first_out * 2, kernel_size=3, stride=2, padding=1),
                        C3(in_channels=first_out * 2, out_channels=first_out * 2, width_multiple=0.5, num_layers=2),
                        CBL(in_channels=first_out * 2, out_channels=first_out * 4, kernel_size=3, stride=2, padding=1),
                        C3(in_channels=first_out * 4, out_channels=first_out * 4, width_multiple=0.5, num_layers=4),
                        CBL(in_channels=first_out * 4, out_channels=first_out * 8, kernel_size=3, stride=2, padding=1),
                        C3(in_channels=first_out * 8, out_channels=first_out * 8, width_multiple=0.5, num_layers=6),
                        CBL(in_channels=first_out * 8, out_channels=first_out * 16, kernel_size=3, stride=2, padding=1),
                        C3(in_channels=first_out * 16, out_channels=first_out * 16, width_multiple=0.5, num_layers=2),
                        SPPF(in_channels=first_out * 16, out_channels=first_out * 16)]
        
        self.neck = nn.ModuleList()
        self.neck += [CBL(in_channels=first_out * 16, out_channels=first_out * 8, kernel_size=1, stride=1, padding=0),
                    C3(in_channels=first_out * 16, out_channels=first_out * 8, width_multiple=0.25, num_layers=2, backbone=False),
                    CBL(in_channels=first_out * 8, out_channels=first_out * 4, kernel_size=1, stride=1, padding=0),
                    C3(in_channels=first_out * 8, out_channels=first_out * 4, width_multiple=0.25, num_layers=2, backbone=False),
                    CBL(in_channels=first_out * 4, out_channels=first_out * 4, kernel_size=3, stride=2, padding=1),
                    C3(in_channels=first_out * 8, out_channels=first_out * 8, width_multiple=0.5, num_layers=2, backbone=False),
                    CBL(in_channels=first_out * 8, out_channels=first_out * 8, kernel_size=3, stride=2, padding=1),
                    C3(in_channels=first_out * 16, out_channels=first_out * 16, width_multiple=0.5, num_layers=2, backbone=False)]
        
        self.head = Head(num_classes, anchors, ch)

    def forward(self, x):
        backbone_connections = []
        neck_connections = []
        outputs = []

        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx == 4 or idx == 6:
                backbone_connections.append(x)

        for idx, layer in enumerate(self.neck):
            x = layer(x)

            if idx == 0 or idx == 2:
                neck_connections.append(x)
                x = nn.Upsample(scale_factor=2)(x)
                x = torch.cat([x, backbone_connections[-1]], dim=1)
                backbone_connections.pop()

            if idx == 4 or idx == 6:
                x = torch.cat([x, neck_connections[-1]], dim=1)
                neck_connections.pop()

            if idx == 3 or idx == 5 or idx == 7:
                outputs.append(x)

        return self.head(outputs)
    
def iou_width_height(gt_boxes, anchors, stride_anchors=True, stride=[8, 16, 32]):
    anchors /= 640
    if stride_anchors:
        anchors = anchors.reshape(9, 2) * torch.tensor(stride).repeat(6, 1).T.reshape(9, 2)

    intersection = torch.min(gt_boxes[..., 0], anchors[..., 0]) * torch.min(gt_boxes[..., 1], anchors[..., 1])
    union = gt_boxes[..., 0] * gt_boxes[..., 1] + anchors[..., 0] * anchors[..., 1] - intersection

    return intersection / (union + 1e-7)

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint", GIOU=True):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
        box1_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
        box1_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
        box1_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2

        box2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
        box2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
        box2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
        box2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    else:
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
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection

    iou = intersection / (union + 1e-7)

    if GIOU:
        cw = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
        ch = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        c_area = cw * ch + 1e-7

        return iou - (c_area - union) / c_area
    
    return iou

def non_max_surpression(boxes, iou_thresh, thresh, box_format="corners", max_detection=300):
    boxes = [box for box in boxes if box[1] > thresh]
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    if len(boxes) > max_detection:
        boxes = boxes[:max_detection]

    boxes_after_nms = []

    while boxes:
        chosen_box = boxes.pop()
        boxes_after_nms.append(chosen_box)

        boxes = [box for box in boxes \
                 if box[0] != chosen_box[0] \
                    or intersection_over_union(torch.tensor(box[2:]), torch.tensor(chosen_box[2:])) < iou_thresh]
        
    return boxes_after_nms

def nms(batch_boxes, iou_thresh, thresh, max_detection=300, tolist=True):
    # batch_boxes: batch_size x boxes_per_image x 6
    boxes_after_nms = []

    for boxes in batch_boxes:
        boxes = [box for box in boxes if box[1] > thresh].reshape(-1, 6)

        # cxcywh to xyxy
        boxes[:, 2] = boxes[:, 2] - boxes[:, 4] / 2
        boxes[:, 3] = boxes[:, 3] - boxes[:, 5] / 2
        boxes[:, 4] = boxes[:, 4] + boxes[:, 2]
        boxes[:, 5] = boxes[:, 5] + boxes[:, 3]

        # class labels are taken into account
        keep = ops.nms(boxes[:, 2:] + boxes[:, 0], boxes[:, 1], iou_thresh)
        boxes = boxes[keep]

        if boxes.shape[0] > max_detection:
            boxes = boxes[:max_detection, :]

        boxes_after_nms.append(boxes.tolist() if tolist else boxes)

    return boxes_after_nms if tolist else torch.cat(boxes_after_nms, dim=0)

def coco_to_yolo(box, image_w=640, image_h=640):
    # left corner, w, h -> midpoint, w, h, also rescale to relative size to the image
    x, y, w, h = box
    new_x = (x + w / 2) / image_w
    new_y = (y + h / 2) / image_h
    new_h = h / image_h
    new_w = w / image_w

    return [new_x, new_y, new_w, new_h]

def coco_to_yolo_tensors(box, image_w=640, image_h=640):
    x, y, w, h = np.split(box, 4, 1)
    new_x = (x + w / 2) / image_w
    new_y = (y + h / 2) / image_h
    new_h = h / image_h
    new_w = w / image_w

    return np.concatenate([new_x, new_y, new_w, new_h], axis=1)

def rescale_box(boxes, starting_size, ending_size):
    sw, sh = starting_size
    ew, eh = ending_size
    y = np.copy(boxes)

    y[:, 0] = np.floor(y[:, 0] / sw * ew * 100) / 100
    y[:, 1] = np.floor(y[:, 1] / sh * eh * 100) / 100
    y[:, 2] = np.floor(y[:, 2] / sw * ew * 100) / 100
    y[:, 3] = np.floor(y[:, 3] / sh * eh * 100) / 100

    return y

def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])

parent_dir = Path(__file__).parent.parent
ROOT_DIR = os.path.join(parent_dir, "datasets", "coco")

# if no yaml file, this must be manually inserted
# nc is number of classes (int)
nc = None
# list containing the labels of classes: i.e. ["cat", "dog"]
labels = None

FIRST_OUT = 48

CLS_PW = 1.0
OBJ_PW = 1.0

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 640

CONF_THRESHOLD = 0.01  # to get all possible bboxes, trade-off metrics/speed --> we choose metrics
NMS_IOU_THRESH = 0.6
# for map 50
MAP_IOU_THRESH = 0.5

# triple check what anchors REALLY are

ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32#
]


TRAIN_TRANSFORMS = A.Compose(
    [
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.4),
        A.Transpose(p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-20, 20), p=0.7),
        A.Blur(p=0.05),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ChannelShuffle(p=0.05),
    ],
    bbox_params=A.BboxParams("yolo", min_visibility=0.4, label_fields=[],),
)

FLIR = [
    'car',
    'person'
]

COCO = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

nc = len(COCO)
labels = COCO

def cells_to_boxes(predictions, anchors, S, is_preds=True):
    batch_size = predictions.shape[0]
    anchors = anchors.reshape(1, 3, 1, 1, 2)
    predicted_boxes = predictions[..., 1:5]

    if is_preds:
        scores = torch.sigmoid(predictions[..., 0])
        best_class = torch.argmax(predictions[..., 5:], dim=-1)
        predicted_boxes[..., 0:2] = 2 * torch.sigmoid(predicted_boxes[..., 0:2]) - 0.5
        predicted_boxes[..., 2:4] = anchors * (2 * torch.sigmoid(predicted_boxes[..., 2:4])) ** 2

    else:
        scores = predictions[..., 0]
        best_class = predictions[..., 5]

    cell_indices = torch.arange(S).repeat(predictions.shape[0], 3, S, 1).unsqueeze(-1).to(predictions.device)
    x = 1 / S * (predicted_boxes[..., 0] * cell_indices)
    y = 1 / S * (predicted_boxes[..., 1] * cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * predicted_boxes[..., 2:4]
    converted_boxes = torch.cat([best_class, scores, x, y, w_h], dim=-1).reshape(batch_size, -1, 6)

    return converted_boxes.tolist()

def plot_image(image, boxes):
    cmap = plt.get_cmap("tab20b")
    class_labels = COCO
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    width, height, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        class_preds = box[0]
        scores = torch.sigmoid(box[1])
        box = box[2:]
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (x1 * width, y1 * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_labels)],
            facecolor="none"
        )
        ax.add_patch(rect)

        plt.text(
            x1 * width, 
            y1 * height,
            s=f"{class_labels[int(class_preds)]}: {round(scores * 100, 2)}%",
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_preds)], "pad": 0}
        )
    
    plt.axis('off')
    plt.show()

class YoloLoss():
    def __init__(self, model):
        self.mse = nn.MSELoss()
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(CLS_PW))
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(OBJ_PW))
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 0.5 * (model.head.num_classes / 80 * 3 / model.head.num_layers)
        self.lambda_obj = ((IMAGE_SIZE / 640) ** 2 * 3 / model.head.num_layers)
        self.lambda_box = 0.05 * (3 / model.head.num_layers)

        self.num_classes = model.head.num_classes
        self.anchors_d = model.head.anchors.clone().detach()
        self.anchors = self.anchors_d.to('cpu')

        self.balance = [4.0, 1.0, 4.0]

        self.num_anchors = 9
        self.num_anchors_per_scale = 3
        self.stride = model.head.stride
        self.S = [80, 40, 20]
        self.ignore_iou_threshold = 0.5

    def __call__(self, preds, targets):
        targets = [self.build_targets(preds, boxes) for boxes in targets]

        t1 = torch.stack([target[0] for target in targets], dim=0).to(DEVICE, non_blocking=True)
        t2 = torch.stack([target[1] for target in targets], dim=0).to(DEVICE, non_blocking=True)
        t3 = torch.stack([target[2] for target in targets], dim=0).to(DEVICE, non_blocking=True)

        loss = self.compute_loss(preds[0], t1, self.anchors_d[0], balance=self.balance[0]) + \
                self.compute_loss(preds[1], t2, self.anchors_d[1], balance=self.balance[1]) + \
                self.compute_loss(preds[2], t3, self.anchors_d[2], balance=self.balance[2])
        
        return loss
    
    def build_targets(self, preds, boxes):
        targets = [torch.zeros(self.num_anchors_per_scale, preds[i].shape[2], preds[i].shape[3], 6) for i in len(self.stride)]

        for box in boxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors_d)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchors = [False] * 3

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchors[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = width * S, height * S
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

        iou = intersection_over_union(pred_boxes, gt_boxes)
        box_loss = (1.0 - iou).mean()

        iou = iou.detach().clamp(0)
        targets[..., 0][obj] *= iou

        obj_loss = self.bce_obj(preds[..., 0], targets[..., 0]) * balance

        target_class = torch.zeros_like(preds[..., 5:][obj], device=DEVICE)
        target_class[torch.arange(target_class.shape[0]), targets[..., 5][obj].long()] = 1.0

        class_loss = self.bce_cls(target_class, preds[..., 5:][obj])

        return batch_size * (self.lambda_box * box_loss + self.lambda_class * class_loss + self.lambda_obj * obj_loss)

def multi_scale(image, target_shape, max_stride=32):
    size = random.randrange(target_shape * 0.5, target_shape + max_stride) // max_stride * max_stride
    sf = size / max(image.shape[2:])
    h, w = image.shape[2:]

    # still don't know why they do this
    ns = [math.ceil(i * sf / max_stride) * max_stride for i in [h, w]] 
    image = F.interpolate(image, ns, mode="bilinear", align_corners=False)

    return image

def resize_image(image, output_size):
    return cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)

class YoloDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, rect_training=False, default_size=640, batch_size=32, boxes_format='coco'):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.rect_training = rect_training
        self.default_size = default_size
        self.batch_size = batch_size
        self.boxes_format = boxes_format

        if train:
            fname = 'images/train'
            annot_file = 'annot_train.csv'
            self.annot_folder = "train"

        else:
            fname = 'images/val'
            annot_file = "annot_val.csv"
            self.annot_folder = "val"

        self.fname = fname
        self.annotations = pd.read_csv(os.path.join(root_dir, "labels", annot_file), header=None, index_col=0).sort_values(by=[0])
        self.annotations = self.annotations.head(len(self.annotations) - 1)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_name = self.annotations.iloc[index, 0]
        label_path = os.path.join(self.root_dir, "labels", self.annot_folder, image_name[:-4] + "txt")
        tg_width, tg_height = self.annotations.iloc[index, 1] if self.rect_training else 640, self.annotations.iloc[index, 2] if self.rect_training else 640
        labels = np.loadtxt(label_path, delimiter=" ", ndmin=2)
        labels = labels[np.all(labels >= 0, axis=1), :]
        labels[:, 3:5] = np.floor(labels[:, 3:5] * 1000) / 1000

        image = np.array(Image.open(os.path.join(self.root_dir, self.fname, image_name)).convert('RGB'))

        if self.boxes_format == 'coco':
            # labels: batch x (x, y, w, h, class)
            labels[:, -1] -= 1
            # -> class, x, y, w, h
            labels = np.roll(labels, shift=1, axis=1)
            labels[:, 1:] = coco_to_yolo_tensors(labels[:, 1:], image.shape[1], image.shape[0])

        image = resize_image(image, (tg_width, tg_height))

        if self.transform:
            batch_n = index // self.batch_size
            if batch_n % 2 == 0:
                self.transform[1].p = 1
            else:
                self.transform[1].p = 0

            augmentations = self.transform(image=image, bboxes=np.roll(labels, shift=4, axis=1))
            image = augmentations["image"]
            labels = np.array(augmentations["bboxes"])
            
        image = image.permute(2, 0, 1)
        image = np.ascontiguousarray(image)

        return torch.tensor(image), labels
    
class YoloDataset_Val(Dataset):
    def __init__(self, root_dir, anchors, transform=None, train=False, S=(8, 16, 32), rect_training=False, default_size=640, batch_size=32, boxes_format='coco'):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.rect_training = rect_training
        self.default_size = default_size
        self.batch_size = batch_size
        self.boxes_format = boxes_format
        self.S = S
        self.num_layers = 3
        self.anchors = torch.tensor(anchors).float().view(3, 3, 2) / torch.tensor(S).repeat(6, 1).T.reshape(3, 3, 2)
        self.num_anchors = 9
        self.num_anchors_per_scale = 3
        self.ignore_iou_threshold = 0.5

        if train:
            fname = 'images/train'
            annot_file = 'annot_train.csv'
            self.annot_folder = "train"

        else:
            fname = 'images/train'
            annot_file = 'annot_val.csv'
            self.annot_folder = "val"

        self.fname = fname
        self.annotations = pd.read_csv(os.path.join(root_dir, "labels", annot_file), header=None, index_col=0).sort_values(by=[0])
        self.annotations = self.annotations.head(len(self.annotations) - 1)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_name = self.annotations.iloc[index, 0]
        label_path = os.path.join(self.root_dir, "labels", image_name[:-4] + "txt")
        tg_width, tg_height = self.annotations.iloc[index, 1] if self.rect_training else 640, self.annotations.iloc[index, 2] if self.rect_training else 640
        labels = np.loadtxt(label_path, delimiter=" ", ndmin=2)
        labels = labels[np.all(labels >= 0, axis=1), :]
        labels[:, 3:5] = np.floor(labels[:, 3:5] * 1000) / 1000

        image = np.array(Image.open(os.path.join(self.root_dir, self.fname, image_name)).convert('RGB'))

        if self.boxes_format == 'coco':
            labels[:, -1] -= 1
            labels = np.roll(labels, shift=1, axis=1)
            labels[:, 1:] = coco_to_yolo_tensors(labels[:, 1:], image.shape[1], image.shape[0])

        image = resize_image(image, (tg_width, tg_height))

        if self.transform:
            batch_n = index // self.batch_size
            if batch_n % 2 == 0:
                self.transform[1].p = 1
            else:
                self.transform[1].p = 0

            augmentations = self.transform(image=image, bboxes=np.roll(labels, shift=4, axis=1))
            image = augmentations["image"]
            labels = np.array(augmentations["bboxes"])

        image = image.permute(2, 0, 1)
        image = np.ascontiguousarray(image)

        targets = [torch.zeros(self.num_anchors_per_scale, int(image.shape[0] / S), int(image.shape[1] / S), 6) for S in self.S]

        for box in labels:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            class_label, x, y, width, height = box
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                S_y = int(image.shape[1] / S)
                S_x = int(image.shape[2] / S)
                i, j = int(S_y * y), int(S_x * x)

                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S_x * x - j, S_y * y - i
                    width_cell, height_cell = S_x * width, S_y * height
                    box_coords = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coords
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_threshold:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return torch.tensor(image), tuple(targets)
    
def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=80):
    average_precision = []
    epsilon = 1e-7

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_boxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_boxes:
            amount_boxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_boxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            gt_in_same_image = [box for box in ground_truths if box[0] == detection[0]]

            best_iou = 0
            for gt_idx, gt_box in enumerate(gt_in_same_image):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt_box[3:]), box_format, False)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou > iou_threshold:
                if amount_boxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_boxes[detection[0]][best_gt_idx] = 1
                
                else:
                    FP[detection_idx] = 1
            
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_boxes + epsilon)
        recalls = torch.cat([torch.tensor([0]), recalls])
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat([torch.tensor([1]), precisions])

        average_precision.append(torch.trapz(precisions, recalls))

    return sum(average_precision) / len(average_precision)

def get_evaluation_boxes(loader, model, iou_threshold, anchors, threshold, box_format="midpoint", device=DEVICE):
    model.eval()
    train_idx = 0
    pred_boxes = []
    true_boxes = []

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)

        with torch.no_grad():
            out = model(x)

        batch_size = x.shape[0]
        boxes = [[] for _ in range(batch_size)]

        for i in range(3):
            S = out[i].shape[2]
            anchor = torch.tensor([*anchors]).to(device) * S
            boxes_scale_i = cells_to_boxes(out[i], anchor, S, True)

            for idx, box in enumerate(boxes_scale_i):
                boxes[idx] += box

        gt_boxes = cells_to_boxes(labels[2], anchor, S, False)

        for i in range(batch_size):
            nms_boxes = non_max_surpression(boxes[i], iou_threshold, threshold, box_format)

            for nms_box in nms_boxes:
                pred_boxes.append([train_idx] + nms_box)

            for box in gt_boxes[i]:
                if box[1] > threshold:
                    true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()

    return pred_boxes, true_boxes

def check_class_accuracy(model, loader, threshold):
    model.eval()
    total_class_preds, correct_class = 0, 0
    total_obj_preds, correct_obj = 0, 0
    total_noobj_preds, correct_noobj = 0, 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(DEVICE)

        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(DEVICE)
            obj = y[i][..., 0] == 1
            noobj = y[i][..., 0] == 0

            correct_class += torch.sum(torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj])
            total_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            total_obj_preds += torch.sum(obj)

            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            total_noobj_preds += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(total_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(total_noobj_preds+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(total_obj_preds+1e-16))*100:2f}%")

    model.train()

def get_mean_std(loader):
    channels_sum, channels_sum_sqrd, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sum_sqrd += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sum_sqrd / num_batches - mean ** 2) ** 0.5

    return mean, std

def get_loaders(root_dir):
    S = [8, 16, 32]
    train_augmentation = TRAIN_TRANSFORMS
    val_augmentation = None

    train_dataset = YoloDataset(root_dir, train_augmentation)
    val_dataset = YoloDataset_Val(root_dir, ANCHORS, val_augmentation, False, S)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader

torch.backends.cudnn.benchmark = True

def train(model, loader, optimizer, loss_fn, scaler, epoch, num_epochs, multi_scale_training=True):
    print(f"Training epoch {epoch}/{num_epochs}")
    batch_size = 32
    accumulate = 1
    last_opt_step = -1

    loop = tqdm(loader)
    loss_epoch = 0
    nb = len(loader)
    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(loop):
        images /= 255.0
        if multi_scale_training:
            images = multi_scale(images, 640, 32)

        images = images.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast():
            out = model(images)
            loss = loss_fn(out, targets)
            loss_epoch += loss

        scaler.scale(loss).backward()

        if batch_idx - last_opt_step >= accumulate or (batch_idx == nb-1):
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            last_opt_step = batch_idx

        freq = 10
        if batch_idx % freq == 0:
            loop.set_postfix(average_loss_batches=avg_batches_loss.item() / freq)
            avg_batches_loss = 0

        print(f"==> training_loss: {(loss_epoch.item() / nb):.2f}")

scaler = torch.cuda.amp.GradScaler()
model = Yolov5(FIRST_OUT, anchors=ANCHORS, ch=(FIRST_OUT * 4, FIRST_OUT * 8, FIRST_OUT * 16)).to(DEVICE)
optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train_loader, val_loader = get_loaders(ROOT_DIR)
loss_fn = YoloLoss(model)

for epoch in range(20):
    train(model, train_loader, optimizer, loss_fn, scaler, epoch, 20)
    model.eval()

    check_class_accuracy(model, val_loader, CONF_THRESHOLD)
    pred_boxes, true_boxes = get_evaluation_boxes(val_loader, model, NMS_IOU_THRESH, ANCHORS, CONF_THRESHOLD)
    MAPval = mean_average_precision(pred_boxes, true_boxes, MAP_IOU_THRESH)

    print(f"MAP: {MAPval.item()}")
    model.train()

    