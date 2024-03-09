import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from tqdm.auto import tqdm
import torch.optim as optim

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            return self.relu(self.bn(self.conv(x)))
        
        return self.conv(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, num_repeats, use_res=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_repeats):
            self.layers += [
                nn.Sequential(CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                              CNNBlock(in_channels // 2, in_channels, kernel_size=3, padding=1))
            ]
        self.use_res = use_res
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_res:
                x = layer(x) + x
            else:
                x = layer(x)

        return x
    
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(CNNBlock(in_channels, in_channels * 2, kernel_size=3, padding=1),
                                  CNNBlock(in_channels * 2, (5 + num_classes) * 3, kernel_size=1, use_bn=False))
        
    def forward(self, x):
        return self.conv(x).reshape(x.shape[0], 3, 5 + self.num_classes, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
    
class Yolov3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self._make_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            if isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _make_conv_layers(self):
        in_channels = self.in_channels
        layers = nn.ModuleList()

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNNBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1 if kernel_size == 3 else 0))
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResBlock(in_channels, num_repeats))

            else:
                if module == "S":
                    layers += [ResBlock(in_channels, 1, False), 
                               CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                               ScalePrediction(in_channels // 2, self.num_classes)]
                    in_channels = in_channels // 2

                else:
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
        
        return layers
    
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "conners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    upper_corner_x = torch.max(box1_x1, box2_x1)
    upper_corner_y = torch.max(box1_y1, box2_y1)

    lower_corner_x = torch.min(box1_x2, box2_x2)
    lower_corner_y = torch.min(box1_y2, box2_y2)

    intersection = torch.max(0, (lower_corner_x - upper_corner_x)) * torch.max(0, (lower_corner_y - upper_corner_y))

    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union = area1 + area2 - intersection

    return intersection / (union + 1e-8)

def iou_width_height(boxes1, boxes2):
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union

def nms(bboxes, iou_threshold, threshold, box_formats='conners'):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_formats=box_formats
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

class YoloLoss(nn.Module):
    def __init__(self):
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, targets, anchors):
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        obj = targets[..., 0] == 1
        noobj = targets[..., 0] == 0

        no_obj_loss = self.bce(predictions[..., 0][noobj], targets[..., 0][noobj])

        box_preds = torch.cat([torch.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        iou = intersection_over_union(box_preds, targets[..., 1:5][obj]).detach()
        obj_loss = self.bce(predictions[..., 0][obj], targets[..., 0][obj] * iou)

        predictions[..., 1:3] = torch.sigmoid(predictions[..., 1:3])
        predictions[..., 3:5] = torch.log(targets[..., 3:5] / anchors)
        box_loss = self.mse(predictions[..., 1:5][obj], targets[..., 1:5][obj])

        class_loss = self.entropy(predictions[..., 5:][obj], targets[..., 5][obj].long())

        return self.lambda_box * box_loss + self.lambda_class * class_loss + self.lambda_noobj * no_obj_loss + self.lambda_obj * obj_loss
    
class YoloDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size=416, S=[13, 26, 52], C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.image_size = image_size
        self.S = S
        self.C = C
        self.transform = transform
        self.num_anchors = len(self.anchors)
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_threshold = 0.5

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        image = np.array(Image.open(image_path).convert('RGB'))
        boxes = np.roll(np.loadtxt(label_path, delimiter=" ", ndmin=2), shift=4, axis=1)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=boxes)
            image = augmentations["image"]
            boxes = augmentations["boxes"]

        targets = [torch.zeros(self.num_anchors_per_scale, S, S, 6) for S in self.S]

        for box in boxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchors = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
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

                elif not anchor_on_scale and iou_anchors[anchor_idx] > self.ignore_iou_threshold:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)
    
import albumentations as A
import cv2
import torch
import os
import numpy as np
import random

from albumentations.pytorch import ToTensorV2

DATASET = 'PASCAL_VOC'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0
NUM_EPOCHS = 5
CONF_THRESHOLD = 0.4
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]


scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                # A.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

COCO_LABELS = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=80):
    # list of [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    average_precisions = []
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
        for key, val in amount_boxes.items():
            amount_boxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_boxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            gt_in_same_image = [box for box in ground_truths if box[0] == detection[0]]
            best_iou = 0

            for gt_idx, gt in enumerate(gt_in_same_image):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)

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
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recalls = torch.cat([torch.tensor([0]), recalls])
        precisions = torch.cat([torch.tensor([1]), precisions])

        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def plot_image(image, boxes):
    cmap = plt.get_cmap("tab20b")
    class_labels = PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = image.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        assert len(box) == 6
        class_pred = box[0]
        scores = torch.sigmoid(box[1])
        box = box[2:]
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (x1 * width, y1 * height),
            box[2] * width, 
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none"
        )
        ax.add_patch(rect)
        plt.text(
            x1 * width,
            y1 * height,
            s=f"{class_labels[int(class_pred)]}: {scores * 100}%",
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0}
        )
    
    plt.show()

def cells_to_boxes(predictions, anchors, S, is_preds=True):
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    anchors = anchors.reshape(1, 3, 1, 1, 2)
    box_predictions = predictions[..., 1:5]

    if is_preds:
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:4] = anchors * (torch.exp(box_predictions[..., 2:4]))
        scores = predictions[..., 0]
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    
    else:
        scores = predictions[..., 0]
        best_class = predictions[..., 5]

    cell_indices = torch.arange(S).repeat(predictions.shape[0], 3, S, 1).unsqueeze(-1).to(predictions.device)

    x = 1 / S * (box_predictions[..., 0] + cell_indices)
    y = 1 / S * (box_predictions[..., 1] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    
    converted_boxes = torch.cat([best_class, scores, x, y, w_h], dim=-1).reshape(batch_size, S * S * num_anchors, 6)

    return converted_boxes.tolist()

def get_evaluation_boxes(loader, model, iou_threshold, anchors, threshold, box_format="midpoint", device="cuda"):
    # out.shape = 3 x batch x 3 x S x S x 85
    model.eval()
    pred_boxes = []
    true_boxes = []
    train_idx = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)

        with torch.no_grad():
            out = model(images)

        batch_size = images.shape[0]
        boxes = [[] for _ in range(batch_size)]

        for i in range(3):
            S = out[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_boxes(out[i], anchor, S)

            for idx, box in enumerate(boxes_scale_i):
                boxes[idx] += box

        gt_boxes = cells_to_boxes(labels[2], anchor, S, False)

        for idx in range(batch_size):
            nms_boxes = nms(boxes[idx], iou_threshold, threshold, box_format)

            for nms_box in nms_boxes:
                pred_boxes.append([train_idx] + nms_box)

            for box in gt_boxes[idx]:
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

    for idx, (x, y) in enumerate(loader):
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

    print(f"Class accuracy is: {(correct_class / (total_class_preds + 1e-16)) * 100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj / (total_noobj_preds + 1e-16)) * 100:2f}%")
    print(f"Obj accuracy is: {(correct_obj / (total_obj_preds + 1e-16))*100:2f}%")

    model.train()

def get_mean_std(loader):
    channels_sum, channels_sum_sqrd, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sum_sqrd += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sum_sqrd / num_batches - mean ** 2) ** 0.5

    return mean, std

def get_loaders(train_csv_path, test_csv_path):
    train_dataset = YoloDataset(train_csv_path, IMG_DIR, LABEL_DIR, ANCHORS, transform=train_transforms)
    test_dataset = YoloDataset(test_csv_path, IMG_DIR, LABEL_DIR, ANCHORS, transform=test_transforms)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, False)

    return train_loader, test_loader

def plot_couple_examples(model, loader, threshold, iou_threshold, anchors):
    model.eval()
    
    x, y = next(iter(loader))
    x = x.to(DEVICE)

    with torch.no_grad():
        out = model(x)
        boxes = [[] for _ in range(x.shape[0])]

        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_boxes(out[i], anchor, S)

            for idx, box in enumerate(boxes_scale_i):
                boxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = nms(boxes[i], iou_threshold, threshold, "midpoint")
        plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes)

torch.backends.cudnn.benchmark = True

def train(loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(loader, leave=True)
    losses = []

    for batch_idx, (images, labels) in enumerate(loop):
        images = images.to(DEVICE)
        y0, y1, y2 = labels[0].to(DEVICE), labels[1].to(DEVICE), labels[2].to(DEVICE)

        with torch.cuda.amp.autocast():
            out = model(images)
            loss = loss_fn(out[0], y0, scaled_anchors[0]) + \
                    loss_fn(out[1], y1, scaled_anchors[1]) + \
                    loss_fn(out[2], y2, scaled_anchors[2])
            
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

def main():
    model = Yolov3().to(DEVICE)
    optimizer = optim.Adam(model.parameters, LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader = get_loaders("PASCAL_VOC/train.csv", "PASCAL_VOC/test.csv")
    scaled_anchors = torch.tensor(ANCHORS) * torch.tensor(S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if epoch > 0 and epoch % 3 == 0:
            model.eval()
            check_class_accuracy(model, test_loader, CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_boxes(test_loader, model, NMS_IOU_THRESH, ANCHORS, CONF_THRESHOLD)
            mapval = mean_average_precision(pred_boxes, true_boxes, MAP_IOU_THRESH)

            print(f"MAP: {mapval.item()}")
            
            model.train()