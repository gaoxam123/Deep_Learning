import random
import numpy as np
import torch 
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from box_utils import *
from plot_utils import *
import config
import cv2

def resize_image(image, output_size):
    # output size is [width, height]
    return cv2.resize(image, dsize=output_size, interpolation=cv2.INTER_LINEAR)

class Training_Dataset(Dataset):
    def __init__(self, root_dir=config.ROOT_DIR, transform=None, train=True, rect_training=False, default_size=640, batch_size=32, boxes_format='coco'):
        super().__init__()
        self.batch_size = batch_size
        self.batch_range = 64
        self.boxex_format = boxes_format
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.default_size = default_size
        self.rect_training = rect_training

        if train:
            fname = 'images/train'
            annot_file = "annot_train.csv"
            self.annot_folder = "train"

        else:
            fname = 'images/val'
            annot_file = "annot_val.csv"
            self.annot_folder = "val"

        self.fname = fname
        self.annotations = pd.read_csv(os.path.join(root_dir, "labels", annot_file), header=None, index_col=0).sort_values(by=[0])
        self.annotations = self.annotations.head((len(self.annotations) - 1)) # remove last line

        if rect_training:
            self.annotations = self.adaptive_shape(self.annotations)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_name = self.annotations.iloc[index, 0]
        tg_width, tg_height = self.annotations.iloc[index, 1] if self.rect_training else 640, self.annotations.iloc[index, 2] if self.rect_training else 640
        label_path = os.path.join(self.root_dir, "labels", self.annot_folder, image_name[:-4] + "txt")
        labels = np.loadtxt(label_path, delimiter=" ", ndmin=2)
        labels = labels[np.all(labels >= 0, axis=1), :]
        labels[:, 3:5] = np.floor(labels[:, 3:5] * 1000) / 1000

        image = np.array(Image.open(os.path.join(self.root_dir, self.fname, image_name)).convert("RGB"))

        if self.boxex_format == 'coco':
            labels[:, -1] -= 1
            labels = np.roll(labels, axis=1, shift=1)
            labels[:, 1:] = coco_to_yolo_tensors(labels[:, 1:], image.shape[1], image.shape[0])

        image = resize_image(image, (int(tg_width), int(tg_height)))

        if self.transform:
            batch_n = index // self.batch_size
            if batch_n % 2 == 0:
                self.transform[1].p = 1
            else:
                self.transform[1].p = 0

            augmentations = self.transform(image=image, bboxes=np.roll(labels, shift=4, axis=1))
            image = augmentations["image"]
            labels = np.array(augmentations["bboxes"])
            if len(labels):
                labels = np.roll(labels, shift=1, axis=1)

        image = image.permute(2, 0, 1)
        image = np.ascontiguousarray(image)

        return torch.tensor(image), labels
    
class Validation_Dataset(Dataset):
    def __init__(self, root_dir, anchors, transform=None, train=True, S=(8, 16, 32), rect_training=False, default_size=640, batch_size=32, boxes_format='coco'):
        super().__init__()
        self.batch_range = 64
        self.batch_size = batch_size
        self.boxes_format = boxes_format
        self.root_dir = root_dir
        self.S = S
        self.rect_training = rect_training
        self.default_size = default_size
        self.num_layers = 3
        self.anchors = torch.tensor(anchors).float().view(3, 3, 2) / torch.tensor(self.S).repeat(6, 1).T.reshape(3, 3, 2)
        self.num_anchors = 9
        self.num_anchors_per_scale = 3
        self.ignore_iou_threshold = 0.5
        self.train = train
        self.transform = transform

        if train:
            fname = 'images/train'
            annot_file = "annot_train.csv"
            self.annot_folder = "train"
        
        else:
            fname = 'images/val'
            annot_file = "annot_val.csv"
            self.annot_folder = "val"

        self.fname = fname
        self.annotations = pd.read_csv(os.path.join(root_dir, "labels", annot_file), header=None, index_col=0).sort_values(by=[0])
        self.annotations = self.annotations.head((len(self.annotations) - 1))
        
        if rect_training:
            self.annotations = self.adaptive_shape(self.annotations)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_name = self.annotations.iloc[index, 0]
        tg_width, tg_height = self.annotations.iloc[index, 1] if self.rect_training else 640, self.annotations.iloc[index, 2] if self.rect_training else 640
        label_path = os.path.join(self.root_dir, "labels", self.annot_folder, image_name[:-4] + "txt")
        labels = np.loadtxt(label_path, delimiter=" ", ndmin=2)
        labels = labels[np.all(labels >= 0, axis=1), :]
        labels[:, 3:5] = np.floor(labels[:, 3:5] * 1000) / 1000

        image = np.array(Image.open(os.path.join(self.root_dir, self.fname, image_name)).convert("RGB"))

        if self.boxes_format == 'coco':
            labels[:, -1] -= 1
            labels = np.roll(labels, shift=1, axis=1)
            labels[:, 1:] = coco_to_yolo_tensors(labels[:, 1:], image.shape[1], image.shape[0])

        image = resize_image(image, (tg_width, tg_height))

        if self.transform:
            batch_n = index // self.bs
            if batch_n % 2 == 0:
                self.transform[2].p = 1
            else:
                self.transform[2].p = 0

            augmentations = self.transform(image=image, bboxes=np.roll(labels, shift=4, axis=1))
            image = augmentations["image"]
            labels = np.array(augmentations["bboxes"])
            if len(labels):
                labels = np.roll(labels, shift=1, axis=1)

        targets = [torch.zeros(self.num_anchors_per_scale, int(image.shape[0] / S), int(image.shape[1] / S), 6) for S in self.S]

        for box in labels:
            iou_anchors = iou_width_height(box[3:5], self.anchors)
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

        image = image.permute(2, 0, 1)
        image = np.ascontiguousarray(image)

        return torch.tensor(image), tuple(targets)