import torch
import numpy as np
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset
import glob
from torch_snippets import *
import albumentations as A
from albumentations.pytorch import ToTensorV2

label2target = {'background': 0, 'wheat_head': 1}
target2label = {value: key for key, value in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)

TG_IMAGE_SIZE = 224
ORI_IMAGE_SIZE = 1024

class FasterRCNNDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, isTest=None, TG_IMAGE_SIZE=TG_IMAGE_SIZE, ORI_IMAGE_SIZE=ORI_IMAGE_SIZE):
        super().__init__()
        self.w, self.h = TG_IMAGE_SIZE, TG_IMAGE_SIZE
        self.img_dir = img_dir
        self.annotations = df.copy()
        self.files = glob.glob(self.img_dir + '/*')
        self.transform = transform
        self.images = df.image_id.unique()
        self.isTest = isTest
        self.TG_IMAGE_SIZE = TG_IMAGE_SIZE
        self.ORI_IMAGE_SIZE = ORI_IMAGE_SIZE

    def __len__(self):
        return len(self.images)
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    def __getitem__(self, index):
        image_path = find(self.images[index], self.files)
        image = np.array(Image.open(image_path).convert('RGB'))
        # image /= 255
        if self.isTest is True:
            if self.transform:
                # image = np.array(image)
                augmentations = self.transform(image)
            
            return image

        data = self.annotations[self.annotations['image_id'] == self.images[index]]
        boxes = data.get('bbox').values.tolist()
        # list of strings to list of lists
        for i in range(len(boxes)):
            boxes[i] = eval(boxes[i])
        boxes = np.array(boxes)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.ORI_IMAGE_SIZE * self.TG_IMAGE_SIZE
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.ORI_IMAGE_SIZE * self.TG_IMAGE_SIZE
        # boxes = boxes.astype(np.uint32).tolist()
        label = ['wheat_head'] * len(data)   

        if self.transform:
            if len(boxes):
                augmentations = self.transform(image=image, bboxes=boxes)
                image = augmentations['image']
                boxes = augmentations['bboxes']
            else:
                augmentations = self.transform(image=image, bboxes=np.zeros((1, 4)))
                image = augmentations['image']
                boxes = augmentations['bboxes']
        
        target = {}
        target['boxes'] = torch.tensor(boxes).float()
        target['labels'] = torch.tensor([label2target[i] for i in label]).long()

        return image, target
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
train_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=int(TG_IMAGE_SIZE)),
        A.PadIfNeeded(
            min_height=int(TG_IMAGE_SIZE),
            min_width=int(TG_IMAGE_SIZE),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=TG_IMAGE_SIZE, height=TG_IMAGE_SIZE),
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
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.4, label_fields=[],),
)
val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
],bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.4, label_fields=[],))