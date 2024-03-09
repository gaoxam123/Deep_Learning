import torch
import os
from PIL import Image
import pandas as pd 
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random

class SiameseDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_path_list = list(Path(img_dir).glob('*/*.pgm'))

    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, index):
        img0_path = self.image_path_list[index]
        same = random.randint(0, 1)

        if same:
            while True:
                img1_path = np.random.choice(self.image_path_list)
                if img0_path.parent.name == img1_path.parent.name:
                    break

        else:
            while True:
                img1_path = np.random.choice(self.image_path_list)
                if img0_path.parent.name != img1_path.parent.name:
                    break

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)

        img0 = img0.convert('L')
        img1 = img1.convert('L')

        if self.transform:
            img0, img1 = self.transform(img0), self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(same)], dtype=np.float32))