import torch
from torch.utils.data import Dataset
import os
import numpy as np 
from PIL import Image
import config

class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform
        self.zebra_imgs = os.listdir(root_zebra)
        self.horse_imgs = os.listdir(root_horse)
        self.length_dataset = max(len(self.zebra_imgs), len(self.horse_imgs))
        self.zebra_length = len(self.zebra_imgs)
        self.horse_length = len(self.horse_imgs)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        zebra_path = self.zebra_imgs[index % self.zebra_length]
        horse_path = self.horse_imgs[index % self.horse_length]

        zebra_img = np.array(Image.open(os.path.join(self.root_zebra, zebra_path)).convert('RGB'))
        horse_img = np.array(Image.open(os.path.join(self.root_horse, horse_path)).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

        return zebra_img, horse_img