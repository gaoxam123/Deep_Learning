import torch 
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CelebDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        img = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            img = self.transform(img)

        return img