import torch 
import os 
import pandas as pd 
from PIL import Image 
from torch.utils.data import Dataset
import config
import numpy as np

class DATASET(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.data = []
        self.class_names = os.listdir(root_dir)

        for idx, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [idx] * len(files)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])
        img_path = os.path.join(root_and_dir, img_file)
        img = np.array(Image.open(img_path).convert('RGB'))
        img = config.both_transforms(image=img)["image"]
        high_res = config.highres_transform(image=img)["image"]
        low_res = config.lowres_transform(image=img)["image"]

        return low_res, high_res