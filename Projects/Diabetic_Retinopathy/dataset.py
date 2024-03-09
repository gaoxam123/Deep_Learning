import torch 
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd 
import config
from tqdm import tqdm
import numpy as np

class RetinaDataset(Dataset):
    def __init__(self, img_folder, path_to_csv, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(path_to_csv)
        self.img_folder = img_folder
        self.train = train
        self.transform = transform
        self.image_files = os.listdir(img_folder)

    def __len__(self):
        return self.data.shape[0] if self.train else len(self.image_files)
    
    def __getitem__(self, index):
        if self.train:
            image_file, label = self.data.iloc[index]

        else:
            image_file, label = self.image_files[index], -1
            image_file = image_file.replace(".jpeg", "")

        image = np.array(Image.open(os.path.join(self.img_folder, image_file + ".jpeg")))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label, image_file