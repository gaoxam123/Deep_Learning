import torch
import numpy as np 
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
from collections import defaultdict

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, image_dir1, image_dir2, transform=None):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.image_dir1 = image_dir1
        self.image_dir2 = image_dir2
        self.transform = transform
        # self.images = os.listdir(self.df)
        dct = defaultdict(list)

        for i, label in enumerate(self.df.dx):
            dct[label].append(i)

        dct = {key: np.array(val) for key, val in dct.items()}
        new_df = pd.DataFrame(np.zeros((self.df.shape[0], len(dct.keys())), dtype=np.int8), columns=dct.keys())

        for key, val in dct.items():
            new_df.loc[val, key] = 1

        self.new_df = new_df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir1, f'{self.df.iloc[index, 1]}.jpg')
        try:
            Image.open(image_path)
        except:
            image_path = os.path.join(self.image_dir2, f'{self.df.iloc[index, 1]}.jpg')
            
        image = Image.open(image_path).convert('RGB')
        label = torch.tensor(self.new_df.iloc[index, :], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label