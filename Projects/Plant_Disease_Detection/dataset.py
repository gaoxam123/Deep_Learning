import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pandas import DataFrame
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# df = pd.read_csv("train.csv")
# dct = defaultdict(list)

# for i, label in enumerate(df.labels):
#     for category in label.split():
#         dct[category].append(i)

# dct = {key: np.array(val) for key, val in dct.items()}

# new_df = pd.DataFrame(np.zeros((df.shape[0], len(dct.keys())), dtype=np.int8), columns=dct.keys())

# for key, val in dct.items():
#     new_df.loc[val, key] = 1



class Fuck(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        super().__init__()
        self.__annotations__ = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv("train.csv")
        dct = defaultdict(list)

        for i, label in enumerate(df.labels):
            for category in label.split():
                dct[category].append(i)

        dct = {key: np.array(val) for key, val in dct.items()}

        new_df = pd.DataFrame(np.zeros((df.shape[0], len(dct.keys())), dtype=np.int8), columns=dct.keys())

        for key, val in dct.items():
            new_df.loc[val, key] = 1

        self.new_df = new_df

    def __len__(self):
        return len(self.__annotations__)
    
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.__annotations__.iloc[index, 0])
        img = Image.open(img_path)
        label = torch.tensor(self.new_df.iloc[index, :], dtype=torch.float32)

        if(self.transform):
            img = self.transform(img)
        
        return img, label
    
data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# dataset = Fuck('train_images', 'train.csv', data_transform)
# img, label = dataset.__getitem__(1)

# img.permute(1, 2, 0)
# plt.imshow(img.numpy())
# plt.show()
# print(label)