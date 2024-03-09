import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import config

class MapDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_path_list = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_path_list[index])
        image = np.array(Image.open(img_path))
        input_img = image[:, :600, :]
        label_img = image[:, 600:, :]

        augmentations = config.both_transform(image=input_img, image0=label_img)
        input_img, label_img = augmentations["image"], augmentations["image0"]

        input_img = config.transform_only_input(input_img)["image"]
        label_img = config.transform_only_mask(label_img)["image"]

        return input_img, label_img