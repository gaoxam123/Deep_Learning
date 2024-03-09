import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET 
import numpy as np
import cv2

class FaceLandmarksDataset(Dataset):
    def __init__(self, transform=None):
        tree = ET.parse('../input/ibug-300w-large-face-landmark-dataset/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = '../input/ibug-300w-large-face-landmark-dataset/ibug_300W_large_face_landmark_dataset'

        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

            self.crops.append(filename[0].attrib)

            landmark = []
            for num in range(68):
                x_coord = int(filename[0][num].attrib['x'])
                y_coord = int(filename[0][num].attrib['y'])
                landmark.append([x_coord, y_coord])

            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('32')
        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        landmarks -= 0.5

        return image, landmarks