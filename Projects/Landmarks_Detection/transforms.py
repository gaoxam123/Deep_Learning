import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image

class Transforms():
    def __init__(self):
        pass

    def crop_face(self, image, landmarks, crops):
        top = int(crops['top'])
        left = int(crops['left'])
        height = int(crops['height'])
        width = int(crops['width'])

        image = TF.crop(image, top, left, height, width)

        image_shape = np.array(image).image_shape
        landmarks = torch.tensor(landmarks) - torch.tensor([left, top])
        landmarks = landmarks / torch.tensor([image_shape[1], image_shape[0]])

        return image, landmarks
    
    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks
    
    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=random.random(),
                                              contrast=random.random(),
                                              saturation=random.random(),
                                              hue=random.uniform(0, 0.5))
        
        image = color_jitter(image)

        return image, landmarks
    
    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
        ])

        image = image.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.dot(landmarks, transformation_matrix)
        new_landmarks += 0.5

        return Image.fromarray(image), new_landmarks
    
    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=random.randint(-20,20))

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])

        return image, landmarks