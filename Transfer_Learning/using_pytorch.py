import torch
import cv2
import torch.nn as nn
import torchvision

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, 10))
model.to(torch.device('cpu'))