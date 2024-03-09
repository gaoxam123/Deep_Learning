import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, in_channels, internal_channels):
        super().__init__()
        self.in_channels = in_channels
        self.squeeze = nn.Conv2d(in_channels, internal_channels)
        self.expand = nn.Conv2d(internal_channels, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.shape[2])
        x = self.squeeze(x)
        x = self.relu(x)
        x = self.expand(x)
        x = self.sigmoid(x)
        x = x.reshape(-1, self.in_channels, 1, 1)

        return inputs * x