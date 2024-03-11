import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.conv(x)
    
class UNetpp(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, deep_supervision=True):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_channels = in_channels
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        
    def forward(self, x):
        # encoder:
        in_channels = self.in_channels
        x0_0 = CNNBlock(in_channels, 64)(x)
        x1_0 = self.pool(CNNBlock(64, 128)(x0_0))
        x2_0 = self.pool(CNNBlock(128, 256)(x1_0))
        x3_0 = self.pool(CNNBlock(256, 512)(x2_0))
        x4_0 = self.pool(CNNBlock(512, 1024)(x3_0))

        x0_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)(x1_0)
        x0_1 = torch.cat([x0_1, TF.resize(x0_0, size=x0_1.shape[2:])], dim=1)
        x0_1 = CNNBlock(128, 64)(x0_1)
        x1_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)(x2_0)
        x1_1 = torch.cat([x1_1, TF.resize(x1_0, size=x1_1.shape[2:])], dim=1)
        x1_1 = CNNBlock(256, 128)(x1_1)
        x2_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)(x3_0)
        x2_1 = torch.cat([x2_1, TF.resize(x2_0, size=x2_1.shape[2:])], dim=1)
        x2_1 = CNNBlock(512, 256)(x2_1)
        x3_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)(x4_0)
        x3_1 = torch.cat([x3_1, TF.resize(x3_0, size=x3_1.shape[2:])], dim=1)
        x3_1 = CNNBlock(1024, 512)(x3_1)

        x0_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)(x1_1)
        x0_2 = torch.cat([x0_2, TF.resize(x0_1, size=x0_2.shape[2:])], dim=1)
        x0_2 = CNNBlock(128, 64)(x0_2)
        x1_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)(x2_1)
        x1_2 = torch.cat([x1_2, TF.resize(x1_1, size=x1_2.shape[2:])], dim=1)
        x1_2 = CNNBlock(256, 128)(x1_2)
        x2_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)(x3_1)
        x2_2 = torch.cat([x2_2, TF.resize(x2_1, size=x2_2.shape[2:])], dim=1)
        x2_2 = CNNBlock(512, 256)(x2_2)

        x0_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)(x1_2)
        x0_3 = torch.cat([x0_3, TF.resize(x0_2, size=x0_3.shape[2:])], dim=1)
        x0_3 = CNNBlock(128, 64)(x0_3)
        x1_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)(x2_2)
        x1_3 = torch.cat([x1_3, TF.resize(x1_2, size=x1_3.shape[2:])], dim=1)
        x1_3 = CNNBlock(256, 128)(x1_3)

        x0_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)(x1_3)
        x0_4 = torch.cat([x0_4, TF.resize(x0_3, size=x0_4.shape[2:])], dim=1)
        x0_4 = CNNBlock(128, 64)(x0_4)

        if self.deep_supervision:
            out = torch.cat([
                nn.Conv2d(64, self.num_classes, 1)(x0_1),
                nn.Conv2d(64, self.num_classes, 1)(x0_2),
                nn.Conv2d(64, self.num_classes, 1)(x0_3),
                nn.Conv2d(64, self.num_classes, 1)(x0_4)
            ], dim=0)

        else:
            out = nn.Conv2d(64, self.num_classes, 1)(x0_4)

        return out
    
model = UNetpp(deep_supervision=False)
x = torch.randn(2, 3, 572, 572)

print(model(x).shape)