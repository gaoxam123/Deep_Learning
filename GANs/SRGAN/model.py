import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, disc=False, use_act=True, use_bn=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True) 
            if disc else nn.PReLU(out_channels) 
        )
        self.use_act = use_act

    def forward(self, x):
        if self.use_act:
            return self.act(self.bn(self.conv(x)))
        
        return self.bn(self.conv(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(upscale_factor=scale_factor) # in_c*4, H, W -> in_c, H*2, W*2
        self.act = nn.PReLU(in_channels)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = CNNBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = CNNBlock(in_channels, in_channels, use_act=False, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_clone = x.clone()
        x = self.block2(self.block1(x))

        return x + x_clone

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = CNNBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_blocks)]
        )
        self.convblock = CNNBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsample = nn.Sequential(
            UpsampleBlock(num_channels, scale_factor=2),
            UpsampleBlock(num_channels, scale_factor=2)
        )
        self.final_conv = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        init = self.initial(x)
        x = self.residuals(init)
        x = self.convblock(x) + init
        x = self.upsample(x)
        x = self.final_conv(x)

        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        block = []
        for idx, feature in enumerate(features):
            block.append(CNNBlock(in_channels, feature), kernel_size=3, stride=1 + idx % 2, padding=1, disc=True, use_bn=False if idx == 0 else True)
            in_channels = feature

        self.block = nn.Sequential(*block)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 36, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        x = self.block(x)
        x = self.fc(x)

        return x