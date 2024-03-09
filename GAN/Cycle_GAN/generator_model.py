import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            Block(in_channels, in_channels, kernel_size=3, padding=1),
            Block(in_channels, in_channels, use_act=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self, in_channels, features, num_res=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )

        self.down = nn.ModuleList([
            Block(features, features * 2, kernel_size=3, stride=2, padding=1),
            Block(features * 2, features * 4, kernel_size=3, stride=2, padding=1)
        ])

        self.res_block = nn.Sequential(
            *[ResBlock(features * 4) for _ in range(num_res)]
        )
        
        self.up = nn.ModuleList([
            Block(features * 4, features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            Block(features * 2, features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
        ])

        self.last = nn.Conv2d(features, in_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down:
            x = layer(x)
        x = self.res_block(x)
        for layer in self.up:
            x = layer(x)

        return torch.tanh(self.last(x))