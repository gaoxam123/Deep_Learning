import torch
import torch.nn as nn

config = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

class DW_Conv(nn.Module):
    def __init__(self, in_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                                  nn.BatchNorm2d(in_channels),
                                  nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.conv(x)
    
class PW_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        return self.conv(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, expansion * in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(expansion * in_channels),
                                   nn.ReLU6(inplace=True))
        self.dw = DW_Conv(expansion * in_channels)
        self.pw = PW_Conv(in_channels * expansion, out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion

    def forward(self, x):
        skip_connection = x.clone()
        if self.expansion == 1:
            x = self.dw(x)
            x = self.pw(x)
        
        else:
            x = self.conv1(x)
            x = self.dw(x)
            x = self.pw(x)

        if self.in_channels == self.out_channels:
            x += skip_connection

        return x
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion):
        super().__init__()
        self.expansion = expansion
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels * expansion),
                                   nn.ReLU6(inplace=True))
        self.dw = DW_Conv(in_channels * expansion, stride=2)
        self.pw = PW_Conv(in_channels * expansion, out_channels)

    def forward(self, x):
        if self.expansion == 1:
            x = self.dw(x)
            x = self.pw(x)

        else:
            x = self.conv1(x)
            x = self.dw(x)
            x = self.pw(x)

        return x
    
class MobileNetV2(nn.Module):
    def __init__(self, in_channels, num_classes, width_multiplier=1):
        super().__init__()
        self.width_multiplier = width_multiplier
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, round(width_multiplier * 32), kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(round(width_multiplier * 32)),
                                   nn.ReLU6(inplace=True))
        self.in_channels = round(width_multiplier * 32)
        self.layers = self._make_conv_layers()

        self.second_last_fc = nn.Sequential(nn.Conv2d(self.in_channels, round(width_multiplier * 1280), kernel_size=1, bias=False),
                                            nn.BatchNorm2d(round(width_multiplier * 1280)),
                                            nn.ReLU6(inplace=True))
        self.avg_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.last_fc = nn.Sequential(nn.Dropout(),
                                     nn.Conv2d(round(width_multiplier * 1280), num_classes, kernel_size=1))
        
    def forward(self, x):
        x = self.conv1(x)
        for layer in self.layers:
            x = layer(x)
        
        x = self.second_last_fc(x)
        x = self.avg_pool(x)
        x = self.last_fc(x)
        x = x.reshape(x.shape[0], -1)

        return x

    def _make_conv_layers(self):
        layers = nn.ModuleList()

        for module in config:
            t, out_channels, n, stride = module
            out_channels = round(out_channels * self.width_multiplier)

            if stride == 1:
                for i in range(n):
                    layers.append(ResBlock(self.in_channels, out_channels, t))
                    self.in_channels = out_channels

            else:
                layers.append(Block(self.in_channels, out_channels, t))
                self.in_channels = out_channels

                for i in range(1, n):
                    layers.append(ResBlock(self.in_channels, out_channels, t))
                    self.in_channels = out_channels

        return layers
    
model = MobileNetV2(3, 1000)
x = torch.rand(32, 3, 224, 224)
print(model(x).shape)