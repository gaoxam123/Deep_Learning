import torch
import torch.nn as nn
from math import ceil

base_model = [
    # expand, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]
# phi, resolution, dropout_rate
phi_values = {'b0': (0, 224, 0.2),
              'b1': (0.5, 240, 0.2),
              'b2': (1, 260, 0.3),
              'b3': (2, 300, 0.3),
              'b4': (3, 380, 0.4),
              'b5': (4, 456, 0.4),
              'b6': (5, 528, 0.5),
              'b7': (6, 600, 0.5)}

def make_div(x, divisor, min_value=None):
    if min_value is None:
        min_value = divisor

    new_x = max(min_value, int(x + divisor / 2) // divisor * divisor)
    if new_x < 0.9 * x:
        new_x += divisor

    return new_x

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class SE(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                  nn.Conv2d(in_channels, mid_channels, 1),
                                  nn.SiLU(inplace=True),
                                  nn.Conv2d(mid_channels, in_channels, 1),
                                  nn.Sigmoid())
        self.in_channels = in_channels

    def forward(self, x):

        return x * self.conv(x)
     
class InvertedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand, reduction=4, survival_prob=0.8): # stochastic depth
        super().__init__()
        self.survival_prob = survival_prob
        self.down_sample = in_channels == out_channels and stride == 1
        mid_channels = in_channels * expand
        self.expand = in_channels != mid_channels
        mid_channels_se = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(CNNBlock(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels),
                                  SE(mid_channels, mid_channels_se),
                                  nn.Conv2d(mid_channels, out_channels, 1, bias=False),
                                  nn.BatchNorm2d(out_channels))
        
    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device = x.device) < self.survival_prob

        return torch.div(x, self.survival_prob) * binary_tensor # just like dropout
        
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.down_sample:
            return self.stochastic_depth(self.conv(x)) + inputs
        
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, mode, num_classes):
        super().__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(mode)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.layers = self._make_conv_layers(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(nn.Dropout(dropout_rate),
                                        nn.Linear(last_channels, num_classes))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)

        return self.classifier(x)

    def calculate_factors(self, mode, alpha=1.2, beta=1.1):
        phi, res, drop = phi_values[mode]
        depth_factor = alpha ** phi
        width_factor = beta ** phi

        return width_factor, depth_factor, drop
    
    def _make_conv_layers(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        layers = nn.ModuleList()
        layers.append(CNNBlock(3, channels, 3, 2, 1))
        in_channels = channels

        for t, out_channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(out_channels * width_factor) / 4)
            repeats = ceil(repeats * depth_factor)

            for i in range(repeats):
                layers.append(InvertedResBlock(in_channels,
                                               out_channels,
                                               kernel_size,
                                               stride=stride if i == 0 else 1,
                                               padding=kernel_size // 2,
                                               expand=t))
                in_channels = out_channels

        layers.append(CNNBlock(in_channels, last_channels, 1, 1, 0))

        return layers
    
model = EfficientNet('b7', 1000).to('cuda')
phi, res, drop = phi_values['b7']
x = torch.randn(2, 3, res, res).to('cuda')
print(model(x).shape)