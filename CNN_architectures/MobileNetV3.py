import torch
import torch.nn as nn

def make_div(x, divisor, min_value=None):
    if min_value is None:
        min_value = divisor

    new_x = max(min_value, int(x + divisor / 2) // divisor * divisor)
    if new_x < 0.9 * x:
        new_x += divisor

    return new_x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
    
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    
class SE(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(nn.Conv2d(in_channels, make_div(in_channels // reduction, 8), kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(make_div(in_channels // reduction, 8), in_channels, kernel_size=1),
                                h_sigmoid())
        self.in_channels = in_channels
        
    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.fc(x).reshape(-1, self.in_channels, 1, 1)

        return inputs * x
    
class Inverted_ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, use_hs, use_se):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and in_channels == out_channels

        if in_channels == mid_channels:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, 
                                                in_channels, 
                                                kernel_size, 
                                                stride, 
                                                (kernel_size - 1) // 2, 
                                                groups=in_channels, 
                                                bias=False), 
                                      nn.BatchNorm2d(in_channels),
                                      h_swish() if use_hs else nn.ReLU(inplace=True),
                                      SE(in_channels) if use_se else nn.Identity(),
                                      nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                      nn.BatchNorm2d(out_channels))
            
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                                      nn.BatchNorm2d(mid_channels),
                                      h_swish() if use_hs else nn.ReLU(inplace=True),
                                      nn.Conv2d(mid_channels,
                                                mid_channels,
                                                kernel_size, 
                                                stride, 
                                                (kernel_size - 1) // 2,
                                                groups=mid_channels, 
                                                bias=False),
                                      nn.BatchNorm2d(mid_channels),
                                      h_swish() if use_hs else nn.ReLU(inplace=True),
                                      SE(mid_channels) if use_se else nn.Identity(),
                                      nn.Conv2d(mid_channels, out_channels, 1, bias=False),
                                      nn.BatchNorm2d(out_channels))
            
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        
        else:
            return self.conv(x)
        
class MobileNetV3(nn.Module):
    def __init__(self, config, mode, num_classes=1000, width_multiplier=1):
        super().__init__()
        self.config = config

        in_channels = make_div(16 * width_multiplier, 8)
        layers = nn.ModuleList()
        layers.append(nn.Sequential(nn.Conv2d(3, in_channels, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(in_channels),
                                    h_swish()))
        
        for kernel_size, t, out_channels, use_se, use_hs, stride in self.config:
            out_channels = make_div(out_channels * width_multiplier, 8)
            mid_channels = make_div(in_channels * t, 8)
            layers.append(Inverted_ResBlock(in_channels, mid_channels, out_channels, kernel_size, stride, use_hs, use_se))
            in_channels = out_channels
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                                  nn.BatchNorm2d(mid_channels),
                                  h_swish())
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        out_channels = {'large': 1280, 'small': 1024}
        out_channels = make_div(out_channels[mode] * width_multiplier, 8) if width_multiplier > 1 else out_channels[mode]

        self.fc = nn.Sequential(nn.Linear(mid_channels, out_channels),
                                h_swish(),
                                nn.Dropout(0.2),
                                nn.Linear(out_channels, num_classes))
        self.layers = layers
        
        self._init_weights()
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                x = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
                layer.weight.data.normal_(0, (2. / x) ** 0.5)
                if layer.bias is not None:
                    layer.bias.data.zero_()

            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

            elif isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.01)
                layer.bias.data.zero_()

config_large = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]

config_small = cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

large_model = MobileNetV3(config_small, 'small')
x = torch.randn(32, 3, 224, 224)
print(large_model(x).shape)