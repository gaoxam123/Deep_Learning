import torch
import torch.nn as nn

config = [(32, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 512), (512, 512), (512, 1024), (1024, 1024)]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.layers(x)

class Depth_Pointwise(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride):
        super().__init__()
        self.depth_wise = nn.Conv2d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=3, 
                                    stride=stride,
                                    padding=1,
                                    groups=in_channels,
                                    bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.relu(self.bn1(x))
        x = self.pointwise(x)
        x = self.relu(self.bn2(x))

        return x
    
class MobileNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.in_channels = in_channels
        self.layers = self._make_conv_layers()
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_conv_layers(self):
        in_channels = self.in_channels
        layers = nn.ModuleList()

        layers.append(CNNBlock(in_channels, 32))
        in_channels = 32

        for idx, module in enumerate(config):
            mid_channels, out_channels = module
            layers.append(Depth_Pointwise(in_channels, mid_channels, out_channels, stride=2 if idx % 2 == 1 else 1))
            in_channels = out_channels
            if idx == 6:
                for _ in range(4):
                    layers.append(Depth_Pointwise(in_channels, mid_channels, out_channels, stride=1))

        return layers
    
model = MobileNet()
x = torch.randn(32, 3, 224, 224)
print(model(x).shape)