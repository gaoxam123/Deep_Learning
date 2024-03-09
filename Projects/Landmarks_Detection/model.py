import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )
        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.clone()
        x = self.conv(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity

        return self.relu(x)
    
class ResNet50(nn.Module):
    def __init__(self, block, image_channels, num_classes, features=[3, 4, 6, 3]):
        super().__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_conv_layers(block, features[0], 64, 1)
        self.layer2 = self._make_conv_layers(block, features[1], 128, 2)
        self.layer3 = self._make_conv_layers(block, features[2], 256, 2)
        self.layer4 = self._make_conv_layers(block, features[3], 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)

        return x

    def _make_conv_layers(self, block, num_repeats, out_channels, stride):
        layers = []
        identity_downsample = None

        if stride == 2 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4

        for i in range(num_repeats - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
model = ResNet50(block, 3, 1000)
x = torch.randn(14, 3, 224, 224)
print(model(x).shape)