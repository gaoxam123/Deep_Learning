import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.stride = stride
        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, intermediate_channels, 3, stride, padding = 1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(intermediate_channels * self.expansion)
        )

    def forward(self, x):
        identity = x.clone()
        x = self.conv(x)
        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x
    
class ResNet50(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2)
        self.layer2 = self._create_conv_layers(block, layers[0], 64, 1)
        self.layer3 = self._create_conv_layers(block, layers[1], 128, 2)
        self.layer4 = self._create_conv_layers(block, layers[2], 256, 2)
        self.layer5 = self._create_conv_layers(block, layers[3], 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _create_conv_layers(self, block, num_repeats, intermediate_channels, stride):
        layers = []
        identity_downsample = None

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(intermediate_channels * 4)
            )

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * 4

        for i in range(num_repeats - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
    
def model(image_channels=3, num_classes=5):
    return ResNet50(block, [3, 4, 6, 3], image_channels, num_classes)