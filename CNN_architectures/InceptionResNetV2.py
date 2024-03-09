import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            return self.relu(self.bn(self.conv(x)))
        
        return self.relu(self.conv(x))
    
class Stem(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, x):
        x = CNNBlock(self.in_channels, 32, kernel_size=3, stride=2)(x)
        x = CNNBlock(32, 32, kernel_size=3)(x)
        x = CNNBlock(32, 64, kernel_size=3, padding=1)(x)
        x0 = x.clone()
        x = CNNBlock(64, 96, kernel_size=3, stride=2)(x)
        x0 = self.max_pool(x0)
        x = torch.cat([x, x0], dim=1)

        x0 = x.clone()
        x0 = CNNBlock(160, 64, kernel_size=1)(x0)
        x0 = CNNBlock(64, 96, kernel_size=3)(x0)
        x = CNNBlock(160, 64, kernel_size=1)(x)
        x = CNNBlock(64, 64, kernel_size=(7, 1), padding=(3, 0))(x)
        x = CNNBlock(64, 64, kernel_size=(1, 7), padding=(0, 3))(x)
        x = CNNBlock(64, 96, kernel_size=3)(x)
        x = torch.cat([x, x0], dim=1)

        x0 = x.clone()
        x0 = CNNBlock(192, 192, kernel_size=3, stride=2)(x0)
        x = self.max_pool(x)
        x = torch.cat([x, x0], dim=1)

        return x
    
class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, scale=0.2):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.in_channels = in_channels
        self.branch1 = CNNBlock(in_channels=self.in_channels, out_channels=32, kernel_size=1)
        self.branch2 = nn.Sequential(CNNBlock(self.in_channels, 32, kernel_size=1),
                                     CNNBlock(32, 32, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(CNNBlock(self.in_channels, 32, kernel_size=1),
                                     CNNBlock(32, 48, kernel_size=3, padding=1),
                                     CNNBlock(48, 64, kernel_size=3, padding=1))
        self.fc = CNNBlock(128, 384, kernel_size=1, use_bn=False)

    def forward(self, x):
        x = self.relu(x)
        x0 = x.clone()
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        x = self.fc(x) * self.scale
        return self.relu(x + x0)
    
class Reduction_A(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch2 = CNNBlock(in_channels, 384, kernel_size=3, stride=2)
        self.branch3 = nn.Sequential(CNNBlock(in_channels, 256, kernel_size=1),
                                     CNNBlock(256, 256, kernel_size=3, padding=1),
                                     CNNBlock(256, 384, kernel_size=3, stride=2))
        
    def forward(self, x):
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        return x
    
class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels, scale=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)
        self.branch1 = CNNBlock(in_channels, 192, kernel_size=1)
        self.branch2 = nn.Sequential(CNNBlock(in_channels, 128, kernel_size=1),
                                     CNNBlock(128, 160, kernel_size=(1, 7), padding=(0, 3)),
                                     CNNBlock(160, 192, kernel_size=(7, 1), padding=(3, 0)))
        self.fc = CNNBlock(384, 1152, False, kernel_size=1)

    def forward(self, x):
        x = self.relu(x)
        x0 = x.clone()
        x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        x = self.fc(x) * self.scale

        return self.relu(x + x0)
    
class Reduction_B(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch2 = nn.Sequential(CNNBlock(in_channels, 256, kernel_size=1),
                                     CNNBlock(256, 384, kernel_size=3, stride=2))
        self.branch3 = nn.Sequential(CNNBlock(in_channels, 256, kernel_size=1),
                                     CNNBlock(256, 288, kernel_size=3, stride=2))
        self.branch4 = nn.Sequential(CNNBlock(in_channels, 256, kernel_size=1),
                                     CNNBlock(256, 288, kernel_size=3, padding=1),
                                     CNNBlock(288, 320, kernel_size=3, stride=2))
        
    def forward(self, x):
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return x
    
class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels, scale=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.branch1 = CNNBlock(in_channels, 192, kernel_size=1)
        self.branch2 = nn.Sequential(CNNBlock(in_channels, 192, kernel_size=1),
                                     CNNBlock(192, 224, kernel_size=(1, 3), padding=(0, 1)),
                                     CNNBlock(224, 256, kernel_size=(3, 1), padding=(1, 0)))
        self.fc = CNNBlock(448, 2144, kernel_size=1, use_bn=False)

    def forward(self, x):
        x = self.relu(x)
        x0 = x.clone()
        x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        x = self.fc(x) * self.scale

        return self.relu(x + x0)
    
class Inception_ResNet_V2(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = nn.ModuleList()

        self.layers.append(Stem(self.in_channels))
        self.in_channels = 384

        for i in range(5):
            self.layers.append(Inception_ResNet_A(self.in_channels))
        
        self.layers.append(Reduction_A(self.in_channels))
        self.in_channels += 384 * 2

        for i in range(10):
            self.layers.append(Inception_ResNet_B(self.in_channels))

        self.layers.append(Reduction_B(self.in_channels))
        self.in_channels += 384 + 288 + 320

        for i in range(5):
            self.layers.append(Inception_ResNet_C(self.in_channels))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(2144, 1000)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        
        return self.fc(x)
    
model = Inception_ResNet_V2()
x = torch.randn(32, 3, 299, 299)
print(model(x).shape)