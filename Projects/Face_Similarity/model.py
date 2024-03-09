import torch
import torch.nn as nn

VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG16Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.conv = self._make_conv_layers(VGG16)
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward_once(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

    def _make_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = out_channels

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

# model = VGG16Net()   
# x1 = torch.randn(16, 1, 224, 224)
# x2 = torch.randn(16, 1, 224, 224)

# output1, output2 = model(x1, x2)

# print(output1.shape, output2.shape)

# print(model(x1).shape)