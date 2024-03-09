import torch 
import torch.nn as nn

# VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

# class VGG16(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.in_channels = 3
#         self.conv = self._create_conv_layers()
#         self.fc = nn.Sequential(nn.Linear(in_features=7*7*512, out_features=4096, bias=False),
#                                 nn.BatchNorm1d(num_features=4096),
#                                 nn.ReLU(),
#                                 nn.Linear(in_features=4096, out_features=4096, bias=False),
#                                 nn.BatchNorm1d(num_features=4096),
#                                 nn.ReLU(),
#                                 nn.Linear(in_features=4096, out_features=1000))

class ColorizationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, 1, 4, 2)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 4, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 4, 2)
        self.conv4 = nn.Conv2d(128, 3, 5, 1, 4, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))

        return x