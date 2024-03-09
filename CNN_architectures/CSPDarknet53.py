import torch
import torch.nn as nn

class CBM(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_bn, **kwargs)
        self.mish = nn.Mish(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            return self.mish(self.bn(self.conv(x)))
        
        return self.conv(x)
    
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_bn, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            return self.relu(self.bn(self.conv(x)))
        
        return self.conv(x)
    
class ResUnit(nn.Module):
    def __init__(self, in_channels, num_repeats, use_res=True):
        super().__init__()
        self.num_repeats = num_repeats
        self.use_res = use_res
        self.layers = nn.ModuleList()

        for i in range(num_repeats):
            self.layers += nn.Sequential(
                CBM(in_channels, in_channels // 2, kernel_size=1),
                CBM(in_channels // 2, in_channels, kernel_size=3, padding=1)
            )
        
    def forward(self, x):
        for layer in self.layers:
            if self.use_res:
                x = layer(x) + x
            
            else:
                x = layer(x)

        return x
    
class CSPX(nn.Module):
    def __init__(self, in_channels, num_repeats):
        super().__init__()
        self.layers1 = nn.ModuleList()
        self.layers2 = CBM(in_channels // 2, in_channels // 2, kernel_size=1)
        self.last_conv = CBM(in_channels, in_channels, kernel_size=1)
        self.last_conv_changing_channels = CBM(in_channels, in_channels * 2, kernel_size=1, stride=2)
        self.layers1.append(CBM(in_channels // 2, in_channels // 2, kernel_size=1))
        self.layers1.append(ResUnit(in_channels // 2, num_repeats))
        self.layers1.append(CBM(in_channels // 2, in_channels // 2, kernel_size=1))
        self.num_repeats = num_repeats
        self.in_channels = in_channels

    def forward(self, x):
        for i in range(self.num_repeats):
            x1, x2 = x[:, 0:self.in_channels, :, :], x[:, self.in_channels:, :, :]
            x1 = self.layers1(x1)
            x2 = self.layers2(x2)
            x = torch.cat([x1, x2], dim=1)

            if i < self.num_repeats - 1:
                x = self.last_conv(x)
            
            else:
                x = self.last_conv_changing_channels(x)

        return x
    
class CSPDarknet53(nn.Module):
    def __init__(self, in_channels, features=[1, 2, 8, 8, 4]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(CBM(in_channels, 32, kernel_size=1))
        in_channels = 32
        for feature in features:
            self.layers.append(CSPX(in_channels, feature))
            in_channels = in_channels * 2

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = self.layer(x)
            if isinstance(layer, CSPX):
                if layer.num_repeats == 8:
                    out = CBL(layer.in_channels * 2, layer.in_channels, kernel_size=3, stride=2, padding=1)
                    outputs.append(out)
                elif layer.num_repeats == 4:
                    outputs.append(x)

        return outputs.reverse()

class SPP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = CBL(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv2 = CBL(in_channels, in_channels, kernel_size=7, padding=3)
        self.conv3 = CBL(in_channels, in_channels, kernel_size=13, padding=6)

    def forward(self, x):
        x4 = x.clone()
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x
    
class PANet_1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        self.in_channels = in_channels
        for _ in range(3):
            self.layers.append(CBL(in_channels, in_channels // 2, kernel_size=1))
            in_channels = in_channels // 2
        self.layers.append(SPP(in_channels))
        for _ in range(3):
            self.layers.append(CBL(in_channels, in_channels, kernel_size=1))

        self.layers += [
            CBL(in_channels, in_channels // 2, kernel_size=1),
            nn.Upsample(scale_factor=2),
        ]

        for _ in range(5):
            if _ < 4:
                self.layers.append(CBL(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            else:
                self.layers.append(CBL(in_channels, in_channels // 2, kernel_size=1))
        in_channels = in_channels // 2

        self.layers += [
            CBL(in_channels, in_channels // 2, kernel_size=1),
            nn.Upsample(scale_factor=2),
        ]

    def forward(self, darknet_outputs):
        outputs = []
        cnt = 0
        count = 0
        last_layer = False
        out = darknet_outputs[cnt]
        for layer in self.layers:
            count += 1
            out = layer(out)
            if count == 7 or count == 14:
                outputs.append(out)
            
            if isinstance(layer, nn.Upsample):
                cnt += 1
                out = torch.cat([out, darknet_outputs[cnt]], dim=1)
                if cnt == 2:
                    last_layer = True

            if last_layer:
                outputs.append(out)

        return outputs
    
class ScalePrediction(nn.Module): # !!!!!!!!!!!!!
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.preds = nn.Sequential(
            CBL(in_channels, in_channels ** 2, kernel_size=1),
            nn.Conv2d(in_channels ** 2, in_channels ** 2 - 1, kernel_size=1)
        )

    def forward(self, x):
        return self.preds(x)
    
class PANet_2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        