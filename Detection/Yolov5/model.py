import torch
import torch.nn as nn
import torch.nn.functional as F

class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                                  nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
                                  nn.SiLU(inplace=True))
        
    def forward(self, x):
        return self.conv(x)
    
class BottleNeck1(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiple=1):
        super().__init__()
        mid_channels = int(in_channels * width_multiple)
        self.conv = nn.Sequential(CBL(in_channels, mid_channels, 1, 1, 0),
                                  CBL(mid_channels, out_channels, 3, 1, 1))
        
    def forward(self, x):
        return self.conv(x) + x
    
class BottleNeck2(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiple=1):
        super().__init__()
        mid_channels = int(in_channels * width_multiple)
        self.conv = nn.Sequential(CBL(in_channels, mid_channels, 1, 1, 0),
                                  CBL(mid_channels, out_channels, 3, 1, 1))
        
    def forward(self, x):
        return self.conv(x)
    
class C3(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiple, num_layers, backbone=True):
        super().__init__()
        mid_channels = int(in_channels * width_multiple)

        self.skipped_layer = CBL(in_channels, mid_channels, 1, 1, 0)
        self.layers = nn.ModuleList()
        self.layers.append(CBL(in_channels, mid_channels, 1, 1, 0))

        for i in range(num_layers):
            if backbone:
                self.layers.append(BottleNeck1(mid_channels, mid_channels, width_multiple=1))
            else:
                self.layers.append(BottleNeck2(mid_channels, mid_channels, width_multiple=1))

        self.layer_out = CBL(mid_channels * 2, out_channels, 1, 1, 0)

    def forward(self, x):
        x0 = x.clone()
        for layer in self.layers:
            x = layer(x)

        x = torch.cat([x, self.skipped_layer(x0)], dim=1)

        return self.layer_out(x)
    
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = int(in_channels // 2)
        self.conv1 = CBL(in_channels, mid_channels, 1, 1, 0)
        self.conv2 = CBL(mid_channels * 4, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        x = torch.cat([x, pool1, pool2, pool3], dim=1)

        return self.conv2(x)
    
class C3_Neck(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiple, num_layers):
        super().__init__()
        mid_channels = int(in_channels * width_multiple)
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.skipped_layer = CBL(in_channels, mid_channels, 1, 1, 0)
        self.out = CBL(mid_channels * 2, out_channels, 1, 1, 0)
        self.silu_block = self._make_silu_block(num_layers)

    def _make_silu_block(self, num_layers):
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(CBL(self.in_channels, self.mid_channels, 1, 1, 0))
            elif i % 2 == 0:
                layers.append(CBL(self.mid_channels, self.mid_channels, 3, 1, 1))
            elif i % 2 != 0:
                layers.append(CBL(self.mid_channels, self.mid_channels, 1, 1, 0))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.cat([self.skipped_layer(x), self.silu_block(x)], dim=1)
        x = self.out(x)

        return x

class Head(nn.Module):
    def __init__(self, num_classes, anchors=(), ch=()):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(anchors)
        self.num_anchors_per_scale = len(anchors[0])

        self.stride = [8, 16, 32]
        anchors_ = torch.tensor(anchors).float().view(self.num_layers, -1, 2) / torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
        self.register_buffer('anchors', anchors_)
        self.out = nn.ModuleList()
        for in_channels in ch:
            self.out += nn.Conv2d(in_channels, (5 + self.num_classes) * self.num_anchors_per_scale, 1)

    def forward(self, x):
        for i in range(self.num_layers):
            x[i] = self.out[i](x[i]) # batch_size x (3 * (5 + 80)) x w x h
            batch_size, _, grid_y, grid_x = x[i].shape
            x[i] = x[i].view(batch_size, self.num_anchors_per_scale, 5 + self.num_classes, grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()

        return x

class Yolov5(nn.Module):
    def __init__(self, first_out, num_classes=80, anchors=(), ch=(), inference=False):
        super().__init__()
        self.inference = inference
        self.backbone = nn.ModuleList()
        self.backbone += [CBL(in_channels=3, out_channels=first_out, kernel_size=6, stride=2, padding=2),
                        CBL(in_channels=first_out, out_channels=first_out*2, kernel_size=3, stride=2, padding=1),
                        C3(in_channels=first_out*2, out_channels=first_out*2, width_multiple=0.5, num_layers=2),
                        CBL(in_channels=first_out*2, out_channels=first_out*4, kernel_size=3, stride=2, padding=1),
                        C3(in_channels=first_out*4, out_channels=first_out*4, width_multiple=0.5, num_layers=4),
                        CBL(in_channels=first_out*4, out_channels=first_out*8, kernel_size=3, stride=2, padding=1),
                        C3(in_channels=first_out*8, out_channels=first_out*8, width_multiple=0.5, num_layers=6),
                        CBL(in_channels=first_out*8, out_channels=first_out*16, kernel_size=3, stride=2, padding=1),
                        C3(in_channels=first_out*16, out_channels=first_out*16, width_multiple=0.5, num_layers=2),
                        SPPF(in_channels=first_out*16, out_channels=first_out*16)]
        
        self.neck = nn.ModuleList()

        self.neck += [CBL(in_channels=first_out*16, out_channels=first_out*8, kernel_size=1, stride=1, padding=0),
                    C3(in_channels=first_out*16, out_channels=first_out*8, width_multiple=0.25, num_layers=2, backbone=False),
                    CBL(in_channels=first_out*8, out_channels=first_out*4, kernel_size=1, stride=1, padding=0),
                    C3(in_channels=first_out*8, out_channels=first_out*4, width_multiple=0.25, num_layers=2, backbone=False),
                    CBL(in_channels=first_out*4, out_channels=first_out*4, kernel_size=3, stride=2, padding=1),
                    C3(in_channels=first_out*8, out_channels=first_out*8, width_multiple=0.5, num_layers=2, backbone=False),
                    CBL(in_channels=first_out*8, out_channels=first_out*8, kernel_size=3, stride=2, padding=1),
                    C3(in_channels=first_out*16, out_channels=first_out*16, width_multiple=0.5, num_layers=2, backbone=False)]
        
        self.head = Head(num_classes, anchors, ch)

    def forward(self, x):
        backbone_connections = []
        neck_connections = []
        outputs = []

        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx == 4 or idx == 6:
                backbone_connections.append(x)
        
        for idx, layer in enumerate(self.neck):
            x = layer(x)

            if idx == 0 or idx == 2:
                neck_connections.append(x)
                x = nn.Upsample(scale_factor=2)(x)
                backbone_connection = backbone_connections[-1]
                backbone_connections.pop()
                x = torch.cat([x, backbone_connection], dim=1)

            if idx == 4 or idx == 6:
                neck_connection = neck_connections[-1]
                neck_connections.pop()
                x = torch.cat([x, neck_connection], dim=1)

            if idx == 3 or idx == 5 or idx == 7:
                outputs.append(x)

        return self.head(outputs)