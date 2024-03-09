import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc
from collections import OrderedDict

from utils import *
from rpn import *
from roi_align import *
from roi_head import *
from transform import *

class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_head = nn.Linear(mid_channels, num_classes)
        self.reg_head = nn.Linear(mid_channels, num_classes * 4)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        scores = self.cls_head(x)
        offsets = self.reg_head(x)

        return scores, offsets
    
class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

class ResBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

        for name, param in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                param.requires_grad_(False)

        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256

        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.body.values():
            x = layer(x)

        x = self.inner_block_module(x)
        x = self.layer_block_module(x)

        return x
    
def maskrcnn_resnet50(pretrained, num_classes, pretrained_backbone=True):
    if pretrained:
        pretrained_backbone = False

    backbone = ResBackbone('resnet50', pretrained_backbone)
    model = MaskRCNN(backbone, num_classes)

    if pretrained:
        model_urls = {
            'maskrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
        model_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'])

        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]

        for i, name in enumerate(msd):
            if i in skip_list:
                continue
            msd[name].copy_(pretrained_msd[i])

        model.load_state_dict(msd)

    return model

class MaskRCNN(nn.Module):
    def __init__(self, backbone, num_classes,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3, rpn_num_samples=256, rpn_pos_fraction=0.5, rpn_reg_weights=(1, 1, 1, 1),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5, box_num_samples=256, box_pos_fraction=0.25, box_reg_weights=(10, 10, 5, 5),
                 box_score_thresh=0.1, box_nms_thresh=0.6, box_num_predictions=100):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels

        #RPN
        anchor_sizes = (128, 256, 512)
        anchor_ratios = (0.5, 1, 2)
        num_anchors = 9
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = RPN(rpn_anchor_generator, rpn_head, rpn_fg_iou_thresh, rpn_bg_iou_thresh, rpn_num_samples, rpn_pos_fraction, rpn_reg_weights, rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        #ROIHeads
        box_roi_pool = ROIAlign(7, 2)

        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)

        self.head = ROIHeads(box_roi_pool, box_predictor, box_fg_iou_thresh, box_bg_iou_thresh, box_num_samples, box_pos_fraction, box_reg_weights, box_score_thresh, box_nms_thresh, box_num_predictions)
        self.head.mask_roi_pool = ROIAlign(14, 2)

        layers = (256, 256, 256, 256)
        dim_reduced = 256
        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)

        self.transformer = Transformer(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]

        image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]
        feature = self.backbone(image)

        proposal, rpn_loss = self.rpn(feature, image_shape, target)
        res, roi_loss = self.head(feature, proposal, image_shape, target)

        if self.training:
            return dict(**rpn_loss, **roi_loss)
        else:
            res = self.transformer.postprocess(res, image_shape, ori_image_shape)
            return res