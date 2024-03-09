import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from PIL import ImageDraw

def iou(boxes1, boxes2):
    box1_x1 = boxes1[:, 0]
    box1_y1 = boxes1[:, 1]
    box1_x2 = boxes1[:, 2]
    box1_y2 = boxes1[:, 3]

    box2_x1 = boxes2[:, 0]
    box2_y1 = boxes2[:, 1]
    box2_x2 = boxes2[:, 2]
    box2_y2 = boxes2[:, 3]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = torch.max(0, x2 - x1) * torch.max(0, y2 - y1)
    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)   
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)

def nms(boxes, thresh=0.5):
    boxes_after_nms_indices = []
    scores = boxes[:, 5]
    indices = torch.argsort(scores)

    while boxes:
        chosen_box = boxes.pop()
        chosen_box_index = indices[len(indices) - 1]
        boxes_after_nms_indices.append(chosen_box_index)

        indices = [index for index in indices if iou(torch.tensor(boxes[index]), torch.tensor(chosen_box)) < thresh]

    return indices

def preprocess(image):
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = (image - 127.5) / 127.5

    return image

def generate_boxes(probs, offsets, scale, thresh):
    indices = torch.where(probs > thresh)
    stride = 2
    kernel_size = 12

    tx1, ty1, tx2, ty2 = [offsets[0, i, indices[0], indices[1]] for i in range(4)]
    offsets = torch.tensor([tx1, ty1, tx2, ty2])
    scores = probs[indices]
    x1 = torch.round(indices[1] * stride + 1.0) / scale
    y1 = torch.round(indices[0] * stride + 1.0) / scale
    x2 = torch.round(indices[1] * stride + 1.0 + kernel_size) / scale
    y2 = torch.round(indices[0] * stride + 1.0 + kernel_size) / scale

    boxes = torch.vstack([x1, y1, x2, y2, scores, offsets])

    return boxes.T

def calibrate_box(boxes, offsets):
    x1, y1, x2, y2 = [boxes[:, i] for i in range(4)]
    tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = w.unsqueeze(1)
    h = h.unsqueeze(1)

    x1 = x1 + tx1 * w
    y1 = y1 + ty1 * h
    x2 = x2 + tx2 * w
    y2 = y2 + ty2 * h

    boxes[:, 0:4] = torch.tensor([x1, y1, x2, y2])

    return boxes

def convert_to_square(boxes):
    squared_boxes = torch.zeros_like(boxes)
    x1, y1, x2, y2 = [boxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    max_side = torch.maximum(w, h)

    squared_boxes[:, 0] = x1 + w / 2 - max_side / 2
    squared_boxes[:, 1] = y1 + h / 2 - max_side / 2
    squared_boxes[:, 2] = squared_boxes[:, 0] + max_side - 1.0
    squared_boxes[:, 3] = squared_boxes[:, 1] + max_side - 1.0

    return squared_boxes

def correct_boxes(boxes, width, height):
    x1, y1, x2, y2 = [boxes[:, i] for i in range(4)]
    num_boxes = len(boxes)
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0

    x, y, ex, ey = x1, y1, x2, y2

    dx, dy = torch.zeros(num_boxes), torch.zeros(num_boxes)
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    indices = torch.where(ex > width - 1.0)[0] # which box
    edx[indices] = w[indices] + width - 2.0 - ex[indices]
    ex[indices] = width - 1.0

    indices = torch.where(ey > height - 1.0)[0]
    edy[indices] = h[indices] + height - 2.0 - ey[indices]
    ey[indices] = height - 1.0

    indices = torch.where(x < 0)[0]
    dx[indices] = 0 - x[indices]
    x[indices] = 0

    indices = torch.where(y < 0)[0]
    dy[indices] = 0 - y[indices]
    y[indices] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list

def get_image_boxes(boxes, image, size=24):
    w, h = image.size
    num_boxes = len(boxes)
    image_boxes = torch.zeros((num_boxes, 3, size, size), dtype=torch.float32)
    dy, edy, dx, edx, y, ey, x, ex, w, h = correct_boxes(boxes, w, h)

    for i in range(num_boxes):
        image_box = torch.zeros((3, w[i], h[i]), dtype=torch.uint8)
        image_array = torch.asarray(image, dtype=torch.uint8)

        image_box[:, dx[i]:(edx[i] + 1), dy[i]:(edy[i] + 1)] = image_array[:, x[i]:(ex[i] + 1), y[i]:(ey[i] + 1)] # crop correct part of the image
        image_box = Image.fromarray(image_box)
        image_box = image_box.resize((3, size, size), Image.BILINEAR)
        image_box = torch.tensor(image_box)

        image_boxes[i, :, :, :] = preprocess(image_box)

    return image_boxes

def show_boxes(image, boxes, facial_landmarks=[]):
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    for box in boxes:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='white')

    for l in facial_landmarks:
        for i in range(5):
            draw.ellipse([(l[i] - 1.0, l[i + 5] - 1.0), (l[i] + 1.0, l[i + 5] + 1.0)], outline='blue')

    return image_copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = dict(
    onet='https://github.com/khrlimam/mtcnn-pytorch/releases/download/0.0.1/onet-60cc8dd5.pth',
    pnet='https://github.com/khrlimam/mtcnn-pytorch/releases/download/0.0.1/pnet-6b6ef92b.pth',
    rnet='https://github.com/khrlimam/mtcnn-pytorch/releases/download/0.0.1/rnet-b13c48bc.pth'
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_state(arch, progress=True):
    state = load_state_dict_from_url(model_urls.get(arch), progress=progress)
    return state

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(3, 2).contiguous()

        return x.reshape(x.shape[0], -1)
    
class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 10, 3, 1),
                                  nn.PReLU(10),
                                  nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                  nn.Conv2d(10, 16, 3, 1),
                                  nn.PReLU(16),
                                  nn.Conv2d(16, 32, 3, 1),
                                  nn.PReLU(32))
        
        self.conv1 = nn.Conv2d(32, 2, 1, 1)
        self.conv2 = nn.Conv2d(32, 4, 1, 1)
        self.load_state_dict(load_state('pnet'))
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.conv(x)
        conf = self.conv1(x)
        reg = self.conv2(x)
        conf = F.softmax(conf, dim=1)

        return reg, conf
    
class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 28, 3, 1),
                                  nn.PReLU(28),
                                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                                  nn.Conv2d(28, 48, 3, 1),
                                  nn.PReLU(48),
                                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                                  nn.Conv2d(48, 64, 2, 1),
                                  nn.PReLU(64),
                                  Flatten(),
                                  nn.Linear(3 * 3 * 64, 128),
                                  nn.PReLU(128))
        
        self.conv1 = nn.Linear(128, 2)
        self.conv2 = nn.Linear(128, 4)

        self.load_state_dict(load_state('rnet'))
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.conv(x)
        conf = self.conv1(x)
        reg = self.conv2(x)
        conf = F.softmax(conf, dim=1)

        return reg, conf
    
class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 3, 1),
                                  nn.PReLU(32),
                                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                                  nn.Conv2d(32, 64, 3, 1),
                                  nn.PReLU(64),
                                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                                  nn.Conv2d(64, 64, 3, 1),
                                  nn.PReLU(64),
                                  nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                  nn.Conv2d(64, 128, 2, 1),
                                  nn.PReLU(128),
                                  Flatten(),
                                  nn.Linear(3 * 3 * 128, 256),
                                  nn.Dropout(0.25),
                                  nn.PReLU(256))
        
        self.conv1 = nn.Linear(256, 2)
        self.conv2 = nn.Linear(256, 4)
        self.conv3 = nn.Linear(256, 10)

        self.load_state_dict(load_state('onet'))
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.conv(x)
        conf = self.conv1(x)
        reg = self.conv2(x)
        coords = self.conv3(x)
        conf = F.softmax(conf, dim=1)

        return coords, reg, conf
    
def run_first_stage(image, model, scale, thresh):
    with torch.no_grad():
        w, h = image.size
        scaled_w, scaled_h = math.ceil(w * scale), math.ceil(h * scale)
        image = image.resize((scaled_w, scaled_h), Image.BILINEAR)
        image = np.asarray(image, dtype=np.float32)
        image = torch.tensor(preprocess(image))
        out = model(image)
        probs = out[1][0, 1, :, :]
        offsets = out[0]

        boxes = generate_boxes(probs, offsets, scale, thresh)
        keep = nms(boxes, thresh)
        boxes = boxes[keep]

    return boxes

def detect_faces(image, min_face_size=20, thresh=[0.6, 0.7, 0.8], nms_thresh=[0.7, 0.7, 0.7]):
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()

    w, h = image.size
    min_len = min(w, h)
    min_detection_size = 12
    factor = 0.07
    scales = []
    m = min_detection_size / min_face_size
    min_len *= m
    factor_count = 0
    while min_len > min_detection_size:
        scales.append(m * factor ** factor_count)
        min_len *= factor
        factor_count += 1
    
    boxes = []

    for scale in scales:
        box = run_first_stage(image, pnet, scale, thresh[0])
        boxes.append(box)

    boxes = [box for box in boxes if box is not None]
    boxes = boxes.flatten(start_dim=0, end_dim=1)

    keep = nms(boxes, nms_thresh[0])
    boxes = boxes[keep]

    boxes = calibrate_box(boxes[:, 0:5], boxes[:, 5:])
    boxes = convert_to_square(boxes)
    boxes[:, 0:4] = torch.round(boxes[:, 0:4])

    with torch.no_grad():
        image_boxes = get_image_boxes(boxes, image)
        out = rnet(image_boxes)
        offsets = out[0] # num_boxes x 4
        probs = out[1] # num_boxes x 2

        keep = torch.where(probs > thresh[1])[0]
        boxes = boxes[keep]
        boxes[:, 4] = probs[keep, 1].reshape(-1)
        offsets = offsets[keep]

        keep = nms(boxes, nms_thresh[1])
        boxes = boxes[keep]
        boxes = calibrate_box(boxes, offsets[keep])
        boxes = convert_to_square(boxes)
        boxes[:, 0:4] = torch.round(boxes[:, 0:4])



        image_boxes = get_image_boxes(boxes, image, 48)
        out = onet(image_boxes)
        landmarks = out[0]
        offsets = out[1]
        probs = out[2]

        keep = torch.where(probs > thresh[2])[0]
        boxes = boxes[keep]
        boxes[:, 4] = probs[keep, 1].reshape(-1)
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # keep = nms(boxes, nms_thresh[2])
        # boxes = boxes[keep]
        # boxes = calibrate_box(boxes, offsets[keep])
        # boxes = convert_to_square(boxes)
        # boxes[:, 0:4] = torch.round(boxes[:, 0:4])
        # landmarks = landmarks[keep]

    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0
    x, y = boxes[:, 0], boxes[:, 1]
    landmarks[:, 0:5] = x.unsqueeze(1) + w.unsqueeze(1) * landmarks[:, 0:5]
    landmarks[:, 5:10] = y.unsqueeze(1) + h.unsqueeze(1) * landmarks[:, 5:10]

    boxes = calibrate_box(boxes, offsets)
    keep = nms(boxes, nms_thresh[2])
    boxes = boxes[keep]
    landmarks = landmarks[keep]

    return boxes, landmarks