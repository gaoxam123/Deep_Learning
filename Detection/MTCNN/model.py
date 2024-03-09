import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
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
    """Run P-Net, generate bounding boxes, and do NMS.

    Arguments:
        image: an instance of PIL.Image.
        net: an instance of pytorch's nn.Module, P-Net.
        scale: a float number,
            scale width and height of the image by this number.
        threshold: a float number,
            threshold on the probability of a face when generating
            bounding boxes from predictions of the net.

    Returns:
        a float numpy array of shape [n_boxes, 9],
            bounding boxes with scores and offsets (4 + 1 + 4).
    """
    with torch.no_grad():
        w, h = image.shape
        scaled_w, scaled_h = math.ceil(w * scale), math.ceil(h * scale)
        img = image.reshape((scaled_w, scaled_h), Image.BILINEAR)
        img = np.asarray(img, dtype=np.float32)

        img = torch.tensor(preprocess(img))
        out = model(img)
        probs = out[1].data.numpy()[0, 1, :, :]
        offsets = out[0].data.numpy()

        boxes = generate_boxes(probs, offsets, scale, thresh)
        if len(boxes) == 0:
            return None
        
        survived = nms(boxes[:, 0:5], 0.5)

    return boxes[survived]

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