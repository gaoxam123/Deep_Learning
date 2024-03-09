import torch
import torch.nn as nn
from model import *
from loss import *
import torch.optim as optim
from box_utils import *
from plot_utils import *
from training_utils import *
from validation_utils import *

first_out = config.FIRST_OUT
scaler = torch.cuda.amp.GradScaler()
model = Yolov5(first_out, anchors=config.ANCHORS, ch=(first_out * 4, first_out * 8, first_out * 16)).to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

train_loader, val_loader = get_loaders("datasets", 32)
loss_fn = YoloLoss(model)
evaluate = Yolo_Eval(config.CONF_THRESHOLD, config.NMS_IOU_THRESH, config.MAP_IOU_THRESH, config.DEVICE)

for epoch in range(20):
    model.train()

    train(model, train_loader, optimizer, loss_fn, scaler, epoch, 20)

    model.eval()

    evaluate.check_class_accuracy(model, val_loader)
    evaluate.calc_map(model, val_loader, model.head.anchors)