import os
import csv
import torch
from tqdm import tqdm
from box_utils import *
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from plot_utils import *

class Yolo_Eval:
    def __init__(self, conf_thresh, nms_thresh, map_thresh, device):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.map_thresh = map_thresh
        self.device = device

        self.class_accuracy = None
        self.noobj_accuracy = None
        self.obj_accuracy = None

    def check_class_accuracy(self, model, loader):
        model.eval()
        total_class_preds, correct_class = 0, 0
        total_obj, correct_obj = 0, 0

        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(self.device)
            images /= 255.0
            with torch.no_grad():
                out = model(images)

            for i in range(3):
                targets[i] = targets[i].to(self.device)
                obj = targets[i][..., 0] == 1

                correct_class += torch.sum(torch.argmax(out[i][..., 5:][obj], dim=-1) == targets[i][..., 5][obj])
                total_class_preds += torch.sum(obj)
                
                obj_preds = torch.sigmoid(out[i][..., 0]) > self.conf_thresh
                correct_obj = torch.sum(obj_preds[obj] == targets[i][..., 0][obj])
                total_obj += torch.sum(obj)

        class_accuracy = correct_class / (total_class_preds + 1e-7)
        obj_accuracy = correct_obj / (total_obj + 1e-7)

        print("Class accuracy: {:.2f}%".format(class_accuracy * 100))
        print("Obj accuracy: {:.2f}%".format(obj_accuracy * 100))

        model.train()

    def calc_map(self, model, loader, anchors):
        model.eval()
        preds = []
        targets = []

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device) / 255.0
            with torch.no_grad():
                predictions = model(images)

            pred_boxes = cells_to_boxes(predictions, anchors, [80, 40, 20])
            true_boxes = cells_to_boxes(labels, anchors, [80, 40, 20], False)
            pred_boxes = nms(pred_boxes, self.nms_thresh, self.conf_thresh, tolist=False)
            true_boxes = nms(true_boxes, self.nms_thresh, self.conf_thresh, tolist=False)

            preds.append(dict(boxes=pred_boxes[..., 2:], scores=pred_boxes[..., 1], labels=pred_boxes[..., 0]))
            targets.append(dict(boxes=targets[..., 2:], labels=targets[..., 0]))

        metric = MeanAveragePrecision()
        metric.update(preds, targets)
        metrics = metric.compute()
        map50 = metrics["map_50"]
        map75 = metrics["map_75"]

        print(f"MAP50: {map50}, \nMAP75: {map75}")

        model.train()