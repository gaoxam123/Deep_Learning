import torch
from torch.utils.data import DataLoader
from model import *
from loss import *
from utils import *
import torch.optim as optim
import torch.nn as nn
from tqdm.auto import tqdm

def train(loader, model, learning_rate, epochs):
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    model.train()
    loss_list = []

    for i in tqdm(range(epochs)):
        total_loss = 0
        for images, gt_boxes, gt_classes in loader:
            loss = model(images, gt_boxes, gt_classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        loss_list.append(total_loss)

    return loss_list