import torch
import torch.nn as nn
import torch.optim as optim

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        loss = self.bce(predictions, targets)  

        return loss.mean()