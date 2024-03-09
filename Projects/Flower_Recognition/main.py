import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from model import model
from dataset import FlowerDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = FlowerDataset(target_dir='data', transform=transform)
train_loader = DataLoader(dataset, batch_sampler=64, shuffle=True, num_workers=8)

model = model().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)

from tqdm import tqdm

for epoch in tqdm(range(100)):
    model.train()
    train_loss = 0
    train_acc = 0

    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        prediction = model(image)
        loss = loss_fn(prediction, label)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        classs = torch.argmax(torch.softmax(prediction, dim=1), dim=1)
        train_acc += (classs == label).sum().item()/len(prediction)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            logits = model(image)
            test_loss += loss_fn(logits, label).item()

            classs = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            test_acc += (classs == label).sum().item()/len(logits)

        test_loss /= len(train_loader)
        test_acc /= len(train_loader)