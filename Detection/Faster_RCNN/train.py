import torch
from torch.utils.data import DataLoader
from model import *
from loss import *
from utils import *
from dataset import *
import torch.optim as optim
import torch.nn as nn
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('train.csv')
train_idx, val_idx = train_test_split(df.image_id.unique(), train_size=0.7, random_state=23)
train_df, val_df = df[df['image_id'].isin(train_idx)], df[df['image_id'].isin(val_idx)]

train_dataset = FasterRCNNDataset(train_df, 'train', train_transform)
val_dataset = FasterRCNNDataset(val_df, 'train', val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=train_dataset.collate_fn, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=val_dataset.collate_fn, drop_last=True)

model = FasterRCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
num_epochs = 20

