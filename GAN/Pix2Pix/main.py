import torch
import torch.optim as optim
import torchvision
import config
import torch.nn as nn
from dataset import MapDataset
from Generator_model import Generator
from Discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.utils import save_image

def train(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce):
    for batch_idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        y_fake = gen(x)
        D_real = disc(x, y)
        D_fake = disc(x, y_fake.detach())
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_fake_loss + D_real_loss) / 2

        opt_disc.zero_grad()
        D_loss.backward(retain_graph=True)
        opt_disc.step()

        output = disc(x, y_fake)
        G_loss = bce(output, torch.ones_like(output))
        L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
        G_loss += L1

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

def main():
    disc = Discriminator().to(config.DEVICE)
    gen = Generator().to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), config.LEARNING_RATE, (0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), config.LEARNING_RATE, (0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    train_dataset = MapDataset("data/maps/train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataset = MapDataset("data/maps/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in tqdm(range(config.NUM_EPOCHS)):
        train(disc, gen, train_loader, opt_disc, opt_gen, l1_loss, bce)

        config.save_some_examples(gen, val_loader, epoch, folder="evaluation")