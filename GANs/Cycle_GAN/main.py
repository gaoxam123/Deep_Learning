import torch
import torch.nn as nn
import torch.optim as optim
from dataset import HorseZebraDataset
import config
from discriminator_model import Discriminator
from generator_model import Generator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.utils import save_image
import sys

def train(disc_H, disc_Z, gen_H, gen_Z, loader, opt_disc, opt_gen, L1, mse):
    for batch_idx, (zebra, horse) in enumerate(tqdm(loader)):
        zebra = zebra.to(device)
        horse = horse.to(device)

        fake_horse = gen_H(zebra)
        D_H_real = disc_H(horse)
        D_H_fake = disc_H(fake_horse.detach())
        D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
        D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
        D_H_loss = (D_H_real_loss + D_H_fake_loss) 

        fake_zebra = gen_Z(zebra)
        D_Z_real = disc_Z(zebra)
        D_Z_fake = disc_Z(fake_zebra.detach())
        D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
        D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
        D_Z_loss = (D_Z_real_loss + D_Z_fake_loss) 

        D_loss = (D_Z_loss + D_H_loss) / 2

        opt_disc.zero_grad()
        D_loss.backward(retain_graph=True)
        opt_disc.step()


        output_H = disc_H(fake_horse)
        output_Z = disc_Z(fake_zebra)
        G_Z_loss = mse(output_Z, torch.ones_like(output_Z))
        G_H_loss = mse(output_H, torch.ones_like(output_H))
        
        cycle_zebra = gen_Z(fake_horse)
        cycle_horse = gen_H(fake_zebra)
        cycle_zebra_loss = L1(zebra, cycle_zebra)
        cycle_horse_loss = L1(horse, cycle_horse)

        identity_zebra = gen_Z(zebra)
        identity_horse = gen_H(horse)
        identity_zebra_loss = L1(zebra, identity_zebra)
        identity_horse_loss = L1(horse, identity_horse)

        G_loss = (
            G_Z_loss +
            G_H_loss +
            cycle_horse_loss * config.LAMBDA_CYCLE +
            cycle_zebra_loss * config.LAMBDA_CYCLE + 
            identity_horse_loss * config.LAMBDA_IDENTITY + 
            identity_zebra_loss * config.LAMBDA_IDENTITY
        )
        
        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

device = config.DEVICE
disc_H = Discriminator().to(device)
disc_Z = Discriminator().to(device)
gen_H = Generator().to(device)
gen_Z = Generator().to(device)

opt_disc = optim.Adam(list(disc_H.parameters()) + list(disc_Z.parameters()), config.LEARNING_RATE, (0.5, 0.999))
opt_gen = optim.Adam(list(gen_H.parameters()) + list(gen_Z.parameters()), config.LEARNING_RATE, (0.5, 0.999))

L1 = nn.L1Loss()
mse = nn.MSELoss()

dataset = HorseZebraDataset(root_horse="/", root_zebra="/", transform=config.transforms)
loader = DataLoader(dataset, config.BATCH_SIZE, True, num_workers=config.NUM_WORKERS)

for epoch in tqdm(range(config.NUM_EPOCHS)):
    train(disc_H, disc_Z, gen_H, gen_Z, loader, opt_disc, opt_gen, L1, mse)