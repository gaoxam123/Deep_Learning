import torch 
import torch.nn as nn
import torch.optim as optim
from utils import *
from loss import *
from model import *
from dataset import *
from config import *
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

def train(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    for epoch in range(tqdm(NUM_EPOCHS)):
        for batch_idx, (input, target) in enumerate(tqdm(loader)):
            low_res = input.to(DEVICE)
            high_res = target.to(DEVICE)

            fake = gen(low_res)
            disc_real = disc(high_res)
            disc_fake = disc(fake.detach())
            disc_loss_real = bce(disc_real, torch.ones_like(disc_real))
            disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = disc_loss_fake + disc_loss_real

            opt_disc.zero_grad()
            disc_loss.backward(retain_graph=True)
            opt_disc.step()

            disc_fake = disc(fake)
            adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
            loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
            gen_loss = adversarial_loss + loss_for_vgg
            
            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            if batch_idx % 200 == 0:
                plot_examples("test_images/", gen)

dataset = DATASET('fuck')
loader = DataLoader(dataset, BATCH_SIZE, True)

gen = Generator().to(DEVICE)
disc = Discriminator().to(DEVICE)
opt_gen = optim.Adam(gen.parameters(), LEARNING_RATE)
opt_disc = optim.Adam(disc.parameters(), LEARNING_RATE)
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
vgg_loss = VGGLoss()

train(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)