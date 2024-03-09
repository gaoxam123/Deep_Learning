import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 2e-4
batch_size = 128
image_size = 64
channels_img = 1
z_dim = 100
num_epochs = 5
features_d = 64
features_g = 64

transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)])
])

dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

gen = Generator(z_dim, channels_img, features_g).to(device)
disc = Discriminator(channels_img, features_d).to(device)
initialize_weights(gen)
initialize_weights(disc)

optimizer_gen = optim.Adam(gen.parameters(), lr, betas=(0.5, 0.999))
optimizer_disc = optim.Adam(disc.parameters(), lr, betas=(0.5, 0.999))
loss_fn = nn.BCELoss()

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = gen(noise)

        # train discriminator max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_fake + loss_disc_real) / 2

        optimizer_disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        optimizer_disc.step()

        # train generator max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = loss_fn(output, torch.ones_like(output))

        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_disc.step()

        if batch_idx % 100 == 0:
            print(f"epoch: {epoch}/{num_epochs} \n batch: {batch_idx}/{len(loader)} \n loss D: {loss_disc} loss G: {loss_gen}")

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("real", img_grid_real, global_step=step)
                writer_fake.add_image("fake", img_grid_real, global_step=step)

                step += 1