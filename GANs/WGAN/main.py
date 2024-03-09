import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, ImageFolder, CIFAR10
from torch.utils.data import DataLoader
from model import Discriminator, Generator, initialize_weights
import matplotlib.pyplot as plt
from utils import gradient_penalty
from dataset import CelebDataset
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-4
batch_size = 64
image_size = 64
channels_img = 3
z_dim = 100
num_epochs = 20
features_d = 16
features_g = 16
critic_iterations = 5
lambda_gp = 10

transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)])
])

# dataset = MNIST(root='dataset/', train=True, transform=transforms, download=True)
dataset = CelebDataset('images', transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

gen = Generator(z_dim, channels_img, features_g).to(device)
disc = Discriminator(channels_img, features_d).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), learning_rate, betas=(0.0, 0.9))
opt_disc = optim.Adam(disc.parameters(), learning_rate, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

gen.train()
disc.train()

for epoch in tqdm(range(num_epochs)):
    for batch_idx, real in enumerate(tqdm(loader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        for _ in range(critic_iterations):
            noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)
            critic_real = disc(real).reshape(-1)
            critic_fake = disc(fake).reshape(-1)
            gp = gradient_penalty(disc, real, fake, device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp 
            
            opt_disc.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_disc.step()

        output = disc(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(f"epoch: {epoch}/{num_epochs} \n batch: {batch_idx}/{len(loader)} \n loss D: {loss_critic} loss G: {loss_gen}")

            with torch.no_grad():
                fake = gen(fixed_noise)
                fake_flattened = fake.detach().cpu().view(-1, 3, 64, 64)
                fake_grid = torchvision.utils.make_grid(fake_flattened[:32], nrow=8)
                plt.imshow(fake_grid.permute(1, 2, 0).squeeze())
                plt.axis('off')
                plt.show()