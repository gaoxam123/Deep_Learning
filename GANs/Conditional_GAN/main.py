import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import DataLoader
from model import Discriminator, Generator, initialize_weights
import matplotlib.pyplot as plt
from utils import gradient_penalty

device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-4
batch_size = 64
image_size = 64
channels_img = 3
z_dim = 100
num_classes = 10
gen_embedding = 100
num_epochs = 200
features_d = 64
features_g = 64
critic_iterations = 5
lambda_gp = 10

transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)])
])

# dataset = MNIST(root='dataset/', train=True, transform=transforms, download=True)
dataset = ImageFolder(root='celeb_dataset', transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

gen = Generator(z_dim, channels_img, features_g, num_classes, img_size=image_size, embed_size=gen_embedding).to(device)
disc = Discriminator(channels_img, features_d, num_classes, image_size).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), learning_rate, betas=(0.0, 0.9))
opt_disc = optim.Adam(disc.parameters(), learning_rate, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

gen.train()
disc.train()

for epoch in range(num_epochs):
    for batch_idx, (real, labels) in enumerate(loader):
        real = real.to(device)
        labels = labels.to(device)

        for _ in range(critic_iterations):
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise, labels)
            critic_real = disc(real, labels).reshape(-1)
            critic_fake = disc(fake, labels).reshape(-1)
            gp = gradient_penalty(disc, labels, real, fake, device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
            
            opt_disc.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_disc.step()

        output = disc(fake, labels).reshape(-1)
        loss_gen = -torch.mean(output)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(f"epoch: {epoch}/{num_epochs} \n batch: {batch_idx}/{len(loader)} \n loss D: {loss_critic} loss G: {loss_gen}")

            with torch.no_grad():
                fake = gen(noise, labels)
                fake_flattened = fake.detach().cpu().view(-1, 3, 64, 64)
                fake_grid = torchvision.utils.make_grid(fake_flattened[:32], nrow=8)
                plt.imshow(fake_grid.permute(1, 2, 0).squeeze())
                plt.axis('off')
                plt.imshow()