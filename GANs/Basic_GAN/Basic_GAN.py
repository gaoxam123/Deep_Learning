import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import torch.optim as optim
from tqdm.auto import tqdm
torch.manual_seed(0)

def show_tensor_images(image_tensor, num_images=32):
    image_flattened = image_tensor.detach().cpu().view(-1, 1, 28, 28)
    image_grid = make_grid(image_flattened[:num_images], nrow=8)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()

class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28*1, hidden_dim=128): # 28 x 28 x 1
        super().__init__()
        self.disc = nn.Sequential(
            self.block(img_dim, hidden_dim * 4),
            self.block(hidden_dim * 4, hidden_dim * 2),
            self.block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def block(self, input_features, output_features):
        return nn.Sequential(
            nn.Linear(input_features, output_features),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim=10, img_dim=784, hidden_dim=128):
        super().__init__()
        self.gen = nn.Sequential(
            self.block(z_dim, hidden_dim),
            self.block(hidden_dim, hidden_dim * 2),
            self.block(hidden_dim * 2, hidden_dim * 4),
            self.block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, img_dim),
            nn.Sigmoid()
        )

    def block(self, input_features, output_features):
        return nn.Sequential(
            nn.Linear(input_features, output_features, bias=False),
            nn.BatchNorm1d(output_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.gen(x)
    
# GAN is sensitive to hyperparameters    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.00001
batch_size = 128
z_dim = 64
num_epochs = 200
loss_fn = nn.BCEWithLogitsLoss()
gen = Generator(z_dim).to(device)
disc = Discriminator().to(device)
optimizer_gen = optim.Adam(gen.parameters(), lr)
optimizer_disc = optim.Adam(disc.parameters(), lr)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = MNIST(root='dataset/', transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_noise(n_samples, z_dim, device='cuda'):
    return torch.randn((n_samples, z_dim)).to(device)

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device=device)
    fake = gen(noise)
    disc_fake = disc(fake.detach())
    loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
    disc_real = disc(real).view(-1)
    loss_real = criterion(disc_real, torch.ones_like(disc_real))
    disc_loss = (loss_fake + loss_real) / 2

    return disc_loss

def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device)
    fake = gen(noise) 
    disc_fake = disc(fake)
    gen_loss = criterion(disc_fake, torch.ones_like(disc_fake))

    return gen_loss

def train(loader, disc, gen, loss_fn, optimizer_gen, optimizer_disc, num_epochs, device):
    for epoch in tqdm(range(num_epochs)):
        for batch_idx, (real, _) in enumerate(tqdm(loader)):
            real = real.reshape(-1, 784).to(device)
            batch_size = real.shape[0]
            
            # train discriminator
            lossD = get_disc_loss(gen, disc, loss_fn, real, batch_size, z_dim, device)

            optimizer_disc.zero_grad()
            lossD.backward(retain_graph=True)
            optimizer_disc.step()

            #train generator
            optimizer_gen.zero_grad()
            lossG = get_gen_loss(gen, disc, loss_fn, batch_size, z_dim, device)
            lossG.backward()
            optimizer_gen.step()

            if batch_idx == 0:
                print(f"epoch: {epoch}/{num_epochs} \n lossD: {lossD} lossG: {lossG}")

                with torch.no_grad():
                    predictions = gen(fixed_noise)
                    show_tensor_images(predictions, 32)
                    show_tensor_images(real, 32)

train(loader, disc, gen, loss_fn, optimizer_gen, optimizer_disc, num_epochs, device)