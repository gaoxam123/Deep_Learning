import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features
    
def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

device = torch.device('cuda')
image_size = 356
loader = transforms.Compose([
    transforms.Resize(size=(image_size, image_size)),
    transforms.ToTensor()
])

original_img = load_image("content.png")
style_img = load_image("style.png")

generated = original_img.clone().requires_grad_(True)

total_step = 2501
learning_rate = 0.001
alpha = 1
beta = 0.01
model = VGG().to(device).eval()
optimizer = optim.Adam([generated], lr=learning_rate)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

for step in range(total_step):
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    style_loss = original_loss = 0

    for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)
        G = gen_feature.view(channel, height * width).mm(gen_feature.view(channel, height * width).t()) # Gram matrix
        A = style_feature.view(channel, height * width).mm(style_feature.view(channel, height * width).t())
        style_loss += torch.mean((G - A) ** 2)
    
    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        image = tensor_to_image(generated)
        plt.imshow(image)
        plt.show()