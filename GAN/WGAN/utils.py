import torch 
import torch.nn as nn

def gradient_penalty(critic, real, fake, device='cpu'):
    N, c, h, w = real.shape
    epsilon = torch.rand((N, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolation = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic(interpolation)
    gradient = torch.autograd.grad(inputs=interpolation, outputs=mixed_scores,
                                   grad_outputs=torch.ones_like(mixed_scores),
                                   create_graph=True, retain_graph=True)[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty