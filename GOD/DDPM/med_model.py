import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(n, d):
    # Alternate implementation of positional embedding
    position = torch.arange(0, n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    embedding = torch.zeros(n, d)
    embedding[:, 0::2] = torch.sin(position * div_term)
    embedding[:, 1::2] = torch.cos(position * div_term)
    return embedding


class DDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10**-4, max_beta=0.02, device=None, image_chw=(1, 512, 512)):
        super(DDPM, self).__init__()
        self.network = network.to(device)
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.register_buffer('betas', torch.linspace(min_beta, max_beta, n_steps).to(device))
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, x0, t, eta=None):
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]
        if eta is None:
            eta = torch.randn_like(x0)
        noisy = torch.sqrt(a_bar).view(n, 1, 1, 1) * x0 + torch.sqrt(1 - a_bar).view(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        return self.network(x, t)


class MyBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_c) if normalize else nn.Identity()
        self.activation = nn.ReLU() if activation is None else activation

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class UNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(UNet, self).__init__()

        self.time_embed = nn.Linear(time_emb_dim, n_steps)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)

        self.layers = nn.ModuleList([
            MyBlock(1, 16, 3, 1, 1),
            MyBlock(16, 32, 3, 1, 1),
            MyBlock(32, 64, 3, 2, 1),
            MyBlock(64, 128, 3, 2, 1),
            MyBlock(128, 256, 3, 2, 1),
            MyBlock(256, 512, 3, 2, 1),
            MyBlock(512, 256, 3, 1, 1),
            MyBlock(256, 128, 3, 1, 1),
            MyBlock(128, 64, 3, 1, 1),
            MyBlock(64, 32, 3, 1, 1),
            MyBlock(32, 16, 3, 1, 1),
            MyBlock(16, 1, 3, 1, 1)
        ])

    def forward(self, x, t):
        time_emb = F.relu(self.time_embed(t))
        for layer in self.layers:
            x = layer(x + time_emb.unsqueeze(-1).unsqueeze(-1))
        return x
