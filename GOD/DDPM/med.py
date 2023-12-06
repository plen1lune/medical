import random
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor, Resize
from argparse import ArgumentParser
from tqdm import tqdm
import imageio
from custom_models import DDPM, UNet

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Constants
STORE_PATH = "ddpm_model.pt"


def display_images(images, title="", save_dir="output"):
    """Display and save a grid of images."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            ax.imshow(images[i][0], cmap='gray')
            ax.axis('off')
        else:
            ax.remove()
    fig.suptitle(title, fontsize=16)
    plt.savefig(os.path.join(save_dir, f"{title}.png"))
    plt.show()


def process_and_save_images(images, title="", folder="saved_images"):
    """Save individual images to a folder."""
    if not os.path.exists(folder):
        os.makedirs(folder)

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    for idx, img in enumerate(images):
        plt.imsave(os.path.join(folder, f"{title}_{idx}.png"), img[0], cmap='gray')


def visualize_batch(data_loader):
    """Visualize the first batch from the loader."""
    for imgs, _ in data_loader:
        display_images(imgs, title="Sample Batch")
        break


def simulate_ddpm_process(ddpm_model, data_loader, device):
    """Simulate and display the DDPM process."""
    for imgs, _ in data_loader:
        display_images(imgs, "Original Images")
        for percent in [0.25, 0.5, 0.75, 1.0]:
            noisy_imgs = ddpm_model(imgs.to(device), int(percent * ddpm_model.n_steps))
            display_images(noisy_imgs.cpu(), f"Noisy Images {int(percent * 100)}%")
        break


def generate_samples(ddpm_model, n_samples=16, device='cpu'):
    """Generate new samples using the DDPM model."""
    ddpm_model.to(device)
    x = torch.randn(n_samples, 1, 28, 28, device=device)
    for t in range(ddpm_model.n_steps - 1, -1, -1):
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = ddpm_model.reverse_step(x, t, noise)

    return x.detach().cpu()


def train_model(model, data_loader, epochs, optimizer, device):
    """Training loop for the DDPM model."""
    model.to(device)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, _ in tqdm(data_loader):
            images = images.to(device)
            optimizer.zero_grad()
            noisy_images, noise = model(images)
            loss = criterion(noisy_images, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    # Data loading
    transform = transforms.Compose([Resize((28, 28)), ToTensor()])
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDPM(UNet(), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    train_model(model, data_loader, args.epochs, optimizer, device)

    # Save
