import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 784),  # 28x28 = 784
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

# Training function
def train_gan(generator, discriminator, dataloader, num_epochs, latent_dim,
              device, save_dir):
    adversarial_loss = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    g_scheduler = optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.99)
    d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.99)

    os.makedirs(save_dir, exist_ok=True)

    g_losses = []
    d_losses = []

    print("Starting training...")
    for epoch in range(num_epochs):
        g_loss_epoch = 0
        d_loss_epoch = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for real_imgs, _ in pbar:
            batch_size = real_imgs.size(0)
            real = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            real_imgs = real_imgs.to(device)

            # Train Generator
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(fake_imgs), real)
            g_loss.backward()
            g_optimizer.step()

            # Train Discriminator
            d_optimizer.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), real)
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

            pbar.set_postfix({
                'D Loss': f'{d_loss.item():.4f}',
                'G Loss': f'{g_loss.item():.4f}'
            })

        g_scheduler.step()
        d_scheduler.step()

        g_loss_epoch /= len(dataloader)
        d_loss_epoch /= len(dataloader)
        g_losses.append(g_loss_epoch)
        d_losses.append(d_loss_epoch)

        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'epoch': epoch,
            }, f'{save_dir}/gan_checkpoint_epoch_{epoch + 1}.pt')

            generate_and_save_samples(generator, latent_dim, device,
                                      f'{save_dir}/samples_epoch_{epoch + 1}.png')

    return g_losses, d_losses

def generate_and_save_samples(generator, latent_dim, device, filename, n_samples=16):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        generated_imgs = generator(z).cpu()
        plt.figure(figsize=(4, 4))
        for i in range(n_samples):
            plt.subplot(4, 4, i + 1)
            plt.imshow(generated_imgs[i, 0, :, :], cmap='gray')
            plt.axis('off')
        plt.savefig(filename)
        plt.close()
    generator.train()

def plot_losses(g_losses, d_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_dir}/loss_plot.png')
    plt.close()

def main():
    # Hyperparameters
    latent_dim = 100
    num_epochs = 200
    batch_size = 128
    save_dir = './gan_results'

    # Ensure CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    mnist_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    dataloader = DataLoader(
        mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Initialize networks
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    torch.backends.cudnn.benchmark = True

    # Train
    g_losses, d_losses = train_gan(generator, discriminator, dataloader,
                                    num_epochs, latent_dim, device, save_dir)

    plot_losses(g_losses, d_losses, save_dir)

    print(f"Training complete. Results saved in {save_dir}")

if __name__ == "__main__":
    main()
