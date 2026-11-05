from typing import Tuple, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(
            self,
            output_dim: int,
            latent_dim: int = 100, 
        ):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            # Sigmoid is used to bound the outputs [0,1]
            nn.Sigmoid()  
        )
    
    def forward(self, z):
        return self.net(z)
    
    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim)
        return self(z)


class Discriminator(nn.Module):
    def __init__(
            self, 
            output_dim: int
        ):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),  
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def gradient_penalty(
            self, 
            real_data: torch.Tensor, 
            fake_data: torch.Tensor
        ) -> torch.Tensor:
        # Calculate interpolation
        alpha = torch.rand(real_data.size(0), 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(real_data.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad = True
        
        # Calculate validity of interpolated examples
        validity_interpolated = self(interpolated)
        fakes = torch.ones(validity_interpolated.size()).to(real_data.device)
        
        # Calculate gradients with respect to examples
        gradients = torch.autograd.grad(
            outputs=validity_interpolated,
            inputs=interpolated,
            grad_outputs=fakes,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    
def train_gan(
        generator: Generator, 
        discriminator: Discriminator, 
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        dataloader: DataLoader, 
        lambda_gp: float = 10,
        batch_size: int = 64, 
        latent_dim: int = 100, 
        n_epochs: int = 100,
        n_critic: int = 5,
        device: str = 'cpu'
    ):
        
    for epoch in range(n_epochs):
        for i, real_data in enumerate(dataloader):

            # Train Discriminator
            optimizer_D.zero_grad()
            batch_size = real_data.size(0)
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            validity_real = discriminator(real_data)
            validity_fake = discriminator(fake_data)
            gradient_penalty = discriminator.gradient_penalty(real_data.data, fake_data.data)                         
            d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            if i % n_critic == 0:
                fake_data = generator(z)
                validity_fake = discriminator(fake_data)
                g_loss = -torch.mean(validity_fake)
                g_loss.backward()
                optimizer_G.step()
        
        print(f"Epoch {epoch} | D loss: {d_loss.item()} | G loss: {g_loss.item()}")


    
