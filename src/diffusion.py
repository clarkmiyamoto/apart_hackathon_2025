import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt

class MLPScore(nn.Module):
    """MLP-based score function for score-based diffusion models."""
    
    def __init__(self, input_dim: int = 784, hidden_dims: list = [512, 512, 256], time_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Main MLP network
        layers = []
        prev_dim = input_dim + time_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer (score has same dimension as input)
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the score function.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            t: Time tensor of shape (batch_size, 1)
            
        Returns:
            Score tensor of shape (batch_size, input_dim)
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Concatenate input and time embedding
        x_t = torch.cat([x, t_emb], dim=-1)
        
        # Forward through MLP
        score = self.mlp(x_t)
        
        return score


class ScoreBasedDiffusion:
    """Score-based diffusion model implementation."""
    
    def __init__(self, 
                 score_fn: nn.Module,
                 beta_min: float = 0.1,
                 beta_max: float = 20.0,
                 N: int = 1000,
                 device: str = 'cuda'):
        """
        Initialize the score-based diffusion model.
        
        Args:
            score_fn: Score function (neural network)
            beta_min: Minimum noise schedule value
            beta_max: Maximum noise schedule value
            N: Number of diffusion steps
            device: Device to run on
        """
        self.score_fn = score_fn.to(device)
        self.device = device
        self.N = N
        
        # Noise schedule
        self.betas = torch.linspace(beta_min, beta_max, N).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) for given t.
        
        Args:
            x_start: Starting point x_0
            t: Time step
            noise: Optional noise to use (for reproducibility)
            
        Returns:
            Noisy sample x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the loss for training the score function.
        
        Args:
            x_start: Starting point x_0
            t: Time step
            noise: Optional noise to use
            
        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_score = self.score_fn(x_noisy, t.unsqueeze(-1).float() / self.N)
        
        # Score matching loss: ||s_θ(x_t, t) + ∇_x log q(x_t | x_0)||^2
        # ∇_x log q(x_t | x_0) = -ε / sqrt(1 - α_t)
        target_score = -noise / self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        loss = F.mse_loss(predicted_score, target_score)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """
        Sample from p_θ(x_{t-1} | x_t) using the score function.
        
        Args:
            x: Current sample x_t
            t: Current time step
            t_index: Current time index
            
        Returns:
            Sample x_{t-1}
        """
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t_index]
        
        # Predict score
        score = self.score_fn(x, t.unsqueeze(-1).float() / self.N)
        
        # Predict x_0
        pred_original = (x - sqrt_one_minus_alphas_cumprod_t * score) * sqrt_recip_alphas_cumprod_t
        
        # Compute mean of q(x_{t-1} | x_t, x_0)
        if t_index > 0:
            noise = torch.randn_like(x)
            sqrt_alphas_cumprod_prev_t = self.sqrt_alphas_cumprod[t_index - 1]
            posterior_variance_t = self.posterior_variance[t_index]
            mean = (
                self.posterior_mean_coef1[t_index] * pred_original +
                self.posterior_mean_coef2[t_index] * x
            )
            return mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return pred_original
    
    @torch.no_grad()
    def sample(self, batch_size: int = 1, image_size: int = 28) -> torch.Tensor:
        """
        Generate samples using the trained score function.
        
        Args:
            batch_size: Number of samples to generate
            image_size: Size of the image (assumed square)
            
        Returns:
            Generated samples of shape (batch_size, image_size * image_size)
        """
        self.score_fn.eval()
        
        # Start from pure noise
        x = torch.randn(batch_size, image_size * image_size).to(self.device)
        
        # Reverse diffusion process
        for i in reversed(range(self.N)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, i)
        
        return x
    
    def train_step(self, x_batch: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        Perform one training step.
        
        Args:
            x_batch: Batch of training data
            optimizer: Optimizer for the score function
            
        Returns:
            Loss value
        """
        self.score_fn.train()
        optimizer.zero_grad()
        
        batch_size = x_batch.shape[0]
        t = torch.randint(0, self.N, (batch_size,), device=self.device)
        
        loss = self.p_losses(x_batch, t)
        loss.backward()
        optimizer.step()
        
        return loss.item()


def visualize_samples(samples: torch.Tensor, num_samples: int = 16, image_size: int = 28):
    """
    Visualize generated samples.
    
    Args:
        samples: Generated samples of shape (batch_size, image_size * image_size)
        num_samples: Number of samples to visualize
        image_size: Size of the image
    """
    samples = samples[:num_samples].cpu().detach()
    samples = samples.view(-1, image_size, image_size)
    
    # Normalize to [0, 1] for visualization
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, sample in enumerate(samples):
        axes[i].imshow(sample, cmap='gray')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show() 