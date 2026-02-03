"""Vector Quantizer for VibeToken.

Simplified for inference-only use. Training-specific features removed.

Reference:
    https://github.com/CompVis/taming-transformers
    https://github.com/google-research/magvit
"""

from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.amp import autocast


class VectorQuantizer(nn.Module):
    """Vector Quantizer module for discrete tokenization.
    
    Converts continuous latent representations to discrete tokens using
    a learned codebook.
    """
    
    def __init__(
        self,
        codebook_size: int = 1024,
        token_size: int = 256,
        commitment_cost: float = 0.25,
        use_l2_norm: bool = False,
    ):
        """Initialize VectorQuantizer.
        
        Args:
            codebook_size: Number of entries in the codebook.
            token_size: Dimension of each codebook entry.
            commitment_cost: Weight for commitment loss (unused in inference).
            use_l2_norm: Whether to L2-normalize embeddings.
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.commitment_cost = commitment_cost
        self.use_l2_norm = use_l2_norm

        self.embedding = nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    @autocast('cuda', enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Quantize input tensor.
        
        Args:
            z: Input tensor of shape (B, C, H, W).
            
        Returns:
            z_quantized: Quantized tensor of shape (B, C, H, W).
            result_dict: Dictionary containing min_encoding_indices and losses.
        """
        z = z.float()
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> (b h w) c')

        if self.use_l2_norm:
            z_flattened = nn.functional.normalize(z_flattened, dim=-1)
            embedding = nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
            
        # Compute distances to codebook entries
        d = (torch.sum(z_flattened**2, dim=1, keepdim=True) + 
             torch.sum(embedding**2, dim=1) - 
             2 * torch.einsum('bd,dn->bn', z_flattened, embedding.T))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
            z_norm = nn.functional.normalize(z, dim=-1)
        else:
            z_norm = z

        # Compute losses (for compatibility, not used in inference)
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z_norm) ** 2)
        codebook_loss = torch.mean((z_quantized - z_norm.detach()) ** 2)
        loss = commitment_loss + codebook_loss

        # Straight-through estimator: preserve gradients
        z_quantized = z_norm + (z_quantized - z_norm).detach()

        # Reshape back to original format
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices.view(
                z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3]
            )
        )

        return z_quantized, result_dict

    @autocast('cuda', enabled=False)
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Get codebook entries for given indices.
        
        Args:
            indices: Token indices, shape (N,) or (N, vocab_size) for soft indices.
            
        Returns:
            Codebook entries, shape (N, token_size).
        """
        indices = indices.long()
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            # Soft indices (weighted sum of embeddings)
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding.weight)
        else:
            raise NotImplementedError(f"Unsupported indices shape: {indices.shape}")
            
        if self.use_l2_norm:
            z_quantized = nn.functional.normalize(z_quantized, dim=-1)
            
        return z_quantized


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution for VAE-style quantization.
    
    Used when quantize_mode='vae' instead of discrete VQ.
    """
    
    @autocast('cuda', enabled=False)
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        """Initialize Gaussian distribution.
        
        Args:
            parameters: Tensor of shape (B, 2*C, H, W) containing mean and logvar.
            deterministic: If True, sample() returns mean (no noise).
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.float(), 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    @autocast('cuda', enabled=False)
    def sample(self) -> torch.Tensor:
        """Sample from the distribution."""
        x = self.mean.float() + self.std.float() * torch.randn(
            self.mean.shape, device=self.parameters.device
        )
        return x

    @autocast('cuda', enabled=False)
    def mode(self) -> torch.Tensor:
        """Return the mode (mean) of the distribution."""
        return self.mean

    @autocast('cuda', enabled=False)
    def kl(self) -> torch.Tensor:
        """Compute KL divergence from standard Gaussian."""
        if self.deterministic:
            return torch.Tensor([0.0])
        return 0.5 * torch.sum(
            torch.pow(self.mean.float(), 2) + self.var.float() - 1.0 - self.logvar.float(),
            dim=[1, 2]
        )
