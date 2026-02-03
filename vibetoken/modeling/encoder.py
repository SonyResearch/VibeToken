"""Resolution-aware encoder for VibeToken.

Vision Transformer-based encoder with flexible patch sizes for variable-resolution inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange
from torch import Tensor, vmap
import numpy as np

from .blocks import ResidualAttentionBlock, _expand_token
from .embeddings import FuzzyEmbedding, to_2tuple


class ResolutionEncoder(nn.Module):
    """Vision Transformer encoder with flexible resolution support.
    
    Encodes images into latent tokens using a ViT architecture with
    support for variable input resolutions and patch sizes.
    """
    
    # Model size configurations
    MODEL_CONFIGS = {
        "small": {"width": 512, "num_layers": 8, "num_heads": 8},
        "base": {"width": 768, "num_layers": 12, "num_heads": 12},
        "large": {"width": 1024, "num_layers": 24, "num_heads": 16},
    }
    
    def __init__(self, config):
        """Initialize ResolutionEncoder.
        
        Args:
            config: OmegaConf config with model parameters.
        """
        super().__init__()
        self.config = config
        
        # Extract config values
        vq_config = config.model.vq_model if hasattr(config.model, 'vq_model') else config.model
        self.patch_size = getattr(vq_config, 'vit_enc_patch_size', 32)
        self.model_size = getattr(vq_config, 'vit_enc_model_size', 'large')
        self.num_latent_tokens = getattr(vq_config, 'num_latent_tokens', 256)
        self.token_size = getattr(vq_config, 'token_size', 256)
        self.is_legacy = getattr(vq_config, 'is_legacy', False)
        
        # Handle VAE mode (doubles token size for mean+std)
        quantize_mode = getattr(vq_config, 'quantize_mode', 'vq')
        if quantize_mode == "vae":
            self.token_size = self.token_size * 2
        
        # Get model dimensions from config
        model_cfg = self.MODEL_CONFIGS[self.model_size]
        self.width = model_cfg["width"]
        self.num_layers = model_cfg["num_layers"]
        self.num_heads = model_cfg["num_heads"]
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
            kernel_size=self.patch_size, stride=self.patch_size, bias=True
        )
        
        # Embeddings
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = FuzzyEmbedding(1024, scale, self.width)
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width)
        )
        self.ln_pre = nn.LayerNorm(self.width)
        
        # Transformer layers
        self.transformer = nn.ModuleList([
            ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)
        
        # Cache for pseudo-inverse matrices
        self.pinvs = {}

    def _resize(self, x: Tensor, shape: Tuple[int, int]) -> Tensor:
        """Bilinear resize of 2D tensor."""
        x_resized = F.interpolate(
            x[None, None, ...], shape, mode="bilinear", antialias=False
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(
        self, 
        old_shape: Tuple[int, int], 
        new_shape: Tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        """Calculate pseudo-inverse of resize matrix for FlexiViT."""
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape, device=device)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(self, patch_embed: Tensor, new_patch_size: Tuple[int, int]) -> Tensor:
        """Resize patch embedding kernel to new patch size (FlexiViT).
        
        Args:
            patch_embed: Original weight tensor (out_ch, in_ch, H, W).
            new_patch_size: Target (H, W) patch size.
            
        Returns:
            Resized weight tensor.
        """
        base_size = to_2tuple(self.patch_size)
        if base_size == new_patch_size:
            return patch_embed

        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(
                base_size, new_patch_size, device=patch_embed.device
            )
        pinv = self.pinvs[new_patch_size]

        def resample_patch_embed(pe: Tensor) -> Tensor:
            h, w = new_patch_size
            original_dtype = pe.dtype
            resampled = pinv @ pe.float().reshape(-1)
            return rearrange(resampled.to(original_dtype), "(h w) -> h w", h=h, w=w)

        v_resample = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)
        return v_resample(patch_embed)

    def apply_flexivit_patch_embed(self, x: Tensor, target_patch_size: Tuple[int, int]) -> Tensor:
        """Apply patch embedding with flexible patch size.
        
        Args:
            x: Input image tensor (B, 3, H, W).
            target_patch_size: Target patch size (H, W).
            
        Returns:
            Patch embeddings (B, C, grid_H, grid_W).
        """
        patch_size = to_2tuple(target_patch_size)
        
        if patch_size == to_2tuple(self.patch_size):
            weight = self.patch_embed.weight
        else:
            weight = self.resize_patch_embed(self.patch_embed.weight, patch_size)

        return F.conv2d(x, weight, bias=self.patch_embed.bias, stride=patch_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
        latent_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encode_patch_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Encode images to latent tokens.
        
        Args:
            pixel_values: Input images (B, 3, H, W), values in [0, 1].
            latent_tokens: Learnable latent tokens (num_latent, width).
            attention_mask: Optional attention mask.
            encode_patch_size: Optional custom patch size for encoding.
            
        Returns:
            Encoded latent features (B, token_size, 1, num_latent).
        """
        batch_size, _, H, W = pixel_values.shape
        
        # Determine patch size
        if encode_patch_size is None:
            target_patch_size = (self.patch_size, self.patch_size)
        elif isinstance(encode_patch_size, int):
            target_patch_size = (encode_patch_size, encode_patch_size)
        else:
            target_patch_size = encode_patch_size
        
        # Apply flexible patch embedding
        x = self.apply_flexivit_patch_embed(pixel_values, target_patch_size)
        
        # Flatten spatial dimensions
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # (B, num_patches, width)
        
        # Add class embedding
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        
        # Compute grid dimensions
        grid_height = H // target_patch_size[0]
        grid_width = W // target_patch_size[1]
        
        # Add positional embeddings to latent tokens
        num_latent = latent_tokens.shape[0]
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)[:num_latent]
        
        # Add positional embeddings to image patches
        x = x + self.positional_embedding(grid_height, grid_width, train=False, dtype=x.dtype)
        
        # Concatenate image patches and latent tokens
        x = torch.cat([x, latent_tokens], dim=1)
        
        # Pre-norm and reshape for transformer
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # (seq_len, B, width)
        
        # Apply transformer layers
        for layer in self.transformer:
            x = layer(x, attention_mask=None)
        
        x = x.permute(1, 0, 2)  # (B, seq_len, width)
        
        # Extract latent tokens
        latent_tokens = x[:, 1 + grid_height * grid_width:]
        latent_tokens = self.ln_post(latent_tokens)
        
        # Reshape and project to token size
        if self.is_legacy:
            latent_tokens = latent_tokens.reshape(batch_size, self.width, num_latent, 1)
        else:
            latent_tokens = latent_tokens.reshape(batch_size, num_latent, self.width, 1).permute(0, 2, 1, 3)
        
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, num_latent)
        
        return latent_tokens
