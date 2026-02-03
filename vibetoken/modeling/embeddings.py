"""Embedding modules for VibeToken.

Includes positional embeddings with fuzzy interpolation for variable resolutions.
"""

import math
from typing import Tuple, Any
import collections.abc
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
from einops import rearrange
from torch import Tensor, vmap


def to_2tuple(x: Any) -> Tuple:
    """Convert input to 2-tuple."""
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


class FuzzyEmbedding(nn.Module):
    """Fuzzy positional embedding with bilinear interpolation.
    
    Supports variable-resolution inputs by interpolating from a base
    positional embedding grid.
    """
    
    def __init__(
        self,
        grid_size: int,
        scale: float,
        width: int,
        apply_fuzzy: bool = False,
    ):
        """Initialize FuzzyEmbedding.
        
        Args:
            grid_size: Base grid size (must be 1024).
            scale: Initialization scale for embeddings.
            width: Embedding dimension.
            apply_fuzzy: Whether to add noise during training.
        """
        super().__init__()
        assert grid_size == 1024, "grid_size must be 1024"
        
        self.grid_size = grid_size
        self.scale = scale
        self.width = width
        self.apply_fuzzy = apply_fuzzy
        
        self.positional_embedding = nn.Parameter(scale * torch.randn(grid_size, width))
        self.class_positional_embedding = nn.Parameter(scale * torch.randn(1, width))

    @torch.amp.autocast('cuda', enabled=False)
    def forward(
        self,
        grid_height: int,
        grid_width: int,
        train: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compute positional embeddings for given grid size.
        
        Args:
            grid_height: Target grid height.
            grid_width: Target grid width.
            train: Whether in training mode (affects fuzzy noise).
            dtype: Output dtype.
            
        Returns:
            Positional embeddings of shape (1 + grid_height * grid_width, width).
        """
        device = self.positional_embedding.device
        
        meshx, meshy = torch.meshgrid(
            torch.arange(grid_height, device=device),
            torch.arange(grid_width, device=device),
            indexing='ij'
        )
        meshx = meshx.to(dtype)
        meshy = meshy.to(dtype)

        # Normalize coordinates to [-1, 1]
        meshx = 2 * (meshx / max(grid_height - 1, 1)) - 1
        meshy = 2 * (meshy / max(grid_width - 1, 1)) - 1
        
        if self.apply_fuzzy and train:
            noise_x = torch.rand_like(meshx) * 0.0008 - 0.0004
            noise_y = torch.rand_like(meshy) * 0.0008 - 0.0004
            meshx = meshx + noise_x
            meshy = meshy + noise_y
        
        grid = torch.stack((meshy, meshx), 2).unsqueeze(0)
        
        # Reshape positional embedding to 2D grid
        base_size = int(math.sqrt(self.grid_size))
        positional_embedding = einops.rearrange(
            self.positional_embedding, "(h w) d -> d h w", h=base_size, w=base_size
        )
        positional_embedding = positional_embedding.to(dtype).unsqueeze(0)

        # Bilinear interpolation
        fuzzy_embedding = F.grid_sample(positional_embedding, grid, align_corners=False)
        fuzzy_embedding = fuzzy_embedding.to(dtype)
        fuzzy_embedding = einops.rearrange(fuzzy_embedding, "b d h w -> b (h w) d").squeeze(0)

        final_embedding = torch.cat([self.class_positional_embedding, fuzzy_embedding], dim=0)
        return final_embedding


class FlexiPatchEmbed:
    """Flexible patch embedding utilities for variable patch sizes.
    
    Based on FlexiViT: https://arxiv.org/abs/2212.08013
    """
    
    def __init__(self, base_patch_size: int, width: int):
        """Initialize FlexiPatchEmbed.
        
        Args:
            base_patch_size: Original patch size of the model.
            width: Embedding dimension.
        """
        self.base_patch_size = base_patch_size
        self.width = width
        self.pinvs = {}
    
    def _resize(self, x: Tensor, shape: Tuple[int, int]) -> Tensor:
        """Bilinear resize of 2D tensor."""
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode="bilinear",
            antialias=False,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(
        self, 
        old_shape: Tuple[int, int], 
        new_shape: Tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        """Calculate pseudo-inverse of resize matrix."""
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape, device=device)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(
        self, 
        patch_embed: Tensor, 
        base_patch_size: Tuple[int, int],
        new_patch_size: Tuple[int, int],
    ) -> Tensor:
        """Resize patch embedding kernel to new patch size.
        
        Args:
            patch_embed: Original patch embedding weight (out_ch, in_ch, H, W).
            base_patch_size: Original patch size.
            new_patch_size: Target patch size.
            
        Returns:
            Resized patch embedding weight.
        """
        if base_patch_size == new_patch_size:
            return patch_embed

        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(
                base_patch_size, new_patch_size, device=patch_embed.device
            )
        pinv = self.pinvs[new_patch_size]

        def resample_patch_embed(patch_embed: Tensor):
            h, w = new_patch_size
            original_dtype = patch_embed.dtype
            patch_embed_float = patch_embed.float()
            resampled_kernel = pinv @ patch_embed_float.reshape(-1)
            resampled_kernel = resampled_kernel.to(original_dtype)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)
        return v_resample_patch_embed(patch_embed)
