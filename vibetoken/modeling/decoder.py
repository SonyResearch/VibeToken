"""Resolution-aware decoder for VibeToken.

Vision Transformer-based decoder with flexible output resolutions.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops.layers.torch import Rearrange

from .blocks import ResidualAttentionBlock, ResizableBlur, _expand_token
from .embeddings import FuzzyEmbedding


class ResolutionDecoder(nn.Module):
    """Vision Transformer decoder with flexible resolution support.
    
    Decodes latent tokens back to images with support for variable
    output resolutions and patch sizes.
    """
    
    # Model size configurations
    MODEL_CONFIGS = {
        "small": {"width": 512, "num_layers": 8, "num_heads": 8},
        "base": {"width": 768, "num_layers": 12, "num_heads": 12},
        "large": {"width": 1024, "num_layers": 24, "num_heads": 16},
    }
    
    def __init__(self, config):
        """Initialize ResolutionDecoder.
        
        Args:
            config: OmegaConf config with model parameters.
        """
        super().__init__()
        self.config = config
        
        # Extract config values
        vq_config = config.model.vq_model if hasattr(config.model, 'vq_model') else config.model
        self.image_size = getattr(config.dataset.preprocessing, 'crop_size', 512) if hasattr(config, 'dataset') else 512
        self.patch_size = getattr(vq_config, 'vit_dec_patch_size', 32)
        self.model_size = getattr(vq_config, 'vit_dec_model_size', 'large')
        self.num_latent_tokens = getattr(vq_config, 'num_latent_tokens', 256)
        self.token_size = getattr(vq_config, 'token_size', 256)
        self.is_legacy = getattr(vq_config, 'is_legacy', False)
        
        if self.is_legacy:
            raise NotImplementedError("Legacy mode is not supported in this inference-only version")
        
        # Get model dimensions
        model_cfg = self.MODEL_CONFIGS[self.model_size]
        self.width = model_cfg["width"]
        self.num_layers = model_cfg["num_layers"]
        self.num_heads = model_cfg["num_heads"]
        
        # Input projection
        self.decoder_embed = nn.Linear(self.token_size, self.width, bias=True)
        
        # Embeddings
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = FuzzyEmbedding(1024, scale, self.width)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
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
        self.ffn = nn.Conv2d(
            self.width, self.patch_size * self.patch_size * 3, 
            kernel_size=1, padding=0, bias=True
        )
        self.rearrange = Rearrange(
            'b (p1 p2 c) h w -> b c (h p1) (w p2)',
            p1=self.patch_size, p2=self.patch_size
        )
        self.down_scale = ResizableBlur(channels=3, max_kernel_size=9, init_type="lanczos")
        self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)

    def _select_patch_size(self, height: int, width: int) -> int:
        """Select appropriate patch size based on target resolution.
        
        Args:
            height: Target image height.
            width: Target image width.
            
        Returns:
            Selected patch size.
        """
        total_pixels = height * width
        min_patches, max_patches = 256, 1024
        
        possible_sizes = []
        for ps in [8, 16, 32]:
            grid_h = height // ps
            grid_w = width // ps
            total_patches = grid_h * grid_w
            if min_patches <= total_patches <= max_patches:
                possible_sizes.append(ps)
        
        if not possible_sizes:
            # Find closest to target range
            patch_counts = []
            for ps in [8, 16, 32]:
                grid_h = height // ps
                grid_w = width // ps
                patch_counts.append((ps, grid_h * grid_w))
            patch_counts.sort(key=lambda x: min(abs(x[1] - min_patches), abs(x[1] - max_patches)))
            return patch_counts[0][0]
        
        return possible_sizes[0]

    def forward(
        self,
        z_quantized: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        decode_patch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode latent tokens to images.
        
        Args:
            z_quantized: Quantized latent features (B, C, H, W).
            attention_mask: Optional attention mask.
            height: Target image height.
            width: Target image width.
            decode_patch_size: Optional custom patch size for decoding.
            
        Returns:
            Decoded images (B, 3, height, width), values in [0, 1].
        """
        N, C, H, W = z_quantized.shape
        
        # Reshape and project input
        x = z_quantized.reshape(N, C * H, W).permute(0, 2, 1)  # (N, seq_len, C*H)
        x = self.decoder_embed(x)
        
        batchsize, seq_len, _ = x.shape
        
        # Default output size
        if height is None:
            height = self.image_size
        if width is None:
            width = self.image_size
        
        # Determine patch size
        if decode_patch_size is None:
            selected_patch_size = self._select_patch_size(height, width)
        else:
            selected_patch_size = decode_patch_size
        
        if isinstance(selected_patch_size, int):
            selected_patch_size = (selected_patch_size, selected_patch_size)
        
        grid_height = height // selected_patch_size[0]
        grid_width = width // selected_patch_size[1]
        
        # Create mask tokens for output positions
        mask_tokens = self.mask_token.repeat(batchsize, grid_height * grid_width, 1).to(x.dtype)
        mask_tokens = torch.cat([
            _expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
            mask_tokens
        ], dim=1)
        
        # Add positional embeddings
        mask_tokens = mask_tokens + self.positional_embedding(
            grid_height, grid_width, train=False
        ).to(mask_tokens.dtype)
        
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)
        
        # Pre-norm and reshape for transformer
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # (seq_len, B, width)
        
        # Apply transformer layers
        for layer in self.transformer:
            x = layer(x, attention_mask=None)
        
        x = x.permute(1, 0, 2)  # (B, seq_len, width)
        
        # Extract output tokens (excluding class token and latent tokens)
        x = x[:, 1:1 + grid_height * grid_width]
        x = self.ln_post(x)
        
        # Reshape to spatial format and project to pixels
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, grid_height, grid_width)
        x = self.ffn(x.contiguous())
        x = self.rearrange(x)
        
        # Downsample to target resolution
        _, _, org_h, org_w = x.shape
        x = self.down_scale(x, input_size=(org_h, org_w), target_size=(height, width))
        x = self.conv_out(x)
        
        return x
