"""Encoder and decoder building blocks for VibeToken.

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    https://github.com/baofff/U-ViT/blob/main/libs/timm.py
"""

import random
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
import einops
from einops.layers.torch import Rearrange
from typing import Optional, Sequence, Tuple, Union
from modeling.modules.fuzzy_embedding import FuzzyEmbedding
import collections.abc
from itertools import repeat
from typing import Any
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from torch import vmap
from torch import Tensor

def to_2tuple(x: Any) -> Tuple:
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))

class PatchMixture():
    def __init__(self, seed=42):
        self.seed = seed
        
    def get_mask(self, x, mask_ratio=0.0, l1_reg=0.0, inverse=False):
        batch_size, num_patches, _ = x.shape
        device = x.device
        num_mask = int(num_patches * mask_ratio)
        num_keep = num_patches - num_mask
        token_magnitudes = x.abs().sum(dim=-1)
        min_mags = token_magnitudes.min(dim=1, keepdim=True)[0]
        max_mags = token_magnitudes.max(dim=1, keepdim=True)[0]
        token_magnitudes = (token_magnitudes - min_mags) / (max_mags - min_mags + 1e-8)
        if inverse:
            adjusted_magnitudes = 1.0 - token_magnitudes
        else:
            adjusted_magnitudes = token_magnitudes
        noise_random = torch.rand(batch_size, num_patches, device=device)
        noise = (1.0 - l1_reg) * noise_random + l1_reg * adjusted_magnitudes
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        ids_mask = ids_shuffle[:, num_keep:]
        mask = torch.ones((batch_size, num_patches), device=device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)
        return {
            'mask': mask,           
            'ids_keep': ids_keep,
            'ids_mask': ids_mask,
            'ids_shuffle': ids_shuffle,
            'ids_restore': ids_restore
        }
    
    def start_route(self, x, mask_info):
        ids_shuffle = mask_info['ids_shuffle']
        num_keep = mask_info['ids_keep'].size(1)
        batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        x_shuffled = x.gather(1, ids_shuffle.unsqueeze(-1).expand(-1, -1, x.size(2)))
        masked_x = x_shuffled[:, :num_keep, :]
        return masked_x
    
    def end_route(self, masked_x, mask_info, original_x=None, mask_token=0.0):
        batch_size, num_patches = mask_info['mask'].shape
        num_keep = masked_x.size(1)
        dim = masked_x.size(2)
        device = masked_x.device
        ids_restore = mask_info['ids_restore']
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1)
        x_unshuffled = torch.empty((batch_size, num_patches, dim), device=device)
        x_unshuffled[:, :num_keep, :] = masked_x
        if original_x is not None:
            x_shuffled = original_x.gather(1, mask_info['ids_shuffle'].unsqueeze(-1).expand(-1, -1, dim))
            x_unshuffled[:, num_keep:, :] = x_shuffled[:, num_keep:, :]
        else:
            x_unshuffled[:, num_keep:, :].fill_(mask_token)
        x_unmasked = x_unshuffled.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, dim))
        return x_unmasked

class ResizableBlur(nn.Module):
    """
    Single-parameter anti‑aliasing layer.
    Call with scale=1,2,4 to downsample by 1× (identity), 2×, or 4×.
    """
    def __init__(self, channels: int,
                 max_kernel_size: int = 9,
                 init_type: str = "gaussian"):
        super().__init__()
        self.C = channels
        K = max_kernel_size                       # e.g. 9 for 4×
        assert K % 2 == 1, "kernel must be odd"

        # ----- initialise the largest kernel ---------------------------------
        if init_type == "gaussian":
            # 2‑D separable Gaussian, σ≈K/6
            ax = torch.arange(-(K//2), K//2 + 1)
            g1d = torch.exp(-0.5 * (ax / (K/6.0))**2)
            g2d = torch.outer(g1d, g1d)
            kernel = g2d / g2d.sum()
        elif init_type == "lanczos":
            a = K//2                           # window size parameter
            x = torch.arange(-a, a+1).float()
            sinc = lambda t: torch.where(t==0, torch.ones_like(t), torch.sin(torch.pi*t)/(torch.pi*t))
            k1d = sinc(x) * sinc(x/a)
            k2d = torch.outer(k1d, k1d)
            kernel = k2d / k2d.sum()
        else:
            raise ValueError("unknown init_type")

        # learnable base kernel (shape 1×1×K×K)
        self.weight = nn.Parameter(kernel.unsqueeze(0).unsqueeze(0))

    # ------------------------------------------------------------------------
    @staticmethod
    def _resize_and_normalise(weight: torch.Tensor, k_size: int) -> torch.Tensor:
        """
        Bilinearly interpolate weight (B,C,H,W) to target k_size×k_size,
        then L1‑normalise over spatial dims so Σ=1.
        """
        if weight.shape[-1] != k_size:
            weight = F.interpolate(weight, size=(k_size, k_size),
                                   mode="bilinear", align_corners=True)
        weight = weight / weight.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        return weight

    # ------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, input_size, target_size) -> torch.Tensor:
        # Unpack input and target dimensions
        input_h, input_w = input_size
        target_h, target_w = target_size
        
        # Calculate scale factors for height and width
        scale_h = input_h / target_h
        scale_w = input_w / target_w
        
        # Determine kernel size based on scale factors
        # Larger scale factors need larger kernels for better anti-aliasing
        k_size_h = min(self.weight.shape[-1], max(1, int(2 * scale_h + 3)))
        k_size_w = min(self.weight.shape[-1], max(1, int(2 * scale_w + 3)))
        
        # Make sure kernel sizes are odd
        k_size_h = k_size_h if k_size_h % 2 == 1 else k_size_h + 1
        k_size_w = k_size_w if k_size_w % 2 == 1 else k_size_w + 1
        
        # Use the maximum for a square kernel, or create a rectangular kernel if needed
        k_size = max(k_size_h, k_size_w)
        
        # Calculate appropriate stride and padding
        stride_h = max(1, round(scale_h))
        stride_w = max(1, round(scale_w))
        pad_h = k_size_h // 2
        pad_w = k_size_w // 2
        
        # Get the kernel and normalize it
        k = self._resize_and_normalise(self.weight, k_size)  # (1,1,k,k)
        k = k.repeat(self.C, 1, 1, 1)  # depth-wise
        
        # Apply convolution with calculated parameters
        result = F.conv2d(x, weight=k, stride=(stride_h, stride_w),
                          padding=(pad_h, pad_w), groups=self.C)
        
        # If the convolution didn't get us exactly to the target size, use interpolation for fine adjustment
        if result.shape[2:] != target_size:
            result = F.interpolate(result, size=target_size, mode='bilinear', align_corners=True)
            
        return result

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ):
        return self.attn(x, x, x, attn_mask=attention_mask, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ):
        attn_output = self.attention(x=self.ln_1(x), attention_mask=attention_mask)
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UViTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class ResolutionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size 
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.model_size = config.model.vq_model.vit_enc_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.apply_fuzzy = config.model.vq_model.get("apply_fuzzy", False)
        self.patch_mixture_start_layer = config.model.vq_model.get("patch_mixture_start_layer", 100)
        self.patch_mixture_end_layer = config.model.vq_model.get("patch_mixture_end_layer", 100)

        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std

        self.is_legacy = config.model.vq_model.get("is_legacy", True)

        self.width = {
                "tiny": 256,
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
              kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))

        self.positional_embedding = FuzzyEmbedding(1024, scale, self.width)
        
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)

        self.patch_mixture = PatchMixture()

        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))

        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)
        self.pinvs = {}

    def apply_flexivit_patch_embed(self, x, target_patch_size):
        patch_size = to_2tuple(target_patch_size)

        # Resize conv weights
        if patch_size == to_2tuple(self.patch_size):
            weight = self.patch_embed.weight
        else:
            weight = self.resize_patch_embed(self.patch_embed.weight, patch_size)

        # Apply conv with resized weights
        x = F.conv2d(x, weight, bias=self.patch_embed.bias, stride=patch_size)
        return x

    def _resize(self, x: Tensor, shape: Tuple[int, int]) -> Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode="bilinear",
            antialias=False,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(
        self, old_shape: Tuple[int, int], new_shape: Tuple[int, int], device=None
    ) -> Tensor:
        # Use the device from patch_embed weights if available
        if device is None and hasattr(self, 'patch_embed'):
            device = self.patch_embed.weight.device
        
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape, device=device)  # Specify device here
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(self, patch_embed: Tensor, new_patch_size: Tuple[int, int]):
        """Resize patch_embed to target resolution via pseudo-inverse resizing"""
        # Return original kernel if no resize is necessary
        if to_2tuple(self.patch_size) == new_patch_size:
            return patch_embed

        # Calculate pseudo-inverse of resize matrix
        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(
                to_2tuple(self.patch_size), new_patch_size, device=patch_embed.device
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

    def get_attention_mask(self, target_shape, attention_mask):
        # Create mask for mask_tokens (all True since we want to attend to all mask tokens)
        mask_token_mask = torch.ones(target_shape).to(attention_mask.device)
        # Combine with input attention mask
        attention_mask = torch.cat((mask_token_mask, attention_mask), dim=1).bool()
        sequence_length = attention_mask.shape[1]
        
        # Create causal attention mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        attention_mask = attention_mask.expand(
            attention_mask.shape[0], 
            self.num_heads,
            sequence_length, 
            sequence_length
        )
        
        # Reshape to [B*num_heads, S, S]
        attention_mask = attention_mask.reshape(
            -1, sequence_length, sequence_length
        )
        
        # Convert boolean mask to float
        attention_mask = attention_mask.float()
        
        # Convert mask values: True -> 0.0, False -> -inf
        attention_mask = attention_mask.masked_fill(
            ~attention_mask.bool(), 
            float('-inf')
        )
        return attention_mask

    def forward(self, pixel_values, latent_tokens, attention_mask=None, encode_patch_size=None, train=True):
        batch_size, _, H, W = pixel_values.shape
        x = pixel_values

        # Apply dynamic patch embedding
        # Determine patch size dynamically based on image resolution
        # Base patch size (32) is for 512x512 images
        # Scale proportionally for other resolutions to maintain ~256 tokens
        base_resolution = 512

        if encode_patch_size is None:
            base_patch_size = random.choice([16, 32])
            target_patch_size = min(int(min(H, W) / base_resolution * base_patch_size), 32) # we force it to be at most 32 otherwise we lose information
        else:
            target_patch_size = encode_patch_size
        
        if isinstance(target_patch_size, int):
            target_patch_size = (target_patch_size, target_patch_size)

        x = self.apply_flexivit_patch_embed(x, target_patch_size)
        
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        # create image_rotary_emb
        grid_height = H // target_patch_size[0]
        grid_width = W // target_patch_size[1]

        mask_ratio = 0.0
        if grid_height*grid_width > 256 and train:
            mask_ratio = torch.empty(1).uniform_(0.5, 0.7).item()

        num_latent_tokens = latent_tokens.shape[0]
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)[:num_latent_tokens]

        x = x + self.positional_embedding(grid_height, grid_width, train=train, dtype=x.dtype)

        # apply attention_mask before concatenating x and latent_tokens
        if attention_mask is not None:
            key_attention_mask = attention_mask.clone()
            attention_mask = self.get_attention_mask((batch_size, x.shape[1]), key_attention_mask)
            full_seq_attention_mask = attention_mask.clone()
        else:
            key_attention_mask = None    
            full_seq_attention_mask = None

        # Concatenate x and latent_tokens first
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            if i == self.patch_mixture_start_layer:
                x = x.permute(1, 0, 2)
                x_D_last = x[:, 1:grid_height*grid_width+1].clone()
                mask_info = self.patch_mixture.get_mask(x[:, 1:grid_height*grid_width+1], mask_ratio=mask_ratio)
                new_x = self.patch_mixture.start_route(x, mask_info)
                x = torch.cat([x[:, :1], new_x, x[:, grid_height*grid_width+1:]], dim=1)
                x = x.permute(1, 0, 2)
                if key_attention_mask is not None:
                    attention_mask = self.get_attention_mask((batch_size, 1+new_x.shape[1]), key_attention_mask)
                else:
                    attention_mask = None

            x = self.transformer[i](x, attention_mask=attention_mask)

            if i == self.patch_mixture_end_layer:
                x = x.permute(1, 0, 2)
                new_x = self.patch_mixture.end_route(x[:, 1:-self.num_latent_tokens], mask_info, original_x=x_D_last)
                x = torch.cat([x[:, :1], new_x, x[:, -self.num_latent_tokens:]], dim=1)
                x = x.permute(1, 0, 2)
                if full_seq_attention_mask is not None:
                    attention_mask = full_seq_attention_mask.clone()
                else:
                    attention_mask = None

        x = x.permute(1, 0, 2)  # LND -> NLD
        
        latent_tokens = x[:, 1+grid_height*grid_width:]
        latent_tokens = self.ln_post(latent_tokens)

        # fake 2D shape
        if self.is_legacy:
            latent_tokens = latent_tokens.reshape(batch_size, self.width, num_latent_tokens, 1)
        else:
            # Fix legacy problem.
            latent_tokens = latent_tokens.reshape(batch_size, num_latent_tokens, self.width, 1).permute(0, 2, 1, 3)
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, num_latent_tokens)
        return latent_tokens

# Keep the original TiTokEncoder as a legacy class
class TiTokEncoder(ResolutionEncoder):
    """Legacy TiTokEncoder - now inherits from ResolutionEncoder for backward compatibility"""
    pass

class ResolutionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.apply_fuzzy = config.model.vq_model.get("apply_fuzzy", False)
        self.patch_mixture_start_layer = config.model.vq_model.get("patch_mixture_start_layer", 100)
        self.patch_mixture_end_layer = config.model.vq_model.get("patch_mixture_end_layer", 100)
        
        self.is_legacy = config.model.vq_model.get("is_legacy", True)
        self.width = {
                "tiny": 256,
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        
        self.positional_embedding = FuzzyEmbedding(1024, scale, self.width)

        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)

        self.patch_mixture = PatchMixture()
 
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        if self.is_legacy:
            raise NotImplementedError("Legacy mode is not implemented for ResolutionDecoder")
        else:
            # Directly predicting RGB pixels
            self.ffn = nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True)
            self.rearrange = Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                p1 = self.patch_size, p2 = self.patch_size)
            self.down_scale = ResizableBlur(channels=3, max_kernel_size=9, init_type="lanczos")
            self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)

    def get_attention_mask(self, target_shape, attention_mask):
        # Create mask for mask_tokens (all True since we want to attend to all mask tokens)
        mask_token_mask = torch.ones(target_shape).to(attention_mask.device)
        # Combine with input attention mask
        attention_mask = torch.cat((mask_token_mask, attention_mask), dim=1).bool()
        sequence_length = attention_mask.shape[1]
        
        # Create causal attention mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        attention_mask = attention_mask.expand(
            attention_mask.shape[0], 
            self.num_heads,
            sequence_length, 
            sequence_length
        )
        
        # Reshape to [B*num_heads, S, S]
        attention_mask = attention_mask.reshape(
            -1, sequence_length, sequence_length
        )
        
        # Convert boolean mask to float
        attention_mask = attention_mask.float()
        
        # Convert mask values: True -> 0.0, False -> -inf
        attention_mask = attention_mask.masked_fill(
            ~attention_mask.bool(), 
            float('-inf')
        )
        return attention_mask
    
    def forward(self, z_quantized, attention_mask=None, height=None, width=None, decode_patch_size=None, train=True):
        N, C, H, W = z_quantized.shape
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        if height is None:
            height = self.image_size
        if width is None:
            width = self.image_size

        # create image_rotary_emb
        if decode_patch_size is None:
            # Calculate total area and determine appropriate patch size
            total_pixels = height * width
            
            # Target patch counts between 256 and 1024
            min_patches = 256
            max_patches = 1024
            
            # Calculate possible patch sizes that would give us patch counts in our target range
            possible_patch_sizes = []
            for patch_size in [8, 16, 32]:
                grid_h = height // patch_size
                grid_w = width // patch_size
                total_patches = grid_h * grid_w
                if min_patches <= total_patches <= max_patches:
                    possible_patch_sizes.append(patch_size)
            
            if not possible_patch_sizes:
                # If no patch size gives us the desired range, pick the one closest to our target range
                patch_counts = []
                for patch_size in [8, 16, 32]:
                    grid_h = height // patch_size
                    grid_w = width // patch_size
                    patch_counts.append((patch_size, grid_h * grid_w))
                
                # Sort by how close the patch count is to our target range
                patch_counts.sort(key=lambda x: min(abs(x[1] - min_patches), abs(x[1] - max_patches)))
                possible_patch_sizes = [patch_counts[0][0]]
            
            selected_patch_size = random.choice(possible_patch_sizes)
        else:
            selected_patch_size = decode_patch_size

        if isinstance(selected_patch_size, int):
            selected_patch_size = (selected_patch_size, selected_patch_size)
        
        grid_height = height // selected_patch_size[0]
        grid_width = width // selected_patch_size[1]

        # if grid_height*grid_width>1024 and train:
        #     grid_height = 32
        #     grid_width = 32

        mask_ratio = 0.0
        if grid_height*grid_width > 256 and train:
            mask_ratio = torch.empty(1).uniform_(0.5, 0.7).item()

        mask_tokens = self.mask_token.repeat(batchsize, grid_height*grid_width, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        
        mask_tokens = mask_tokens + self.positional_embedding(grid_height, grid_width, train=train).to(mask_tokens.dtype)
        
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)

        if attention_mask is not None:
            key_attention_mask = attention_mask.clone()
            attention_mask = self.get_attention_mask((batchsize, 1+grid_height*grid_width), key_attention_mask)
            full_seq_attention_mask = attention_mask.clone()
        else:
            key_attention_mask = None
            full_seq_attention_mask = None

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            if i == self.patch_mixture_start_layer:
                x = x.permute(1, 0, 2)
                x_D_last = x[:, 1:grid_height*grid_width+1].clone()
                mask_info = self.patch_mixture.get_mask(x[:, 1:grid_height*grid_width+1], mask_ratio=mask_ratio)
                new_x = self.patch_mixture.start_route(x, mask_info)
                x = torch.cat([x[:, :1], new_x, x[:, grid_height*grid_width+1:]], dim=1)
                x = x.permute(1, 0, 2)
                if key_attention_mask is not None:
                    attention_mask = self.get_attention_mask((batchsize, 1+new_x.shape[1]), key_attention_mask)
                else:
                    attention_mask = None

            x = self.transformer[i](x, attention_mask=attention_mask)

            if i == self.patch_mixture_end_layer:
                x = x.permute(1, 0, 2)
                new_x = self.patch_mixture.end_route(x[:, 1:-self.num_latent_tokens], mask_info, original_x=x_D_last)
                x = torch.cat([x[:, :1], new_x, x[:, -self.num_latent_tokens:]], dim=1)
                x = x.permute(1, 0, 2)
                if full_seq_attention_mask is not None:
                    attention_mask = full_seq_attention_mask.clone()
                else:
                    attention_mask = None

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+grid_height*grid_width] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, grid_height, grid_width)
        x = self.ffn(x.contiguous())
        x = self.rearrange(x)
        _, _, org_h, org_w = x.shape
        x = self.down_scale(x, input_size=(org_h, org_w), target_size=(height, width))
        x = self.conv_out(x)

        return x

# Keep the original TiTokDecoder as a legacy class that inherits from ResolutionDecoder
class TiTokDecoder(ResolutionDecoder):
    """Legacy TiTokDecoder - now inherits from ResolutionDecoder for backward compatibility"""
    
    def __init__(self, config):
        # Override config to disable patch mixture and other advanced features for legacy mode
        config_copy = type(config)()
        for attr in dir(config):
            if not attr.startswith('__'):
                try:
                    setattr(config_copy, attr, getattr(config, attr))
                except:
                    pass
        
        # Disable patch mixture for legacy mode
        if hasattr(config_copy.model.vq_model, 'patch_mixture_start_layer'):
            config_copy.model.vq_model.patch_mixture_start_layer = -1
        if hasattr(config_copy.model.vq_model, 'patch_mixture_end_layer'):
            config_copy.model.vq_model.patch_mixture_end_layer = -1
            
        super().__init__(config_copy)
        
        # Override grid_size for legacy compatibility
        self.grid_size = self.image_size // self.patch_size
        
        # Replace ResolutionDecoder's advanced final layers with legacy ones if needed
        if self.is_legacy:
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv2d(2 * self.width, 1024, 1, padding=0, bias=True),
            )
            self.conv_out = nn.Identity()
        else:
            # Use simpler final layers for backward compatibility
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
                Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                    p1 = self.patch_size, p2 = self.patch_size),)
            self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)
    
    def forward(self, z_quantized, attention_mask=None, height=None, width=None, decode_patch_size=None, train=True):
        # Legacy compatibility: use fixed grid size if height/width not provided
        if height is None:
            height = self.image_size
        if width is None:
            width = self.image_size
            
        # Force decode_patch_size to be the original patch_size for legacy compatibility
        if decode_patch_size is None:
            decode_patch_size = self.patch_size
            
        # Use the parent's forward method but with legacy parameters
        return super().forward(z_quantized, attention_mask, height, width, decode_patch_size, train)


class TATiTokDecoder(ResolutionDecoder):
    def __init__(self, config):
        super().__init__(config)
        scale = self.width ** -0.5
        self.text_context_length = config.model.vq_model.get("text_context_length", 77)
        self.text_embed_dim = config.model.vq_model.get("text_embed_dim", 768)
        self.text_guidance_proj = nn.Linear(self.text_embed_dim, self.width)
        self.text_guidance_positional_embedding = nn.Parameter(scale * torch.randn(self.text_context_length, self.width))
        
        # Add grid_size for backward compatibility
        self.grid_size = self.image_size // self.patch_size

    def forward(self, z_quantized, text_guidance, attention_mask=None, height=None, width=None, decode_patch_size=None, train=True):
        N, C, H, W = z_quantized.shape
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape
        
        # Use fixed grid size for backward compatibility
        if height is None:
            height = self.image_size
        if width is None:
            width = self.image_size
        if decode_patch_size is None:
            decode_patch_size = self.patch_size
            
        grid_height = height // decode_patch_size
        grid_width = width // decode_patch_size

        mask_tokens = self.mask_token.repeat(batchsize, grid_height*grid_width, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding(grid_height, grid_width, train=train).to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)

        text_guidance = self.text_guidance_proj(text_guidance)
        text_guidance = text_guidance + self.text_guidance_positional_embedding
        x = torch.cat([x, text_guidance], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+grid_height*grid_width] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, grid_height, grid_width)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x


class WeightTiedLMHead(nn.Module):
    def __init__(self, embeddings, target_codebook_size):
        super().__init__()
        self.weight = embeddings.weight
        self.target_codebook_size = target_codebook_size

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        # Get the weights for the target codebook size
        weight = self.weight[:self.target_codebook_size]  # Shape: [target_codebook_size, embed_dim]
        # Compute the logits by matrix multiplication
        logits = torch.matmul(x, weight.t())  # Shape: [batch_size, seq_len, target_codebook_size]
        return logits


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)