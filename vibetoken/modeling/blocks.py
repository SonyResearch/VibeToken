"""Transformer building blocks for VibeToken.

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    https://github.com/baofff/U-ViT/blob/main/libs/timm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional
import einops


# Determine attention mode based on available implementations
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except ImportError:
        ATTENTION_MODE = 'math'


class Attention(nn.Module):
    """Multi-head self-attention with support for flash/xformers/math backends."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        else:  # math
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ResidualAttentionBlock(nn.Module):
    """Residual attention block with MLP."""
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        
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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.attn(x, x, x, attn_mask=attention_mask, need_weights=False)[0]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_output = self.attention(x=self.ln_1(x), attention_mask=attention_mask)
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """MLP block with GELU activation."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UViTBlock(nn.Module):
    """U-ViT block with optional skip connection."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
        skip: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip, use_reentrant=False)
        return self._forward(x, skip)

    def _forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ResizableBlur(nn.Module):
    """Anti-aliasing layer for downsampling with learnable blur kernel."""
    
    def __init__(
        self,
        channels: int,
        max_kernel_size: int = 9,
        init_type: str = "gaussian",
    ):
        super().__init__()
        self.C = channels
        K = max_kernel_size
        assert K % 2 == 1, "kernel must be odd"

        if init_type == "gaussian":
            ax = torch.arange(-(K // 2), K // 2 + 1)
            g1d = torch.exp(-0.5 * (ax / (K / 6.0)) ** 2)
            g2d = torch.outer(g1d, g1d)
            kernel = g2d / g2d.sum()
        elif init_type == "lanczos":
            a = K // 2
            x = torch.arange(-a, a + 1).float()
            sinc = lambda t: torch.where(
                t == 0, torch.ones_like(t), 
                torch.sin(torch.pi * t) / (torch.pi * t)
            )
            k1d = sinc(x) * sinc(x / a)
            k2d = torch.outer(k1d, k1d)
            kernel = k2d / k2d.sum()
        else:
            raise ValueError(f"Unknown init_type: {init_type}")

        self.weight = nn.Parameter(kernel.unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _resize_and_normalise(weight: torch.Tensor, k_size: int) -> torch.Tensor:
        if weight.shape[-1] != k_size:
            weight = F.interpolate(weight, size=(k_size, k_size), mode="bilinear", align_corners=True)
        weight = weight / weight.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        return weight

    def forward(self, x: torch.Tensor, input_size: tuple, target_size: tuple) -> torch.Tensor:
        input_h, input_w = input_size
        target_h, target_w = target_size
        
        scale_h = input_h / target_h
        scale_w = input_w / target_w
        
        k_size_h = min(self.weight.shape[-1], max(1, int(2 * scale_h + 3)))
        k_size_w = min(self.weight.shape[-1], max(1, int(2 * scale_w + 3)))
        k_size_h = k_size_h if k_size_h % 2 == 1 else k_size_h + 1
        k_size_w = k_size_w if k_size_w % 2 == 1 else k_size_w + 1
        k_size = max(k_size_h, k_size_w)
        
        stride_h = max(1, round(scale_h))
        stride_w = max(1, round(scale_w))
        pad_h = k_size_h // 2
        pad_w = k_size_w // 2
        
        k = self._resize_and_normalise(self.weight, k_size)
        k = k.repeat(self.C, 1, 1, 1)
        
        result = F.conv2d(x, weight=k, stride=(stride_h, stride_w),
                         padding=(pad_h, pad_w), groups=self.C)
        
        if result.shape[2:] != target_size:
            result = F.interpolate(result, size=target_size, mode='bilinear', align_corners=True)
            
        return result


def _expand_token(token: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Expand a single token to batch size."""
    return token.unsqueeze(0).expand(batch_size, -1, -1)
