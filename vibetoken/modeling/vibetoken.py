"""VibeToken model - Main model class for image tokenization.

Combines encoder, quantizer, and decoder into a unified model for
encoding images to discrete tokens and decoding back to images.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union
from einops import rearrange
from omegaconf import OmegaConf

from .encoder import ResolutionEncoder
from .decoder import ResolutionDecoder
from ..quantizer import VectorQuantizer, VectorQuantizerMVQ


class VibeToken(nn.Module):
    """VibeToken image tokenizer model.
    
    A Vision Transformer-based image tokenizer that encodes images into
    discrete tokens and decodes them back to images. Supports multiple
    quantization modes (VQ, MVQ, VAE) and variable resolutions.
    """
    
    def __init__(self, config: Union[dict, Any]):
        """Initialize VibeToken model.
        
        Args:
            config: Configuration dict or OmegaConf object with model parameters.
        """
        super().__init__()
        
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        
        self.config = config
        
        # Get model config
        vq_config = config.model.vq_model if hasattr(config.model, 'vq_model') else config.model
        
        # Quantization mode
        self.quantize_mode = getattr(vq_config, 'quantize_mode', 'vq')
        if self.quantize_mode not in ["vq", "vae", "mvq"]:
            raise ValueError(f"Unsupported quantize mode: {self.quantize_mode}")
        
        # Build encoder and decoder
        self.encoder = ResolutionEncoder(config)
        self.decoder = ResolutionDecoder(config)
        
        # Latent tokens (learnable queries for encoder)
        self.num_latent_tokens = getattr(vq_config, 'num_latent_tokens', 256)
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width)
        )
        
        # Build quantizer based on mode
        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=getattr(vq_config, 'codebook_size', 32768),
                token_size=getattr(vq_config, 'token_size', 256),
                commitment_cost=getattr(vq_config, 'commitment_cost', 0.25),
                use_l2_norm=getattr(vq_config, 'use_l2_norm', False),
            )
        elif self.quantize_mode == "mvq":
            self.quantize = VectorQuantizerMVQ(
                codebook_size=getattr(vq_config, 'codebook_size', 32768),
                token_size=getattr(vq_config, 'token_size', 256),
                commitment_cost=getattr(vq_config, 'commitment_cost', 0.25),
                use_l2_norm=getattr(vq_config, 'use_l2_norm', False),
                num_codebooks=getattr(vq_config, 'num_codebooks', 8),
            )
        elif self.quantize_mode == "vae":
            # VAE mode uses DiagonalGaussianDistribution directly in forward
            from ..quantizer.vector_quantizer import DiagonalGaussianDistribution
            self.quantize = DiagonalGaussianDistribution
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights for module."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def encode(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encode_patch_size: Optional[Tuple[int, int]] = None,
        length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encode images to quantized latent tokens.
        
        Args:
            x: Input images (B, 3, H, W), values in [0, 1].
            attention_mask: Optional attention mask.
            encode_patch_size: Optional custom patch size for encoding.
            length: Optional token length limit.
            
        Returns:
            z_quantized: Quantized latent features.
            result_dict: Dictionary with token indices and optional losses.
        """
        # Select latent tokens
        if length is not None:
            latent_tokens = self.latent_tokens[:length + 1]
        else:
            latent_tokens = self.latent_tokens
        
        # Encode through ViT encoder
        z = self.encoder(
            pixel_values=x,
            latent_tokens=latent_tokens,
            attention_mask=attention_mask,
            encode_patch_size=encode_patch_size,
        )
        
        # Quantize
        if self.quantize_mode in ["vq", "mvq"]:
            z_quantized, result_dict = self.quantize(z)
        elif self.quantize_mode == "vae":
            posteriors = self.quantize(z)
            z_quantized = posteriors.sample()
            result_dict = {"posteriors": posteriors}
        
        return z_quantized, result_dict

    def decode(
        self,
        z_quantized: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        decode_patch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode quantized features to images.
        
        Args:
            z_quantized: Quantized latent features.
            attention_mask: Optional attention mask.
            height: Target image height.
            width: Target image width.
            decode_patch_size: Optional custom patch size for decoding.
            
        Returns:
            Decoded images (B, 3, height, width), values approximately in [0, 1].
        """
        z_quantized = z_quantized.to(self.decoder.decoder_embed.weight.dtype)
        decoded = self.decoder(
            z_quantized,
            attention_mask=attention_mask,
            height=height,
            width=width,
            decode_patch_size=decode_patch_size,
        )
        return decoded

    def decode_tokens(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        decode_patch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode discrete tokens directly to images.
        
        This is the main interface used by generators.
        
        Args:
            tokens: Token indices. Shape depends on quantize_mode:
                - VQ: (B, seq_len) or (B, 1, seq_len)
                - MVQ: (B, num_codebooks, seq_len) or (B, seq_len, 1)
                - VAE: Continuous latent (B, C, H, W)
            attention_mask: Optional attention mask.
            height: Target image height.
            width: Target image width.
            decode_patch_size: Optional custom patch size for decoding.
            
        Returns:
            Decoded images (B, 3, height, width), values approximately in [0, 1].
        """
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1)
            batch, seq_len = tokens.shape
            z_quantized = self.quantize.get_codebook_entry(tokens.reshape(-1))
            z_quantized = z_quantized.reshape(batch, 1, seq_len, -1)
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        elif self.quantize_mode == "mvq":
            z_quantized = self.quantize.get_codebook_entry(tokens)
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        
        z_quantized = z_quantized.to(self.decoder.decoder_embed.weight.dtype)
        decoded = self.decode(
            z_quantized,
            attention_mask=attention_mask,
            height=height,
            width=width,
            decode_patch_size=decode_patch_size,
        )
        return decoded

    def forward(
        self,
        x: torch.Tensor,
        key_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Full encode-decode forward pass.
        
        Args:
            x: Input images (B, 3, H, W), values in [0, 1].
            key_attention_mask: Optional attention mask.
            height: Target output height (default: same as input).
            width: Target output width (default: same as input).
            
        Returns:
            decoded: Reconstructed images.
            result_dict: Dictionary with token indices and losses.
        """
        if height is None:
            _, _, height, width = x.shape
        
        z_quantized, result_dict = self.encode(x, attention_mask=key_attention_mask)
        z_quantized = z_quantized.to(self.decoder.decoder_embed.weight.dtype)
        decoded = self.decode(z_quantized, attention_mask=key_attention_mask, height=height, width=width)
        
        return decoded, result_dict

    @classmethod
    def from_pretrained(
        cls,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "VibeToken":
        """Load pretrained model from config and checkpoint.
        
        Args:
            config_path: Path to YAML config file.
            checkpoint_path: Path to model checkpoint (.pt or .bin).
            device: Device to load model on.
            dtype: Optional dtype for model parameters.
            
        Returns:
            Loaded VibeToken model.
        """
        config = OmegaConf.load(config_path)
        model = cls(config)
        
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        model.eval()
        model.requires_grad_(False)
        model.to(device)
        
        if dtype is not None:
            model.to(dtype)
        
        return model
