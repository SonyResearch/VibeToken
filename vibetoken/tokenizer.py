"""High-level VibeToken Tokenizer API.

Provides a clean, user-friendly interface for image tokenization.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

from .modeling.vibetoken import VibeToken


def center_crop_to_multiple(image: Image.Image, multiple: int = 32) -> Image.Image:
    """Center crop image to dimensions divisible by multiple.
    
    Args:
        image: PIL Image
        multiple: Dimensions must be divisible by this value (default: 32)
        
    Returns:
        Cropped PIL Image
    """
    w, h = image.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple
    
    if new_w == w and new_h == h:
        return image
    
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    
    return image.crop((left, top, right, bottom))


def get_auto_patch_size(height: int, width: int) -> int:
    """Automatically determine optimal patch size based on resolution.
    
    Rules:
        - <= 256x256: patch size 8
        - <= 512x512: patch size 16
        - > 512x512: patch size 32
    
    Args:
        height: Image height
        width: Image width
        
    Returns:
        Optimal patch size as int
    """
    # Check <= 256x256
    if height <= 256 and width <= 256:
        return 8
    
    # Check <= 512x512
    if height <= 512 and width <= 512:
        return 16
    
    # Larger resolutions
    return 32


def auto_preprocess_image(image: Image.Image, verbose: bool = True) -> Tuple[Image.Image, int, dict]:
    """Automatically preprocess image: center crop to divisible by 32 and determine optimal patch size.
    
    Args:
        image: PIL Image
        verbose: Whether to print preprocessing info
        
    Returns:
        image: Preprocessed PIL Image
        patch_size: Optimal patch size
        info: Dictionary with preprocessing details
    """
    original_size = image.size  # (W, H)
    
    # Center crop to dimensions divisible by 32
    image = center_crop_to_multiple(image, multiple=32)
    cropped_size = image.size  # (W, H)
    
    # Get optimal patch size
    patch_size = get_auto_patch_size(cropped_size[1], cropped_size[0])  # (H, W)
    
    info = {
        "original_size": original_size,
        "cropped_size": cropped_size,
        "patch_size": patch_size,
        "was_cropped": original_size != cropped_size,
    }
    
    if verbose:
        if info["was_cropped"]:
            print(f"Center cropped: {original_size[0]}x{original_size[1]} -> {cropped_size[0]}x{cropped_size[1]} (divisible by 32)")
        print(f"Auto patch size: {patch_size}x{patch_size}")
    
    return image, patch_size, info


class VibeTokenTokenizer:
    """High-level API for VibeToken image tokenization.
    
    Provides simple encode/decode methods for converting images to/from
    discrete tokens.
    
    Example:
        >>> tokenizer = VibeTokenTokenizer.from_config("configs/vibetoken_ll.yaml", "model.bin")
        >>> tokens = tokenizer.encode(images)  # (B, num_codebooks, seq_len)
        >>> reconstructed = tokenizer.decode(tokens, height=512, width=512)
    """
    
    def __init__(
        self,
        model: VibeToken,
        device: str = "cuda",
    ):
        """Initialize tokenizer with a VibeToken model.
        
        Args:
            model: Loaded VibeToken model.
            device: Device for inference.
        """
        self.model = model
        self.device = device
        self.num_codebooks = getattr(model.config.model.vq_model, 'num_codebooks', 1) \
            if hasattr(model.config.model, 'vq_model') else 1

    @classmethod
    def from_config(
        cls,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "VibeTokenTokenizer":
        """Create tokenizer from config and checkpoint files.
        
        Args:
            config_path: Path to YAML config file.
            checkpoint_path: Path to model checkpoint.
            device: Device for inference.
            dtype: Optional dtype for model.
            
        Returns:
            VibeTokenTokenizer instance.
        """
        model = VibeToken.from_pretrained(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=dtype,
        )
        return cls(model=model, device=device)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "VibeTokenTokenizer":
        """Load tokenizer by model name.
        
        Args:
            model_name: Model variant name ("ll" for Large-Large, "sl" for Small-Large).
            checkpoint_path: Path to model checkpoint.
            device: Device for inference.
            dtype: Optional dtype for model.
            
        Returns:
            VibeTokenTokenizer instance.
        """
        # Get config path based on model name
        package_dir = Path(__file__).parent.parent
        config_map = {
            "ll": "vibetoken_ll.yaml",
            "sl": "vibetoken_sl.yaml",
        }
        
        if model_name not in config_map:
            raise ValueError(f"Unknown model name: {model_name}. Choose from: {list(config_map.keys())}")
        
        config_path = package_dir / "configs" / config_map[model_name]
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        return cls.from_config(
            config_path=str(config_path),
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=dtype,
        )

    def preprocess(
        self,
        images: Union[torch.Tensor, np.ndarray, Image.Image, list],
    ) -> torch.Tensor:
        """Preprocess images for encoding.
        
        Args:
            images: Input images. Can be:
                - torch.Tensor (B, 3, H, W) or (3, H, W), values in [0, 1]
                - numpy array (B, H, W, 3) or (H, W, 3), values in [0, 255]
                - PIL Image
                - List of PIL Images
                
        Returns:
            Preprocessed tensor (B, 3, H, W), values in [0, 1].
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(images, list):
            # List of PIL images
            tensors = []
            for img in images:
                if isinstance(img, Image.Image):
                    arr = np.array(img.convert("RGB"))
                    tensor = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
                    tensors.append(tensor)
                else:
                    raise ValueError(f"Unsupported image type in list: {type(img)}")
            images = torch.stack(tensors)
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = images[None]  # Add batch dim
            # Assume (B, H, W, C) format
            images = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
        elif isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)  # Add batch dim
            # Assume already in [0, 1] range
            if images.max() > 1.0:
                images = images / 255.0
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")
        
        return images.to(self.device)

    @torch.no_grad()
    def encode(
        self,
        images: Union[torch.Tensor, np.ndarray, Image.Image, list],
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Encode images to discrete tokens.
        
        Args:
            images: Input images (see preprocess for supported formats).
            patch_size: Optional encoding patch size. Can be int or tuple (H, W).
            return_dict: Whether to return result dictionary with additional info.
            
        Returns:
            tokens: Token indices. Shape depends on quantize_mode:
                - VQ: (B, seq_len)
                - MVQ: (B, num_codebooks, seq_len) 
            result_dict (optional): Dictionary with additional encoding info.
        """
        images = self.preprocess(images)
        
        # Convert int to tuple if needed
        if isinstance(patch_size, int):
            encode_ps = (patch_size, patch_size)
        else:
            encode_ps = patch_size  # Already tuple or None
        _, result_dict = self.model.encode(images, encode_patch_size=encode_ps)
        
        # Extract token indices
        tokens = result_dict['min_encoding_indices']
        
        # Reshape based on quantize mode
        if self.model.quantize_mode == "mvq":
            # Already in (B, num_codebooks, H, W) format, flatten spatial
            B, num_cb, H, W = tokens.shape
            tokens = tokens.reshape(B, num_cb, H * W)
        else:
            # VQ mode: (B, H, W) -> (B, H*W)
            B, H, W = tokens.shape
            tokens = tokens.reshape(B, H * W)
        
        if return_dict:
            return tokens, result_dict
        return tokens

    @torch.no_grad()
    def decode(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        clamp: bool = True,
    ) -> torch.Tensor:
        """Decode tokens back to images.
        
        Args:
            tokens: Token indices from encode().
            height: Target image height.
            width: Target image width.
            patch_size: Optional decoding patch size. Can be int or tuple (H, W).
            clamp: Whether to clamp output to [0, 1].
            
        Returns:
            Decoded images (B, 3, height, width), values in [0, 1].
        """
        tokens = tokens.to(self.device)
        
        # Convert int to tuple if needed
        if isinstance(patch_size, int):
            decode_ps = (patch_size, patch_size)
        else:
            decode_ps = patch_size  # Already tuple or None
        
        decoded = self.model.decode_tokens(
            tokens,
            height=height,
            width=width,
            decode_patch_size=decode_ps,
        )
        
        if clamp:
            decoded = torch.clamp(decoded, 0.0, 1.0)
        
        return decoded

    @torch.no_grad()
    def reconstruct(
        self,
        images: Union[torch.Tensor, np.ndarray, Image.Image, list],
        encode_patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        decode_patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode and decode images (full reconstruction).
        
        Args:
            images: Input images.
            encode_patch_size: Optional encoding patch size. Can be int or tuple (H, W).
            decode_patch_size: Optional decoding patch size. Can be int or tuple (H, W).
            target_height: Target output height (default: same as input).
            target_width: Target output width (default: same as input).
            
        Returns:
            Reconstructed images (B, 3, H, W), values in [0, 1].
        """
        images = self.preprocess(images)
        B, C, H, W = images.shape
        
        if target_height is None:
            target_height = H
        if target_width is None:
            target_width = W
        
        # Encode with optional patch size
        tokens = self.encode(images, patch_size=encode_patch_size)
        
        # Decode with optional patch size
        decoded = self.decode(
            tokens, 
            height=target_height, 
            width=target_width,
            patch_size=decode_patch_size,
        )
        
        return decoded

    def to_pil(self, images: torch.Tensor) -> list:
        """Convert tensor images to PIL Images.
        
        Args:
            images: Tensor (B, 3, H, W), values in [0, 1].
            
        Returns:
            List of PIL Images.
        """
        images = torch.clamp(images * 255, 0, 255)
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return [Image.fromarray(img) for img in images]

    @torch.no_grad()
    def auto_reconstruct(
        self,
        image: Image.Image,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """Automatically reconstruct image with optimal settings.
        
        This method:
        1. Center crops input to dimensions divisible by 32
        2. Auto-determines optimal patch sizes based on resolution
        3. Returns output at same size as (cropped) input
        
        Args:
            image: Input PIL Image
            verbose: Whether to print auto-selected settings
            
        Returns:
            reconstructed: Reconstructed image tensor (1, 3, H, W)
            info: Dictionary with auto-selected settings
        """
        # Use centralized auto_preprocess_image
        original_size = image.size
        image, patch_size, preprocess_info = auto_preprocess_image(image, verbose=verbose)
        width, height = preprocess_info["cropped_size"]
        
        # Encode and decode
        tokens = self.encode(image, patch_size=patch_size)
        reconstructed = self.decode(tokens, height=height, width=width, patch_size=patch_size)
        
        info = {
            "original_size": original_size,
            "cropped_size": (width, height),
            "patch_size": patch_size,
            "token_shape": tokens.shape,
        }
        
        return reconstructed, info

    @property
    def codebook_size(self) -> int:
        """Get total codebook size."""
        vq_config = self.model.config.model.vq_model if hasattr(self.model.config.model, 'vq_model') else self.model.config.model
        return getattr(vq_config, 'codebook_size', 32768)

    @property
    def num_latent_tokens(self) -> int:
        """Get number of latent tokens."""
        return self.model.num_latent_tokens
