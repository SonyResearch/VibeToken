#!/usr/bin/env python3
"""Basic encode-decode example for VibeToken.

Demonstrates how to:
1. Load the tokenizer from config and checkpoint
2. Encode an image to discrete tokens
3. Decode tokens back to an image
4. Save the reconstructed image

Usage:
    # Auto mode (recommended)
    python examples/encode_decode.py --auto \
        --config configs/vibetoken_ll.yaml \
        --checkpoint path/to/checkpoint.bin \
        --image path/to/image.jpg \
        --output reconstructed.png

    # Manual mode
    python examples/encode_decode.py \
        --config configs/vibetoken_ll.yaml \
        --checkpoint path/to/checkpoint.bin \
        --image path/to/image.jpg \
        --output reconstructed.png \
        --encoder_patch_size 16,32 \
        --decoder_patch_size 16
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vibetoken import VibeTokenTokenizer, auto_preprocess_image, center_crop_to_multiple


def parse_patch_size(value):
    """Parse patch size from string. Supports single int or tuple (e.g., '16' or '16,32')."""
    if value is None:
        return None
    if ',' in value:
        parts = value.split(',')
        return (int(parts[0]), int(parts[1]))
    return int(value)


def main():
    parser = argparse.ArgumentParser(description="VibeToken encode-decode example")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="reconstructed.png", help="Output image path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    # Auto mode
    parser.add_argument("--auto", action="store_true",
                        help="Auto mode: automatically determine optimal settings")
    
    parser.add_argument("--height", type=int, default=None, help="Output height (default: input height)")
    parser.add_argument("--width", type=int, default=None, help="Output width (default: input width)")
    parser.add_argument("--encoder_patch_size", type=str, default=None, 
                        help="Encoder patch size: single int (e.g., 16) or tuple (e.g., 16,32 for H,W)")
    parser.add_argument("--decoder_patch_size", type=str, default=None,
                        help="Decoder patch size: single int (e.g., 16) or tuple (e.g., 16,32 for H,W)")
    parser.add_argument("--num_tokens", type=int, default=None, help="Number of tokens to encode")

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Loading tokenizer from {args.config}")
    tokenizer = VibeTokenTokenizer.from_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print(f"Tokenizer loaded: codebook_size={tokenizer.codebook_size}, "
          f"num_latent_tokens={tokenizer.num_latent_tokens}")

    # Load image
    print(f"Loading image from {args.image}")
    image = Image.open(args.image).convert("RGB")
    original_size = image.size  # (W, H)
    print(f"Original image size: {original_size[0]}x{original_size[1]}")

    if args.auto:
        # AUTO MODE - use centralized auto_preprocess_image
        print("\n=== AUTO MODE ===")
        image, patch_size, info = auto_preprocess_image(image, verbose=True)
        encoder_patch_size = patch_size
        decoder_patch_size = patch_size
        height, width = info['cropped_size'][1], info['cropped_size'][0]
        print("=================\n")
        
        # Encode to tokens
        print("Encoding image to tokens...")
        print(f"  Using encoder patch size: {encoder_patch_size}")
        tokens = tokenizer.encode(image, patch_size=encoder_patch_size)
        print(f"Token shape: {tokens.shape}")
        
        # Decode back to image
        print(f"Decoding tokens to image ({width}x{height})...")
        print(f"  Using decoder patch size: {decoder_patch_size}")
        reconstructed = tokenizer.decode(
            tokens, height=height, width=width, patch_size=decoder_patch_size
        )
        
    else:
        # MANUAL MODE
        # Parse patch sizes
        encoder_patch_size = parse_patch_size(args.encoder_patch_size)
        decoder_patch_size = parse_patch_size(args.decoder_patch_size)

        # Always center crop to ensure dimensions divisible by 32
        image = center_crop_to_multiple(image, multiple=32)
        cropped_size = image.size  # (W, H)
        if cropped_size != original_size:
            print(f"Center cropped to {cropped_size[0]}x{cropped_size[1]} (divisible by 32)")

        # Encode to tokens
        print("Encoding image to tokens...")
        if encoder_patch_size:
            print(f"  Using encoder patch size: {encoder_patch_size}")
        tokens = tokenizer.encode(image, patch_size=encoder_patch_size, num_tokens=args.num_tokens)
        print(f"Token shape: {tokens.shape}")
        
        if tokenizer.model.quantize_mode == "mvq":
            print(f"  - Batch size: {tokens.shape[0]}")
            print(f"  - Num codebooks: {tokens.shape[1]}")
            print(f"  - Sequence length: {tokens.shape[2]}")
        else:
            print(f"  - Batch size: {tokens.shape[0]}")
            print(f"  - Sequence length: {tokens.shape[1]}")

        # Decode back to image (use cropped size as default)
        height = args.height or cropped_size[1]
        width = args.width or cropped_size[0]
        print(f"Decoding tokens to image ({width}x{height})...")
        if decoder_patch_size:
            print(f"  Using decoder patch size: {decoder_patch_size}")
        
        reconstructed = tokenizer.decode(
            tokens, height=height, width=width, patch_size=decoder_patch_size
        )
    
    print(f"Reconstructed image shape: {reconstructed.shape}")

    # Convert to PIL and save
    output_images = tokenizer.to_pil(reconstructed)
    output_path = Path(args.output)
    output_images[0].save(output_path)
    print(f"Saved reconstructed image to {output_path}")

    # Compute PSNR (compare with cropped image)
    import numpy as np
    original_np = np.array(image).astype(np.float32)
    recon_np = np.array(output_images[0]).astype(np.float32)
    if original_np.shape == recon_np.shape:
        mse = np.mean((original_np - recon_np) ** 2)
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            print(f"PSNR: {psnr:.2f} dB")


if __name__ == "__main__":
    main()
