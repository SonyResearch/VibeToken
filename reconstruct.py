#!/usr/bin/env python3
"""Simple reconstruction script for VibeToken.

Usage:
    # Auto mode (recommended) - automatically determines optimal settings
    python reconstruct.py --auto \
        --config configs/vibetoken_ll.yaml \
        --checkpoint /path/to/checkpoint.bin \
        --image assets/example_1.jpg \
        --output assets/reconstructed.png

    # Manual mode - specify all parameters
    python reconstruct.py \
        --config configs/vibetoken_ll.yaml \
        --checkpoint /path/to/checkpoint.bin \
        --image assets/example_1.jpg \
        --output assets/reconstructed.png \
        --input_height 512 --input_width 512 \
        --encoder_patch_size 16,32 \
        --decoder_patch_size 16
"""

import argparse
from PIL import Image
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
    parser = argparse.ArgumentParser(description="VibeToken image reconstruction")
    parser.add_argument("--config", type=str, default="configs/vibetoken_ll.yaml",
                        help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default="assets/example_1.jpg",
                        help="Path to input image")
    parser.add_argument("--output", type=str, default="./assets/reconstructed.png",
                        help="Path to output image")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    
    # Auto mode
    parser.add_argument("--auto", action="store_true",
                        help="Auto mode: automatically determine optimal input resolution and patch sizes")
    
    # Input resolution (optional - resize input before encoding)
    parser.add_argument("--input_height", type=int, default=None,
                        help="Resize input to this height before encoding (default: original)")
    parser.add_argument("--input_width", type=int, default=None,
                        help="Resize input to this width before encoding (default: original)")
    
    # Output resolution (optional - decode to this size)
    parser.add_argument("--output_height", type=int, default=None,
                        help="Decode to this height (default: same as input)")
    parser.add_argument("--output_width", type=int, default=None,
                        help="Decode to this width (default: same as input)")
    
    # Patch sizes (optional) - supports single int or tuple like "16,32"
    parser.add_argument("--encoder_patch_size", type=str, default=None,
                        help="Encoder patch size: single int (e.g., 16) or tuple (e.g., 16,32 for H,W)")
    parser.add_argument("--decoder_patch_size", type=str, default=None,
                        help="Decoder patch size: single int (e.g., 16) or tuple (e.g., 16,32 for H,W)")
    
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.config}")
    tokenizer = VibeTokenTokenizer.from_config(
        args.config,
        args.checkpoint,
        device=args.device,
    )

    # Load image
    print(f"Loading image from {args.image}")
    image = Image.open(args.image).convert("RGB")
    original_size = image.size  # (W, H)
    print(f"Original image size: {original_size[0]}x{original_size[1]}")

    if args.auto:
        # AUTO MODE - use centralized auto_preprocess_image
        print("\n=== AUTO MODE ===")
        image, patch_size, info = auto_preprocess_image(image, verbose=True)
        input_width, input_height = info["cropped_size"]
        output_width, output_height = input_width, input_height
        encoder_patch_size = patch_size
        decoder_patch_size = patch_size
        print("=================\n")
        
    else:
        # MANUAL MODE
        # Parse patch sizes
        encoder_patch_size = parse_patch_size(args.encoder_patch_size)
        decoder_patch_size = parse_patch_size(args.decoder_patch_size)
        
        # Resize input if specified
        if args.input_width or args.input_height:
            input_width = args.input_width or original_size[0]
            input_height = args.input_height or original_size[1]
            print(f"Resizing input to {input_width}x{input_height}")
            image = image.resize((input_width, input_height), Image.LANCZOS)
        
        # Always center crop to ensure dimensions divisible by 32
        image = center_crop_to_multiple(image, multiple=32)
        input_width, input_height = image.size
        if (input_width, input_height) != original_size:
            print(f"Center cropped to {input_width}x{input_height} (divisible by 32)")
        
        # Determine output size
        output_height = args.output_height or input_height
        output_width = args.output_width or input_width

    # Encode image to tokens
    print("Encoding image to tokens...")
    if encoder_patch_size:
        print(f"  Using encoder patch size: {encoder_patch_size}")
    tokens = tokenizer.encode(image, patch_size=encoder_patch_size)
    print(f"Token shape: {tokens.shape}")

    # Decode back to image
    print(f"Decoding to {output_width}x{output_height}...")
    if decoder_patch_size:
        print(f"  Using decoder patch size: {decoder_patch_size}")
    reconstructed = tokenizer.decode(
        tokens, 
        height=output_height, 
        width=output_width,
        patch_size=decoder_patch_size
    )
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Convert tensor to PIL and save
    output_images = tokenizer.to_pil(reconstructed)
    output_images[0].save(args.output)
    print(f"Saved reconstructed image to {args.output}")


if __name__ == "__main__":
    main()
