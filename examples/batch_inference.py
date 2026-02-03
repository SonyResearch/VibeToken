#!/usr/bin/env python3
"""Batch inference example for VibeToken.

Demonstrates how to process multiple images efficiently in batches.

Usage:
    # Auto mode (recommended)
    python examples/batch_inference.py --auto \
        --config configs/vibetoken_ll.yaml \
        --checkpoint path/to/checkpoint.bin \
        --input_dir path/to/images/ \
        --output_dir path/to/output/ \
        --batch_size 4

    # Manual mode
    python examples/batch_inference.py \
        --config configs/vibetoken_ll.yaml \
        --checkpoint path/to/checkpoint.bin \
        --input_dir path/to/images/ \
        --output_dir path/to/output/ \
        --batch_size 4 \
        --resolution 512 \
        --encoder_patch_size 16,32 \
        --decoder_patch_size 16
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
import numpy as np

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


def load_and_preprocess_image(path: Path, target_size: tuple = None, auto_mode: bool = False) -> tuple:
    """Load and preprocess image.
    
    Args:
        path: Path to image
        target_size: Optional target size (width, height) for resizing
        auto_mode: If True, use auto_preprocess_image for cropping
        
    Returns:
        image: numpy array
        patch_size: auto-determined patch size (if auto_mode) or None
    """
    img = Image.open(path).convert("RGB")
    
    if auto_mode:
        # Use centralized auto_preprocess_image
        img, patch_size, info = auto_preprocess_image(img, verbose=False)
        return np.array(img), patch_size, info
    else:
        if target_size:
            img = img.resize(target_size, Image.LANCZOS)
        # Always center crop to ensure dimensions divisible by 32
        img = center_crop_to_multiple(img, multiple=32)
        return np.array(img), None, None


def main():
    parser = argparse.ArgumentParser(description="VibeToken batch inference example")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output images")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    # Auto mode
    parser.add_argument("--auto", action="store_true",
                        help="Auto mode: automatically determine optimal settings per image")
    
    # Manual mode options
    parser.add_argument("--resolution", type=int, default=512, help="Target resolution (manual mode)")
    parser.add_argument("--encoder_patch_size", type=str, default=None,
                        help="Encoder patch size: single int (e.g., 16) or tuple (e.g., 16,32 for H,W)")
    parser.add_argument("--decoder_patch_size", type=str, default=None,
                        help="Decoder patch size: single int (e.g., 16) or tuple (e.g., 16,32 for H,W)")
    args = parser.parse_args()
    
    # Parse patch sizes
    encoder_patch_size = parse_patch_size(args.encoder_patch_size)
    decoder_patch_size = parse_patch_size(args.decoder_patch_size)

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {args.config}")
    tokenizer = VibeTokenTokenizer.from_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    if args.auto:
        print("Running in AUTO MODE - optimal settings determined per image")
    else:
        print(f"Running in MANUAL MODE - resolution: {args.resolution}")
        if encoder_patch_size:
            print(f"  Encoder patch size: {encoder_patch_size}")
        if decoder_patch_size:
            print(f"  Decoder patch size: {decoder_patch_size}")

    # Find all images
    input_dir = Path(args.input_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions]
    print(f"Found {len(image_paths)} images")

    if not image_paths:
        print("No images found!")
        return

    # Process in batches
    target_size = (args.resolution, args.resolution) if not args.auto else None
    total_time = 0
    num_processed = 0

    if args.auto:
        # AUTO MODE: Process images one by one since each may have different sizes
        for i, path in enumerate(image_paths):
            try:
                img_array, patch_size, info = load_and_preprocess_image(path, auto_mode=True)
                batch_array = img_array[np.newaxis, ...]  # Add batch dim
                
                start_time = time.time()
                
                # Reconstruct with auto-determined patch size
                height, width = info['cropped_size'][1], info['cropped_size'][0]
                reconstructed = tokenizer.reconstruct(
                    batch_array,
                    encode_patch_size=patch_size,
                    decode_patch_size=patch_size,
                    target_height=height,
                    target_width=width,
                )
                
                if args.device == "cuda":
                    torch.cuda.synchronize()
                
                batch_time = time.time() - start_time
                total_time += batch_time
                num_processed += 1
                
                # Save output
                output_images = tokenizer.to_pil(reconstructed)
                output_path = output_dir / f"{path.stem}_recon.png"
                output_images[0].save(output_path)
                
                print(f"[{i+1}/{len(image_paths)}] {path.name}: "
                      f"{info['cropped_size'][0]}x{info['cropped_size'][1]}, "
                      f"patch_size={patch_size}, {batch_time:.2f}s")
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
    else:
        # MANUAL MODE: Batch processing with uniform size
        for batch_start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[batch_start:batch_start + args.batch_size]
            batch_names = [p.stem for p in batch_paths]
            
            # Load batch
            batch_images = []
            for path in batch_paths:
                try:
                    img_array, _, _ = load_and_preprocess_image(path, target_size, auto_mode=False)
                    batch_images.append(img_array)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if not batch_images:
                continue

            # Stack into batch tensor
            batch_array = np.stack(batch_images, axis=0)
            
            # Measure time
            start_time = time.time()
            
            # Reconstruct
            reconstructed = tokenizer.reconstruct(
                batch_array,
                encode_patch_size=encoder_patch_size,
                decode_patch_size=decoder_patch_size,
                target_height=args.resolution,
                target_width=args.resolution,
            )
            
            # Synchronize if GPU
            if args.device == "cuda":
                torch.cuda.synchronize()
            
            batch_time = time.time() - start_time
            total_time += batch_time
            num_processed += len(batch_images)
            
            # Save outputs
            output_images = tokenizer.to_pil(reconstructed)
            for name, img in zip(batch_names[:len(output_images)], output_images):
                output_path = output_dir / f"{name}_recon.png"
                img.save(output_path)
            
            print(f"Processed batch {batch_start // args.batch_size + 1}: "
                  f"{len(batch_images)} images in {batch_time:.2f}s "
                  f"({len(batch_images) / batch_time:.2f} img/s)")

    # Summary
    if num_processed > 0:
        print(f"\nTotal: {num_processed} images in {total_time:.2f}s")
        print(f"Average: {num_processed / total_time:.2f} images/sec")
        print(f"Per image: {total_time / num_processed * 1000:.1f}ms")


if __name__ == "__main__":
    main()
