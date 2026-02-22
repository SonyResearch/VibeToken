"""Overfit test for VibeTokenAE.

Loads 1-2 real batches from the ImageNet webdataset, caches them on GPU,
and trains on those fixed batches to verify the training loop works.

Usage:
    python scripts/overfit_ae.py --config configs/training/VibeToken_AE_small.yaml
    python scripts/overfit_ae.py --model_size tiny --steps 300 --save_every 25
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.insert(0, parent_dir)

import torch
from torch.optim import AdamW
from omegaconf import OmegaConf

from modeling.vibetoken_ae_model import VibeTokenAEModel
from modeling.modules.losses import ReconstructionLoss_AE
from data import SimpleImageDataset
from utils.viz_utils import make_viz_from_samples, make_viz_from_refinement_steps


def parse_args():
    p = argparse.ArgumentParser(description="VibeTokenAE overfit test")
    p.add_argument("--config", type=str,
                   default="configs/training/VibeToken_AE_small.yaml")
    p.add_argument("--model_size", type=str, default="small",
                   choices=["tiny", "small", "base", "large"])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--num_batches", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--disc_start", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="overfit_output")
    p.add_argument("--mixed_precision", action="store_true", default=False)
    return p.parse_args()


def build_dataloader(config, batch_size):
    """Build a minimal dataloader from the config shard paths."""
    preproc = config.dataset.preprocessing
    ds_params = config.dataset.params
    dataset = SimpleImageDataset(
        train_shards_path=ds_params.train_shards_path_or_url,
        eval_shards_path=ds_params.eval_shards_path_or_url,
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=batch_size,
        global_batch_size=batch_size,
        num_workers_per_gpu=4,
        resize_shorter_edge=preproc.resize_shorter_edge,
        crop_size=preproc.crop_size,
        random_crop=False,
        random_flip=False,
        dataset_with_class_label=ds_params.get("dataset_with_class_label", True),
        dataset_with_text_label=ds_params.get("dataset_with_text_label", False),
        res_ratio_filtering=preproc.get("res_ratio_filtering", False),
        min_tokens=preproc.get("min_tokens", 32),
        max_tokens=preproc.get("max_tokens", 64),
    )
    return dataset.train_dataloader


def save_recon(model, images, step, output_dir):
    """Run encode/decode and save side-by-side comparison."""
    model.eval()
    with torch.no_grad():
        z, _ = model.encode(images, train=False)
        z = z.to(model.decoder.decoder_embed.weight.dtype)
        _, _, h, w = images.shape
        recon = model.decode(z, height=h, width=w, train=False)

    imgs_save, _ = make_viz_from_samples(images, recon)
    out = Path(output_dir) / "recon_images"
    out.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs_save):
        img.save(out / f"step{step:05d}_sample{i}.png")
    model.train()


def save_generation(model, step, output_dir, num_images=4, height=256, width=256):
    """Generate one-shot and iterative-refinement images and save grids."""
    # One-shot
    oneshot_steps = model.generate(num_images, height, width, num_steps=1)
    oneshot_img, _ = make_viz_from_refinement_steps(oneshot_steps)
    out_oneshot = Path(output_dir) / "gen_oneshot"
    out_oneshot.mkdir(parents=True, exist_ok=True)
    oneshot_img.save(out_oneshot / f"step{step:05d}.png")

    # Iterative refinement (4 steps)
    refine_steps = model.generate(num_images, height, width, num_steps=4,
                                  refine_noise_deg=5.0)
    refine_img, _ = make_viz_from_refinement_steps(refine_steps)
    out_refine = Path(output_dir) / "gen_refinement"
    out_refine.mkdir(parents=True, exist_ok=True)
    refine_img.save(out_refine / f"step{step:05d}.png")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = OmegaConf.load(args.config)
    config.model.vq_model.vit_enc_model_size = args.model_size
    config.model.vq_model.vit_dec_model_size = args.model_size
    config.losses.discriminator_start = args.disc_start

    # --- Load real data ---
    print(f"Loading {args.num_batches} batch(es) from webdataset (batch_size={args.batch_size})...")
    dataloader = build_dataloader(config, args.batch_size)
    cached_batches = []
    for batch in dataloader:
        images = batch["image"].to(device)
        cached_batches.append(images)
        print(f"  Cached batch {len(cached_batches)}: shape {images.shape}, "
              f"range [{images.min():.3f}, {images.max():.3f}]")
        if len(cached_batches) >= args.num_batches:
            break
    print(f"Cached {len(cached_batches)} batch(es) on {device}.")

    # --- Create model + loss ---
    print(f"Creating VibeTokenAEModel (encoder/decoder size: {args.model_size})...")
    model = VibeTokenAEModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {num_params:,}")

    print("Creating ReconstructionLoss_AE...")
    loss_module = ReconstructionLoss_AE(config).to(device)
    disc_params = sum(p.numel() for p in loss_module.discriminator.parameters())
    print(f"  Discriminator params: {disc_params:,}")

    # --- Optimizers ---
    model_optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    disc_optimizer = AdamW(loss_module.parameters(), lr=args.lr, weight_decay=1e-4)

    # --- Output dir ---
    os.makedirs(args.output_dir, exist_ok=True)
    save_recon(model, cached_batches[0], step=0, output_dir=args.output_dir)
    h, w = cached_batches[0].shape[2], cached_batches[0].shape[3]
    save_generation(model, step=0, output_dir=args.output_dir, num_images=4,
                    height=h, width=w)
    print(f"Saved initial recon + generation to {args.output_dir}/")

    # --- Training loop ---
    print(f"\nStarting overfit loop for {args.steps} steps (disc_start={args.disc_start})...\n")
    model.train()
    use_amp = args.mixed_precision and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for step in range(1, args.steps + 1):
        t0 = time.time()
        images = cached_batches[(step - 1) % len(cached_batches)]

        # --- Generator step ---
        model_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            x_small, extra = model(images, train=True)
            gen_loss, gen_dict = loss_module(
                images, x_small, extra, global_step=step, mode="generator")

        if scaler is not None:
            scaler.scale(gen_loss).backward()
            scaler.step(model_optimizer)
            scaler.update()
        else:
            gen_loss.backward()
            model_optimizer.step()

        # --- Discriminator step ---
        disc_dict = {}
        if loss_module.should_discriminator_be_trained(step):
            disc_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                disc_loss, disc_dict = loss_module(
                    images, x_small.detach(), extra, global_step=step, mode="discriminator")
            if scaler is not None:
                scaler.scale(disc_loss).backward()
                scaler.step(disc_optimizer)
                scaler.update()
            else:
                disc_loss.backward()
                disc_optimizer.step()

        dt = time.time() - t0

        # --- Logging ---
        total = gen_dict["total_loss"].item()
        recon = gen_dict["reconstruction_loss"].item()
        pix_con = gen_dict["l_pixcon"].item()
        lat_con = gen_dict["l_latcon"].item()
        gan = gen_dict["weighted_gan_loss"].item()
        disc_str = ""
        if disc_dict:
            disc_str = (f"  D_loss={disc_dict['discriminator_loss'].item():.4f}"
                        f"  real={disc_dict['logits_real'].item():.3f}"
                        f"  fake={disc_dict['logits_fake'].item():.3f}")

        print(f"[{step:4d}/{args.steps}] "
              f"total={total:.4f}  recon={recon:.4f}  "
              f"pixcon={pix_con:.4f}  latcon={lat_con:.6f}  "
              f"gan={gan:.4f}{disc_str}  "
              f"({dt:.2f}s)")

        # --- Save reconstruction + generation ---
        if step % args.save_every == 0 or step == args.steps:
            save_recon(model, cached_batches[0], step=step, output_dir=args.output_dir)
            h, w = cached_batches[0].shape[2], cached_batches[0].shape[3]
            save_generation(model, step, args.output_dir, num_images=4,
                            height=h, width=w)
            print(f"  -> Saved recon + generation images at step {step}")

    print(f"\nDone. Outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
