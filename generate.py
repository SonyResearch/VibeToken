# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py

"""Example run:
python generate.py \
    --gpt-ckpt ./checkpoints/VibeTokenGen-xxl-dynamic-65_750k.pt \
    --gpt-model GPT-XXL --num-output-layer 4 \
    --num-codebooks 8 --codebook-size 32768 \
    --image-size 256 --cfg-scale 2.0 --top-k 0 --temperature 1.0 \
    --class-dropout-prob 0.1 \
    --extra-layers "QKV" \
    --latent-size 65 \
    --config ./configs/vibetoken_ll.yaml \
    --vq-ckpt ./checkpoints/VibeToken_LL.bin \
    --sample-dir ./assets/ \
    --skip-folder-creation \
    --compile \
    --decoder-patch-size 16,16 \
    --target-resolution 1024,1024 \
    --llamagen-target-resolution 256,256 \
    --precision bf16
"""

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import sys
from omegaconf import OmegaConf

from vibetokengen.model import GPT_models
from vibetokengen.generate import generate

from vibetoken import VibeTokenTokenizer, auto_preprocess_image, center_crop_to_multiple


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Set global seed for reproducibility
    torch.manual_seed(args.global_seed)
    np.random.seed(args.global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]

    # Load VibeToken model
    vq_model = VibeTokenTokenizer.from_config(
        args.config,
        args.vq_ckpt,
        device=device,
        dtype=precision,
    )
    print(f"VibeToken image tokenizer is loaded")

    # create and load gpt model
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=args.latent_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        num_codebooks=args.num_codebooks,
        n_output_layer=args.num_output_layer,
        class_dropout_prob=args.class_dropout_prob,
        extra_layers=args.extra_layers,
        capping=args.capping,
    ).to(device=device, dtype=precision)
    print(f"GPT model is loaded")

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
    if args.from_fsdp:  # fsdp
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:  # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()
    del checkpoint

    print(f"GPT model weights are loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        )  # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile")

    print(f"GPT model is compiled")

    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    if args.from_fsdp:
        ckpt_string_name = args.gpt_ckpt.split('/')[-2]
    else:
        ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    folder_name = f"{model_string_name}-{ckpt_string_name}-target-resolution-{args.target_resolution}-llamagen-target-resolution-{args.llamagen_target_resolution}-vibetoken-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    if args.skip_folder_creation:
        sample_folder_dir = args.sample_dir
    else:
        sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    multiplier = 2 if args.cfg_scale > 1.0 else 1

    # Use fixed class labels
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    c_indices = torch.tensor(class_labels, device=device)
    n = len(class_labels)
    nrow = 4  # 2 rows x 4 columns for 8 images

    index_sample = generate(
        gpt_model, c_indices, args.latent_size, args.num_codebooks,
        cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
        target_h=torch.tensor(args.llamagen_target_resolution[0]/1536, device=device, dtype=precision).unsqueeze(0).repeat(len(c_indices)*multiplier, 1),
        target_w=torch.tensor(args.llamagen_target_resolution[1]/1536, device=device, dtype=precision).unsqueeze(0).repeat(len(c_indices)*multiplier, 1),
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True,
    )

    # Use VibeToken decode_tokens method
    # VibeToken expects tokens in shape (batch_size, seq_len, 1)
    index_sample = index_sample.unsqueeze(2)
    samples = vq_model.decode(
        index_sample, 
        height=args.target_resolution[0], 
        width=args.target_resolution[1], 
        patch_size=args.decoder_patch_size
    )
    
    # VibeToken output is in [0, 1] range, clamp and convert to uint8
    samples = torch.clamp(samples, 0, 1)

    # Create a grid of images (2 rows x 4 columns)
    from torchvision.utils import make_grid
    grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
    
    # Convert to PIL and save
    grid_np = (grid.permute(1, 2, 0).to(torch.float32).cpu().numpy() * 255).astype('uint8')
    Image.fromarray(grid_np).save(f"{sample_folder_dir}/generated_images.png")
    print(f"Saved grid of {n} images to {sample_folder_dir}/generated_images.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i",
                        help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true', default=True)
    # parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--config", type=str, required=True, help="Path to VibeToken config file")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0) # not used
    parser.add_argument("--top-k", type=int, default=500, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--num-codebooks", type=int, default=1)
    parser.add_argument("--num-output-layer", type=int, default=1)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--extra-layers", type=str, choices=['QK', 'QKV', 'FC', 'cap', 'clip', 'QK_cap', 'QKV_cap', 'QK_clip', 'QKV_clip', 'QK_FC_cap', 'QKV_FC_cap', 'QK_FC_clip', 'QKV_FC_clip'], default=None,
                        help="Type of extra layers to add: QK (query-key), QKV (query-key-value), FC (fully connected), cap (caption), clip (clip), QK_cap (query-key-caption), QKV_cap (query-key-value-caption), QK_clip (query-key-clip), QKV_clip (query-key-value-clip), QK_FC_cap (query-key-fully-connected-caption), QKV_FC_cap (query-key-value-fully-connected-caption), QK_FC_clip (query-key-fully-connected-clip), QKV_FC_clip (query-key-value-fully-connected-clip)")
    parser.add_argument("--capping", type=float, default=50.0, help="Capping for attention softmax")

    # VibeToken dynamic
    parser.add_argument("--decoder-patch-size", type=str, default="8,8", help="Decoder patch size as 'width,height'")
    parser.add_argument("--target-resolution", type=str, default="256,256", help="Target resolution as 'width,height'")
    parser.add_argument("--llamagen-target-resolution", type=str, default="256,256", help="LlamaGen target resolution as 'width,height'")

    parser.add_argument("--latent-size", type=int, default=16, help="Latent size")
    parser.add_argument("--skip-folder-creation", action='store_true', default=False, help="skip folder creation")

    args = parser.parse_args()

    args.decoder_patch_size = tuple(map(int, args.decoder_patch_size.split(",")))
    args.target_resolution = tuple(map(int, args.target_resolution.split(",")))
    args.llamagen_target_resolution = tuple(map(int, args.llamagen_target_resolution.split(",")))

    main(args)