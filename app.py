"""
VibeToken-Gen Gradio Demo
Class-conditional ImageNet generation with dynamic resolution support.
"""

import os
import random

import gradio as gr
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

from huggingface_hub import hf_hub_download
from PIL import Image

from vibetokengen.generate import generate
from vibetokengen.model import GPT_models
from vibetoken import VibeTokenTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO = "mpatel57/VibeToken"
USE_XXL = os.environ.get("VIBETOKEN_XXL", "0") == "1"

if USE_XXL:
    GPT_MODEL_NAME = "GPT-XXL"
    GPT_CKPT_FILENAME = "VibeTokenGen-xxl-dynamic-65_750k.pt"
    NUM_OUTPUT_LAYER = 4
    EXTRA_LAYERS = "QKV"
else:
    GPT_MODEL_NAME = "GPT-B"
    GPT_CKPT_FILENAME = "VibeTokenGen-b-fixed65_dynamic_1500k.pt"
    NUM_OUTPUT_LAYER = 4
    EXTRA_LAYERS = "QKV"

VQ_CKPT_FILENAME = "VibeToken_LL.bin"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "vibetoken_ll.yaml")

CODEBOOK_SIZE = 32768
NUM_CODEBOOKS = 8
LATENT_SIZE = 65
NUM_CLASSES = 1000
CLS_TOKEN_NUM = 1
CLASS_DROPOUT_PROB = 0.1
CAPPING = 50.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
COMPILE = os.environ.get("VIBETOKEN_NO_COMPILE", "0") != "1" and DEVICE == "cuda"

# ---------------------------------------------------------------------------
# ImageNet class labels (curated popular subset)
# ---------------------------------------------------------------------------

IMAGENET_CLASSES = {
    "Golden Retriever": 207,
    "Labrador Retriever": 208,
    "German Shepherd": 235,
    "Siberian Husky": 250,
    "Pembroke Corgi": 263,
    "Tabby Cat": 281,
    "Persian Cat": 283,
    "Siamese Cat": 284,
    "Tiger": 292,
    "Lion": 291,
    "Cheetah": 293,
    "Brown Bear": 294,
    "Giant Panda": 388,
    "Red Fox": 277,
    "Arctic Fox": 279,
    "Timber Wolf": 269,
    "Bald Eagle": 22,
    "Macaw": 88,
    "Flamingo": 130,
    "Peacock": 84,
    "Goldfish": 1,
    "Great White Shark": 2,
    "Jellyfish": 107,
    "Monarch Butterfly": 323,
    "Ladybug": 301,
    "Snail": 113,
    "Red Sports Car": 817,
    "School Bus": 779,
    "Steam Locomotive": 820,
    "Sailboat": 914,
    "Space Shuttle": 812,
    "Castle": 483,
    "Church": 497,
    "Lighthouse": 437,
    "Volcano": 980,
    "Lakeside": 975,
    "Cliff": 972,
    "Coral Reef": 973,
    "Valley": 979,
    "Seashore": 978,
    "Mushroom": 947,
    "Broccoli": 937,
    "Pizza": 963,
    "Ice Cream": 928,
    "Cheeseburger": 933,
    "Espresso": 967,
    "Acoustic Guitar": 402,
    "Grand Piano": 579,
    "Violin": 889,
    "Balloon": 417,
}

GENERATOR_RESOLUTION_PRESETS = {
    "256 × 256": (256, 256),
    "384 × 256": (384, 256),
    "256 × 384": (256, 384),
    "384 × 384": (384, 384),
    "512 × 256": (512, 256),
    "256 × 512": (256, 512),
    "512 × 512": (512, 512),
}

OUTPUT_RESOLUTION_PRESETS = {
    "Same as generator": None,
    "256 × 256": (256, 256),
    "384 × 384": (384, 384),
    "512 × 512": (512, 512),
    "768 × 768": (768, 768),
    "1024 × 1024": (1024, 1024),
    "512 × 256 (2:1)": (512, 256),
    "256 × 512 (1:2)": (256, 512),
    "768 × 512 (3:2)": (768, 512),
    "512 × 768 (2:3)": (512, 768),
    "1024 × 512 (2:1)": (1024, 512),
    "512 × 1024 (1:2)": (512, 1024),
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

vq_model = None
gpt_model = None


def download_checkpoint(filename: str) -> str:
    return hf_hub_download(repo_id=HF_REPO, filename=filename)


def _make_res_tensors(gen_h: int, gen_w: int, multiplier: int):
    """Create normalized resolution tensors for the GPT generator."""
    th = torch.tensor(gen_h / 1536, device=DEVICE, dtype=DTYPE).unsqueeze(0).repeat(multiplier, 1)
    tw = torch.tensor(gen_w / 1536, device=DEVICE, dtype=DTYPE).unsqueeze(0).repeat(multiplier, 1)
    return th, tw


def _warmup(model):
    """Run a throwaway generation to trigger torch.compile and warm CUDA caches."""
    print("Warming up (first call triggers compilation, may take ~30-60s)...")
    dummy_cond = torch.tensor([0], device=DEVICE)
    th, tw = _make_res_tensors(256, 256, multiplier=2)
    with torch.inference_mode():
        generate(
            model, dummy_cond, LATENT_SIZE, NUM_CODEBOOKS,
            cfg_scale=4.0, cfg_interval=-1,
            target_h=th, target_w=tw,
            temperature=1.0, top_k=500, top_p=1.0, sample_logits=True,
        )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    print("Warmup complete — subsequent generations will be fast.")


def load_models():
    global vq_model, gpt_model

    print("Downloading checkpoints (if needed)...")
    vq_path = download_checkpoint(VQ_CKPT_FILENAME)
    gpt_path = download_checkpoint(GPT_CKPT_FILENAME)

    print(f"Loading VibeToken tokenizer from {vq_path}...")
    vq_model = VibeTokenTokenizer.from_config(
        CONFIG_PATH, vq_path, device=DEVICE, dtype=DTYPE,
    )
    print("VibeToken tokenizer loaded.")

    print(f"Loading {GPT_MODEL_NAME} from {gpt_path}...")
    gpt_model = GPT_models[GPT_MODEL_NAME](
        vocab_size=CODEBOOK_SIZE,
        block_size=LATENT_SIZE,
        num_classes=NUM_CLASSES,
        cls_token_num=CLS_TOKEN_NUM,
        model_type="c2i",
        num_codebooks=NUM_CODEBOOKS,
        n_output_layer=NUM_OUTPUT_LAYER,
        class_dropout_prob=CLASS_DROPOUT_PROB,
        extra_layers=EXTRA_LAYERS,
        capping=CAPPING,
    ).to(device=DEVICE, dtype=DTYPE)

    checkpoint = torch.load(gpt_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        weights = checkpoint["model"]
    elif "module" in checkpoint:
        weights = checkpoint["module"]
    elif "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
    else:
        weights = checkpoint
    gpt_model.load_state_dict(weights, strict=True)
    gpt_model.eval()
    del checkpoint
    print(f"{GPT_MODEL_NAME} loaded.")

    if COMPILE:
        print("Compiling GPT model with torch.compile (max-autotune)...")
        gpt_model = torch.compile(gpt_model, mode="max-autotune", fullgraph=True)
        _warmup(gpt_model)
    else:
        print("Skipping torch.compile (set VIBETOKEN_NO_COMPILE=0 to enable).")


# ---------------------------------------------------------------------------
# Decoder patch-size heuristic
# ---------------------------------------------------------------------------

def auto_decoder_patch_size(h: int, w: int) -> tuple[int, int]:
    max_dim = max(h, w)
    if max_dim <= 256:
        ps = 8
    elif max_dim <= 512:
        ps = 16
    else:
        ps = 32
    return (ps, ps)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_image(
    class_name: str,
    class_id: int,
    gen_resolution_preset: str,
    out_resolution_preset: str,
    decoder_ps_choice: str,
    cfg_scale: float,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    randomize_seed: bool,
):
    if vq_model is None or gpt_model is None:
        raise gr.Error("Models are still loading. Please wait a moment and try again.")

    if randomize_seed:
        seed = random.randint(0, 2**31 - 1)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

    if class_name and class_name != "Custom (enter ID below)":
        cid = IMAGENET_CLASSES[class_name]
    else:
        cid = int(class_id)
    cid = max(0, min(cid, NUM_CLASSES - 1))

    gen_h, gen_w = GENERATOR_RESOLUTION_PRESETS[gen_resolution_preset]

    out_res = OUTPUT_RESOLUTION_PRESETS[out_resolution_preset]
    if out_res is None:
        out_h, out_w = gen_h, gen_w
    else:
        out_h, out_w = out_res

    if decoder_ps_choice == "Auto":
        dec_ps = auto_decoder_patch_size(out_h, out_w)
    else:
        ps = int(decoder_ps_choice)
        dec_ps = (ps, ps)

    multiplier = 2 if cfg_scale > 1.0 else 1

    c_indices = torch.tensor([cid], device=DEVICE)
    th, tw = _make_res_tensors(gen_h, gen_w, multiplier)

    index_sample = generate(
        gpt_model,
        c_indices,
        LATENT_SIZE,
        NUM_CODEBOOKS,
        cfg_scale=cfg_scale,
        cfg_interval=-1,
        target_h=th,
        target_w=tw,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        sample_logits=True,
    )

    index_sample = index_sample.unsqueeze(2)
    samples = vq_model.decode(
        index_sample,
        height=out_h,
        width=out_w,
        patch_size=dec_ps,
    )
    samples = torch.clamp(samples, 0, 1)

    img_np = (samples[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
    pil_img = Image.fromarray(img_np)

    return pil_img, seed


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

HEADER_MD = """
# VibeToken-Gen: Dynamic Resolution Image Generation

<p style="margin-top:4px;">
  <b>Maitreya Patel, Jingtao Li, Weiming Zhuang, Yezhou Yang, Lingjuan Lyu</b>
  &nbsp;|&nbsp;
</p>
<h3>CVPR 2026 (Main Conference)</h3>

<p>
  <a href="https://huggingface.co/mpatel57/VibeToken" target="_blank">🤗 Model</a> &nbsp;|&nbsp;
  <a href="https://github.com/patel-maitreya/VibeToken" target="_blank">💻 GitHub</a>
</p>

Generate ImageNet class-conditional images at **arbitrary resolutions** using only **65 tokens**.
VibeToken-Gen maintains a constant **179G FLOPs** regardless of output resolution.
"""

CITATION_MD = """
### Citation
```bibtex
@inproceedings{vibetoken2026,
  title     = {VibeToken: Scaling 1D Image Tokenizers and Autoregressive Models for Dynamic Resolution Generations},
  author    = {Patel, Maitreya and Li, Jingtao and Zhuang, Weiming and Yang, Yezhou and Lyu, Lingjuan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
"""

class_choices = ["Custom (enter ID below)"] + sorted(IMAGENET_CLASSES.keys())

with gr.Blocks(
    title="VibeToken-Gen Demo",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(HEADER_MD)

    with gr.Row():
        # ---- Left column: controls ----
        with gr.Column(scale=1):
            class_dropdown = gr.Dropdown(
                label="ImageNet Class",
                choices=class_choices,
                value="Golden Retriever",
                info="Pick a class or choose 'Custom' to enter an ID manually.",
            )
            class_id_input = gr.Number(
                label="Custom Class ID (0–999)",
                value=207,
                minimum=0,
                maximum=999,
                step=1,
                visible=False,
            )
            gen_resolution_dropdown = gr.Dropdown(
                label="Generator Resolution",
                choices=list(GENERATOR_RESOLUTION_PRESETS.keys()),
                value="256 × 256",
                info="Internal resolution for the AR generator (max 512×512).",
            )
            out_resolution_dropdown = gr.Dropdown(
                label="Output Resolution (Decoder)",
                choices=list(OUTPUT_RESOLUTION_PRESETS.keys()),
                value="Same as generator",
                info="Final image resolution. Set higher for super-resolution (e.g. generate at 256, decode at 1024).",
            )
            decoder_ps_dropdown = gr.Dropdown(
                label="Decoder Patch Size",
                choices=["Auto", "8", "16", "32"],
                value="Auto",
                info="'Auto' selects based on output resolution. Larger = faster but coarser.",
            )

            with gr.Accordion("Advanced Sampling Parameters", open=False):
                cfg_slider = gr.Slider(
                    label="CFG Scale",
                    minimum=1.0, maximum=20.0, value=4.0, step=0.5,
                    info="Classifier-free guidance strength.",
                )
                temp_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.1, maximum=2.0, value=1.0, step=0.05,
                )
                topk_slider = gr.Slider(
                    label="Top-k",
                    minimum=0, maximum=2000, value=500, step=10,
                    info="0 disables top-k filtering.",
                )
                topp_slider = gr.Slider(
                    label="Top-p",
                    minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                    info="1.0 disables nucleus sampling.",
                )
                seed_input = gr.Number(
                    label="Seed", value=0, minimum=0, maximum=2**31 - 1, step=1,
                )
                randomize_cb = gr.Checkbox(label="Randomize seed", value=True)

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

        # ---- Right column: output ----
        with gr.Column(scale=2):
            output_image = gr.Image(label="Generated Image", type="pil", height=512)
            used_seed = gr.Number(label="Seed used", interactive=False)

    # Show/hide custom class ID field
    def toggle_custom_id(choice):
        return gr.update(visible=(choice == "Custom (enter ID below)"))

    class_dropdown.change(
        fn=toggle_custom_id,
        inputs=[class_dropdown],
        outputs=[class_id_input],
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[
            class_dropdown,
            class_id_input,
            gen_resolution_dropdown,
            out_resolution_dropdown,
            decoder_ps_dropdown,
            cfg_slider,
            temp_slider,
            topk_slider,
            topp_slider,
            seed_input,
            randomize_cb,
        ],
        outputs=[output_image, used_seed],
    )

    gr.Markdown(CITATION_MD)


if __name__ == "__main__":
    load_models()
    demo.launch()
