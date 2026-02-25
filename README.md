# [CVPR 2026] VibeToken: Scaling 1D Image Tokenizers and Autoregressive Models for Dynamic Resolution Generations

<p align="center">
  <img src="assets/teaser.png" alt="VibeToken Teaser" width="100%">
</p>

<p align="center">
  <b>CVPR 2026</b> &nbsp;|&nbsp;
  <a href="#">Paper</a> &nbsp;|&nbsp;
  <a href="#">Project Page</a> &nbsp;|&nbsp;
  <a href="#-checkpoints">Checkpoints</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CVPR-2026-blue" alt="CVPR 2026">
  <img src="https://img.shields.io/badge/arXiv-TODO-b31b1b" alt="arXiv">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
  <a href="https://huggingface.co/mpatel57/VibeToken"><img src="https://img.shields.io/badge/🤗-Model-yellow" alt="HuggingFace"></a>
</p>

---

We introduce an efficient, resolution-agnostic autoregressive (AR) image synthesis approach that generalizes to **arbitrary resolutions and aspect ratios**, narrowing the gap to diffusion models at scale. At its core is **VibeToken**, a novel resolution-agnostic 1D Transformer-based image tokenizer that encodes images into a dynamic, user-controllable sequence of 32--256 tokens, achieving state-of-the-art efficiency and performance trade-off. Building on VibeToken, we present **VibeToken-Gen**, a class-conditioned AR generator with out-of-the-box support for arbitrary resolutions while requiring significantly fewer compute resources.

### 🔥 Highlights

| | |
|---|---|
| 🎯 **1024×1024 in just 64 tokens** | Achieves **3.94 gFID** vs. 5.87 gFID for diffusion-based SOTA (1,024 tokens) |
| ⚡ **Constant 179G FLOPs** | 63× more efficient than LlamaGen (11T FLOPs at 1024×1024) |
| 🌐 **Resolution-agnostic** | Supports arbitrary resolutions and aspect ratios out of the box |
| 🎛️ **Dynamic token count** | User-controllable 32--256 tokens per image |
| 🔍 **Native super-resolution** | Supports image super-resolution out of the box |


## 📰 News

- **[Feb 2026]** 🎉 VibeToken is accepted at **CVPR 2026**!
- **[Feb 2026]** Training scripts released.
- **[Feb 2026]** Inference code and checkpoints released.


## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/<your-org>/VibeToken.git
cd VibeToken
uv venv --python=3.11.6
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Download a checkpoint (see Checkpoints section below)
mkdir -p checkpoints
wget https://huggingface.co/mpatel57/VibeToken/resolve/main/VibeToken_LL.bin -O ./checkpoints/VibeToken_LL.bin

# 3. Reconstruct an image
python reconstruct.py --auto \
  --config configs/vibetoken_ll.yaml \
  --checkpoint ./checkpoints/VibeToken_LL.bin \
  --image ./assets/example_1.png \
  --output ./assets/reconstructed.png
```


## 📦 Checkpoints

All checkpoints are hosted on [Hugging Face](https://huggingface.co/mpatel57/VibeToken).

#### Reconstruction Checkpoints

| Name | Resolution | rFID (256 tokens) | rFID (64 tokens) | Download |
|------|:----------:|:-----------------:|:----------------:|----------|
| VibeToken-LL | 1024×1024 | 3.76 | 4.12 | [VibeToken_LL.bin](https://huggingface.co/mpatel57/VibeToken/resolve/main/VibeToken_LL.bin) |
| VibeToken-LL | 256×256 | 5.12 | 0.90 | same as above |
| VibeToken-SL | 1024×1024 | 4.25 | 2.41 | [VibeToken_SL.bin](https://huggingface.co/mpatel57/VibeToken/resolve/main/VibeToken_SL.bin) |
| VibeToken-SL | 256×256 | 5.44 | 0.40 | same as above |

#### Generation Checkpoints

| Name | Training Resolution(s) | Tokens | Best gFID | Download |
|------|:----------------------:|:------:|:---------:|----------|
| VibeToken-Gen-B | 256×256 | 65 | 7.62 | [VibeTokenGen-b-fixed65_dynamic_1500k.pt](https://huggingface.co/mpatel57/VibeToken/resolve/main/VibeTokenGen-b-fixed65_dynamic_1500k.pt) |
| VibeToken-Gen-B | 1024×1024 | 65 | 7.37 | same as above |
| VibeToken-Gen-XXL | 256×256 | 65 | 3.62 | [VibeTokenGen-xxl-dynamic-65_750k.pt](https://huggingface.co/mpatel57/VibeToken/resolve/main/VibeTokenGen-xxl-dynamic-65_750k.pt) |
| VibeToken-Gen-XXL | 1024×1024 | 65 | **3.54** | same as above |


## 🛠️ Setup

```bash
uv venv --python=3.11.6
source .venv/bin/activate
uv pip install -r requirements.txt
```

> **Tip:** If you don't have `uv`, install it via `pip install uv` or see [uv docs](https://github.com/astral-sh/uv). Alternatively, use `python -m venv .venv && pip install -r requirements.txt`.


## 🖼️ VibeToken Reconstruction

Download the VibeToken-LL checkpoint (see [Checkpoints](#-checkpoints)), then:

```bash
# Auto mode (recommended) -- automatically determines optimal patch sizes
python reconstruct.py --auto \
  --config configs/vibetoken_ll.yaml \
  --checkpoint ./checkpoints/VibeToken_LL.bin \
  --image ./assets/example_1.png \
  --output ./assets/reconstructed.png

# Manual mode -- specify patch sizes explicitly
python reconstruct.py \
  --config configs/vibetoken_ll.yaml \
  --checkpoint ./checkpoints/VibeToken_LL.bin \
  --image ./assets/example_1.png \
  --output ./assets/reconstructed.png \
  --encoder_patch_size 16 \
  --decoder_patch_size 16
```

> **Note:** For best performance, the input image resolution should be a multiple of 32. Images with other resolutions are automatically rescaled to the nearest multiple of 32.


## 🎨 VibeToken-Gen: ImageNet-1k Generation

Download both the VibeToken-LL and VibeToken-Gen-XXL checkpoints (see [Checkpoints](#-checkpoints)), then:

```bash
python generate.py \
    --gpt-ckpt ./checkpoints/VibeTokenGen-xxl-dynamic-65_750k.pt \
    --gpt-model GPT-XXL --num-output-layer 4 \
    --num-codebooks 8 --codebook-size 32768 \
    --image-size 256 --cfg-scale 4.0 --top-k 500 --temperature 1.0 \
    --class-dropout-prob 0.1 \
    --extra-layers "QKV" \
    --latent-size 65 \
    --config ./configs/vibetoken_ll.yaml \
    --vq-ckpt ./checkpoints/VibeToken_LL.bin \
    --sample-dir ./assets/ \
    --skip-folder-creation \
    --compile \
    --decoder-patch-size 32,32 \
    --target-resolution 1024,1024 \
    --llamagen-target-resolution 256,256 \
    --precision bf16 \
    --global-seed 156464151
```

The `--target-resolution` controls the tokenizer output resolution, while `--llamagen-target-resolution` controls the generator's internal resolution (max 512×512; for higher resolutions, the tokenizer handles upscaling).


## 🏋️ Training

To train the VibeToken tokenizer from scratch, please refer to [TRAIN.md](TRAIN.md) for detailed instructions.


## 🙏 Acknowledgement

We would like to acknowledge the following repositories that inspired our work and upon which we directly build:
[1d-tokenizer](https://github.com/bytedance/1d-tokenizer),
[LlamaGen](https://github.com/FoundationVision/LlamaGen), and
[UniTok](https://github.com/FoundationVision/UniTok).


## 📝 Citation

If you find VibeToken useful in your research, please consider citing:

```bibtex
@inproceedings{vibetoken2026,
  title     = {VibeToken: Scaling 1D Image Tokenizers and Autoregressive Models for Dynamic Resolution Generations},
  author    = {Patel, Maitreya and Li, Jingtao and Zhuang, Weiming and Yang, Yezhou and Lyu, Lingjuan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

If you have any questions, feel free to open an issue or reach out!
