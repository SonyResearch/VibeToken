# VibeToken: Scaling 1D Image Tokenizers and Autoregressive Models for Dynamic Resolution Generations

<p align="center">
  <img src="assets/teaser.png" alt="VibeToken Teaser" width="100%">
</p>


We introduce an efficient, resolution-agnostic autoregressive (AR) image synthesis approach that generalizes to arbitrary resolutions and aspect ratios, narrowing the gap to diffusion models at scale. At its core is VibeToken, a novel resolution-agnostic 1D Transformer-based image tokenizer that encodes images into a dynamic, user-controllable sequence of 32–256 tokens, achieving a state-of-the-art efficiency and performance trade-off. Building on VibeToken, we present VibeToken-Gen, a class-conditioned AR generator with out-of-the-box support for arbitrary resolutions while requiring significantly fewer compute resources. Notably, VibeToken-Gen synthesizes 1024x1024 images using only 64 tokens and achieves 3.94 gFID; by comparison, a diffusion-based state-of-the-art alternative requires 1,024 tokens and attains 5.87 gFID. In contrast to fixed-resolution AR models such as LlamaGen—whose inference FLOPs grow quadratically with resolution (11T FLOPs at 1024x1024)—VibeToken-Gen maintains a constant 179G FLOPs (63.4 efficient) independent of resolution. We hope VibeToken can help unlock the wide adoption of AR visual generative models in production use cases.


## Releases

- [x] Inference Code
- [x] Checkpoint Release
- [ ] Training Scripts


## Checkpoints


### VibeToken Reconstruction Checkpoints

Note: these links are random and not valid

| Name                   | Resolution | rFID (256 tokens) | rFID (64 tokens) | Download Link                                           |
|------------------------|:---------------------:|:-----------------:|:----------------:|---------------------------------------------------------|
| VibeToken-LL      | 1024x1024                 | 3.76              | 4.53             | [hf.co/VibeToken/L-32x32](https://huggingface.co/VibeToken/L-32x32) |
| VibeToken-LL      | 256x256                   | 5.12              | 5.96             | [hf.co/VibeToken/L-32x32](https://huggingface.co/VibeToken/L-32x32) (same as above) |
| VibeToken-SL      | 1024x1024                 | 4.25              | 4.97             | [hf.co/VibeToken/L-32x32](https://huggingface.co/VibeToken/L-32x32) |
| VibeToken-SL      | 256x256                   | 5.44              | 6.23             | [hf.co/VibeToken/L-32x32](https://huggingface.co/VibeToken/L-32x32) (same as above) |


### VibeToken-Gen Generation Checkpoints

| Name                        | Training Resolution(s) | Tokens    | Best gFID | Download Link                                           |
|-----------------------------|:---------------------:|:-------------:|:---------:|---------------------------------------------------------|
| VibeToken-Gen-B         | 256x256               | 64            | 5.26      | [hf.co/VibeToken/Gen-B](https://huggingface.co/VibeToken/Gen-B)      |
| VibeToken-Gen-B         | 1024x1024             | 64            | 3.94      | [hf.co/VibeToken/Gen-B](https://huggingface.co/VibeToken/Gen-B)   (same as above)   |
| VibeToken-Gen-XXL       | 256x256               | 64            | 4.80      | [hf.co/VibeToken/Gen-XXL](https://huggingface.co/VibeToken/Gen-XXL)  |
| VibeToken-Gen-XXL       | 1024x1024             | 64            | 3.50      | [hf.co/VibeToken/Gen-XXL](https://huggingface.co/VibeToken/Gen-XXL) (same as above) |


## Setup

```bash
uv venv --python=3.11.6
source .venv/bin/activate
uv pip install -r requirements.txt

# to download two key checkpoints VibeToken-LL and VibeToken-Gen XXL/65
bash setup.sh
```


## VibeToken Reconstruction

```python
# select auto for our suggested adaptation to arbitrary resolution
python reconstruct.py     \
  --config configs/vibetoken_ll.yaml     \
  --checkpoint /mnt/localssd/vibetoken_mvq_ll.bin     \
  --image ./assets/example_1.png     \
  --output assets/reconstructed.png \
  --auto

# or manually define the parameters
python reconstruct.py     \
  --config configs/vibetoken_ll.yaml     \
  --checkpoint /mnt/localssd/vibetoken_mvq_ll.bin     \
  --image ./assets/example_1.png     \
  --output assets/reconstructed.png \
  --encoder_patch_size 16 \
  --decoder_patch_size 16
```
Note: We require the input image resolution to be the factor of 32 for the best performance. Hence, for any other rnadom resolutions, we reslace the image to nearest multiple of 32.

## VibeToken-Gen ImageNet1k Generations


```bash
python generate.py \
    --gpt-ckpt /mnt/localssd/vibetoken/gpt-xxl-dynamic-65_750k.pt \
    --gpt-model GPT-XXL --num-output-layer 4 \
    --num-codebooks 8 --codebook-size 32768 \
    --image-size 256 --cfg-scale 4.0 --top-k 500 --temperature 1.0 \
    --class-dropout-prob 0.1 \
    --extra-layers "QKV" \
    --latent-size 65 \
    --config ./configs/vibetoken_ll.yaml \
    --vq-ckpt /mnt/localssd/vibetoken/MVQ_LL_590k.bin \
    --sample-dir ./assets/ \
    --skip-folder-creation \
    --compile \
    --decoder-patch-size 32,32 \
    --target-resolution 1024,1024 \ # this is for tokenizer
    --llamagen-target-resolution 256,256 \ # this is for the generator (maximum is 512,512 for higher resolution handle this via tokenizer)
    --precision bf16 \
    --skip-folder-creation \
    --global-seed 156464151
```


## Acknowledgement

We would like to acknowledge the following repository which inspired our works and directly builds upon their source code: 1d-tokenizer, LLamaGen, and UniTok.

## Citation
