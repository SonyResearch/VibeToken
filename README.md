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

All checkpoints are avaiable on S3. Please refer to the following S3 access keys.

```bash
pip install awscli

export AWS_ACCESS_KEY_ID=AKIATXRXFEIWPN2VLGUE
export AWS_SECRET_ACCESS_KEY=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

aws s3 cp <s3-path> <output-path>
```

| Name                   | Resolution | rFID (256 tokens) | rFID (64 tokens) | Download Link                                           |
|------------------------|:---------------------:|:-----------------:|:----------------:|---------------------------------------------------------|
| VibeToken-LL      | 1024x1024                 | 3.76              | 4.12             | s3://vibetoken/checkpoints/VibeToken_LL.bin |
| VibeToken-LL      | 256x256                   | 5.12              | 0.90             | s3://vibetoken/checkpoints/VibeToken_LL.bin (same as above) |
| VibeToken-SL      | 1024x1024                 | 4.25              | 2.41             | s3://vibetoken/checkpoints/VibeToken_SL.bin |
| VibeToken-SL      | 256x256                   | 5.44              | 0.40             | s3://vibetoken/checkpoints/VibeToken_SL.bin (same as above) |


### VibeToken-Gen Generation Checkpoints

| Name                        | Training Resolution(s) | Tokens    | Best gFID | Download Link                                           |
|-----------------------------|:---------------------:|:-------------:|:---------:|---------------------------------------------------------|
| VibeToken-Gen-B         | 256x256               | 65            | 7.62      | s3://vibetoken/checkpoints/VibeTokenGen-b-fixed65_dynamic_1500k.pt      |
| VibeToken-Gen-B         | 1024x1024             | 65            | 7.37      | s3://vibetoken/checkpoints/VibeTokenGen-b-fixed65_dynamic_1500k.pt   (same as above)   |
| VibeToken-Gen-XXL       | 256x256               | 65            | 3.62      | s3://vibetoken/checkpoints/VibeTokenGen-xxl-dynamic-65_750k.pt  |
| VibeToken-Gen-XXL       | 1024x1024             | 65            | 3.54      | s3://vibetoken/checkpoints/VibeTokenGen-xxl-dynamic-65_750k.pt (same as above) |


## Setup

```bash
uv venv --python=3.11.6
source .venv/bin/activate
uv pip install -r requirements.txt # this is not maintained
```


## VibeToken Reconstruction

- Download the VibeToken-LL

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

- Download the VibeToken-LL
- Download the VibeToken-Gen-XXL


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

## Train

To train the model, please refer to the [TRAIN.md](TRAIN.md) script for detailed instructions.

## Acknowledgement

We would like to acknowledge the following repository which inspired our works and directly builds upon their source code: 1d-tokenizer, LLamaGen, and UniTok.

## Citation
