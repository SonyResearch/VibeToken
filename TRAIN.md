# Training Instructions

## VibeToken MVQ Tokenizer

This repository contains the training code for our tokenizer.
We provide the example config [VibeToken-Small](configs/training/VibeToken_small.yaml) that trains the small encoder/decoder architecture with 32-64 tokens.

### Data Preparation

All data paths are controlled by the `DATA_DIR` environment variable. Set it once to point to your preferred storage location:

```bash
export DATA_DIR=/path/to/your/storage   # defaults to ./data if unset
```

Download ImageNet-1k and convert to WebDataset format:

```bash
source .venv/bin/activate

# Option 1: Use the setup script (recommended)
bash setup.sh

# Option 2: Run steps manually
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset --local-dir "${DATA_DIR}/imagenet-1k"
python data/convert_imagenet_to_wds.py \
    --input_dir "${DATA_DIR}/imagenet-1k" \
    --output_dir "${DATA_DIR}/imagenet_wds"
```

After preparation, update the shard paths in your training config to match your `DATA_DIR`:

```yaml
dataset:
    params:
        train_shards_path_or_url: "<DATA_DIR>/imagenet_wds/imagenet-train-{000001..000128}.tar"
        eval_shards_path_or_url: "<DATA_DIR>/imagenet_wds/imagenet-val-{000001..000004}.tar"
```

### Launch Training

Start training on 1 node with 8 GPUs:

```bash
source .venv/bin/activate
bash train_tokenizer.sh
```

### Config Reference

Below are the important hyperparameters to manage the training.

```yaml
model:
    vq_model:
        vit_enc_model_size: "small"     # this can be small/base/large
        vit_dec_model_size: "small"     # this can be small/base/large
        num_latent_tokens: 64           # in paper we set this to 256

losses:
    discriminator_start: 100_000        # set based on convergence, in paper we set this to 250_000

dataset:
    params:
        pretokenization: True           # keep this true if using the current setup
        train_shards_path_or_url: "./data/imagenet_wds/imagenet-train-{000001..000128}.tar"
        eval_shards_path_or_url: "./data/imagenet_wds/imagenet-val-{000001..000004}.tar"
    preprocessing:
        resize_shorter_edge: 512        # maximum size during pretraining but can be any value
        crop_size: 512                  # maximum size during pretraining but can be any value
        min_tokens: 32                  # minimum number of tokens to generate
        max_tokens: 64                  # maximum number of tokens to generate

training:
    gradient_accumulation_steps: 1      # increase for LL model that does not fit on single node
    per_gpu_batch_size: 32              # decrease to 16 for LL model; during GAN training this is halved
    max_train_steps: 400_000            # in paper we train up to 650_000; model may diverge after 600_000
    num_generated_images: 2             # for validation
    variable_resolution:                # any-to-any resolution training
        any2any: True
        dim:
          - [256, 256]
          - [512, 512]
          - [384, 256]
          - [256, 384]
          - [512, 384]
          - [384, 512]
        ratio: [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]   # probability per resolution; must sum to 1.0


# Remove patch mixture parameters unless the model does not fit in memory.
# This will slow down training and may hurt performance.
# We do not use this in our normal setup.
model:
    vq_model:
        encoder:
            patch_mixture_start_layer: 2
            patch_mixture_end_layer: 22
        decoder:
            patch_mixture_start_layer: 2
            patch_mixture_end_layer: 22
```


### Reproduced Results on Small Baseline

> Note: Our released checkpoints are from a different codebase and may observe +/- changes in results.

Below we report the performance on the above training script on the small baseline. This baseline is not reported in the paper but achieves competitive performance as expected.
