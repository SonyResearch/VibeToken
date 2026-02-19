# Training Instructions

## VibeToken MVQ tokenizer

This repository contains the training code for our tokenizer. 
We provide the example config [VibeToken-Small](configs/training/VibeToken_small.yaml) that trains the small encoder/deocder architecture with 32-64 tokens.
To train the model:

```bash
source .venv/bin/activate

# download the data and prepare the imagenet shards
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download ILSVRC/imagenet-1k --repo-type dataset --local-dir /mnt/localssd/datasets/imagenet-1k
python data/convert_imagenet_to_wds.py --output_dir /mnt/localssd/datasets/imagenet_wds

# start the training on 1 node with 8 GPUs
bash train_tokenizer.sh
```

### Instruction to manage the config

Below are the important hyperparameters to manage the training. 

```yaml
model:
    vq_model:
        vit_enc_model_size: "small"     # this can be small/base/large
        vit_dec_model_size: "small"     # this can be small/base/large
        num_latent_tokens: 64           # in paper we set this to 256

losses:
    discriminator_start: 100_000        # set this parameter based on the convergence, in paper we set this to 250_000

dataset:
    params:
        pretokenization: True           # keep this true if using the current setup
        train_shards_path_or_url: "/mnt/localssd/datasets/imagenet_wds/imagenet-train-{000001..000128}.tar" # path to training shards
        eval_shards_path_or_url: "/mnt/localssd/datasets/imagenet_wds/imagenet-val-{000001..000004}.tar"    # path to val shards
    preprocessing:
        resize_shorter_edge: 512        # this is maximum size during pretraining but can be any value
        crop_size: 512                  # this is maximum size during pretraining but can be any value
        min_tokens: 32                  # minimum number of tokens at least to generate
        max_tokens: 64                  # maximum number of tokens at most to generate

training:
    gradient_accumulation_steps: 1      # our LL model does not fit in single node so we increase the grad-accumulation
    per_gpu_batch_size: 32              # our LL model does not fit in single node so we decrease the batch size to 16; note that during GAN training this will be reduced by half
    max_train_steps: 400_000            # in paper, we train upto 650_000 but after 600_000 model starts diverging so we peak 600_000 checkpoint
    num_generated_images: 2             # for valdiation
    variable_resolution:                # this for any to any resolution training
        any2any: True
        dim:
          - [256, 256]
          - [512, 512]
          - [384, 256]
          - [256, 384]
          - [512, 384]
          - [384, 512]
        ratio: [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]   # probability of selecting the certain resolution in order of above; sum must be equal to 1.0


# remove any patch mixture related parameters unless not able to fit the model
# this will slow down the speed and may hurt the performance 
# we do not use this in our normal setup
model:
    vq_model:
        encoder: # patch mixture is not supported
            patch_mixture_start_layer: 2
            patch_mixture_end_layer: 22
        decoder: # patch mixture is not supported
            patch_mixture_start_layer: 2
            patch_mixture_end_layer: 22
```


### Reproduced Results on Small Baseline

> Note: Our released checkpoints are from different codebase and may observe +/- changes in results.

Below we report the performance on above training script on small baseline. This baseline is not reported in the paper but achieves the competitive performance as expected.
