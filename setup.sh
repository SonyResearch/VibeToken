#!/bin/bash
# Data preparation script for VibeToken training.
# Set DATA_DIR to control where datasets are stored (defaults to ./data).
#
# Usage:
#   export DATA_DIR=/mnt/fastssd/datasets   # optional, defaults to ./data
#   bash setup.sh

DATA_DIR="${DATA_DIR:-./data}"

echo "Using DATA_DIR=${DATA_DIR}"

# Download ImageNet-1k via HuggingFace
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset --local-dir "${DATA_DIR}/imagenet-1k"

# Convert to WebDataset format
python data/convert_imagenet_to_wds.py \
    --input_dir "${DATA_DIR}/imagenet-1k" \
    --output_dir "${DATA_DIR}/imagenet_wds"
