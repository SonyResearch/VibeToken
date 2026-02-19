#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1
hf download ILSVRC/imagenet-1k --repo-type dataset --local-dir /mnt/localssd/datasets/imagenet-1k


# convert the imagenet to wds
python data/convert_imagenet_to_wds.py --output_dir /mnt/localssd/datasets/imagenet_wds