# Adapted from https://github.com/webdataset/webdataset-imagenet/blob/main/convert-imagenet.py

import argparse
import os
import sys
import time

import webdataset as wds
from datasets import load_dataset


def convert_imagenet_to_wds(input_dir, output_dir, max_train_samples_per_shard, max_val_samples_per_shard):
    assert not os.path.exists(os.path.join(output_dir, "imagenet-train-000000.tar"))
    assert not os.path.exists(os.path.join(output_dir, "imagenet-val-000000.tar"))

    opat = os.path.join(output_dir, "imagenet-train-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_train_samples_per_shard)
    dataset = load_dataset(input_dir, split="train")
    now = time.time()
    for i, example in enumerate(dataset):
        if i % max_train_samples_per_shard == 0:
            print(i, file=sys.stderr)
        img, label = example["image"], example["label"]
        output.write({"__key__": "%08d" % i, "jpg": img.convert("RGB"), "cls": label})
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {i+1} train examples in {time_taken // 3600} hours.")

    opat = os.path.join(output_dir, "imagenet-val-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_val_samples_per_shard)
    dataset = load_dataset(input_dir, split="validation")
    now = time.time()
    for i, example in enumerate(dataset):
        if i % max_val_samples_per_shard == 0:
            print(i, file=sys.stderr)
        img, label = example["image"], example["label"]
        output.write({"__key__": "%08d" % i, "jpg": img.convert("RGB"), "cls": label})
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {i+1} val examples in {time_taken // 60} min.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the ImageNet-1k dataset (HuggingFace format).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory for WebDataset shards.")
    parser.add_argument("--max_train_samples_per_shard", type=int, default=10000,
                        help="Maximum number of training samples per shard.")
    parser.add_argument("--max_val_samples_per_shard", type=int, default=10000,
                        help="Maximum number of validation samples per shard.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    convert_imagenet_to_wds(args.input_dir, args.output_dir, args.max_train_samples_per_shard, args.max_val_samples_per_shard)