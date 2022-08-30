# To prepare imagenet127 train/val/test splits after downloading from http://image-net.org/download-images
# python prepare_imagenet127_index.py ./ --root=/scratch/fercus/ --seed=1
import argparse
import math
import os
import shutil
import time
import random
import numpy as np
import pandas as pd
import gdown
import torch
import torch.utils.data
from pathlib import Path

from utils import extract_archive
import data.datasets as datasets

parser = argparse.ArgumentParser(description='PyTorch Dataset Index Preparation')
parser.add_argument('data', metavar='DIR',
                    help='path to downloaded dataset')
parser.add_argument('--root', default=None, type=str, metavar='DIR',
                    help='path to root (specify if need to transfer dataset from args.data to a compute node)')
parser.add_argument('--index_name', default='default', type=str,
                    help='name of index')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for split train/val sets. ')
parser.add_argument('--train-ratio', type=float, default=0.08,
                    help='ratio of train data')
parser.add_argument('--val-ratio', type=float, default=0.02,
                    help='ratio of val data')


def main():
    args = parser.parse_args()
    print(args)

    data = Path(args.data)
    assert (data / 'imagenet').exists(), "imagenet folder not found"

    if args.root is not None:
        root = Path(args.root) / "imagenet"
        if not root.exists():
            print(f"Copy data from {data}/imagenet to {root}")
            print(f"It could be faster to run rsync: ")
            print(f"rsync -rtuvp {data / 'semi_inat'}/ {root}/")
            shutil.copytree(data / "imagenet", root)
        else:
            print(f"{root} already exists.")
    else:
        root = data / "imagenet"

    index_dir = Path("indexes") / "imagenet127" / f"{args.index_name}"
    if not index_dir.exists():
        index_dir.mkdir(parents=True)
    train_index_file = index_dir / "train.csv"
    val_index_file = index_dir / "val.csv"
    unlabeled_index_file = index_dir / "unlabeled.csv"
    test_index_file = Path("indexes") / "imagenet127" / f"test.csv"

    all_dataset, test_dataset = datasets.get_imagenet127_datasets(root)
    if not test_index_file.exists():
        # make test index
        test_samples = test_dataset.samples
        test_indexes = [idx for idx, _ in enumerate(test_samples)]
        test_paths = [s[0][len(str(root))+1:] for s in test_samples] # relative to root
        test_targets = [s[1] for s in test_samples]
        test_df = pd.DataFrame({"Index": test_indexes, "Path": test_paths, "Target": test_targets})
        test_df.to_csv(test_index_file, index=False)
        print(f"Saved at {test_index_file}")
    assert test_index_file.exists()

    if not train_index_file.exists() \
            or not val_index_file.exists() \
            or not unlabeled_index_file.exists():
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
        assert 0 < args.val_ratio + args.train_ratio < 1

        all_samples = all_dataset.samples
        all_indexes = [idx for idx, _ in enumerate(all_samples)]
        all_paths = [s[0][len(str(root))+1:] for s in all_samples] # relative to root
        all_targets = [s[1] for s in all_samples]

        num_classes = max(all_targets) + 1

        train_index, unlabeled_index, val_index = datasets.x_u_v_split(
            all_targets, args.train_ratio, args.val_ratio, num_classes)
        assert len(train_index) + len(val_index) + len(unlabeled_index) == len(all_indexes)
        train_pd = pd.DataFrame(
            {"Index": train_index, 
             "Path": [all_paths[i] for i in train_index], 
             "Target": [all_targets[i] for i in train_index]}
        )
        val_pd = pd.DataFrame(
            {"Index": val_index,
             "Path": [all_paths[i] for i in val_index],
             "Target": [all_targets[i] for i in val_index]}
        )
        unlabeled_pd = pd.DataFrame(
            {"Index": unlabeled_index,
             "Path": [all_paths[i] for i in unlabeled_index],
             "Target": [all_targets[i] for i in unlabeled_index]}
        )
        train_pd.to_csv(train_index_file, index=False)
        val_pd.to_csv(val_index_file, index=False)
        unlabeled_pd.to_csv(unlabeled_index_file, index=False)

        print(f"Saved at {train_index_file}")
        print(f"Saved at {val_index_file}")
        print(f"Saved at {unlabeled_index_file}")
    else:
        print(f"Index files already exists.")

if __name__ == '__main__':
    main()