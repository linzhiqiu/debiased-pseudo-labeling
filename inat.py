# (1) prepare inat test split by merging official train and val -- "l_train_and_val" 
# (2) make "u_train_in" a pytorch image folder
# after downloading from https://github.com/cvl-umass/semi-inat-2021
# python inat.py --data ./
import argparse
import os
import shutil
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
from pathlib import Path

import data.datasets as datasets

parser = argparse.ArgumentParser(description='PyTorch Dataset Index Preparation')
parser.add_argument('--data', metavar='DIR', default="./",
                    help='path to downloaded dataset')

def main():
    args = parser.parse_args()
    print(args)

    data = Path(args.data) / "inat"
    original_l_train_path = data / "l_train"
    original_val_path = data / "val"

    new_test_path = data / "l_train_and_val"
    if new_test_path.exists():
        print(f"{new_test_path} already exists.")
    else:
        assert original_l_train_path.exists(), f"{original_l_train_path} does not exist."
        assert original_val_path.exists(), f"{original_val_path} does not exist."
        new_test_path.mkdir()
        # get all folder names under original_l_train_path
        folder_names = [f.name for f in original_l_train_path.iterdir() if f.is_dir()]
        for folder_name in folder_names:
            l_train_folder = original_l_train_path / folder_name
            val_folder = original_val_path / folder_name
            new_test_folder = new_test_path / folder_name
            new_test_folder.mkdir()
            # copy all files from l_train_folder to new_test_folder
            for f in l_train_folder.iterdir():
                shutil.copy(f, new_test_folder)
            # copy all files from val_folder to new_test_folder
            for f in val_folder.iterdir():
                # get file name without extension
                file_name = f.name.split(".")[0]
                new_file_name = file_name + "_val" + ".jpg"
                shutil.copy(f, new_test_folder / new_file_name)
    
    folder_names = [f.name for f in new_test_path.iterdir() if f.is_dir()]
    # read from u_train_in.txt
    u_train_in_path = data / "u_train_in.txt"
    assert u_train_in_path.exists(), f"{u_train_in_path} does not exist."
    with open(u_train_in_path, "r") as f:
        lines = f.readlines()
        # split by space
        lines = [line.strip().split(" ") for line in lines]
    
    
    u_train_path = data / "u_train"
    u_train_in_path = data / "u_train_in"
    if u_train_in_path.exists():
        print(f"{u_train_in_path} already exists.")
    else:
        assert u_train_path.exists(), f"{u_train_path} does not exist."
        u_train_in_path.mkdir()
        for folder_name in folder_names:
            u_train_in_folder = u_train_in_path / folder_name
            u_train_in_folder.mkdir()
        for path, label in tqdm(lines):
            u_train_in_folder = u_train_in_path / label
            shutil.copy(data / path, u_train_in_folder)
    
if __name__ == '__main__':
    main()