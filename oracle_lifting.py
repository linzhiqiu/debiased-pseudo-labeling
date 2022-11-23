# python oracle_lifting.py --dataset inat --seed=1 --index_name default --threshold 18 --budget 6 --lift tail_uniform
# python oracle_lifting.py --dataset inat --seed=1 --index_name default --threshold 18 --budget 6 --lift uniform
# python oracle_lifting.py --dataset imagenet127 --seed=1 --index_name default --threshold 25 --budget 12 --lift tail_uniform
# python oracle_lifting.py --dataset imagenet127 --seed=1 --index_name default --threshold 25 --budget 12 --lift uniform
# python oracle_lifting.py --dataset inat --seed=1 --index_name default --threshold 18 --lift random
# python oracle_lifting.py --dataset inat --seed=1 --index_name default --threshold 18 --lift tail
# python oracle_lifting.py --dataset imagenet127 --seed=1 --index_name default --threshold 25 --lift random
# python oracle_lifting.py --dataset imagenet127 --seed=1 --index_name default --threshold 25 --lift tail
import argparse
import shutil
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from pathlib import Path

parser = argparse.ArgumentParser(description='PyTorch Dataset Index Preparation')
parser.add_argument('--dataset', default='imagenet127', type=str,
                    choices=['inat', 'imagenet127'],
                    help='dataset to use, choices: [inat, imagenet127]')
parser.add_argument('--index_name', default='default', type=str,
                    help='name of index dir (the directory under indexes/)')
parser.add_argument('--lift', default='random', type=str,
                    choices=['random', 'tail', 'tail_uniform', 'uniform'],
                    help='lifting method')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for lifting tail. ')
parser.add_argument('--threshold', type=int, default=20,
                    help='to lift tail such that all classes are above this threshold')
parser.add_argument('--budget', type=int, default=10,
                    help='lifting budget per tail class')


def lift_tail(train_targets, unlabeled_targets, num_classes, threshold):
    train_perclass_count = [len(np.where(train_targets == i)[0]) for i in range(num_classes)]
    train_perclass_count = np.array(train_perclass_count)
    lifted_tail_samples = np.array([], dtype=np.int64)
    num_tail_classes = 0
    tail_classes = []
    for i in range(num_classes):
        if train_perclass_count[i] < threshold:
            num_tail_classes += 1
            tail_classes.append(i)
            to_lift_num = threshold - train_perclass_count[i]
            lifted_tail_samples = np.concatenate((lifted_tail_samples, np.where(unlabeled_targets == i)[0][:to_lift_num]))
    return lifted_tail_samples, num_tail_classes, tail_classes


def lift_uniform(train_targets, unlabeled_targets, num_classes, threshold, budget):
    train_perclass_count = [len(np.where(train_targets == i)[0]) for i in range(num_classes)]
    train_perclass_count = np.array(train_perclass_count)
    lifted_tail_samples = np.array([], dtype=np.int64)
    num_tail_classes = 0
    tail_classes = []
    for i in range(num_classes):
        if train_perclass_count[i] < threshold:
            num_tail_classes += 1
            tail_classes.append(i)
            lifted_tail_samples = np.concatenate((lifted_tail_samples, np.random.choice(np.where(unlabeled_targets == i)[0], budget, replace=False)))
    return lifted_tail_samples, num_tail_classes, tail_classes

def main():
    args = parser.parse_args()
    print(args)

    index_dir = Path("indexes") / f"{args.dataset}" / f"{args.index_name}"
    assert index_dir.exists(), "index dir not found"
    train_index_file = index_dir / "train.csv"
    val_index_file = index_dir / "val.csv"
    unlabeled_index_file = index_dir / "unlabeled.csv"
    import pdb; pdb.set_trace()

    new_index_dir = Path("indexes") / f"{args.dataset}" / f"{args.index_name}_lift_{args.lift}"
    if not new_index_dir.exists():
        new_index_dir.mkdir(parents=True)
        shutil.copy(val_index_file, new_index_dir)
    new_train_index_file = new_index_dir / "train.csv"
    new_unlabeled_index_file = new_index_dir / "unlabeled.csv"


    if not new_train_index_file.exists() \
            or not new_unlabeled_index_file.exists():
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
        
        train_pd = pd.read_csv(train_index_file)
        unlabeled_pd = pd.read_csv(unlabeled_index_file)

        train_indexes = np.array(train_pd['Index'].tolist())
        train_paths = np.array(train_pd['Path'].tolist())
        train_targets = np.array(train_pd['Target'].tolist())
        unlabeled_indexes = np.array(unlabeled_pd['Index'].tolist())
        unlabeled_paths = np.array(unlabeled_pd['Path'].tolist())
        unlabeled_targets = np.array(unlabeled_pd['Target'].tolist())
        
        num_classes = max(train_targets) + 1

        if args.lift in ['random', 'tail']:
            lifted_tail_samples, num_tail_classes, tail_classes = lift_tail(
                train_targets, unlabeled_targets, num_classes, args.threshold)
            if args.lift == 'random':
                lifted_tail_samples = np.random.choice(
                    np.arange(len(unlabeled_indexes)), size=len(lifted_tail_samples), replace=False
                )
        elif args.lift in ['uniform', 'tail_uniform']:
            lifted_tail_samples, num_tail_classes, tail_classes = lift_uniform(
                train_targets, unlabeled_targets, num_classes, args.threshold, args.budget)
            if args.lift == 'uniform':
                lifted_tail_samples = np.random.choice(
                    np.arange(len(unlabeled_indexes)), size=len(lifted_tail_samples), replace=False
                )

        print("lifted_tail_samples length:", len(lifted_tail_samples))
        new_train_indexes = np.concatenate((train_indexes, unlabeled_indexes[lifted_tail_samples]))
        new_train_paths = np.concatenate((train_paths, unlabeled_paths[lifted_tail_samples]))
        new_train_targets = np.concatenate((train_targets, unlabeled_targets[lifted_tail_samples]))
        
        new_unlabeled_indexes = np.delete(unlabeled_indexes, lifted_tail_samples)
        new_unlabeled_paths = np.delete(unlabeled_paths, lifted_tail_samples)
        new_unlabeled_targets = np.delete(unlabeled_targets, lifted_tail_samples)

        new_train_pd = pd.DataFrame(
            {"Index": list(new_train_indexes),
             "Path": list(new_train_paths),
             "Target": list(new_train_targets)}
        )
        new_unlabeled_pd = pd.DataFrame(
            {"Index": list(new_unlabeled_indexes),
             "Path": list(new_unlabeled_paths),
             "Target": list(new_unlabeled_targets)}
        )
        new_train_pd.to_csv(new_train_index_file, index=False)
        new_unlabeled_pd.to_csv(new_unlabeled_index_file, index=False)

        print(f"Saved at {new_train_index_file}")
        print(f"Saved at {new_unlabeled_index_file}")
    else:
        print(f"Index files already exists.")

if __name__ == '__main__':
    main()