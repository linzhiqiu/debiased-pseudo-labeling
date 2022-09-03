import numpy as np
import os
import pandas as pd


def _get_perclass_count(train_index_file, val_index_file, test_index_file, unlabeled_index_file, verbose=True):
    train_targets = np.array(pd.read_csv(train_index_file)['Target'].tolist())
    val_targets = np.array(pd.read_csv(val_index_file)['Target'].tolist())
    test_targets = np.array(pd.read_csv(test_index_file)['Target'].tolist())
    unlabeled_targets = np.array(pd.read_csv(unlabeled_index_file)['Target'].tolist())

    num_classes = max(test_targets)
    all_targets = np.concatenate((train_targets, val_targets, unlabeled_targets))
    all_perclass_count = [len(np.where(all_targets == i)[0]) for i in range(num_classes)]

    sorted_classes = [item[0] for item in sorted(
        enumerate(all_perclass_count), key=lambda x: x[1], reverse=True)]

    train_perclass_count = [len(np.where(train_targets == i)[0]) for i in range(num_classes)]
    val_perclass_count = [len(np.where(val_targets == i)[0]) for i in range(num_classes)]
    test_perclass_count = [len(np.where(test_targets == i)[0]) for i in range(num_classes)]
    unlabeled_perclass_count = [len(np.where(unlabeled_targets == i)[0]) for i in range(num_classes)]

    train_perclass_count = [train_perclass_count[idx] for idx in sorted_classes]
    val_perclass_count = [val_perclass_count[idx] for idx in sorted_classes]
    test_perclass_count = [test_perclass_count[idx] for idx in sorted_classes]
    unlabeled_perclass_count = [unlabeled_perclass_count[idx] for idx in sorted_classes]
    if verbose:
        print(f"Number of classes: {num_classes}")
        print("Unlabeled " + str(len(unlabeled_targets)))
        print("Train " + str(len(train_targets)))
        print("Val " + str(len(val_targets)))
        print("Test " + str(len(test_targets)))

        print("Unlabeled (min)" + str(min(unlabeled_perclass_count)))
        print("Train (min)" + str(min(train_perclass_count)))
        print("Val (min)" + str(min(val_perclass_count)))
        print("Unlabeled (max)" + str(max(unlabeled_perclass_count)))
        print("Train (max)" + str(max(train_perclass_count)))
        print("Val (max)" + str(max(val_perclass_count)))
    return {
        'train': train_perclass_count,
        'val': val_perclass_count,
        'test': test_perclass_count,
        'unlabeled': unlabeled_perclass_count,
        'sorted_classes': sorted_classes,
    }

def get_perclass_count(dataset='imagenet127',
                       index_dir='indexes/',
                       index_name='default',
                       verbose=True):
    index_dir_path = f"{index_dir}/{dataset}"
    train_index_file = os.path.join(index_dir_path, index_name, "train.csv")
    val_index_file = os.path.join(index_dir_path, index_name, "val.csv")
    unlabeled_index_file = os.path.join(index_dir_path, index_name, "unlabeled.csv")
    test_index_file = os.path.join(index_dir_path, "test.csv")

    return _get_perclass_count(
        train_index_file, val_index_file, test_index_file, unlabeled_index_file,
        verbose=verbose)

def get_tail_info(dataset='imagenet127',
                  index_dir='indexes/',
                  index_name='default',
                  threshold=20):
    count_dict = get_perclass_count(dataset, index_dir, index_name, verbose=False)
    train_count = np.array(count_dict['train'])
    tail_class_count = np.sum(train_count < threshold)
    tail_sample_count = np.sum((threshold-train_count)[train_count < threshold])
    print(f"Classes < {threshold} images: " + str(tail_class_count))
    print(f"Samples to lift = {tail_sample_count} images")
    return tail_class_count, tail_sample_count
