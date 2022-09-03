import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, default_loader
from . import transforms as data_transforms
import os
from pathlib import Path
import pandas as pd


IMAGENET127_CLASSES_PATH = os.path.join(
    'imagenet127', 'synset_words_up_down_127.txt')
IMAGENET127_TRAIN_PATH = os.path.join(
    'imagenet127', 'train_up_down_127.txt')
IMAGENET127_VAL_PATH = os.path.join(
    'imagenet127', 'val_up_down_127.txt')


def readlines(file_path):
    # read txt file and return a list of strings
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def read_paths_and_labels(file_path, root):
    # read txt file and return a list of tuples as
    # (image_path:absolute path, image_label:int)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(' ') for line in lines]
        lines = [(os.path.join(root, line[0]), int(line[1]))
                 for line in lines]
    return lines


def get_transforms_from_type(train_type, val_type, weak_type, strong_type, multiviews):
    transform_x = data_transforms.get_transforms(train_type)
    weak_transform = data_transforms.get_transforms(weak_type)
    strong_transform = data_transforms.get_transforms(strong_type)
    if multiviews:
        weak_transform2 = data_transforms.get_transforms(weak_type)
        strong_transform2 = data_transforms.get_transforms(strong_type)
        transform_u = data_transforms.FourCropsTransform(weak_transform, weak_transform2, strong_transform, strong_transform2)
    else:
        transform_u = data_transforms.TwoCropsTransform(weak_transform, strong_transform)
    transform_val = data_transforms.get_transforms(val_type)
    return transform_x, transform_u, transform_val


class ImageDataset(Dataset):
    def __init__(self, samples, transform, class_names=None):
        self.samples = samples # list of (path, label)
        self.transform = transform
        self.class_names = class_names
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        path, label = self.samples[index]
        sample = default_loader(path)
        sample = self.transform(sample)
        return sample, label


def get_imagenet127_datasets(
        root, train_type='DefaultTrain', val_type='DefaultVal',
        IMAGENET127_VAL_PATH=IMAGENET127_VAL_PATH,
        IMAGENET127_TRAIN_PATH=IMAGENET127_TRAIN_PATH,
        IMAGENET127_CLASSES_PATH=IMAGENET127_CLASSES_PATH):
    # return all / test sets of imagenet127 from root
    traindir = os.path.join(root, 'train')
    testdir = os.path.join(root, 'val')
    assert os.path.exists(traindir), '{} does not exist'.format(traindir)
    assert os.path.exists(testdir), '{} does not exist'.format(testdir)
    transform_train = data_transforms.get_transforms(train_type)
    transform_val = data_transforms.get_transforms(val_type)

    classes = readlines(IMAGENET127_CLASSES_PATH)
    train_samples = read_paths_and_labels(
        IMAGENET127_TRAIN_PATH, root=Path(root) / 'train')
    test_samples = read_paths_and_labels(
        IMAGENET127_VAL_PATH, root=Path(root) / 'val')

    train_dataset = ImageDataset(
        train_samples, transform=transform_train, class_names=classes)
    test_dataset = ImageDataset(
        test_samples, transform=transform_val, class_names=classes)

    return train_dataset, test_dataset


class SortedImageFolder(ImageFolder):
    # Same as ImageFolder, but sorts the classes by their integer value
    def find_classes(self, directory):
        classes = sorted(int(entry.name) for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder with integer class name in {directory}.")

        class_to_idx = {str(cls_name): i for i, cls_name in enumerate(classes)}
        classes = [str(name) for name in classes]
        return classes, class_to_idx


def get_semi_inat_datasets(root, train_type='DefaultTrain', val_type='DefaultVal'):
    # return all / test sets of semi_inat from root
    traindir = os.path.join(root, 'l_train_and_u_train_in')
    testdir = os.path.join(root, 'val')
    assert os.path.exists(traindir)
    assert os.path.exists(testdir)
    transform_train = data_transforms.get_transforms(train_type)
    transform_val = data_transforms.get_transforms(val_type)

    train_dataset = SortedImageFolder(
        traindir, transform=transform_train)
    test_dataset = SortedImageFolder(
        testdir, transform=transform_val)

    return train_dataset, test_dataset


def x_u_v_split(labels, train_ratio, val_ratio, num_classes, least_num_per_class=3):
    # x is labeled trainset, u is unlabeled trainset, v is val set
    assert 0 < val_ratio + train_ratio < 1
    labels = np.array(labels)
    train_index = []
    unlabeled_index = []
    val_index = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        train_per_class = max(least_num_per_class, round(train_ratio * len(idx)))
        val_per_class = max(least_num_per_class, round(val_ratio * len(idx)))
        assert len(idx) > train_per_class + val_per_class
        np.random.shuffle(idx)
        train_index.extend(idx[:train_per_class])
        val_index.extend(idx[train_per_class:train_per_class + val_per_class])
        unlabeled_index.extend(idx[train_per_class + val_per_class:])
    print('train_index ({}): {}, ..., {}'.format(len(train_index), train_index[:5], train_index[-5:]))
    print('unlabeled_index ({}): {}, ..., {}'.format(len(unlabeled_index), unlabeled_index[:5], unlabeled_index[-5:]))
    print('val_index ({}): {}, ..., {}'.format(len(val_index), val_index[:5], val_index[-5:]))
    return train_index, unlabeled_index, val_index


def get_semi_inat_ssl_datasets(
        root,
        train_index_file,
        val_index_file,
        unlabeled_index_file,
        test_index_file,
        train_type='DefaultTrain',
        val_type='DefaultVal',
        weak_type='DefaultTrain',
        strong_type='RandAugment',
        multiviews=False):
    traindir = os.path.join(root, 'l_train_and_u_train_in')
    testdir = os.path.join(root, 'val')
    assert os.path.exists(traindir)
    assert os.path.exists(testdir)

    transform_x, transform_u, transform_val = get_transforms_from_type(
        train_type, val_type, weak_type, strong_type, multiviews)

    train_pd = pd.read_csv(train_index_file)
    val_pd = pd.read_csv(val_index_file)
    unlabeled_pd = pd.read_csv(unlabeled_index_file)
    test_pd = pd.read_csv(test_index_file)

    train_indexes = train_pd['Index'].tolist()
    train_paths = train_pd['Path'].tolist()
    train_targets = train_pd['Target'].tolist()
    val_indexes = val_pd['Index'].tolist()
    val_paths = val_pd['Path'].tolist()
    val_targets = val_pd['Target'].tolist()
    unlabeled_indexes = unlabeled_pd['Index'].tolist()
    unlabeled_paths = unlabeled_pd['Path'].tolist()
    unlabeled_targets = unlabeled_pd['Target'].tolist()
    test_indexes = test_pd['Index'].tolist()
    test_paths = test_pd['Path'].tolist()
    test_targets = test_pd['Target'].tolist()

    train_dataset = SortedImageFolderWithIndex(
        traindir, train_indexes, transform=transform_x)
    for idx, s in enumerate(train_dataset.imgs):
        assert s[0] == os.path.join(str(root), train_paths[idx])
        assert s[1] == train_targets[idx]
    
    val_dataset = SortedImageFolderWithIndex(
        traindir, val_indexes, transform=transform_val)
    for idx, s in enumerate(val_dataset.imgs):
        assert s[0] == os.path.join(str(root), val_paths[idx])
        assert s[1] == val_targets[idx]

    unlabeled_dataset = SortedImageFolderWithIndex(
        traindir, unlabeled_indexes, transform=transform_u)
    for idx, s in enumerate(unlabeled_dataset.imgs):
        assert s[0] == os.path.join(str(root), unlabeled_paths[idx])
        assert s[1] == unlabeled_targets[idx]

    test_dataset = SortedImageFolderWithIndex(
        testdir, test_indexes, transform=transform_val)
    for idx, s in enumerate(test_dataset.imgs):
        assert s[0] == os.path.join(str(root), test_paths[idx])
        assert s[1] == test_targets[idx]

    return train_dataset, unlabeled_dataset, val_dataset, test_dataset


def get_imagenet127_ssl_datasets(
        root,
        train_index_file,
        val_index_file,
        unlabeled_index_file,
        test_index_file,
        train_type='DefaultTrain',
        val_type='DefaultVal',
        weak_type='DefaultTrain',
        strong_type='RandAugment',
        multiviews=False,
        IMAGENET127_VAL_PATH=IMAGENET127_VAL_PATH,
        IMAGENET127_TRAIN_PATH=IMAGENET127_TRAIN_PATH,
        IMAGENET127_CLASSES_PATH=IMAGENET127_CLASSES_PATH):
    traindir = os.path.join(root, 'train')
    testdir = os.path.join(root, 'val')
    assert os.path.exists(traindir)
    assert os.path.exists(testdir)

    transform_x, transform_u, transform_val = get_transforms_from_type(
        train_type, val_type, weak_type, strong_type, multiviews)

    classes = readlines(IMAGENET127_CLASSES_PATH)
    train_samples = read_paths_and_labels(
        IMAGENET127_TRAIN_PATH, root=Path(root) / 'train')
    test_samples = read_paths_and_labels(
        IMAGENET127_VAL_PATH, root=Path(root) / 'val')

    train_pd = pd.read_csv(train_index_file)
    val_pd = pd.read_csv(val_index_file)
    unlabeled_pd = pd.read_csv(unlabeled_index_file)
    test_pd = pd.read_csv(test_index_file)

    train_indexes = train_pd['Index'].tolist()
    train_paths = train_pd['Path'].tolist()
    train_targets = train_pd['Target'].tolist()
    val_indexes = val_pd['Index'].tolist()
    val_paths = val_pd['Path'].tolist()
    val_targets = val_pd['Target'].tolist()
    unlabeled_indexes = unlabeled_pd['Index'].tolist()
    unlabeled_paths = unlabeled_pd['Path'].tolist()
    unlabeled_targets = unlabeled_pd['Target'].tolist()
    test_indexes = test_pd['Index'].tolist()
    test_paths = test_pd['Path'].tolist()
    test_targets = test_pd['Target'].tolist()

    train_dataset = ImageDatasetWithIndex(
        train_samples, transform_x, indexs=train_indexes, class_names=classes)
    for idx, s in enumerate(train_dataset.samples):
        assert s[0] == os.path.join(str(root), train_paths[idx])
        assert s[1] == train_targets[idx]
    
    val_dataset = ImageDatasetWithIndex(
        train_samples, transform_val, indexs=val_indexes, class_names=classes)
    for idx, s in enumerate(val_dataset.samples):
        assert s[0] == os.path.join(str(root), val_paths[idx])
        assert s[1] == val_targets[idx]

    unlabeled_dataset = ImageDatasetWithIndex(
        train_samples, transform_u, indexs=unlabeled_indexes, class_names=classes)
    for idx, s in enumerate(unlabeled_dataset.samples):
        assert s[0] == os.path.join(str(root), unlabeled_paths[idx])
        assert s[1] == unlabeled_targets[idx]

    test_dataset = ImageDatasetWithIndex(
        test_samples, transform_val, indexs=test_indexes, class_names=classes)
    for idx, s in enumerate(test_dataset.samples):
        assert s[0] == os.path.join(str(root), test_paths[idx])
        assert s[1] == test_targets[idx]

    return train_dataset, unlabeled_dataset, val_dataset, test_dataset


class SortedImageFolderWithIndex(SortedImageFolder):

    def __init__(self, root, indexs=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]
            self.targets = [self.targets[i] for i in indexs]
            self.imgs = self.samples

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageDatasetWithIndex(ImageDataset):

    def __init__(self, samples, transform, indexs=None, class_names=None):
        super().__init__(samples, transform, class_names)
        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]