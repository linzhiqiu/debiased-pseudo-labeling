# python eval.py --threshold 18 --budget 6 --dataset inat --algo supervised --index_name default --model_name wd_1e_5 --lift default_lift_supervised_wd_1e_5_pl_thre_18_budget_6
# python eval.py --threshold 18 --budget 6 --dataset inat --algo supervised --index_name default --model_name wd_1e_5 --lift default_lift_supervised_wd_1e_5_pl_entropy_thre_18_budget_6
# python eval.py --threshold 18 --budget 6 --dataset inat --algo supervised --index_name default --model_name wd_1e_5 --lift default
# python eval.py --threshold 18 --budget 6 --dataset inat --algo supervised --index_name default --model_name wd_1e_5 --lift default_lift_supervised_wd_1e_5_pl_entropy_all_thre_18_budget_6
# python eval.py --threshold 18 --budget 6 --dataset inat --algo supervised --index_name default --model_name wd_1e_5 --lift default_lift_supervised_wd_1e_5_pl_entropy_all_tp_thre_18_budget_6
# python eval.py --threshold 18 --budget 6 --dataset inat --algo supervised --index_name default --model_name wd_1e_5 --lift default_lift_supervised_wd_1e_5_pl_entropy_all_fp_thre_18_budget_6
# python eval.py --threshold 18 --budget 6 --dataset inat --algo fixmatch --index_name default --model_name wd_1e_4 --lift default_lift_fixmatch_wd_1e_4_pl_thre_18_budget_6
# python eval.py --threshold 18 --budget 6 --dataset inat --algo fixmatch --index_name default --model_name wd_1e_4 --lift default_lift_fixmatch_wd_1e_4_pl_entropy_thre_18_budget_6
# python eval.py --threshold 18 --budget 6 --dataset inat --algo fixmatch --index_name default --model_name wd_1e_4 --lift default
# python eval.py --threshold 18 --budget 6 --dataset inat --algo debiased --index_name default --model_name wd_1e_4 --lift default_lift_debiased_wd_1e_4_pl_thre_18_budget_6
# python eval.py --threshold 18 --budget 6 --dataset inat --algo debiased --index_name default --model_name wd_1e_4 --lift default_lift_debiased_wd_1e_4_pl_entropy_thre_18_budget_6
# python eval.py --threshold 18 --budget 6 --dataset inat --algo debiased --index_name default --model_name wd_1e_4 --lift default

# python eval.py --threshold 25 --budget 12 --dataset imagenet127 --algo supervised --index_name default --model_name wd_1e_4 --lift default_lift_supervised_wd_1e_5_pl_thre_25_budget_12
# python eval.py --threshold 25 --budget 12 --dataset imagenet127 --algo supervised --index_name default --model_name wd_1e_4 --lift default_lift_supervised_wd_1e_5_pl_entropy_thre_25_budget_12
import argparse
import random
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader, RandomSampler
from pathlib import Path
from engine import validate_perclass
from models import get_fixmatch_model
import backbone as backbone_models
from utils import utils
backbone_model_names = sorted(name for name in backbone_models.__dict__
                              if name.islower() and not name.startswith("__")
                              and callable(backbone_models.__dict__[name]))
from utils import get_norm
from data.datasets import get_inat_ssl_datasets, get_imagenet127_ssl_datasets
from results_second_round import RESULT_DICT

parser = argparse.ArgumentParser(description='Eval for second round')
parser.add_argument('--dataset', default='imagenet127', type=str,
                    choices=['inat', 'imagenet127'],
                    help='dataset to use, choices: [inat, imagenet127]')
parser.add_argument('--index_name', default='default', type=str,
                    help='name of index dir (the directory under indexes/)')
parser.add_argument('--lift', default=None, type=str,
                    help='lifting method')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for lifting tail. ')
parser.add_argument('--threshold', type=int, default=20,
                    help='to lift tail such that all classes are above this threshold')
parser.add_argument('--budget', type=int, default=10,
                    help='lifting budget per tail class')
parser.add_argument('--algo', type=str, default="supervised", choices=["supervised", "fixmatch", "debiased"],
                    help='algorithm name in results.py')
parser.add_argument('--model_name', type=str, default=None,
                    help='model name in results.py')
# below are default (no need to modify)
parser.add_argument('--root', metavar='DIR', default='/ssd0/fercus/',
                    help='path to dataset')
parser.add_argument('--arch', metavar='ARCH', default='FixMatch',
                    help='model architecture (default is FixMatch)')
parser.add_argument('--backbone', default='resnet50_encoder',
                    choices=backbone_model_names,
                    help='model architecture: ' +
                        ' | '.join(backbone_model_names) +
                        ' (default: resnet50_encoder)')
parser.add_argument('--norm', default='BN', type=str,
                    help='the normalization for backbone (default: BN)')
parser.add_argument('--eman', action='store_true', default=True,
                    help='use EMAN')
parser.add_argument('--ema-m', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--multiviews', action='store_true', default=False,
                    help='augmentation invariant mapping')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')


def get_tailinfo(train_targets, num_classes, threshold):
    train_perclass_count = [len(np.where(train_targets == i)[0]) for i in range(num_classes)]
    train_perclass_count = np.array(train_perclass_count)
    head_classes = []
    tail_classes = []
    for i in range(num_classes):
        if train_perclass_count[i] < threshold:
            tail_classes.append(i)
        else:
            head_classes.append(i)
    return head_classes, tail_classes

# def lifting(train_targets, unlabeled_targets, unlabeled_dataset, lift_method, num_classes, threshold, budget):
def calc_accuracy(perclass_acc, head_classes, tail_classes):
    mean_acc = np.mean(perclass_acc)
    head_acc = np.mean([perclass_acc[i] for i in head_classes])
    tail_acc = np.mean([perclass_acc[i] for i in tail_classes])
    return head_acc, tail_acc, mean_acc


def get_model(pretrained, arch, backbone, norm, num_classes, ema_m=0.999, eman=True):
    # create model
    print("=> creating model '{}' with backbone '{}'".format(arch, backbone))
    model_func = get_fixmatch_model(arch)
    norm = get_norm(norm)
    model = model_func(
        backbone_models.__dict__[backbone],
        eman=eman,
        momentum=ema_m,
        norm=norm,
        num_classes=num_classes,
    )
    # print(model)
    print("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))

    assert os.path.isfile(pretrained)
    print("=> loading pretrained model from '{}'".format(pretrained))
    checkpoint = torch.load(pretrained, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        new_key = k.replace("module.", "")
        state_dict[new_key] = state_dict[k]
        del state_dict[k]
    model_num_cls = state_dict['ema.fc.weight'].shape[0]
    assert model_num_cls == num_classes
    model.load_state_dict(state_dict)
    print("=> loaded pre-trained model '{}' (epoch {})".format(pretrained, checkpoint['epoch']))
    return model.cuda().eval()

def main():
    args = parser.parse_args()
    print(args)

    index_dir = Path("indexes") / f"{args.dataset}"
    assert index_dir.exists(), "index dir not found"
    train_index_file = index_dir / f"{args.index_name}" / "train.csv"
    val_index_file = index_dir / f"{args.index_name}" / "val.csv"
    unlabeled_index_file = index_dir / f"{args.index_name}" / "unlabeled.csv"
    test_index_file = index_dir / 'test.csv'

    if args.dataset == 'inat':
        dataset_func = get_inat_ssl_datasets
        dataset_name = "inat"
    elif args.dataset == 'imagenet127':
        dataset_func = get_imagenet127_ssl_datasets
        dataset_name = "imagenet"
    else:
        raise ValueError("Invalid dataset")
    
    train_dataset, unlabeled_dataset, val_dataset, test_dataset = dataset_func(
        Path(args.root) / dataset_name,
        train_index_file,
        val_index_file,
        unlabeled_index_file,
        test_index_file,
        train_type='DefaultVal',
        val_type='DefaultVal',
        weak_type='DefaultVal',
        strong_type='DefaultVal',
        multiviews=args.multiviews,
    )
    print("train_dataset:\n{}".format(len(train_dataset)))
    print("unlabeled_dataset:\n{}".format(len(unlabeled_dataset)))
    print("val_dataset:\n{}".format(len(val_dataset)))
    print("test_dataset:\n{}".format(len(test_dataset)))

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
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
    args.cls = num_classes # for engine.py
    pretrained_path = RESULT_DICT[args.dataset]['default'][args.algo][args.index_name][args.model_name][args.lift]
    pretrained_path = Path(pretrained_path) / "best_val_model.pth.tar"
    pretrained_model = get_model(pretrained_path, args.arch, args.backbone, args.norm, num_classes, ema_m=args.ema_m, eman=args.eman)
    head_classes, tail_classes = get_tailinfo(train_targets, num_classes, args.threshold)
    val_result = validate_perclass(
        val_loader, pretrained_model, criterion, args, prefix='Val')
    test_result = validate_perclass(
        test_loader, pretrained_model, criterion, args)
    val_head_acc, val_tail_acc, val_acc = calc_accuracy(val_result['all'], head_classes, tail_classes)
    test_head_acc, test_tail_acc, test_acc = calc_accuracy(test_result['all'], head_classes, tail_classes)
    print("Val set: Head: {:.3%}, Tail: {:.3%}, Mean: {:.3%}".format(val_head_acc, val_tail_acc, val_acc))
    print("Test set: Head: {:.3%}, Tail: {:.3%}, Mean: {:.3%}".format(test_head_acc, test_tail_acc, test_acc))
       
if __name__ == '__main__':
    main()