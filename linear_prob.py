import argparse
import builtins
import math
import os
import shutil
import time
from copy import deepcopy
import warnings
import numpy as np
import pandas as pd
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from data.datasets import get_inat_ssl_datasets, get_imagenet127_ssl_datasets
import backbone as backbone_models
from models import get_fixmatch_model
from utils import utils, lr_schedule, get_norm, dist_utils
from torch.utils.tensorboard import SummaryWriter

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

LRS = [3., 0.3, 0.03, 0.003, 0.0003, 0.00003]
WDS = [0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

backbone_model_names = sorted(name for name in backbone_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbone_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Supervised Training')
parser.add_argument('root', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default='imagenet127', type=str,
                    choices=['inat', 'imagenet127'],
                    help='dataset to use, choices: [inat, imagenet127]')
parser.add_argument('--index_name', default='default', type=str,
                    help='name of index dir (the directory under indexes/)')
parser.add_argument('--arch', metavar='ARCH', default='FixMatch',
                    help='model architecture')
parser.add_argument('--backbone', default='resnet50_encoder',
                    choices=backbone_model_names,
                    help='model architecture: ' +
                        ' | '.join(backbone_model_names) +
                        ' (default: resnet50_encoder)')
parser.add_argument('--cls', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epoch', default=0, type=int, metavar='N',
                    help='number of epochs for learning warmup')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[1000, 2000], nargs='*', type=int, # this and cos are exclusive
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule (has priority over step lr schedule)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='use nesterov momentum')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--self-pretrained', default='', type=str, metavar='PATH',
                    help='path to MoCo pretrained model (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--eman', action='store_true', default=True,
                    help='use EMAN')
parser.add_argument('--ema-m', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--train-type', default='RandAugment', type=str,
                    help='the type for train augmentation')
parser.add_argument('--norm', default='BN', type=str,
                    help='the normalization for backbone (default: BN)')
# online_net.backbone for BYOL
parser.add_argument('--model-prefix', default='encoder_q', type=str,
                    help='the model prefix of self-supervised pretrained state_dict')
# additional hyperparameters
parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--output', default='checkpoints/', type=str,
                    help='the path to checkpoints')


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index]
    
    def __len__(self):
        return self.input_tensor.size(0)


def validate_perclass(val_loader, model, criterion, args, prefix='Test'):
    # losses = utils.AverageMeter('Loss', ':.4e')
    # acc = utils.AverageMeter('Acc@1', ':6.2f')
    # progress = utils.ProgressMeter(
    #     len(val_loader),
    #     [losses, acc],
    #     prefix=f'{prefix}: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (features, target) in enumerate(val_loader):
            features = features.cuda()
            target = target.cuda()

            # compute output
            output = model(features)

            # measure accuracy and record loss
            probs = torch.softmax(output, dim=-1)
            _, preds = probs.topk(1, 1, True, True)
            preds = preds.view(-1)
            if i == 0:
                preds_list = preds.cpu()
                target_list = target.cpu()
            else:
                preds_list = torch.cat((preds_list, preds.cpu()), dim=0)
                target_list = torch.cat((target_list, target.cpu()), dim=0)

            # acc1, _ = utils.accuracy(output, target, topk=(1, 5))
            # acc.update(acc1[0], features.size(0))


        perclass_acc = [0 for _ in range(args.cls)]
        for c in range(args.cls):
            perclass_acc[c] = ((preds_list == target_list) * (target_list == c)).sum().float() / max((target_list == c).sum(), 1)
        acc = (preds_list == target_list).sum().float() / len(target_list)
        mean_acc = sum(perclass_acc) / args.cls
        min_perclass_acc = min(perclass_acc)
        max_perclass_acc = max(perclass_acc)
        # print(' * Acc@1 {acc:.3%} mAcc {mean_acc:.3%} minAcc {min_perclass_acc:.3%} maxAcc {max_perclass_acc:.3%} Loss {loss.avg:.4f}'
            #   .format(acc=acc, mean_acc=mean_acc, min_perclass_acc=min_perclass_acc, max_perclass_acc=max_perclass_acc, loss=losses))

    return {
        'top1' : acc, 
        'mean' : mean_acc,
        'all' : perclass_acc
    }


def main():
    args = parser.parse_args()
    assert args.warmup_epoch < args.schedule[0]
    print(args)

    if args.seed is not None:
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # ngpus_per_node = torch.cuda.device_count()
    # if args.multiprocessing_distributed:
    #     # Since we have ngpus_per_node processes per node, the total world_size
    #     # needs to be adjusted accordingly
    #     args.world_size = ngpus_per_node * args.world_size
    #     # Use torch.multiprocessing.spawn to launch distributed processes: the
    #     # main_worker process function
    #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # else:
        # Simply call main_worker function
    main_worker(args)


def get_n_classes(dataset):
    # loop through dataset to get the number of samples per class
    
    n_classes = {}
    for _, label in dataset:
        if not label in n_classes:
            n_classes[label] = 1
        else:
            n_classes[label] += 1
    return [n_classes[label] for label in range(max(n_classes.keys())+1)]

def get_effective_weights(n_classes, beta=0.999):
    effective_num = 1.0 - torch.pow(beta, torch.FloatTensor(n_classes))
    weights = (1.0 - beta) / effective_num
    return weights.cuda()


def extract(model, loader):
    features = None
    labels = None
    
    with torch.no_grad():
        for images_i, labels_i in tqdm(loader):
            features_i = model(images_i.cuda()).cpu()
            if features is None:
                features = features_i
                labels = labels_i
            else:
                features = torch.cat((features, features_i), dim=0)
                labels = torch.cat((labels, labels_i), dim=0)
    return features, labels

def main_worker(args):
    # get head and tail classes first
    index_dir = Path("indexes") / f"{args.dataset}"
    assert index_dir.exists(), "index dir not found"
    default_index_file = index_dir / f"default" / "train.csv"
    train_pd = pd.read_csv(default_index_file)
    train_targets = np.array(train_pd['Target'].tolist())
    num_classes = max(train_targets) + 1
    assert args.cls == num_classes, "number of classes not match"
    from lifting import get_tailinfo, calc_accuracy
    if args.dataset == "inat":
        threshold = 18
    elif args.dataset == "imagenet127":
        threshold = 25
    head_classes, tail_classes = get_tailinfo(train_targets, num_classes, threshold)
    
    # create model
    print("=> creating model '{}' with backbone '{}'".format(args.arch, args.backbone))
    model_func = get_fixmatch_model(args.arch)
    norm = get_norm(args.norm)
    model = model_func(
        backbone_models.__dict__[args.backbone],
        eman=args.eman,
        momentum=args.ema_m,
        norm=norm,
        num_classes=args.cls,
    )
    print(model)
    print("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))

    if args.self_pretrained:
        if os.path.isfile(args.self_pretrained):
            print("=> loading checkpoint '{}'".format(args.self_pretrained))
            checkpoint = torch.load(args.self_pretrained, map_location="cpu")

            # rename self pre-trained keys to model.main keys
            state_dict = checkpoint['state_dict']
            model_prefix = 'module.' + args.model_prefix
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith(model_prefix) and not k.startswith(model_prefix + '.fc'):
                    # replace prefix
                    new_key = k.replace(model_prefix, "main.backbone")
                    state_dict[new_key] = state_dict[k]
                    if model.ema is not None:
                        new_key = k.replace(model_prefix, "ema.backbone")
                        state_dict[new_key] = state_dict[k].clone()
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            if len(msg.missing_keys) > 0:
                print("missing keys:\n{}".format('\n'.join(msg.missing_keys)))
            if len(msg.unexpected_keys) > 0:
                print("unexpected keys:\n{}".format('\n'.join(msg.unexpected_keys)))
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.self_pretrained, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.self_pretrained))
    elif args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained model from '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                new_key = k.replace("module.", "")
                state_dict[new_key] = state_dict[k]
                del state_dict[k]
            model_num_cls = state_dict['ema.fc.weight'].shape[0]
            assert model_num_cls == args.cls
            model.load_state_dict(state_dict)
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))
        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained))

    model.cuda()
    # if args.amp_opt_level != "O0":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    # if args.distributed:
    #     # For multiprocessing distributed, DistributedDataParallel constructor
    #     # should always set the single device scope, otherwise,
    #     # DistributedDataParallel will use all available devices.
    #     if args.gpu is not None:
    #         torch.cuda.set_device(args.gpu)
    #         model.cuda(args.gpu)
    #         # When using a single GPU per process and per
    #         # DistributedDataParallel, we need to divide the batch size
    #         # ourselves based on the total number of GPUs we have
    #         args.batch_size = int(args.batch_size / ngpus_per_node)
    #         args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     else:
    #         model.cuda()
    #         # DistributedDataParallel will divide and allocate batch_size to all
    #         # available GPUs if device_ids are not set
    #         model = torch.nn.parallel.DistributedDataParallel(model)
    # elif args.gpu is not None:
    model = model.cuda().eval()
    model = model.main
    old_fc = deepcopy(model.fc)
    model.fc = torch.nn.Identity()
    # else:
    #     # DataParallel will divide and allocate batch_size to all available GPUs
    #     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #         model.features = torch.nn.DataParallel(model.features)
    #         model.cuda()
    #     else:
    #         model = torch.nn.DataParallel(model).cuda()

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%m-%d-%Y-%H:%M")
    print("date and time =", dt_string)	
    args.output = os.path.join(args.output, dt_string)

    cudnn.benchmark = True

    # Supervised Data loading code
    if args.dataset == 'inat':
        index_dir = Path('indexes') / 'inat'
        dataset_func = get_inat_ssl_datasets
    elif args.dataset == 'imagenet127':
        index_dir = Path('indexes') / 'imagenet127'
        dataset_func = get_imagenet127_ssl_datasets
    else:
        raise ValueError("Invalid dataset")
    
    train_index_file = index_dir / args.index_name / 'train.csv'
    val_index_file = index_dir / args.index_name / 'val.csv'
    unlabeled_index_file = index_dir / args.index_name / 'unlabeled.csv'
    test_index_file = index_dir / 'test.csv'

    train_dataset, _, val_dataset, test_dataset = dataset_func(
        args.root,
        train_index_file,
        val_index_file,
        unlabeled_index_file,
        test_index_file,
        train_type=args.train_type,
        val_type='DefaultVal',
        weak_type=args.train_type,
        strong_type=args.train_type,
        multiviews=False,
    )
    n_classes = get_n_classes(train_dataset)
    effective_weights = get_effective_weights(n_classes)
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss(weight=effective_weights).cuda()
    print("train_dataset:\n{}".format(len(train_dataset)))
    print("val_dataset:\n{}".format(len(val_dataset)))
    print("test_dataset:\n{}".format(len(test_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    train_features, train_labels = extract(model, train_loader)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0)
    val_features, val_labels = extract(model, val_loader)
    val_loader = DataLoader(
        TensorDataset(val_features, val_labels),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0)
    test_features, test_labels = extract(model, test_loader)
    test_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0)

    all_results = {}
    for lr in LRS:
        for weight_decay in WDS:
            optim = f"lr_{lr}_wd_{weight_decay}"
            # define optimizer
            
            fc = torch.nn.Linear(2048, len(n_classes), bias=True)
            optimizer = torch.optim.SGD(fc.parameters(), lr,
                                        momentum=args.momentum,
                                        weight_decay=weight_decay,
                                        nesterov=args.nesterov)

            best_val_epoch = 0
            best_test_epoch = 0
            best_val_acc = None
            best_test_acc = None
            all_results[optim] = {
                'val_top1': [], 
                'test_top1': [],
                'val_acc': [], # mean class accuracy on validation set
                'test_acc': [], # mean class accuracy on test set
                'val_acc_all': [], # mean perclass accuracy on validation set
                'test_acc_all': [], # mean perclass accuracy on test set
            }

            def update_result(
                    val_result, test_result):
                all_results[optim]['val_top1'].append(float(val_result['top1']))
                all_results[optim]['test_top1'].append(float(test_result['top1']))
                all_results[optim]['val_acc'].append(float(val_result['mean']))
                all_results[optim]['val_acc_all'].append(val_result['all'])
                all_results[optim]['test_acc'].append(float(test_result['mean']))
                all_results[optim]['test_acc_all'].append(test_result['all'])

            for epoch in tqdm(range(args.start_epoch, args.epochs)):
                if epoch >= args.warmup_epoch:
                    lr_schedule.adjust_learning_rate(optimizer, epoch, args)

                # train for one epoch
                train(
                    train_loader, fc,
                    optimizer, epoch, criterion, args)

                is_best_val = False
                is_best_test = False

                # evaluate on validation set
                val_result = validate_perclass(val_loader, fc, criterion, args, prefix='Val')
                # remember best acc@1 and save checkpoint
                is_best_val = best_val_acc is None or val_result['mean'] > best_val_acc
                if is_best_val:
                    best_val_acc = val_result['mean']
                    best_val_epoch = epoch

                # import pdb; pdb.set_trace()
                
                # evaluate on validation set
                test_result = validate_perclass(test_loader, fc, criterion, args)
                # remember best acc@1 and save checkpoint
                is_best_test = best_test_acc is None or test_result['mean'] > best_test_acc
                if is_best_val:
                    best_test_acc_at_best_val = {
                        'mean': test_result['mean'],
                        'all': test_result['all'],
                    }
                if is_best_test:
                    best_test_acc = test_result['mean']
                    best_test_epoch = epoch

                # torch.distributed.barrier()
                update_result(
                    val_result, test_result
                )
            
            all_results[optim]['final'] = {
                'val_acc': best_val_acc,
                'test_acc': best_test_acc_at_best_val['mean'],
                'test_acc_all': best_test_acc_at_best_val['all'],
            }
                # if args.rank in [-1, 0]:
                #     writer.add_scalar('test/1.test_acc', test_result['mean'], epoch)
                #     writer.add_scalar('test/2.val_acc', val_result['mean'], epoch)

                # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                #         and args.rank % ngpus_per_node == 0):
                    # torch.save(
                    #     results, os.path.join(args.output, 'results.pth')
                    # )
                    
                    # if is_best_val:
                    #     torch.save({
                    #         'epoch': epoch + 1,
                    #         'state_dict': model.state_dict(),
                    #         'best_val_acc': best_val_acc,
                    #         'best_val_epoch': best_val_epoch,
                    #         'best_test_acc': best_test_acc,
                    #         'best_test_epoch': best_test_epoch,
                    #         'optimizer' : optimizer.state_dict(),
                    #     }, os.path.join(args.output, 'best_val_model.pth.tar'))
                    # if is_best_test:
                    #     torch.save({
                    #         'epoch': epoch + 1,
                    #         'state_dict': model.state_dict(),
                    #         'best_val_acc': best_val_acc,
                    #         'best_val_epoch': best_val_epoch,
                    #         'best_test_acc': best_test_acc,
                    #         'best_test_epoch': best_test_epoch,
                    #         'optimizer' : optimizer.state_dict(),
                    #     }, os.path.join(args.output, 'best_test_model.pth.tar'))

            # print('Best Val Acc {0} @ epoch {1}'.format(
            #     best_val_acc, best_val_epoch))
            # print('Best Test Acc {0} @ Best val epoch {1}'.format(
            #     all_results[optim]['test_acc'][best_val_epoch], best_val_epoch))

    sorted_optims = sorted(list(all_results.keys()), key=lambda optim: all_results[optim]['final']['val_acc'], reverse=False)
    
    for optim in sorted_optims:
        test_head_acc, test_tail_acc, test_acc = calc_accuracy(all_results[optim]['final']['test_acc_all'], head_classes, tail_classes)
        print(
            f"Optim: {optim} | Val: {all_results[optim]['final']['val_acc']:.2%} | TestAll: {test_acc:.2%} TestHead: {test_head_acc:.2%} TestTail: {test_tail_acc:.2%}")

def train(loader, fc, optimizer, epoch, criterion, args):
    # switch to train mode
    fc.train().cuda()
    for i, (features_x, targets_x) in enumerate(loader):
        features_x = features_x.cuda()

        targets_x = targets_x.cuda()

        # warmup learning rate
        if epoch < args.warmup_epoch:
            warmup_step = args.warmup_epoch * len(loader)
            curr_step = epoch * len(loader) + i + 1
            lr_schedule.warmup_learning_rate(optimizer, curr_step, warmup_step, args)

        # model forward
        logits_x = fc(features_x)

        # loss for labeled samples
        loss_x = criterion(logits_x, targets_x)

        loss = loss_x

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
