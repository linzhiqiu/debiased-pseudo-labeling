import argparse
import builtins
import math
import os
import shutil
import time
import warnings
import numpy as np
import pandas as pd
import random
from datetime import datetime
from pathlib import Path

import torch
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
from engine import validate_perclass
from torch.utils.tensorboard import SummaryWriter

# TODO: Use EMA model to test/validate
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

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
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
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

def main():
    args = parser.parse_args()
    assert args.warmup_epoch < args.schedule[0]
    print(args)

    if args.seed is not None:
        seed = args.seed + dist_utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

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

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    if args.amp_opt_level != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%m-%d-%Y-%H:%M")
    print("date and time =", dt_string)	
    args.output = os.path.join(args.output, dt_string)

    if args.rank in [-1, 0]:
        os.makedirs(args.output, exist_ok=True)
        writer = SummaryWriter(args.output)
        print("Writer is initialized")
    else:
        writer = None

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
    print("train_dataset:\n{}".format(len(train_dataset)))
    print("val_dataset:\n{}".format(len(val_dataset)))
    print("test_dataset:\n{}".format(len(test_dataset)))

    # Data loading code
    train_sampler = DistributedSampler if args.distributed else RandomSampler

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler(train_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=True)


    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    best_val_epoch = args.start_epoch
    best_test_epoch = args.start_epoch
    best_val_acc = None
    best_test_acc = None
    results = {
        'val_top1': [], 
        'test_top1': [],
        'val_acc': [], # mean class accuracy on validation set
        'test_acc': [], # mean class accuracy on test set
        'val_acc_all': [], # mean perclass accuracy on validation set
        'test_acc_all': [], # mean perclass accuracy on test set
    }

    def update_result(
            val_result, test_result):
        results['val_top1'].append(float(val_result['top1']))
        results['test_top1'].append(float(test_result['top1']))
        results['val_acc'].append(float(val_result['mean']))
        results['val_acc_all'].append(val_result['all'])
        results['test_acc'].append(float(test_result['mean']))
        results['test_acc_all'].append(test_result['all'])

    for epoch in range(args.start_epoch, args.epochs):
        if epoch >= args.warmup_epoch:
            lr_schedule.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(
            train_loader, model,
            optimizer, epoch, args, writer)

        is_best_val = False
        is_best_test = False

        # evaluate on validation set
        val_result = validate_perclass(val_loader, model, criterion, args, prefix='Val')
        # remember best acc@1 and save checkpoint
        is_best_val = best_val_acc is None or val_result['mean'] > best_val_acc
        if is_best_val:
            best_val_acc = val_result['mean']
            best_val_epoch = epoch
        
        # evaluate on validation set
        test_result = validate_perclass(test_loader, model, criterion, args)
        # remember best acc@1 and save checkpoint
        is_best_test = best_test_acc is None or test_result['mean'] > best_test_acc
        if is_best_test:
            best_test_acc = test_result['mean']
            best_test_epoch = epoch

        torch.distributed.barrier()
        update_result(
            val_result, test_result
        )
        if args.rank in [-1, 0]:
            writer.add_scalar('test/1.test_acc', test_result['mean'], epoch)
            writer.add_scalar('test/2.val_acc', val_result['mean'], epoch)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            torch.save(
                results, os.path.join(args.output, 'results.pth')
            )
            
            if is_best_val:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_val_epoch': best_val_epoch,
                    'best_test_acc': best_test_acc,
                    'best_test_epoch': best_test_epoch,
                    'optimizer' : optimizer.state_dict(),
                }, os.path.join(args.output, 'best_val_model.pth.tar'))
            if is_best_test:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_val_epoch': best_val_epoch,
                    'best_test_acc': best_test_acc,
                    'best_test_epoch': best_test_epoch,
                    'optimizer' : optimizer.state_dict(),
                }, os.path.join(args.output, 'best_test_model.pth.tar'))

    print('Best Val Acc {0} @ epoch {1}'.format(
        best_val_acc, best_val_epoch))
    print('Best Test Acc {0} @ Best val epoch {1}'.format(
        results['test_acc'][best_val_epoch], best_val_epoch))
    print('Best Test Acc {0} @ epoch {1}'.format(
        best_test_acc, best_test_epoch))
    print('checkpoint saved in: ', args.output)
    if args.rank in [-1, 0]:
        writer.close()

def train(train_loader, model, optimizer, epoch, args, writer=None):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1_x = utils.AverageMeter('Acc_x@1', ':6.2f')
    curr_lr = utils.InstantMeter('LR', '')
    progress = utils.ProgressMeter(
        len(train_loader),
        [curr_lr, batch_time, data_time, losses, top1_x],
        prefix="Epoch: [{}/{}]\t".format(epoch, args.epochs))

    if args.distributed:
        print("set epoch={} for labeled sampler".format(epoch))
        train_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    if args.eman:
        print("setting the ema model to eval mode")
        if hasattr(model, 'module'):
            model.module.ema.eval()
        else:
            model.ema.eval()

    end = time.time()
    for i, (images_x, targets_x) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images_x = images_x.cuda(args.gpu, non_blocking=True)

        targets_x = targets_x.cuda(args.gpu, non_blocking=True)

        # warmup learning rate
        if epoch < args.warmup_epoch:
            warmup_step = args.warmup_epoch * len(train_loader)
            curr_step = epoch * len(train_loader) + i + 1
            lr_schedule.warmup_learning_rate(optimizer, curr_step, warmup_step, args)
        curr_lr.update(optimizer.param_groups[0]['lr'])

        # model forward
        logits_x = model(images_x)

        # loss for labeled samples
        loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')

        loss = loss_x
        # measure accuracy and record loss
        losses.update(loss_x.item())
        acc1_x, _ = utils.accuracy(logits_x, targets_x, topk=(1, 5))
        top1_x.update(acc1_x[0], logits_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        # update the ema model
        if args.eman:
            if hasattr(model, 'module'):
                model.module.momentum_update_ema()
            else:
                model.momentum_update_ema()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    if args.rank in [-1, 0]:
        writer.add_scalar('train/1.train_loss', losses.avg, epoch)

if __name__ == '__main__':
    main()
