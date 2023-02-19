# python lifting.py --dataset inat --index_name default --threshold 18 --budget 6 --lift pl --model_name wd_1e_5 --algo supervised
# python lifting.py --dataset inat --index_name default --threshold 18 --budget 6 --lift pl_entropy --model_name wd_1e_5 --algo supervised
# python lifting.py --dataset imagenet127 --index_name default --threshold 25 --budget 12 --lift pl --model_name wd_1e_5 --algo supervised
# python lifting.py --dataset imagenet127 --index_name default --threshold 25 --budget 12 --lift pl_entropy --model_name wd_1e_5 --algo supervised

# python lifting.py --dataset inat --index_name default --threshold 18 --budget 6 --lift pl_entropy_all --model_name wd_1e_5 --algo supervised
# python lifting.py --dataset inat --index_name default --threshold 18 --budget 6 --lift pl_entropy_all_tp --model_name wd_1e_5 --algo supervised
# python lifting.py --dataset inat --index_name default --threshold 18 --budget 6 --lift pl_entropy_all_fp --model_name wd_1e_5 --algo supervised
import argparse
import shutil
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
from results import RESULT_DICT

parser = argparse.ArgumentParser(description='PyTorch Dataset Index Preparation')
parser.add_argument('--dataset', default='imagenet127', type=str,
                    choices=['inat', 'imagenet127'],
                    help='dataset to use, choices: [inat, imagenet127]')
parser.add_argument('--index_name', default='default', type=str,
                    help='name of index dir (the directory under indexes/)')
parser.add_argument('--lift', default='pl', type=str,
                    choices=['pl', 'pl_entropy', 'pl_entropy_all', 'pl_entropy_all_tp', 'pl_entropy_all_fp', 'random'],
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
parser.add_argument('--root', metavar='DIR', default='/scratch/fercus/',
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

def lifting(model, unlabeled_targets, unlabeled_loader, lift_method, head_classes, tail_classes, num_classes, budget):
    # for target, item in zip(unlabeled_targets, unlabeled_loader):
    #     assert target == item[1]
    preds_list = None
    with torch.no_grad():
        for images, target in tqdm(unlabeled_loader):
            images = images[0].cuda() # Default Unlabeled set uses TwoCropsTransform; taking the first crop here
            target = target.cuda()
            # compute output
            output = model(images)

            # measure accuracy and record loss
            probs = torch.softmax(output, dim=-1)
            entropys = -torch.sum(probs * torch.log(probs), dim=-1)
            _, preds = probs.topk(1, 1, True, True)
            preds = preds.view(-1)
            if preds_list is None:
                preds_list = preds.cpu()
                probs_list = probs.cpu()
                target_list = target.cpu()
                entropys_list = entropys.cpu()
            else:
                preds_list = torch.cat((preds_list, preds.cpu()), dim=0)
                probs_list = torch.cat((probs_list, probs.cpu()), dim=0)
                target_list = torch.cat((target_list, target.cpu()), dim=0)
                entropys_list = torch.cat((entropys_list, entropys.cpu()), dim=0)
            acc1, _ = utils.accuracy(output, target, topk=(1, 5))

        perclass_acc = [0 for _ in range(num_classes)]
        for c in range(num_classes):
            perclass_acc[c] = ((preds_list == target_list) * (target_list == c)).sum().float() / max((target_list == c).sum(), 1)
        acc = (preds_list == target_list).sum().float() / len(target_list)
        mean_acc = sum(perclass_acc) / num_classes
        min_perclass_acc = min(perclass_acc)
        max_perclass_acc = max(perclass_acc)
        print(' * Acc@1 {acc:.3%} mAcc {mean_acc:.3%} minAcc {min_perclass_acc:.3%} maxAcc {max_perclass_acc:.3%}'
              .format(acc=acc, mean_acc=mean_acc, min_perclass_acc=min_perclass_acc, max_perclass_acc=max_perclass_acc))

    # return {
    #     'top1' : acc, 
    #     'mean' : mean_acc,
    #     'all' : perclass_acc
    # }
    for target_gt, target in zip(target_list, unlabeled_targets):
        assert target_gt == target
    lifted_tail_samples = np.array([], dtype=np.int64)
    tailaccs = {}
    all_count = 0.
    fp_count = 0.
    fp_count_tail = 0. # when fps label is actually from tail class
    fp_count_head = 0. # when fps label is actually from head class
    for tail_index in range(len(tail_classes)):
        if lift_method == 'random':
            tail_samples = np.random.choice(
                np.arange(len(unlabeled_targets)), size=budget, replace=False
            )
        elif lift_method == 'pl':
            prob_list = probs_list[:, tail_index]
            tail_samples = np.argsort(prob_list)[-budget:]
        elif lift_method == 'pl_entropy':
            prob_list = probs_list[:, tail_index]
            tail_samples = np.argsort(prob_list)[-2*budget:]
            entropy_list = entropys_list[tail_samples]
            tail_samples = tail_samples[np.argsort(entropy_list)[-budget:]]
        elif lift_method == 'pl_entropy_all':
            prob_list = probs_list[:, tail_index]
            tail_samples = np.argsort(prob_list)[-2*budget:]
        elif lift_method == 'pl_entropy_all_tp':
            prob_list = probs_list[:, tail_index]
            tail_samples = np.argsort(prob_list)[-2*budget:]
        elif lift_method == 'pl_entropy_all_fp':
            prob_list = probs_list[:, tail_index]
            tail_samples = np.argsort(prob_list)[-2*budget:]
        else:
            raise NotImplementedError()
        
        # calculate fp ratio
        all_count += len(tail_samples)
        fp_count += (target_list[tail_samples] != tail_index).sum().float()
        for sample in tail_samples[(target_list[tail_samples] != tail_index)]:
            if target_list[sample] in head_classes:
                fp_count_head += 1
            elif target_list[sample] in tail_classes:
                fp_count_tail += 1

        tailaccs[tail_index] = (target_list[tail_samples] == tail_index).sum().float() / len(tail_samples)
        if lift_method == 'pl_entropy_all':
            correct_tail_samples = tail_samples
        elif lift_method == "pl_entropy_all_fp":
            correct_tail_samples = tail_samples[target_list[tail_samples] != tail_index]
        else:
            correct_tail_samples = tail_samples[target_list[tail_samples] == tail_index]
        lifted_tail_samples = np.concatenate((lifted_tail_samples, correct_tail_samples))
    print(f"Average of tail accs: {sum(tailaccs.values())/len(tailaccs)}")
    print(f"FP ratio (out of all tail samples): {fp_count/all_count}")
    print(f"FP ratio (head): {fp_count_head/fp_count}")
    print(f"FP ratio (tail): {fp_count_tail/fp_count}")
    import pdb; pdb.set_trace()
    return lifted_tail_samples


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

    if args.lift == "random":
        lift_name = f"{args.index_name}_lift_{args.lift}_thre_{args.threshold}_budget_{args.budget}"
    else:
        lift_name = f"{args.index_name}_lift_{args.algo}_{args.model_name}_{args.lift}_thre_{args.threshold}_budget_{args.budget}"
    new_index_dir = Path("indexes") / f"{args.dataset}" / lift_name
    if not new_index_dir.exists():
        new_index_dir.mkdir(parents=True)
        shutil.copy(val_index_file, new_index_dir)
    new_train_index_file = new_index_dir / "train.csv"
    new_unlabeled_index_file = new_index_dir / "unlabeled.csv"


    # if not new_train_index_file.exists() \
    #         or not new_unlabeled_index_file.exists():
    if True:
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
        pretrained_path = RESULT_DICT[args.dataset][args.index_name][args.algo]["default"][args.model_name]["result"]
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
        lifted_tail_samples = lifting(
            pretrained_model, unlabeled_targets, unlabeled_loader, args.lift, head_classes, tail_classes, num_classes, args.budget)
        print("lifted_tail_samples length:", len(lifted_tail_samples))
        new_train_indexes = np.concatenate((train_indexes, unlabeled_indexes[lifted_tail_samples]))
        new_train_paths = np.concatenate((train_paths, unlabeled_paths[lifted_tail_samples]))
        new_train_targets = np.concatenate((train_targets, unlabeled_targets[lifted_tail_samples]))

        # load index files
        # new_train_pd = pd.read_csv(new_train_index_file)
        # old_new_train_indexes = np.array(new_train_pd['Index'].tolist())
        # old_new_train_paths = np.array(new_train_pd['Path'].tolist())
        # old_new_train_targets = np.array(new_train_pd['Target'].tolist())
        # import pdb; pdb.set_trace()
        
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
        # load index files
        new_train_pd = pd.read_csv(new_train_index_file)
        new_train_indexes = np.array(new_train_pd['Index'].tolist())
        new_train_paths = np.array(new_train_pd['Path'].tolist())
        new_train_targets = np.array(new_train_pd['Target'].tolist())

        

if __name__ == '__main__':
    main()