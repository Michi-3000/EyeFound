# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import pandas as pd

import torch
torch.manual_seed(42)
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit

from engine_finetune_multi import train_one_epoch, evaluate
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from sklearn.model_selection import StratifiedShuffleSplit

class CustomImageFolder(Dataset):
    def __init__(self, dt, transform=None):
        self.dt = dt
        self.transform = transform
        self.samples = self._make_dataset()

    def _make_dataset(self):
        images = []
        for path, cls in zip(self.dt['impath'].to_list(), self.dt['class'].to_list()):
            item = (path, cls)
            images.append(item)
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target, path

def build_dataset(is_train, dt, args):    
    transform = build_transform(is_train, args)
    dataset = CustomImageFolder(dt, transform=transform)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    parser.add_argument('--now_epoch', default=0)

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--data_name', default='', type=str, metavar='NAME',
                        help='Name of dataset')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--id_map', default=None, help='A custom dictionary argument')
    parser.add_argument('--test_num', default=5 ,type=int,
                        help='Number of tests')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/danli/data/public', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir_downstream',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_downstream',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--save_mem', default=True,
                        help='to save memory')
    parser.add_argument('--test_dir', default='',type=str)
    parser.add_argument('--ref_dir', default='',type=str)
    parser.add_argument('--output_dir2', default='',type=str)
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    #test_dt = pd.read_csv(os.path.join(args.test_dir, args.data_name, 'epoch33/seed4', 'test.csv'))
    test_dt = pd.read_csv("/home/danli/data/public/MAE_process/"+args.data_name+'.csv')
    cudnn.benchmark = True
    for dataname in ['MESSIDOR2_v2', 'Aptos2019','IDRiD']:#用别的external weight去predict当前数据的test.csv
        if dataname == args.data_name:
            continue
        #/home/danli/workspace/pretrain/mae_zwy/output_dir_downstream/all_dataset_2024-02-18-0013_testtimes5/MESSIDOR2_v2/epoch33/seed4
        args.ref_dir = os.path.join(args.test_dir, dataname, 'epoch0/seed4')#epoch33
        args.output_dir = os.path.join(args.output_dir2, dataname)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        test_dt.to_csv(os.path.join(args.output_dir, "test.csv"))

        dataset_test = build_dataset(is_train='test', dt=test_dt, args=args)

        if True:  # args.distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            
            if args.dist_eval:
                if len(dataset_test) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            else:
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
                

        if global_rank == 0 and args.log_dir is not None and not args.eval:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir+args.task)
        else:
            log_writer = None

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        
        
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)
        
        model = models_vit.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

        if args.finetune and not args.eval:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            if args.global_pool:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            else:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)

        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        
        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        loss_scaler = NativeScaler()


        model = misc.load_model_test2(args=args, epoch="best_epoch", model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, best_name=args.save_mem)
        test_stats,auc_roc,_ = evaluate(data_loader_test, model, device, os.path.join(os.path.dirname(os.path.dirname(args.output_dir)), "test"), mode='test',num_class=args.nb_classes, epoch=args.now_epoch, id_map=args.id_map, best_epoch=dataname, finetune=args.finetune)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    from datetime import datetime
    current_time = datetime.now()
    #args.output_dir = os.path.join(args.output_dir, "all_dataset_"+current_time.strftime("%Y-%m-%d-%H%M")+"_testtimes"+str(args.test_num), args.data_name, "epoch"+str(args.now_epoch))
    args.output_dir = os.path.join(args.output_dir+"_testtimes"+str(args.test_num), args.data_name, "epoch"+str(args.now_epoch))
    #args.output_dir = os.path.join('/home/danli/workspace/pretrain/mae_zwy/output_dir_downstream/all_dataset_2024-01-19-0223_testtimes1', "epoch"+str(args.now_epoch))
    print(args.output_dir)
    or_dir = args.output_dir
    #if args.index_map!=None:
    #    try:
    #        args.index_map = json.loads(args.index_map)
    #        print(args.index_map)
    #    except json.JSONDecodeError as e:
    #        print(f'Error decoding JSON: {e}')
    if args.eval:
        args.output_dir = os.path.join(args.output_dir, "test")

    if args.data_name in ['IDRiD', 'MESSIDOR2', 'MESSIDOR2_v2', 'Aptos2019']:
        args.id_map = {"no DR":0, "mild DR":1, "moderate DR":2, "severe DR":3, "proliferative DR":4}
    elif args.data_name in ['Retina']:
        args.id_map = {'normal':0, 'cataract':1, 'glaucoma':2, 'retina disease':3}
    elif args.data_name in ['Glaucoma_Fundus']:
        args.id_map = {'normal control':0, 'early glaucoma':1, 'advanced glaucoma':2}
    elif args.data_name in ['OCTID']:
        args.id_map = {'normal':0, 'AMRD':1, 'CSR':2, 'DR':3, 'macula hole':4}
    
    if args.id_map !=None:
        args.nb_classes = len(args.id_map)
    if args.data_name in ['JSIEC']:
        args.nb_classes = 39
    elif args.data_name in ['PAPILA', 'PAPILA_v2']:
        args.nb_classes = 3

    
    for i in range(args.test_num):
        args.seed = i
        args.output_dir = os.path.join(or_dir, "seed"+str(args.seed))
        args.log_dir = args.output_dir
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.output_dir2 = args.output_dir
        main(args)
    
    '''
    all_dt = pd.DataFrame()
    for n in os.listdir(or_dir):
        dt = pd.read_csv(os.path.join(or_dir, n, 'test', 'metrics_test.csv'))
        all_dt = all_dt.append(dt)
    all_dt.to_csv(os.path.join(or_dir, 'metrics_test_all.csv'))

    summary_df = []
    for task in ['auc_roc', 'auc_pr']:
        aurocs = all_dt[task].to_list()
        mean_auroc = np.mean(aurocs)
        std_auroc = np.std(aurocs)

        std_error_auroc = std_auroc / np.sqrt(len(aurocs))

        confidence_level = 0.95

        lower_bound = mean_auroc - 1.96 * (std_auroc / np.sqrt(len(aurocs)))
        upper_bound = mean_auroc + 1.96 * (std_auroc / np.sqrt(len(aurocs)))
        summary_df.append({
            'Task': task,
            'Mean': mean_auroc,
            'Standard Deviation': std_auroc,
            'Standard Error': std_error_auroc,
            'Confidence Interval': f'[{lower_bound:.4f}, {upper_bound:.4f}]'
        })
    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(os.path.join(or_dir, 'confidence_interval.csv'))
    print(summary_df)
    '''


