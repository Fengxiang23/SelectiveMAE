# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from timm.data.loader import MultiEpochsDataLoader
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from transformer_utils import handle_flash_attn
import models_mae
import models_selectivemae
from engine_pretrain import train_one_epoch
from util.datasets import AllDataset
from infobatch import InfoBatch
from token_selected_smooth import TokenSelect_smooth


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--decoder_depth', default=8, type=int,
                        help='depth of decoder')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (1 - percentage of remained patches).')
    parser.add_argument('--kept_mask_ratio', default=0.75, type=float,
                        help='Amongst the all tokens, the percentage of the mask that are kept')
    parser.add_argument('--inverse_lr', action='store_true', default=False, help='Use inverse lr scheduler')
    parser.add_argument('--no_lr_scale', action='store_true', default=False, help='Do not scale lr by mask_ratio')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument(
        '--find_unused_parameters', action='store_true',
        help="distributed ddp find unused parameters")
    parser.set_defaults(find_unused_parameters=True)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--multi_epochs_dataloader', action='store_true',
                        help='Use MultiEpochsDataLoader to prevent reinitializing dataloader per epoch')

    # distributed training parameterssmooth_scale
    parser.add_argument('--world_size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # MAE or cross-MAE
    parser.add_argument('--selectivemae', action='store_true', default=False)
    parser.add_argument('--weight_fm', action='store_true', default=False,
                        help='Weight the feature maps for decoder when running cross-mae')
    parser.add_argument('--use_fm', nargs='+', type=int, default=[-1],
                        help='Feature maps to use for decoder')
    parser.add_argument('--use_input', action='store_true', default=False,
                        help="use input as a feature map")
    parser.add_argument('--self_attn', action='store_true', default=False, help="use self attention in decoder")

    parser.add_argument('--enable_flash_attention2', action='store_true', default=False, help="Use flash attntion 2")

    parser.add_argument('--dataset', default='opticalrs-4m', type=str,
                        choices=['opticalrs-13m', 'opticalrs-4m',  'millionaid'],
                        help='type of dataset')
    parser.add_argument('--dataset_path', default='/home/wangfengxiang/data/imagenet/imagenet/train', type=str,
                        help='type of dataset')
    parser.add_argument('--smooth_scale_epoch', default=20, type=int)
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    handle_flash_attn(args)

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_path = args.dataset_path
    dataset_train = AllDataset(dataset_path, train=True, transform=transform_train)
    print(dataset_train)
    ratio = 0.5
    delta = 0.875
    dataset_train = InfoBatch(dataset_train, args.epochs, ratio, delta)
    global_rank = misc.get_rank()
    sampler_train = dataset_train.sampler
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    dataloader_cls = MultiEpochsDataLoader if args.multi_epochs_dataloader else torch.utils.data.DataLoader

    data_loader_train = dataloader_cls(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.selectivemae:
        model = models_selectivemae.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            decoder_depth=args.decoder_depth,
        )
    else:
        model = models_mae.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            decoder_depth=args.decoder_depth,
        )

    model.token_select = TokenSelect_smooth(expansion_step=[0, 300 - args.smooth_scale_epoch, 300, 600 - args.smooth_scale_epoch, 600],
                                    keep_rate=[1-args.mask_ratio, 1-args.mask_ratio, 1-args.mask_ratio, 1-args.mask_ratio, 1-args.mask_ratio],
                                    initialization_keep_rate=(1-args.mask_ratio)/2,
                                    expansion_multiple_stage= 2,
                                    smooth_scale_epoch=args.smooth_scale_epoch)

    model.to(device)
    model_without_ddp = model
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    print("Model = %s" % str(model_without_ddp))
    print('args.accum_iter', args.accum_iter)
    print('misc.get_world_size()', misc.get_world_size())
    print('args.batch_size', args.batch_size)
    print('eff_batch_size', eff_batch_size)

    if args.lr is None:
        base_ratio = args.kept_mask_ratio / args.mask_ratio  # base ratio for MAE
        if args.no_lr_scale:
            scale_kmr = 1
        elif args.inverse_lr:
            scale_kmr = 1 / base_ratio
        else:
            scale_kmr = base_ratio
        args.lr = scale_kmr * args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_parameters
        )
        model_without_ddp = model.module
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)



        model_without_ddp.token_select.update_current_stage(epoch)
        model_without_ddp.token_select.sparse_inference = True
        train_stats = train_one_epoch(
                model,  data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer, dataset_train=dataset_train, args=args)


        if args.output_dir:
            if epoch % 200 == 0:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, save_latest_model_only=False)
            elif epoch % 10 == 0:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, save_latest_model_only=False)
            if epoch + 1 == args.epochs:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, save_latest_model_only=True)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    assert args.kept_mask_ratio <= args.mask_ratio, "Cannot reconstruct more than what is masked"
    if args.log_dir is None:
        args.log_dir = args.output_dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
