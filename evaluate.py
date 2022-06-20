import torch
import time
import os
import sys
import numpy as np
import json
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.backends import cudnn
from torch.optim import SGD, Adam, lr_scheduler

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from opts import parse_opts
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from utils import AverageMeter, calculate_accuracy, calculate_precision_recall_f1, each_accuracy, accuracy
from utils import Logger, worker_init_fn, get_lr, EvalLogger
from configuration import build_config
from dataloaders.all_frame_dataset import TSNdataset
from tqdm import tqdm


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

def test_epoch(
        data_loader,
        model,
        device,
        model_type,
        num_classes,
        logger,
):

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    ###
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()

    end_time = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(data_loader)):

            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)

            """修改损失函数"""
            outputs = model(inputs)
            """"""
            ###
            probs = nn.Softmax(dim=1)(outputs)
            [prec1, prec5], index = each_accuracy(probs, targets, topk=(1, 5))
            test_top1.update(prec1.item(), targets.shape[0])
            test_top5.update(prec5.item(), targets.shape[0])

            batch_time.update(time.time() - end_time)
            end_time = time.time()


        print(test_top1.sum, test_top5.sum)
        print(test_top1.count, test_top5.count)
        print(test_top1.avg, test_top5.avg)


def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    # assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model


def resume_train_utils(resume_path, begin_epoch, optimizer, scheduler):
    print('loading checkpoint {} train myutils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, optimizer, scheduler


def get_mean_std(value_scale, dataset):
    assert dataset in ['tinyvirat', 'activitynet', 'kinetics', '0.5', "ucf101", "hmdb51"]

    if dataset == 'activitynet':
        mean = [0.4477, 0.4209, 0.3906]
        std = [0.2767, 0.2695, 0.2714]
    elif dataset == 'kinetics':
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
    elif dataset == '0.5':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif dataset == 'tinyvirat':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean, std


def get_test_utils(opt, cfg):
    # Get validation data
    test_data = TSNdataset(cfg, 'test', 1.0, num_frames=opt.num_frames, seg_method=opt.seg_method,
                           input_size=opt.sample_size)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=opt.n_threads)
    # pin_memory=True,
    # sampler=val_sampler,
    # worker_init_fn=worker_init_fn)

    out_file_path = 'evaluate_{}.txt'.format(opt.model)
    test_logger = EvalLogger(opt.result_path / out_file_path)

    return test_loader, test_logger


def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.video_path = opt.root_path / opt.video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.sub_path is None:
        opt.sub_path = opt.model

    if opt.sub_path is not None:
        opt.result_path = opt.result_path / opt.sub_path
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3
    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]

    opt.no_cuda = True if not torch.cuda.is_available() else False
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')

    if opt.distributed:
        opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)

    return opt


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)


def main_worker(opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    cfg = build_config(opt.dataset)

    model = generate_model(opt)
    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if opt.pretrain_path:
        model = load_pretrained_model(model, opt.pretrain_path, opt.model,
                                      opt.n_finetune_classes)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)
    model = make_data_parallel(model, opt.distributed, opt.device)

    # print(model)

    # test epoch
    test_loader, test_logger = get_test_utils(opt, cfg)
    test_epoch(test_loader, model, opt.device, opt.model, opt.n_classes, test_logger)


if __name__ == '__main__':

    opt = get_opt()

    if not opt.no_cuda:
        cudnn.benchmark = True

    main_worker(opt)
