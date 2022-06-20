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
from torch.optim import SGD, Adam, lr_scheduler, AdamW

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from opts import parse_opts
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from utils import AverageMeter, calculate_accuracy, calculate_precision_recall_f1, accuracy
from utils import Logger, worker_init_fn, get_lr
from configuration import build_config
from dataloaders.all_frame_dataset import TSNdataset


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                model_type,
                current_lr,
                epoch_logger,
                tb_writer=None,
                distributed=False):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ###
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):

        data_time.update(time.time() - end_time)

        targets = targets.to(device, non_blocking=True)
        """修改损失函数"""
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        """"""
        ###
        probs = nn.Softmax(dim=1)(outputs)
        prec1, prec5 = accuracy(probs, targets, topk=(1, 3))
        train_top1.update(prec1.item(), targets.shape[0])
        train_top5.update(prec5.item(), targets.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'train_top1: {3:.3f}, train_top5: {4:.3f}\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch,
                                                            i + 1,
                                                            len(data_loader),
                                                            train_top1.avg, train_top5.avg,
                                                            batch_time=batch_time,
                                                            data_time=data_time,
                                                            loss=losses))
    '''
    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()
    '''

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'train_top1': train_top1.avg,
            'train_top5': train_top5.avg,
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/train_top1', train_top1.avg, epoch)
        tb_writer.add_scalar('train/train_top5', train_top5.avg, epoch)
        tb_writer.add_scalar('train/lr', current_lr, epoch)


def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              model_type,
              logger,
              tb_writer=None,
              distributed=False):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # accuracies = AverageMeter()
    precision_all = []
    recall_all = []
    f1_all = []

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):

            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            if model_type == 'i3d':
                outputs = torch.max(outputs, dim=2)[0]

            loss = criterion(outputs, targets)
            # acc = calculate_accuracy(outputs, targets)
            precision, recall, f1 = calculate_precision_recall_f1(outputs, targets, 26, 0.5)

            losses.update(loss.item(), inputs.size(0))
            # accuracies.update(acc, inputs.size(0))
            precision_all.append(precision)
            recall_all.append(recall)
            f1_all.append(f1)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'precision: {3:.3f}, recall: {4:.3f}, f1:{5:.3f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch,
                                                                i + 1,
                                                                len(data_loader),
                                                                precision, recall, f1,
                                                                batch_time=batch_time,
                                                                data_time=data_time,
                                                                loss=losses))
    '''
    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()
    '''

    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'precision': np.average(np.array(precision_all)),
                    'recall': np.average(np.array(recall_all)),
                    'f1': np.average(np.array(f1_all))})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/f1', np.average(np.array(f1_all)), epoch)

    return losses.avg


def test_epoch(epoch,
               data_loader,
               model,
               criterion,
               device,
               model_type,
               logger,
               tb_writer=None,
               distributed=False):
    print('test at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ###
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()

    end_time = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):

            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)

            """修改损失函数"""
            outputs= model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            """"""
            ###
            probs = nn.Softmax(dim=1)(outputs)
            prec1, prec5 = accuracy(probs, targets, topk=(1, 3))
            test_top1.update(prec1.item(), targets.shape[0])
            test_top5.update(prec5.item(), targets.shape[0])

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'test_top1: {3:.3f}, test_top5: {4:.3f},\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch,
                                                                i + 1,
                                                                len(data_loader),
                                                                test_top1.avg, test_top5.avg,
                                                                batch_time=batch_time,
                                                                data_time=data_time,
                                                                loss=losses))
    '''
    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()
    '''

    if logger is not None:
        logger.log(
            {
                'epoch': epoch, 'loss': losses.avg,
                'test_top1': test_top1.avg,
                'test_top5': test_top5.avg,
            }
        )


    """save best"""
    save_best_path = opt.result_path / 'best.pth'
    if os.path.exists(save_best_path):
        best_pth = torch.load(save_best_path)
    else:
        best_pth = {
            'epoch': 0,
            'test_top1': 0.,
            'test_top5': 0.,
            "state_dict": None,
        }

    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    if test_top1.avg >= best_pth["test_top1"]:
        best_pth = {
            'epoch': epoch,
            'test_top1': test_top1.avg,
            'test_top5': test_top5.avg,
            "state_dict": model_state_dict,
        }

    torch.save(best_pth, save_best_path)
    """end"""


    if tb_writer is not None:
        tb_writer.add_scalar('test/loss', losses.avg, epoch)
        tb_writer.add_scalar('test/test_top1', test_top1.avg, epoch)
        tb_writer.add_scalar('test/test_top5', test_top5.avg, epoch)


def resume_model(resume_path, model):
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


def get_train_utils(opt, cfg, model_parameters):
    # Get training data
    train_data = TSNdataset(cfg, 'train', 1.0, num_frames=opt.num_frames, seg_method=opt.seg_method,
                            input_size=opt.sample_size)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.n_threads)
    # pin_memory=True,
    # sampler=train_sampler,
    # worker_init_fn=worker_init_fn)

    out_file_path = 'train_{}.log'.format(opt.model)
    train_logger = Logger(opt.result_path / out_file_path, ['epoch', 'loss', 'train_top1', 'train_top5', 'lr'])

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    if opt.optimizer == 'sgd':
        optimizer = SGD(model_parameters,
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        dampening=dampening,
                        weight_decay=opt.weight_decay,
                        nesterov=opt.nesterov)
    elif opt.optimizer == 'adam':
        optimizer = AdamW(model_parameters,
                         lr=opt.learning_rate,
                         weight_decay=opt.weight_decay,
                         eps=1e-8)
    else:
        print("=" * 40)
        print("Invalid optimizer mode: ", opt.optimizer)
        print("Select [sgd, adam]")
        exit(0)

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             opt.multistep_milestones, gamma=opt.gama)

    return (train_loader, train_logger, optimizer, scheduler)


def get_val_utils(opt, cfg):
    # Get validation data
    val_data = TSNdataset(cfg, 'val', 1.0, num_frames=opt.num_frames, seg_method=opt.seg_method,
                          input_size=opt.sample_size)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(1),
                                             shuffle=False,
                                             num_workers=opt.n_threads)
    # pin_memory=True,
    # sampler=val_sampler,
    # worker_init_fn=worker_init_fn)

    out_file_path = 'val_{}.log'.format(opt.model)
    val_logger = Logger(opt.result_path / out_file_path, ['epoch', 'loss', 'precision', 'recall', 'f1'])

    return val_loader, val_logger


def get_test_utils(opt, cfg):
    # Get validation data
    test_data = TSNdataset(cfg, 'test', 1.0, num_frames=opt.num_frames, seg_method=opt.seg_method,
                           input_size=opt.sample_size)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              num_workers=opt.n_threads)
    # pin_memory=True,
    # sampler=val_sampler,
    # worker_init_fn=worker_init_fn)

    out_file_path = 'test_{}.log'.format(opt.model)
    test_logger = Logger(opt.result_path / out_file_path, ['epoch', 'loss', 'test_top1', 'test_top5', ])

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
    """

    :param opt:
    :return:
    """

    """设置随机种子"""
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    # torch.manual_seed(opt.manual_seed)
    """"""

    cfg = build_config(opt.dataset)

    model = generate_model(opt)
    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if opt.pretrain_path:
        model = load_pretrained_model(model, opt.pretrain_path, opt.model,
                                      opt.n_finetune_classes)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, model)
    model = make_data_parallel(model, opt.distributed, opt.device)

    if opt.pretrain_path:
        parameters = model.parameters()
    else:
        parameters = model.parameters()

    print(model)

    criterion = nn.BCEWithLogitsLoss().to(opt.device)

    if not opt.no_train:
        (train_loader, train_logger, optimizer, scheduler) = get_train_utils(opt, cfg, parameters)
        if opt.resume_path is not None:
            opt.begin_epoch, optimizer, scheduler = resume_train_utils(
                opt.resume_path, opt.begin_epoch, optimizer, scheduler)
            if opt.overwrite_milestones:
                scheduler.milestones = opt.multistep_milestones

    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt, cfg)

    if not opt.no_test:
        test_loader, test_logger = get_test_utils(opt, cfg)

    if opt.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    prev_val_loss = None


    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            current_lr = get_lr(optimizer)
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, opt.model, current_lr, train_logger,
                        tb_writer)

            if i % opt.checkpoint == 0:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)

        if not opt.no_val:
            prev_val_loss = val_epoch(i, val_loader, model, criterion,
                                      opt.device, opt.model, val_logger, tb_writer)

        if not opt.no_test:
            test_epoch(i, test_loader, model, criterion,
                       opt.device, opt.model, test_logger, tb_writer)

        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)


if __name__ == '__main__':

    opt = get_opt()

    if not opt.no_cuda:
        cudnn.benchmark = True

    main_worker(opt)
