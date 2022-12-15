import time
import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.backends import cudnn

from torch.utils.data import DataLoader
from setting import get_opt, resume_model
from utils import AverageMeter, calculate_accuracy, calculate_precision_recall_f1, each_accuracy
from utils import Logger, worker_init_fn, get_lr, EvalLogger
from configuration import build_config
from dataloaders.all_frame_dataset import TSNdataset
from tqdm import tqdm


def test_epoch(
        data_loader,
        model,
        device,
        model_type,
        num_classes,
        logger,
        test_logger_each_class,
):
    print('########## evaling ##########')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    ###
    test_top1 = torch.zeros(size=(1, num_classes))
    test_top2 = torch.zeros(size=(1, num_classes))
    all_label = torch.zeros(size=(1, num_classes))

    end_time = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(data_loader)):
            # inputs = [bs, c, T, h, w]
            data_time.update(time.time() - end_time)

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)

            probs = nn.Softmax(dim=1)(outputs)
            [prec1, prec2], index = each_accuracy(probs, targets, topk=(1, 2))
            logger.write(f"Label:{torch.argmax(targets).item()} --> Prob:{torch.argmax(probs).item()}")

            test_top1[0][index] += prec1.item()
            test_top2[0][index] += prec2.item()
            all_label[0][index] += 1

            batch_time.update(time.time() - end_time)
            end_time = time.time()

        """
        save top1，top2
        """
        print(test_top1)
        print(test_top2)
        print(all_label)
        """"""
        top1 = test_top1 / all_label
        top2 = test_top2 / all_label
        print("class_top1:", top1)
        print("class_top2:", top2)
        print('test_top1: {:.4f}, test_top2: {:.4f}'.format(top1.mean().item(), top2.mean().item()))

        # logger
        logger.write(f"top1_acc:{test_top1}")
        logger.write(f"top2_acc:{test_top2}")
        logger.write(f"all_label:{all_label}")
        logger.write(f"class_top1:{top1}")
        logger.write(f"class_top2:{top2}")
        logger.write('test_top1: {:.4f}, test_top2: {:.4f}\n'.format(top1.mean().item(), top2.mean().item()))

        # save accuracy
        # save_path = os.path.join(opt.result_path, "save_dict.pth")
        # dict = {
        #     "class_top1": top1,
        #     "class_top2": top2,
        #     "test_top1": top1.mean().item(),
        #     "test_top2": top2.mean().item(),
        # }
        # torch.save(dict, save_path)


def get_test_utils(opt, cfg):
    # Get validation data
    test_data = TSNdataset(cfg, 'test', 1.0, num_frames=opt.num_frames, seg_method=opt.seg_method,
                           input_size=opt.sample_size, seed=opt.seed)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=opt.n_threads)
    # pin_memory=True,
    # sampler=val_sampler,
    # worker_init_fn=worker_init_fn)

    out_file_path = 'evaluate_{}.txt'.format(opt.model)    # draw confuse matrix
    test_logger = EvalLogger(opt.result_path / out_file_path)

    out_file_path_1 = 'evaluate_each_class_{}.txt'.format(opt.model)
    test_logger_each_class = EvalLogger(opt.result_path / out_file_path_1)

    return test_loader, test_logger, test_logger_each_class


def main_worker(opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    cfg = build_config(opt.dataset)

    from STR_transformer import STR_Transformer
    # set model here
    model = STR_Transformer(at_type=opt.at_type, num_classes=opt.n_classes, lstm_channel=opt.num_frames)
    model = resume_model(opt.resume_path, model)
    model.to(opt.device)

    # test epoch
    test_loader, test_logger, test_logger_each_class = get_test_utils(opt, cfg)
    test_epoch(test_loader, model, opt.device, opt.model, opt.n_classes, test_logger, test_logger_each_class)


if __name__ == '__main__':

    opt = get_opt()

    if not opt.no_cuda:
        cudnn.benchmark = True

    main_worker(opt)


# python evaluate.py --result_path result/eval1215 --sub_path STR_Transformer_DTM --model STR_Transformer --at_type DTM --n_classes 7 --resume_path best.pth --num_frames 8 --sample_size 224 --dataset myaction --batch_size 1 --n_threads 4
#  --seg_method tsn
