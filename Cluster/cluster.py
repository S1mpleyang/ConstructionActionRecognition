import time

import torch
from torchvision.models import resnet50, mobilenet_v3_large
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F


class CNN_TRX(nn.Module):

    def __init__(self, backbone='resnet'):
        super(CNN_TRX, self).__init__()

        if backbone == 'resnet':
            resnet = resnet50(pretrained=True)
            last_layer_idx = -1  # [0-9]
            self.backbone = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        elif backbone == "mobilenet_v3_large":
            mobilenet = mobilenet_v3_large(pretrained=True)
            last_layer_idx = -1  # [0-9]
            self.backbone = nn.Sequential(*list(mobilenet.children())[:last_layer_idx])
        elif backbone == "pool":
            self.backbone = nn.Sequential(
                nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            )

        self.freeze_backbone()
        self.backbone.eval()

    def freeze_backbone(self):
        """
        冻结特征提取
        :return:
        """
        for i, j in self.backbone.named_parameters():
            j.requires_grad = False

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.backbone(x)
        y = y.reshape(bs, -1)
        # y = torch.sigmoid(y)        是否使用sigmod归一化
        return y


def tsn_seg(frame_len, num_frames, seed=None):
    """
    按照 tsn 方法划分数据
    :param seed:
    :param frame_len: 总的帧长度
    :param num_frames: 选择的帧长度
    :return:
    """
    frame_count = frame_len
    seg = frame_count // num_frames
    frame_seg = np.arange(0, seg * num_frames, step=seg)
    # 设置随机数种子
    if seed:
        np.random.seed(seed)
    offset = np.random.randint(0, seg, size=num_frames)
    frame_indicies = frame_seg + offset
    return frame_indicies


def random_seg(frame_len, num_frames, seed=None):
    """
    随机采样
    :param frame_len:
    :param num_frames:
    :return:
    """
    # frame_indicies = []
    # for i in range(num_frames):
    #     start = frame_indicies[-1]+1 if frame_indicies else i
    #     k = np.random.randint(low=start, high=frame_len-num_frames+i)
    #     frame_indicies.append(k)
    # 设置随机数种子
    if seed:
        np.random.seed(seed)
    frame_indicies = np.random.choice(frame_len, num_frames, replace=False)
    frame_indicies = np.sort(frame_indicies)
    return frame_indicies


def max_pool_seg(frame_len, num_frames):
    """
    Temporal Aggregate Representation
    每一段的视频帧 池化为一帧
    :param frame_len:
    :param num_frames:
    :return:
    """
    frame_count = frame_len
    seg = frame_count // num_frames
    frame_seg = np.arange(0, seg * num_frames, step=seg)
    frame_seg = np.insert(frame_seg, len(frame_seg), values=frame_len)
    return frame_seg


def avg_seg(frame_len, num_frames, seed=None):
    """
    均匀采样
    :param frame_len:
    :param num_frames:
    :return:
    """
    frame_count = frame_len
    seg = frame_count // num_frames
    if seed:
        np.random.seed(seed)
    start = np.random.randint(0, seg, 1)
    frame_indicies = np.arange(start, seg * num_frames + start, seg)
    return frame_indicies


# 返回排序后的tensor
def pool_seg(frames, num_frames, seg_method, seed=None):
    """

    :param frames: tensor:Tensor
    :param num_frames: number:int
    :param seg_method: ["tsn", "avg", "random", "max_pool", "mean_pool", ]
    :return: tensor
    """
    assert seg_method in ["tsn", "avg", "random", "max_pool", "mean_pool", ], print(seg_method + " not support!")
    frame_count = frames.shape[0]
    if seg_method == "avg":
        # 均匀采样
        frame_indicies = avg_seg(frame_count, num_frames, seed)
    elif seg_method == "tsn":
        # tsn 采样
        frame_indicies = tsn_seg(frame_count, num_frames, seed)
    elif seg_method == "random":
        frame_indicies = random_seg(frame_count, num_frames, seed)
    else:
        frame_indicies = max_pool_seg(frame_count, num_frames)
        f_list = []
        for i in range(len(frame_indicies) - 1):
            j = i + 1
            start = frame_indicies[i]
            end = frame_indicies[j]
            if seg_method == "max_pool":
                f_list.append(torch.max(frames[start:end], dim=0).values)
            elif seg_method == "mean_pool":
                f_list.append(torch.mean(frames[start:end], dim=0))
        return_frames = torch.stack(f_list)
        return return_frames

    f_list = []
    for i in frame_indicies:
        f_list.append(frames[i])

    return_frames = torch.stack(f_list).squeeze(0)
    return return_frames


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # model = CNN_TRX(backbone="pool")
    # print(model)
    #
    # x = torch.rand(16, 3, 224, 224)
    # y = model(x)
    # print(y.max())
    # print(y.min())
    # print(y.mean())
    # print(y.shape)

    a = tsn_seg(40, 8, seed=1)
    print(a)
    print(len(a))
