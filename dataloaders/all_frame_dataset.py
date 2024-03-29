import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
import random
import numpy as np
from tqdm import tqdm
import time
from Cluster.cluster import tsn_seg, avg_seg, random_seg, max_pool_seg
import configuration
import torch.nn.functional as F


def resize(frames, size, interpolation='bicubic'):
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(frames.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(frames, size=size, scale_factor=scale, mode=interpolation,
                                           align_corners=False)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


def to_normalized_float_tensor(vid):
    # return vid.permute(3, 0, 1, 2).to(torch.float32) / 255
    return vid.permute(3, 0, 1, 2)


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


def normalize(vid, mean, std):
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class TSNdataset(Dataset):
    def __init__(self, cfg, data_split, data_percentage, num_frames, seg_method, input_size, shuffle=False, seed=None):

        self.num_classes = cfg.num_classes
        self.data_folder = cfg.data_folder
        self.data_split = data_split
        self.seed = seed
        if data_split == "train":
            self.train_split = cfg.train_split
        elif data_split == "val":
            self.train_split = cfg.val_split
        else:
            self.train_split = cfg.test_split
        self.seg_method = seg_method
        assert self.seg_method in ["tsn", "avg", "max_pool", "mean_pool", "all", "random"], print(f"use {self.seg_method} seg method")

        # 获取视频数据列表
        with open(self.train_split, 'r') as f:
            self.video_data = f.readlines()
        for i in range(len(self.video_data)):
            self.video_data[i] = self.video_data[i].strip().split(" ")

        if shuffle:
            random.shuffle(self.video_data)
        len_data = int(len(self.video_data) * data_percentage)
        self.video_ids = self.video_data[0:len_data]
        self.num_frames = num_frames
        self.input_size = input_size
        self.resize = Resize((self.input_size, self.input_size))
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 是否需要归一化
        # self.transform = transforms.Compose([ToFloatTensorInZeroOne(), self.resize, self.normalize])
        self.transform = transforms.Compose(
            [
                ToFloatTensorInZeroOne(),
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224)
            ]
        )


    def __len__(self):
        return len(self.video_data)

    def load_all_frames(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret = True
        frames = []
        while ret:
            ret, frame = vidcap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        vidcap.release()
        frame_count = len(frames)
        frames = torch.from_numpy(np.stack(frames))
        return frames

    def tsn_seg(self, frames, num_frames):
        frame_count = frames.shape[0]
        if self.seg_method == "avg":
            # 均匀采样
            frame_indicies = avg_seg(frame_count, num_frames, seed=self.seed)
        elif self.seg_method == "tsn":
            # tsn 采样
            frame_indicies = tsn_seg(frame_count, num_frames, seed=self.seed)
        elif self.seg_method == "random":
            frame_indicies = random_seg(frame_count, num_frames, seed=self.seed)

        f_list = []
        for i in frame_indicies:
            f_list.append(frames[i])

        return_frames = torch.stack(f_list).squeeze(0)
        return return_frames

    def pool_seg(self, frames, num_frames):
        frame_count = frames.shape[0]
        frame_indicies = max_pool_seg(frame_count, num_frames)
        f_list = []
        for i in range(len(frame_indicies) - 1):
            j = i + 1
            start = frame_indicies[i]
            end = frame_indicies[j]
            if self.seg_method == "max_pool":
                f_list.append(torch.max(frames[start:end], dim=0).values)
            elif self.seg_method == "mean_pool":
                f_list.append(torch.mean(frames[start:end], dim=0))
        return_frames = torch.stack(f_list)
        return return_frames

    def build_tsn_clip(self, video_path):
        frames = self.load_all_frames(video_path)
        frames = frames.to(torch.float32) / 255
        # print("frames.shape:", frames.shape)
        frames = self.tsn_seg(frames, self.num_frames)
        # print("frames.shape:", frames.shape)
        frames = self.transform(frames)
        return frames

    def build_pool_clip(self, video_path):
        frames = self.load_all_frames(video_path)
        frames = frames.to(torch.float32) / 255
        frames = self.pool_seg(frames, self.num_frames)
        frames = self.transform(frames)
        return frames

    def build_all_clip(self, video_path):
        frames = self.load_all_frames(video_path)
        frames = frames.to(torch.float32) / 255
        # frames = self.pool_seg(frames, self.num_frames)
        frames = self.transform(frames)
        return frames

    def build_consecutive_clips(self, video_path):
        frames = self.load_all_frames(video_path)
        if len(frames) % self.num_frames != 0:
            frames = frames[:len(frames) - (len(frames) % self.num_frames)]
        clips = torch.stack([self.transform(x) for x in chunks(frames, self.num_frames)])
        return clips

    def __getitem__(self, index):
        """

        :param index:
        :return:  clips [bs, c, T, h, w] , label: np.zeros(num_classes)
        """
        data_path = self.data_folder
        video_path, video_class = self.video_data[index]
        if self.seg_method in ["tsn", "avg", "random"]:
            clips = self.build_tsn_clip(os.path.join(data_path, video_path))
        elif self.seg_method == "all":
            clips = self.build_all_clip(os.path.join(data_path, video_path))
            return clips, int(video_class)
        else:
            clips = self.build_pool_clip(os.path.join(data_path, video_path))

        label = np.zeros(self.num_classes)
        label[int(video_class)] = 1
        return clips, label


if __name__ == '__main__':
    shuffle = False
    batch_size = 2

    dataset = 'myaction'
    cfg = configuration.build_config(dataset)
    data_generator = TSNdataset(cfg, 'train', 1.0, num_frames=8, seg_method="random", input_size=224)
    dataloader = DataLoader(data_generator, batch_size, shuffle=shuffle, num_workers=0)

    start = time.time()

    for epoch in range(0, 1):
        for i, (clips, labels) in enumerate(tqdm(dataloader)):
            print(clips.shape)
            print(labels.shape)
            break
    print("time taken : ", time.time() - start)
