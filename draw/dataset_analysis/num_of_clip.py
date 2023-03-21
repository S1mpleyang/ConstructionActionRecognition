import os
import cv2
import numpy as np

def video_time(file_path):
    cap = cv2.VideoCapture(file_path)
    # file_path是文件的绝对路径，防止路径中含有中文时报错，需要解码
    if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
        # get方法参数按顺序对应下表（从0开始编号)
        rate = cap.get(5)  # 帧速率
        FrameNumber = cap.get(7)  # 视频文件的帧数
        duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
    return duration


def read(path):
    vp, label = [], []
    with open(path, 'r') as f:
        data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].strip().split(" ")
        vp.append(data[i][0])
        label.append(data[i][1])
    return vp, label


path = r"D:\DATASET\myaction"
txt = "list.txt"
video_p = "ConstructionSafety"
VP, L = read(os.path.join(path, txt))
during = []

during_dict = [[0 for i in range(20)] for i in range(7)]

for i in range(len(VP)):
    item = VP[i]
    lab = int(L[i])
    vpath = os.path.join(path, video_p, item)
    duri = video_time(vpath)
    time_class = int(duri)
    during_dict[lab][time_class]+=1
    during.append(duri)

np.save("outdict", np.array(during_dict))
np.save("out", np.array(during))