import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from myutils.read_txt import read


def fun3():
    plt.figure()
    x = np.arange(0, len(seg_methods))

    width = 0.5
    move = width / 2
    plt.bar(x, mean_seg_point, width=width, color="#87CEFA", edgecolor="black", label="distance")
    plt.plot(x, test_top1, color="red", marker="o", linestyle="--", label="top1")
    plt.plot(x, test_top5, color="green", marker="x", linestyle="--", label="top5")

    plt.xlabel('seg methods')
    plt.xticks(x, seg_methods)

    plt.ylim([0, 1.1])
    plt.legend()
    plt.title(f"Mean Accuracy on ALL Class")
    plt.show()
    plt.close()


def fun2():
    if not os.path.exists("draw_result"):
        os.makedirs("draw_result")
    for i in range(class_number):
        # if i > 0:
        #     break
        plt.figure()
        mid1 = [top1[j].squeeze(0)[i].item() for j in range(len(seg_methods))]
        mid5 = [top5[j].squeeze(0)[i].item() for j in range(len(seg_methods))]
        x = np.arange(0, len(seg_methods))
        dis = class_seg_Point[i]

        width = 0.5
        move = width / 2
        plt.bar(x, dis, width=width, color="#87CEFA", edgecolor="black", label="distance")
        plt.plot(x, mid1, color="red", marker="o", linestyle="--", label="top1")
        plt.plot(x, mid5, color="green", marker="x", linestyle="--", label="top5")

        plt.xlabel('seg methods')
        plt.xticks(x, seg_methods)

        plt.ylim([0, 1.1])
        plt.legend()
        plt.title(f"Accuracy on {ClassIdMap[i]} class")
        # plt.show()
        plt.savefig(f"draw_result/{ClassIdMap[i]}.png")
        plt.close()


def fun1():
    for i in range(len(path_list)):
        if i > 0:
            break
        plt.figure()
        mid1 = top1[i].squeeze(0)
        mid5 = top5[i].squeeze(0)
        x = np.arange(0, len(mid1))
        dis = class_seg_Point[:, i]

        width = 0.5
        move = width / 2
        plt.bar(x, dis, width=width, color="#87CEFA", edgecolor="black", label="distance")
        plt.plot(x, mid1, color="red", marker="o", linestyle="--", label="top1")
        # plt.plot(x, mid5, color="green", marker="x", linestyle="--", label="top5")

        plt.xlabel('classes')
        plt.xticks(x, read("../draw/ucf101_classInd.txt"), rotation=90, fontsize=8)

        plt.ylim([0, 1])
        plt.legend()
        plt.show()



top1, top5 = [], []
test_top1, test_top5 = [], []
path_list = []

DatasetDistance = torch.load("../Cluster/c_result/class_seg_point_seed=5.pth")
seg_methods = DatasetDistance["seg_method_list"]
class_seg_Point = DatasetDistance["class_seg_point"]
mean_seg_point = DatasetDistance["mean_seg_point"]

model = "resnet2p1d_50"
ClassIdMap = read("../draw/ucf101_classInd.txt")
class_number = 101

for seg_method in seg_methods:
    # p = f"../experiments_eval/{model}_224_random2{seg_method}/save_dict.pth"
    p = f"../experiments_eval_seed_5/{model}_224_random2{seg_method}/save_dict.pth"
    assert os.path.exists(p), print(p + " not exist!")
    path_list.append(p)

for i in path_list:
    k = torch.load(i)
    top1.append(k["class_top1"])
    top5.append(k["class_top5"])
    test_top1.append(k["test_top1"])
    test_top5.append(k["test_top5"])

fun3()
#####################################

