import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def fun1():
    top1 = []
    top5 = []
    path_list = []
    seg_methods = ["avg", "tsn", "max_pool", "mean_pool", "random"]
    for seg_method in seg_methods:
        p = f"../experiments_eval/r3d_50_224_{seg_method}/save_dict.pth"
        assert os.path.exists(p), print(p + " not exist!")
        path_list.append(p)

    for i in path_list:
        k = torch.load(i)
        top1.append(k["test_top1"])
        top5.append(k["test_top5"])

    plt.figure()
    x = np.arange(0, len(top1))
    dis = [0.76, 0.65, 0.45, 0.33]

    width = 0.5
    move = width / 2
    plt.bar(x, dis, width=width, color="#87CEFA", edgecolor="black", label="distance")
    plt.plot(x, top1, color="red", marker="o", linestyle="--", label="top1")
    plt.plot(x, top5, color="green", marker="x", linestyle="--", label="top5")

    plt.xlabel('seg methods')
    plt.xticks(x, seg_methods)

    plt.ylim([0, 1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fun1()