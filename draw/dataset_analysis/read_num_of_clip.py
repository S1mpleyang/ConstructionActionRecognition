import numpy as np
import matplotlib.pyplot as plt

# 显示中文
from pylab import *
mpl.rcParams['font.sans-serif'] = ["SimHei"]


k = np.load("outdict.npy")
print(k)

label = ["0.0 - 2.0 秒", "2.0 - 4.0 秒", "4.0 - 6.0 秒", "6.0 - 8.0 秒", "> 8.0 秒"]
# label = ["0.0 - 2.0 Sec", "2.0 - 4.0 Sec", "4.0 - 6.0 Sec", "6.0 - 8.0 Sec", "> 8.0 Sec"]
m = np.zeros(shape=(7, 5))
m[:, 0] = k[:, 0] + k[:, 1]
m[:, 1] = k[:, 2] + k[:, 3] + m[:, 0]
m[:, 2] = k[:, 4] + k[:, 5] + m[:, 1]
m[:, 3] = k[:, 6] + k[:, 7] + m[:, 2]
m[:, 4] = k[:, 8] + k[:, 9] + k[:, 10] + m[:, 3]
print(m)

plt.figure()
x = [1, 2, 3, 4, 5, 6, 7]
plt.ylabel("视频片段数目")
plt.xticks(x, ["摔倒", "吸烟", "站立", "爬梯", "交谈", "戴头盔", "行走"], rotation="0")

# plt.ylabel("Number of Clips")
# plt.xticks(x, ["Climb\n Ladder", "Fall\n Down", "Smoke", "Stand", "Talk", "Walk", "Wear\n Helmet"], rotation="0")

plt.bar(
    x,
    height=m[:, 0],
    label=label[0],
    width=0.7
)
plt.bar(x, height=m[:, 1] - m[:, 0], bottom=m[:, 0], label=label[1],width=0.7, color="#ffd965")
plt.bar(x, height=m[:, 2] - m[:, 1], bottom=m[:, 1], label=label[2],width=0.7, color="#e5b9b5")
plt.bar(x, height=m[:, 3] - m[:, 2], bottom=m[:, 2], label=label[3],width=0.7, color="#00b0b0")
plt.bar(x, height=m[:, 4] - m[:, 3], bottom=m[:, 3], label=label[4],width=0.7, color="#aa00aa")
plt.ylim(0, 400)
plt.legend()
plt.grid(axis="y", linestyle=":")
plt.show()

print(k.sum(axis=1))
