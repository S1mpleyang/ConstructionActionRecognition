sum = [1595, 1055, 540]
climb = [140, 95, 45]
fall_domn = [315, 215, 100]
smoke = [365, 215, 150]
stand = [315, 215, 100]
talk = [160, 115, 45]
walk = [195, 140, 55]
wear_helmet = [105, 60, 45]

import matplotlib.pyplot as plt
import numpy as np


# 显示中文
from pylab import *
mpl.rcParams['font.sans-serif'] = ["SimHei"] #["Microsoft YaHei"]

x = np.array([ 315, 365, 315, 140, 160, 105, 195])
labels = ["摔倒", "吸烟", "站立", "爬梯", "交谈", "戴头盔", "行走"]
# labels = ["Fall Down", "Smoke", "Stand", "Climb Ladder", "Talk", "Wear Helmet", "Walk"]
colors = [ "#ffd965", "#e5b9b5", "#00b0f0", "#00ffff", "#ab9ac0", "#00ff99", "#ff3398"]

plt.figure()
plt.pie(
    x,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    textprops={"fontsize": 14},
)
plt.show()
