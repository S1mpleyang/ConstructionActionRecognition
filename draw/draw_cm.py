import matplotlib.pyplot as plt
import numpy as np

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='混淆矩阵', cmap=plt.cm.Oranges):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    x = np.arange(len(classes))
    plt.xticks(x, classes, rotation=45)
    plt.yticks(x, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    for x in range(7):
        for y in range(7):
            # info = float(format('%0.2f' % cm[y][x]))
            info = format('%0.2f' %  cm[y][x])
            if x == y:
                plt.text(x, y, info, verticalalignment='center', horizontalalignment='center', color="white")
            else:
                plt.text(x, y, info, verticalalignment='center', horizontalalignment='center')
    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')

    #plt.title("Normalized confusion matrix of baseline method. \n average precision: 84.4%")
    # plt.savefig('./test.pdf')
    plt.show()


from pylab import *
mpl.rcParams['font.sans-serif'] = ["SimHei"]

path = "STR-Transformer"  # path to your file
with open(path, 'r') as f:
    data = f.readlines()

L = []
for i in range(len(data)):
    data[i] = data[i].strip().split("-->")
    a, b = data[i]
    a = a.split(":")[1]
    b = b.split(":")[1]
    L.append([a, b])

ConfuseMatrix = np.zeros(shape=(7, 7))
for a, b in L:
    ConfuseMatrix[int(a)][int(b)] += 1

plot_confusion_matrix(
    ConfuseMatrix,
    # ["Climb Ladder", "Fall Down", "Smoke", "Stand", "Talk", "Walk", "Wear Helmet"],
    ["摔倒", "吸烟", "站立", "爬梯", "交谈", "戴头盔", "行走"],
    normalize=True
)

