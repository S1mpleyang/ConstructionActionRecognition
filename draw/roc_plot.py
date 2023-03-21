# 引入必要的库
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

import torch
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp




"""
evaluate_and_outthepredict.py 生成 resnet_label.pth， resnet_tensor.pth
roc_plot.py绘图
"""

def plot_roc(
        model_name,
        label,
        predict,
        color,
        linestyle=":",
        linewidth=2,
):
    y_test = torch.load(label)
    y_test = y_test.numpy().astype(np.int32)
    try:
        y_score = torch.load(predict)
        y_score = y_score.numpy().astype(np.float64)
    except:
        y_score = np.load(predict)
        y_score = y_score[0:, ].astype(np.float64)
    n_classes = 7

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve of SwimTransformer(area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='red', linestyle='-', linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"],
             label='{0} (AUC = {1:.3f})'.format(model_name, roc_auc["macro"]),
             color=color, linestyle=linestyle, linewidth=linewidth)


##ours
plot_roc(
    model_name="STR-Transformer",
    label=r"STR_Transformer_label.pth",
    predict=r"STR_Transformer_tensor.pth",
    color="k",
    linestyle="--"
)

"""
STR_Transformer_label.pth 
-size [540,7]
-one-hot code of true label
[1., 0., 0.,  ..., 0., 0., 0.]

STR_Transformer_tensor.pth
-size [540,7]
-real prediction score
[ 8.497159  , -3.47669744, -2.80618858, ..., -7.54730511, -0.45979768, -3.15947509]
"""



"""end"""

# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
