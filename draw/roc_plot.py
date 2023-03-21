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
    y_test = y_test.numpy()[1:, ].astype(np.int32)
    try:
        y_score = torch.load(predict)
        y_score = y_score.numpy()[1:, ].astype(np.float64)
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


##TSN
plot_roc(
    model_name="TSN",
    label=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\resnet_224_8_0509_newdataset\resnet_label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\tsn_out.npy",
    color="blue",
)

##TSM
plot_roc(
    model_name="TSM",
    label=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\resnet_224_8_0509_newdataset\resnet_label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\tsm_out.npy",
    color="darkorange",
)

##SlowFast
plot_roc(
    model_name="SlowFast",
    label=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\resnet_224_8_0509_newdataset\resnet_label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\slowfast_out.npy",
    color="yellowgreen",
)

##R3D
plot_roc(
    model_name="R3D",
    label=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\resnet_224_8_0509_newdataset\resnet_label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\resnet_224_8_0509_newdataset\resnet_tensor.pth",
    color="gray",
)

##R(2+1)D
plot_roc(
    model_name="R(2+1)D",
    label=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\resnet2p1d_224_8_0509_newdataset\resnet2p1d_label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\resnet2p1d_224_8_0509_newdataset\resnet2p1d_tensor.pth",
    color="purple",
)

##TEA
plot_roc(
    model_name="TEA",
    label=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\tea_224_8_0530\label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\tea_224_8_0530\tensor.pth",
    color="cyan",
)


##ViT-B
plot_roc(
    model_name="ViT-B",
    label=r"D:\yangmeng_workspace\STR-Transformer\exp_0808\ViT\experiments_myaction_8_vitbase_0807\eval\vitbase_224_8_0807\vitbase_label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp_0808\ViT\experiments_myaction_8_vitbase_0807\eval\vitbase_224_8_0807\vitbase_tensor.pth",
    color="g",
)

##ViT-L
plot_roc(
    model_name="ViT-L",
    label=r"D:\yangmeng_workspace\STR-Transformer\exp_0808\ViT\experiments_myaction_8_vitlarge_0807\eval\vitlarge_224_8_0807\vitlarge_label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp_0808\ViT\experiments_myaction_8_vitlarge_0807\eval\vitlarge_224_8_0807\vitlarge_tensor.pth",
    color="g",
    linestyle="--"
)

##SwinV1-B
plot_roc(
    model_name="SwinV1-B",
    label=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\SwinTransformer_AT_224_8_0530\SwinTransformer_AT_label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\experiments_each_class\SwinTransformer_AT_224_8_0530\SwinTransformer_AT_tensor.pth",
    color="red",
)

##SwinV1-L
plot_roc(
    model_name="SwinV1-L",
    label=r"D:\yangmeng_workspace\STR-Transformer\exp_0808\swinViT\eval\swinVlarge_224_8_0809\swinVlarge_label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp_0808\swinViT\eval\swinVlarge_224_8_0809\swinVlarge_tensor.pth",
    color="red",
    linestyle="--"
)

# ##Video Swin
# plot_roc(
#     model_name="Video Swin",
#     label= r"D:\yangmeng_workspace\STR-Transformer\exp_0808\videoswin\eval\videoswin_224_8_0807\videoswin_label.pth",
#     predict=r"D:\yangmeng_workspace\STR-Transformer\exp_0808\videoswin\eval\videoswin_224_8_0807\videoswin_tensor.pth",
#     color="sienna",
# )

##MViT
plot_roc(
    model_name="MViT",
    label= r"D:\yangmeng_workspace\STR-Transformer\exp_0808\MViT\eval\mvitbase_224_8_0807\mvitbase_label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp_0808\MViT\eval\mvitbase_224_8_0807\mvitbase_tensor.pth",
    color="sienna",
)

plot_roc(
    model_name="TimeSformer",
    label=r"D:\yangmeng_workspace\STR-Transformer\exp_0808\Timesformer-B-16\eval\timesformer_224_8_0811\timesformer_label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp_0808\Timesformer-B-16\eval\timesformer_224_8_0811\timesformer_tensor.pth",
    color="m",
    linestyle="-."
)

##ours
plot_roc(
    model_name="STR-Transformer",
    label=r"D:\yangmeng_workspace\STR-Transformer\exp_0808\STR-Trans\eval\STR_Transformer_distance_attention_224_8_0808\STR_Transformer_label.pth",
    predict=r"D:\yangmeng_workspace\STR-Transformer\exp_0808\STR-Trans\eval\STR_Transformer_distance_attention_224_8_0808\STR_Transformer_tensor.pth",
    # label=r"D:\yangmeng_workspace\STR-Transformer\0829\eval\STR_Transformer_distance_attention_224_8_0829\STR_Transformer_label.pth",
    # predict=r"D:\yangmeng_workspace\STR-Transformer\0829\eval\STR_Transformer_distance_attention_224_8_0829\STR_Transformer_tensor.pth",
    color="k",
    linestyle="--"
)


"""end"""

# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
