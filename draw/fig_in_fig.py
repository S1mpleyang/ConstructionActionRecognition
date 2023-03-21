from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import re
import matplotlib.patches as patches


def read_log(file):
    """
    读取日志文件,进行数据重组,写入mysql
    :return:
    """
    with open(file) as f:
        list1, list2, list3 = [], [], []
        data = f.readlines()
        for i in range(len(data)):
            # data[i] = re.sub('\s'," ", data[i])
            if i > 0:
                data[i] = data[i].strip().split("\t")
                if len(data[i]) > 3:
                    list1.append(float(data[i][0]))
                    list2.append(float(data[i][1]))
                    list3.append(float(data[i][2]))

        return list1, list2, list3


"""
绘制损失曲线
"""

label = ["R3D", "R(2+1)D", "TEA", "ViT-B", "ViT-L", "SwinV1-B", "SwinV1-L", "MViT", "TimeSformer", "STR-Transformer"]
color = ["gray", "purple", "cyan", "g", "g", "red", "red", "sienna", "m", "k"]
linestyle = [":", ":", ":", ":", "--", ":", "--", ":", "-.", ":"]
filelist = [
    r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\resnet_224_8_0530\events.out.tfevents.1652138266.DESKTOP-AHAEOLU.23572.0",
    r"D:\yangmeng_workspace\STR-Transformer\exp\experiments_myaction_8frame_results_0509_newdataset\resnet2p1d_224_8_0509_newdataset\events.out.tfevents.1652122066.DESKTOP-AHAEOLU.24100.0",
    r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\tea_224_8_0530\events.out.tfevents.1653884472.DESKTOP-AHAEOLU.5428.0",
    r"D:\yangmeng_workspace\STR-Transformer\exp_0811_loss\vitbase\vitbase_224_8_0807\train_vitbase.log",
    r"D:\yangmeng_workspace\STR-Transformer\exp_0811_loss\vitlarge\vitlarge_224_8_0807\train_vitlarge.log",
    r"D:\yangmeng_workspace\STR-Transformer\exp_0811_loss/swinVbase/swinVbase_224_8_0807/train_swinVbase.log",
    r"D:\yangmeng_workspace\STR-Transformer\exp_0811_loss/swinVlarge/swinVlarge_224_8_0807/train_swinVlarge.log",
    r"D:\yangmeng_workspace\STR-Transformer\exp_0808/MViT/mvitbase_224_8_0807/train_mvitbase.log",
    r"D:\yangmeng_workspace\STR-Transformer\exp_0808\Timesformer-B-16\timesformer_224_8_0811\train_timesformer.log",
    r"D:\yangmeng_workspace\STR-Transformer\exp_0811_loss/STR-Trans/STR_Transformer_distance_attention_224_8_0808/train_STR_Transformer.log",
]

# label = ["R(2+1)D", "SwimTransformer", "STR-Transformer", "R3D", "TEA"]
# color = ["purple", "red", "green", "gray", "cyan"]
#
# filelist = [
#     r"D:\yangmeng_workspace\STR-Transformer\exp\experiments_myaction_8frame_results_0509_newdataset\resnet2p1d_224_8_0509_newdataset\events.out.tfevents.1652122066.DESKTOP-AHAEOLU.24100.0",
#     # 原始的
#     # r"D:\yangmeng_workspace\ym-action-recognize-v2\exp0515\experiments_myaction_8frame_results_0509_newdataset\SwinTransformer_AT_moblenet_no_attention_224_8_0509_newdataset\events.out.tfevents.1652154197.DESKTOP-AHAEOLU.13540.0",
#     r"D:\yangmeng_workspace\STR-Transformer\exp\experiments_myaction_8_STR_Transformer/STR_Transformer_no_attention_224_8_0604\events.out.tfevents.1654329350.DESKTOP-AHAEOLU.9884.0",
#     r"D:\yangmeng_workspace\STR-Transformer\exp0521\experiments_myaction_8_SwinTransformer_R2plus1d\SwinTransformer_R2plus1d_distance_attention_224_8_0515\events.out.tfevents.1652519491.DESKTOP-AHAEOLU.11376.0",
#     r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\resnet_224_8_0530\events.out.tfevents.1652138266.DESKTOP-AHAEOLU.23572.0",
#     r"D:\yangmeng_workspace\STR-Transformer\exp0531\experiments_myaction_8frame_results_0530\tea_224_8_0530\events.out.tfevents.1653884472.DESKTOP-AHAEOLU.5428.0",
# ]

all_train_loss_epoch, all_train_loss_value = [], []
all_train_acc_value = []
all_test_acc_epoch, all_test_acc_value = [], []

for path in filelist:
    try:
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
        # print(ea.scalars.Keys())
        train_loss = ea.scalars.Items("train/loss")
        train_loss_epoch, train_loss_value = [], []
        for item in train_loss:
            if item.step <= 25:
                train_loss_epoch.append(item.step)
                train_loss_value.append(item.value)

        train_acc = ea.scalars.Items("train/train_top1")
        train_acc_value = []
        for item in train_acc:
            if item.step <= 25:
                train_acc_value.append(item.value)

        test_acc = ea.scalars.Items("test/test_top1")
        test_acc_epoch, test_acc_value = [], []
        for item in test_acc:
            if item.step <= 25:
                test_acc_epoch.append(item.step)
                test_acc_value.append(item.value)

        # add to list
        all_train_loss_epoch.append(train_loss_epoch)
        all_train_loss_value.append(train_loss_value)
        all_train_acc_value.append(train_acc_value)

        all_test_acc_epoch.append(test_acc_epoch)
        all_test_acc_value.append(test_acc_value)
    except:
        train_loss_epoch, train_loss_value, train_acc_value = read_log(path)
        all_train_loss_epoch.append(train_loss_epoch)
        all_train_loss_value.append(train_loss_value)
        all_train_acc_value.append(train_acc_value)
        # print(path + " error!")
        # continue

# plot
# UCF101=120epoch  HMDB51=150epoch
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.xlabel("epoch")
# plt.ylabel("Loss")
# for i in range(len(filelist)):
#     plt.plot(all_train_loss_epoch[0], all_train_loss_value[i], label=label[i], color=color[i], linestyle=linestyle[i])
# plt.legend()
# plt.grid()
#
# plt.subplot(1, 2, 2)
# plt.xlabel("epoch")
# plt.ylabel("Loss")
# for i in range(3, len(filelist)):
#     if i == 7:
#         continue
#     plt.plot(all_train_loss_epoch[0][21:-1], all_train_loss_value[i][21:-1], label=label[i], color=color[i],
#              linestyle=linestyle[i])
# plt.grid()


# define figure
fig = plt.figure()

left, bottom, width, height = 0.1, 0.1, 0.8, 0.8

ax1 = fig.add_axes([left, bottom, width, height])
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
for i in range(len(filelist)):
    ax1.plot(all_train_loss_epoch[0], all_train_loss_value[i], label=label[i], color=color[i], linestyle=linestyle[i],
             linewidth=2)
    print(label[i], all_train_loss_value[i][0], all_train_loss_value[i][-2])

ax1.legend(loc='upper right')
# ax1.set_title('Loss during training')

# Method 1
left, bottom, width, height = 0.6, 0.3, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
for i in range(3, len(filelist)):
    if i == 7:
        continue
    ax2.plot(all_train_loss_epoch[0][21:-1], all_train_loss_value[i][21:-1], label=label[i], color=color[i],
             linestyle=linestyle[i], linewidth=2)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
# ax2.set_xticks([])
ax2.set_yticks([0, 2.5e-5, 5e-5, 7.5e-5, 1e-4])
ax2.set_yticklabels(["0", "2.5e-5", "5e-5", "7.5e-5", "1e-4"])

# ax1.add_patch(
#     patches.Rectangle(
#         (0.1, 0.1),   # (x,y)
#         0.5,          # width
#         0.5,          # height
#     )
# )


plt.show()
