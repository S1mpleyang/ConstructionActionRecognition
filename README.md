# ConstructionActionRecognition

This is repository for the ["Transformer-based deep learning model and video dataset for unsafe action identification in construction projects"](https://www.sciencedirect.com/science/article/pii/S0926580522005738) published in Automation in Construction (JCR Q1, IF=9.6).

This work apply computer vision technology into construction industry for understanding the various actions of workers. In this work, we propose a dataset about actions of construction workers as well as a new designed model.

With these codes, you can reapper the results in our paper, or you can design your own model and train/test it through our codes. 


## update

Our new work ["A teacher–student deep learning strategy for extreme low resolution unsafe action recognition in construction projects"](https://www.sciencedirect.com/science/article/abs/pii/S1474034623004226) is pubilshed in Advanced Engineering Informatics (JCR Q1, IF=8.0). We focus on extramely low resolution (28 pixels * 28 pixels) action recognition task in construction site using extramely low-resolution CMA dataset. You can gain more details in the publish paper.


# Step1
Create virtual environment
```
conda create -n STRT python=3.6
conda activate STRT
pip install -r requirements.txt
```

# Step2 

## Build your own Dataset
You can build your dataset like this:
```
dataset
|-annotations
|    |-classInd.txt
|-myaction
|    |--class_name_0
|    |    |---video_0
|    |    |---video_1
|    |--class_name_1
...

|-train.txt
|-test.txt
```

The classInd.txt save the class name and its ID, like this:
```
0  class_name_0 # ID class_name
1  class_name_1
...

```

train.txt and test.txt save the video path for training and testing, like this:
```
class_name_0/video_0 0  # path ID
class_name_0/video_1 0
...
```

##  Construction Meta Action (CMA) Dataset
Or you can use the Construction Meta Action (CMA) Dataset. CMA dataset defines seven construction worker actions, including 1595 video clips, 
please read our paper for more details.

If you are interested in this dataset, please **leave this repository a star** and **send email** to zl.yang@siat.ac.cn and Cc to ck.wu@siat.ac.cn, yangmengjlu@gmail.com with your institution email.
Note that the Construction Meta Action (CMA) is built only for research,please do not share it with anyone or use it for commercial purposes.

The number of video clips, duration time of each clip are shown below:
![image](https://github.com/S1mpleyang/ConstructionActionRecognition/blob/main/img/numofclip.png)

## Modify the path to the dataset
Then modify the **configuration.py** to make sure the path to your dataset is correct.
```
cfg.data_folder =   # location to $myaction$
cfg.train_split =   # location to train split
cfg.test_split =    # location to test split
cfg.dataset_path =  # location to $dataset$
cfg.num_classes =   # number of classes
```



## Change the backbone model
if you want to apply our training method to train another model (e.g. Resnet, private model)

You should modify line 371 in **main.py** to your own model
```
 /# set model here
model = # your own model
model = model.to(opt.device)
```


# Step3
For training, run the script below:
```
python train_script.py
```

For testing, run the script below:
```
python test_script.py
```
The pretrain-weight can be downloaded from [Google disk](https://drive.google.com/file/d/1z5nWkpQxLxXOQn-5K4eQ9riOBWqm-xkz/view?usp=share_link)

# Evaluation Results
The ROC/AUC curve is shown below:

![image](https://github.com/S1mpleyang/ConstructionActionRecognition/blob/main/img/ROC.png)

Find more detail in **"draw"** directory



# Citation
If you find our work is helpful, please leave us a star and cite our paper below.
```
@article{YANG2023104703,
title = {Transformer-based deep learning model and video dataset for unsafe action identification in construction projects},
journal = {Automation in Construction},
volume = {146},
pages = {104703},
year = {2023},
issn = {0926-5805},
doi = {https://doi.org/10.1016/j.autcon.2022.104703},
url = {https://www.sciencedirect.com/science/article/pii/S0926580522005738},
author = {Meng Yang and Chengke Wu and Yuanjun Guo and Rui Jiang and Feixiang Zhou and Jianlin Zhang and Zhile Yang},
keywords = {Action recognition, Construction safety, Transformer, Deep learning},
}

@article{YANG2024102294,
title = {A teacher–student deep learning strategy for extreme low resolution unsafe action recognition in construction projects},
journal = {Advanced Engineering Informatics},
volume = {59},
pages = {102294},
year = {2024},
issn = {1474-0346},
doi = {https://doi.org/10.1016/j.aei.2023.102294},
url = {https://www.sciencedirect.com/science/article/pii/S1474034623004226},
author = {Meng Yang and Chengke Wu and Yuanjun Guo and Yong He and Rui Jiang and Junjie Jiang and Zhile Yang},
keywords = {Construction safety, Low resolution, Action recognition, Knowledge distillation},
}
```

# Acknowledgement
This repository is based on 

[kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch.git) --> (R3D, R2+1D), 

[rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models.git) --> (ViT, SwinT),

[mmaction2](https://github.com/open-mmlab/mmaction2.git) --> (SlowFast, TSN, TSM),

[m-bain/video-transformers](https://github.com/m-bain/video-transformers) --> (TimeSformer),

[mx-mark/VideoTransformer-pytorch](https://github.com/mx-mark/VideoTransformer-pytorch) --> (ViViT),

Thanks for these researchers!

# Contact
Any question please contact yangmengjlu@gmail.com.

