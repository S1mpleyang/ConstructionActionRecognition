# ConstructionActionRecognition

This is repository for the ["Transformer-based deep learning model and video dataset for unsafe action identification in construction projects"](https://www.sciencedirect.com/science/article/pii/S0926580522005738).

This repository is based on 

[kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch.git) --> (R3D, R2+1D), 

[rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models.git) --> (ViT, SwinT),

[mmaction2](https://github.com/open-mmlab/mmaction2.git) --> (SlowFast, TSN, TSM),

[m-bain/video-transformers](https://github.com/m-bain/video-transformers) --> (TimeSformer),

[mx-mark/VideoTransformer-pytorch](https://github.com/mx-mark/VideoTransformer-pytorch) --> (ViViT),

Thanks for these researchers!


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

If you are interested in this dataset, please send email to yangmeng@siat.ac.cn and Cc to zl.yang@siat.ac.cn with your institution email.
Note that the Construction Meta Action (CMA) is built only for research,please do not share it with anyone or use it for commercial purposes.

## modify the path to the dataset
Then modify the configuration.py to make sure the path to your dataset is correct.
```
cfg.data_folder =   # location to $myaction$
cfg.train_split =   # location to train split
cfg.test_split =    # location to test split
cfg.dataset_path =  # location to $dataset$
cfg.num_classes =   # number of classes
```

# step3
For training, run the script below:
```
python train_script.py
```

For testing, run the script below:
```
python test_script.py
```
The pretrain-weight can be downloaded from [Google disk](https://drive.google.com/file/d/1z5nWkpQxLxXOQn-5K4eQ9riOBWqm-xkz/view?usp=share_link)

# Citation
If you find our work is helpful, please give us a star so that more people can find our Projects. If you use our dataset or code in your work, please cite our paper as below.
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
```

# Contact
Any question please contact yangmeng@siat.ac.cn and zl.yang@siat.ac.cn

