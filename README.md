# ConstructionActionRecognition

This is repository for the "Transformer-based deep learning model and video dataset for unsafe action identification in construction projects".

This repository is based on 

[kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch.git) --> (R3D, R2+1D), 

[rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models.git) --> (ViT, SwinT),

[mmaction2](https://github.com/open-mmlab/mmaction2.git) --> (SlowFast, TSN, TSM),

[m-bain/video-transformers](https://github.com/m-bain/video-transformers) --> (TimeSformer),

[mx-mark/VideoTransformer-pytorch](https://github.com/mx-mark/VideoTransformer-pytorch) --> (ViViT),



# Step1:create virtual environment
Create virtual environment
```
conda create -n STRT python=3.6
conda activate STRT
pip install -r requirements.txt
```

# Step2 
## Pretained weight
Download pretrain-weight from [Baidu disk](https://pan.baidu.com/s/15qpLsPcBtyY4oc7Mzg_4LQ)

## Dataset
You can build your dataset like this:
```
dataset
-annotations
--classInd.txt
-myaction
--class_name_0
---video_0
---video_1
--class_name_1
...

-train.txt
-test.txt
```

The classInd.txt save the class name and its ID, like this:
```
ID class_name
0  class_name_0
1  class_name_1
...

```

train.txt and test.txt save the video path for training and testing, like this:
```
path ID
class_name_0/video_0 0
class_name_0/video_1 0
...
```


Or you can use the Construction Meta Action (CMA) Dataset.
If you are interested in this dataset, please contact yangmeng@siat.ac.cn and zl.yang@siat.ac.cn.
Note that the Construction Meta Action (CMA) is built only for research,please do not share it with anyone or use it for commercial purposes.

Then modify the configuration.py to make sure the path to your dataset is correct.

# step3
For training, run the script below:
```
python train.py
```

For testing, run the script below:
```
python test_script.py
```

# Contact
Any question please contact yangmeng@siat.ac.cn and zl.yang@siat.ac.cn

