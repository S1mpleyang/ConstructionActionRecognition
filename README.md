# ConstructionActionRecognition

This is repository for the "Transformer-based deep learning model and video dataset for unsafe action identification in construction projects".

This repository is based on 

[kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch.git) --> (R3D, R2+1D), 

[rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models.git) --> (ViT, SwinT),

[mmaction2](https://github.com/open-mmlab/mmaction2.git) --> (SlowFast, TSN, TSM),

[m-bain/video-transformers](https://github.com/m-bain/video-transformers) --> (TimeSformer),

[mx-mark/VideoTransformer-pytorch](https://github.com/mx-mark/VideoTransformer-pytorch) --> (ViViT),


# Code

## step1
Create virtual environment
```
conda create -n STRT python=3.6
conda activate STRT
pip install -r requirements.txt
```

## step2 
Download pretrain-weight from [Baidu disk](https://pan.baidu.com/s/15qpLsPcBtyY4oc7Mzg_4LQ)

Download dataset then modify the configuration.py to make sure the path to your dataset is correct.

## step3
### For training

Run the script below:
```
python train.py
```

### For testing
Run the script below:
```
python test_script.py
```


# Dataset

The Construction Meta Action (CMA) Dataset is built only for research, if you are interested in this dataset, please contact yangmeng@siat.ac.cn and zl.yang@siat.ac.cn

