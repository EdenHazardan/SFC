# SFC
This repository is released for double-blind submission, which can reproduce the main results (our proposed SFC) of the experiment on VIPER to Cityscapes-Seq.  Experiments on the SYNTHIA-Seq to Cityscapes-Seq can be easily implemented by slightly modifying the dataset and setting.

## Install & Requirements

The code has been tested on pytorch=1.8.0 and python3.8. Please refer to ``requirements.txt`` for detailed information.

### To Install python packages

```
pip install -r requirements.txt
```
## Download Pretrained Weights
We put the weight of flownet pretrained on Flying Chairs in ``SFC/model_weights/flownet_flyingchairs_pretrained.pth``. As for the segmentation model initialization, following DA-VSN, we start with a model pretrained on ImageNet: [Download](http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth)


## Data preparation
You need to download the [VIPER](https://playing-for-benchmarks.org/download/) datasets and [Cityscapes-Seq](https://www.cityscapes-dataset.com/) datasets.

Your directory tree should be look like this:
```
./SFC/data
├── Cityscapes
|  ├── gtFine
|  |  |—— train
|  |  └── val
|  └── leftImg8bit_sequence 
│       ├── train
│       └── val
├── VIPER
|  ├── train
|  |  |—— cls
|  |  └── img
|  └── val 
│       ├── cls
│       └── img
```

## Training 
SFC contains three training phases. The first is to train a FlowNet in source domain. Here we choose to train Accel source-only, which can indirectly train a flownet. The second is to train SFM in source domain. Finally, we use the well-trained SFM and FlowNet to train SFC from scratch.

### FlowNet pretrained
```
# Firstly, we need to train segmentation models source-only in Accel stage one.

# train update baseline model
cd ./SFC/exp/FlowNet_pretrain/stage_one/script/
bash update.sh
# train reference baseline model
bash reference.sh
# the checkpoints in Accel stage one would be saved in ./SFC/save_results/FlowNet_pretrain/stage_one/  

# Then, we use the stage one model to train Accel source-only in stage two
cd ./SFC/exp/FlowNet_pretrain/stage_two/script/
bash train.sh
# the well-trained FlowNet checkpoint would be saved in ./SFC/save_results/FlowNet_pretrain/stage_two/  
```

### SFM pretrained
```
cd ./SFC/exp/SFM/script/
bash train.sh
# the well-trained SFM checkpoint would be saved in ./SFC/save_results/SFM/  
```

### SFC training
```
# Firstly, we need to train segmentation models with SFC in Accel stage one.

# train update baseline model
cd ./SFC/exp/SFC/stage_one/script/
bash update.sh
# train reference baseline model
bash reference.sh
# the checkpoints in Accel stage one would be saved in ./SFC/save_results/SFC/stage_one/  

# Then, we use the stage one model to train Accel with SFC in stage two
cd ./SFC/exp/SFC/stage_two/script/
bash train.sh
```

