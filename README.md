# Exploit Domain-robust Optical Flow in Domain Adaptive Video Semantic Segmentation （AAAI 2023 Oral）
This repository is released that can reproduce the main results (our proposed SFC) of the experiment on VIPER to Cityscapes-Seq.  Experiments on the SYNTHIA-Seq to Cityscapes-Seq can be easily implemented by slightly modifying the dataset and setting.

## Paper
![image](https://github.com/EdenHazardan/SFC/blob/master/SFC.PNG)
Exploit domain-robust optical flow in domain adaptive video semantic segmentation.
Yuan Gao, Zilei Wang, Jiafan Zhuang, Yixin, Zhang and Junjie Li. 
University of Science and Technology of China.

Proceedings of the AAAI Conference on Artificial Intelligence(AAAI),  2023(Oral)

If you find this code useful for your research, please cite our paper:

```
@inproceedings{gao2023exploit,
  title={Exploit domain-robust optical flow in domain adaptive video semantic segmentation},
  author={Gao, Yuan and Wang, Zilei and Zhuang, Jiafan and Zhang, Yixin and Li, Junjie},
  booktitle={AAAI},
  year={2023}
}

```

## Abstract
Domain adaptive semantic segmentation aims to exploit the pixel-level annotated samples on source domain to assist the segmentation of unlabeled samples on target domain. For such a task, the key is to construct reliable supervision signals on target domain. However, existing methods can only pro- vide unreliable supervision signals constructed by segmen- tation model (SegNet) that are generally domain-sensitive. In this work, we try to find a domain-robust clue to con- struct more reliable supervision signals. Particularly, we ex- perimentally observe the domain-robustness of optical flow in video tasks as it mainly represents the motion characteris- tics of scenes. However, optical flow cannot be directly used as supervision signals of semantic segmentation since both of them essentially represent different information. To tackle this issue, we first propose a novel Segmentation-to-Flow Module (SFM) that converts semantic segmentation maps to optical flows, named the segmentation-based flow (SF), and then propose a Segmentation-based Flow Consistency (SFC) method to impose consistency between SF and optical flow, which can implicitly supervise the training of segmentation model. The extensive experiments on two challenging bench- marks demonstrate the effectiveness of our method, and it outperforms previous state-of-the-art methods with consider- able performance improvement.

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

