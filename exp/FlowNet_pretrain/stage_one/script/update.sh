#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/gaoy/miniconda3/envs/davss/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:/home/gaoy/SFC

cd /home/gaoy/SFC && \
python3 exp/FlowNet_pretrain/stage_one/python/train.py \
        --exp_name stage_one \
        --weight_deeplabv2_res101 /home/gaoy/SFC/model_weights/DeepLab_resnet_pretrained_init-f81d91e8.pth \
        --model_name update \
        --lr 2.5e-4 \
        --distance 2 \
        --source_batch_size 1 \
        --train_num_workers 2 \
        --test_batch_size 1 \
        --test_num_workers 2 \
        --train_iterations 80000 \
        --log_interval 100 \
        --val_interval 2000 \
        --work_dirs /home/gaoy/SFC/save_results/FlowNet_pretrain \