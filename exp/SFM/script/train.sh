#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/gaoy/miniconda3/envs/davss/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:/home/gaoy/SFC

cd /home/gaoy/SFC && \
python3 exp/SFM/python/train.py \
        --exp_name SFM \
        --weight_flownet_global /home/gaoy/SFC/save_results/FlowNet_pretrain/stage_two/flownet_source_well_trained.pth \
        --weight_SFM /home/gaoy/SFC/save_results/FlowNet_pretrain/stage_two/flownet_source_well_trained.pth \
        --lr 0.0001 \
        --distance 2 \
        --train_batch_size 1 \
        --train_num_workers 4 \
        --test_batch_size 1 \
        --test_num_workers 2 \
        --train_iterations 80000 \
        --log_interval 100 \
        --val_interval 4000 \
        --work_dirs /home/gaoy/SFC/save_results \