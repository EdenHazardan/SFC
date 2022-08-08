#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/gaoy/miniconda3/envs/davss/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:/home/gaoy/SFC

cd /home/gaoy/SFC && \
python3 exp/SFC/stage_two/python/train.py \
        --exp_name stage_two \
        --weight_update /home/gaoy/SFC/save_results/SFC/stage_one/SFC_one_stage_update.pth \
        --weight_reference /home/gaoy/SFC/save_results/SFC/stage_one/SFC_one_stage_reference.pth \
        --weight_flownet /home/gaoy/SFC/model_weights/flownet_flyingchairs_pretrained.pth \
        --weight_flownet_global /home/gaoy/SFC/save_results/FlowNet_pretrain/stage_two/flownet_source_well_trained.pth \
        --weight_SFM /home/gaoy/SFC/save_results/SFM/SFM_source_well_trained.pth \
        --lr 0.0005 \
        --distance 2 \
        --loss_flow 0.001 \
        --source_batch_size 1 \
        --target_batch_size 1 \
        --train_num_workers 2 \
        --test_batch_size 1 \
        --test_num_workers 2 \
        --train_iterations 160000 \
        --log_interval 100 \
        --val_interval 2000 \
        --work_dirs /home/gaoy/SFC/save_results/SFC \