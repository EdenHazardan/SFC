#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/gaoy/miniconda3/envs/davss/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:/home/gaoy/SFC

cd /home/gaoy/SFC && \
python3 temporal_semantic_visualization/python/train.py \
        --exp_name temporal_semantic_visualization \
        --weight_update /home/gaoy/SFC/save_results/SFC/stage_one/SFC_one_stage_update.pth \
        --weight_reference /home/gaoy/SFC/save_results/SFC/stage_one/SFC_one_stage_reference.pth \
        --weight_flownet /home/gaoy/SFC/model_weights/flownet_flyingchairs_pretrained.pth \
        --lr 0.0005 \
        --numpy_transform \
        --distance 2 \
        --train_batch_size 4 \
        --train_num_workers 4 \
        --test_batch_size 1 \
        --test_num_workers 2 \
        --train_iterations 40000 \
        --log_interval 50 \
        --val_interval 2000 \
        --work_dirs /data/gaoy/SFC/save_results \