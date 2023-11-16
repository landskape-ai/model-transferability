#!/bin/bash

wandb online
python3 experiments/ilm_vp_clip.py \
        --sparse_chkpt /clip_chkpts/clip_large_retrieval_coco_2x_compressed.pth \
        --seed 0 \
        --use_wandb \
        --batch_size 16 \
        --dataset cifar10 \
        --results_path /clip_chkpts/ilm_vp_clip \
        --run_name cifar10_clip_ilm_vp_compressed_2x
