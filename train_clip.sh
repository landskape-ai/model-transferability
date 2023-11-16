#!/bin/bash
Download the uncompressed,2x and 4x compressed checkpoints from the [Upop (Retrieval-clip-coco)](https://github.com/sdc17/UPop/tree/main#-image-text-and-text-image-retrieval-on-the-coco-dataset-with-clip)
wandb online
python3 experiments/ilm_vp_clip.py \
        --sparse_chkpt /clip_chkpts/clip_large_retrieval_coco_2x_compressed.pth \
        --seed 0 \
        --use_wandb \
        --batch_size 16 \
        --dataset cifar10 \
        --results_path /clip_chkpts/ilm_vp_clip \
        --run_name cifar10_clip_ilm_vp_compressed_2x
