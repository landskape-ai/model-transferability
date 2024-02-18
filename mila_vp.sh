#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH --job-name=VP_ViT_C10
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --reservation=ubuntu1804
#SBATCH --signal=SIGUSR1@90 # 90 seconds before time limit
#SBATCH --output=$SCRATCH/vpvit-%j.out
#SBATCH --error=$SCRATCH/vpvit-%j.err

pyfile=/home/mila/d/diganta.misra/projects/model-transferability/experiments/cnn/ilm_vp.py

module load anaconda/3

conda activate /home/mila/d/diganta.misra/.conda/envs/sparse

wandb login bd67cef57b7227730fe3edf96e11d954558a9d0d

ulimit -Sn $(ulimit -Hn)

wandb online

WANDB_CACHE_DIR=$SCRATCH

python3 $pyfile \
    --model $1 \
    --n_shot $2 \
    --seed $3 \
    --batch_size $4  \
    --dataset $5 \
    --results_path $6 \
    --wandb

