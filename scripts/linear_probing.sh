#!/bin/bash
#SBATCH --job-name=1shot_baseline
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --output=/home/mila/d/diganta.misra/projects/prune_train/logs/1shot_baseline.out
#SBATCH --error=/home/mila/d/diganta.misra/projects/prune_train/logs/1shot_baseline.err

module load anaconda/3

conda activate /home/mila/d/diganta.misra/.conda/envs/sparse

wandb login bd67cef57b7227730fe3edf96e11d954558a9d0d

ulimit -Sn $(ulimit -Hn)

python3 experiments/cnn/linear_probing.py \
        --network LT \
        --dataset cifar10 \
        --pretrained_dir /data/jaygala/ILM-VP/artifacts/ImageNetCheckpoint_LT \
        --results_path /data/jaygala/ILM-VP/results \
        --sparsity 9 \
        --train_data_fraction 0.5 \
        --wandb \
        --run_name linear_probing \
