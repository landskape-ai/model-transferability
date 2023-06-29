#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 8
#SBATCH --time=05:00:00
#SBATCH --partition=long
#SBATCH --job-name=lth_imagenet_reprogram
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -o  /home/mila/m/mai-thi.ho/scratch/Reprogram.txt
#SBATCH --no-requeue


module load anaconda/3

conda activate /home/mila/m/mai-thi.ho/.conda/envs/ssl

wandb login aa51b1dce7dcffa818a66449cfa00197dab68bfe
nvidia-smi


python experiments/cnn/rlm_vp.py --network LT --dataset cifar10
