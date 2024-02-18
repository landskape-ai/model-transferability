#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model=$1    # check `--model` choices in `experiments/cnn/ilm_vp.py` script
dataset=$2  # check `--dataset` choices in `experiments/cnn/ilm_vp.py` script
n_shots=(-1 1 10 50 100)
seeds=(1 2 3)

declare -A batch_sizes
batch_sizes=( [-1]=256 [1]=8 [10]=32 [50]=64 [100]=128 )


for n_shot in "${n_shots[@]}"; do
    for seed in "${seeds[@]}"; do
        batch_size=${batch_sizes[$n_shot]}
    
        if [ -z "$batch_size" ]; then
            echo "Unsupported n_shot value: $n_shot"
            exit 1
        fi

        folder_name=f"/home/mila/d/diganta.misra/scratch/mamvit/lp/${model}_${n_shot}_${seed}_${batch_size}_${dataset}"

        # ---------------------------------------------------------------------------
        #                                LP experiment
        # ---------------------------------------------------------------------------
        sbatch /home/mila/d/diganta.misra/projects/model-transferability/mila_lp.sh $model $n_shot $seed $batch_size $dataset $folder_name


    done
done
