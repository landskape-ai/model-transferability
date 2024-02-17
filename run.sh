export CUDA_VISIBLE_DEVICES=0

model=$1    # check `--model` choices in `experiments/cnn/ilm_vp.py` script
dataset=$2  # check `--dataset` choices in `experiments/cnn/ilm_vp.py` script
n_shots=(1 2 5 10 20 50 100 -1)
seeds=(1 2 3)


for n_shot in "${n_shots[@]}"; do
    for seed in "${seeds[@]}"; do
        # ---------------------------------------------------------------------------
        #                               ILM-VP experiment
        # ---------------------------------------------------------------------------
        python3 experiments/cnn/ilm_vp.py \
            --model $model \
            --n_shot $n_shot \
            --seed $seed \
            --batch_size $dataset  \
            --dataset cifar10 \
            --results_path results \
            --wandb

        # ---------------------------------------------------------------------------
        #                                LP experiment
        # ---------------------------------------------------------------------------
        python3 experiments/cnn/linear_probing.py \
            --model $model \
            --n_shot $n_shot \
            --seed $seed \
            --batch_size 128 \
            --dataset $dataset \
            --results_path results \
            --wandb

    done
done
