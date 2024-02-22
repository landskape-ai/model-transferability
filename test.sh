export CUDA_VISIBLE_DEVICES=1

# bash run.sh vim_tiny cifar10
# bash run.sh vim_small cifar10
# bash run.sh vssm_tiny cifar10
# bash run.sh vssm_small cifar10
# bash run.sh vssm_base cifar10

python3 experiments/cnn/linear_probing.py --model vim_small --n_shot 100 --seed 2 --batch_size 128 --dataset cifar10 --results_path results --wandb
# python3 experiments/cnn/ilm_vp.py --model vim_tiny --n_shot 100 --seed 2 --batch_size 128 --dataset cifar10 --results_path results --wandb
# python3 experiments/cnn/linear_probing.py --model vim_tiny --n_shot 100 --seed 2 --batch_size 128 --dataset cifar10 --results_path results --wandb
# python3 experiments/cnn/ilm_vp.py --model vim_tiny --n_shot 100 --seed 3 --batch_size 128 --dataset cifar10 --results_path results --wandb
# python3 experiments/cnn/linear_probing.py --model vim_tiny --n_shot 100 --seed 3 --batch_size 128 --dataset cifar10 --results_path results --wandb

