<h1 align="center">Uncovering the Hidden Cost of Model Compression</h1>
<p align="center">Diganta Misra<sup>*,1,2,3</sup>, Agam Goyal<sup>*,4</sup>, Bharat Runwal<sup>*,1,2</sup>, Pin Yu Chen<sup>5</sup></p>
<p align="center"><sup>1</sup> Mila - Quebec AI Institute,<sup>2</sup> Landskape AI,<sup>3</sup> UdeM,<sup>4</sup> UW-Madison,<sup>5</sup> IBM Research</p>
<p align="center"><sup>*</sup> Equal Contribution </p>
<p align="center">
  <a href="https://arxiv.org/abs/2308.14969" alt="ArXiv">
        <img src="https://img.shields.io/badge/Preprint-arXiv-blue.svg" /></a>
  <a href="https://wandb.ai/landskape/Reprogram-Sparse" alt="Dashboard">
        <img src="https://img.shields.io/badge/WandB-Dashboard-gold.svg" /></a>
</p>

<p align="center">
  <img src ="vp.png"  width="1000"/>
</p>

In the era of resource-intensive foundation models, efficient adaptation in downstream tasks has become paramount. Visual Prompting (VP), inspired by prompting in Large Language Models (LLMs), has emerged as a key transfer learning method in computer vision. Aligned with the growing significance of efficiency, research in model compression has become pivotal to alleviate the computational burden in both training and deploying over-parameterized neural networks. A key goal in model compression is the development of sparse models capable of matching or surpassing the performance of their over-parameterized, dense counterparts. While prior research has explored the impact of model sparsity on transfer learning, its effects on visual prompting-based transfer remain unclear. This study addresses this gap, revealing that model sparsity adversely affects the performance of visual prompting-based transfer, particularly in low-data-volume scenarios. Furthermore, our findings highlight the negative influence of sparsity on the calibration of downstream visual-prompted models. This empirical exploration calls for a nuanced understanding beyond accuracy in sparse settings, opening avenues for further research in Visual Prompting for sparse models.

## Dependencies

Run `pip3 install -r requirements.txt`.

## Run the scripts

### Transferring a dense model

To run the Linear Probing script to transfer the dense model onto the full CIFAR-10 dataset with default parameters, use the following command:

```
python3 experiments/cnn/linear_probing.py \
        --network dense \
        --n_shot -1 \
        --batch_size 128 \
        --dataset cifar10 \
        --results_path results \
```

Note that `n_shot = -1` indicated that the entire data is being used. To use other N-shot data budgets, the user can provide a custon value.

Similarly, to run the ILM-VP script to transfer the dense model onto the full CIFAR-10 dataset with default parameters, use the following command:

```
python3 experiments/cnn/ilm_vp.py \
        --network dense \
        --n_shot -1 \
        --batch_size 128 \
        --dataset cifar10 \
        --results_path results \
```

### Transferring a Lottery Ticket

To run the Linear Probing script to transfer lottery ticket at sparsity state `8` onto the full CIFAR-10 dataset with default parameters, use the following command:

```
python3 experiments/cnn/linear_probing.py \
        --network LT \
        --sparsity 8 \
        --pretrained_dir pretrained_dir_name \
        --n_shot -1 \
        --batch_size 128 \
        --dataset cifar10 \
        --results_path results \
```

Note that `n_shot = -1` indicated that the entire data is being used. To use other N-shot data budgets, the user can provide a custon value.

Similarly, to run the ILM-VP script to transfer lottery ticket at sparsity state `8` onto the full CIFAR-10 dataset with default parameters, use the following command:

```
python3 experiments/cnn/ilm_vp.py \
        --network LT \
        --sparsity 8 \
        --pretrained_dir pretrained_dir_name \
        --n_shot -1 \
        --batch_size 128 \
        --dataset cifar10 \
        --results_path results \
```

**Note:** The ResNet-50 lottery ticket checkpoints pretrained on ImageNet-1k used in this study may be made available upon reasonable request.

## Cite:

```
@article{misra2023reprogramming,
  title   = {Reprogramming under constraints: Revisiting efficient and reliable transferability of lottery tickets},
  author  = {Diganta Misra and Agam Goyal and Bharat Runwal and Pin Yu Chen},
  year    = {2023},
  journal = {arXiv preprint arXiv: 2308.14969}
}
```
