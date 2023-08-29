# Reprogramming Under Constraints

[arXiv] ["Reprogramming under constraints: Revisiting efficient and reliable transferability of lottery tickets"](https://github.com/landskape-ai/ILM-VP)

> Diganta Misra*, Agam Goyal*, Bharat Runwal*, Pin-Yu Chen

In the era of foundation models with huge pre-training budgets, the downstream tasks have been shifted to the narrative of efficient and fast adaptation. For classification-based tasks in the domain of computer vision, the two most efficient approaches have been **linear probing** (LP) and **visual prompting**/**reprogramming** (VP); the former aims to learn a classifier in the form of a linear head on the features extracted by the pre-trained model, while the latter maps the input data to the domain of the source data on which the model was originally pre-trained on. Although extensive studies have demonstrated the differences between LP and VP in terms of downstream performance, we explore the capabilities of the two aforementioned methods via the sparsity axis: (a) **Data sparsity**: the impact of few-shot adaptation and (b) **Model sparsity**: the impact of lottery tickets (LT). We demonstrate that <u>LT</u> are not universal reprogrammers, i.e., for certain target datasets, reprogramming an LT yields significantly lower performance than the reprogrammed dense model although their corresponding upstream performance is similar. Further, we demonstrate that the calibration of dense models is always superior to that of their lottery ticket counterparts under both LP and VP regimes. Our empirical study opens a new avenue of research into VP for sparse models and encourages further understanding of the performance beyond the accuracy achieved by VP under constraints of sparsity. Code and logs can be accessed at [https://github.com/landskape-ai/ILM-VP](https://github.com/landskape-ai/ILM-VP) and [https://wandb.ai/landskape/Reprogram-Sparse](https://wandb.ai/landskape/Reprogram-Sparse) respectively.

![Overview](vp.pdf)

