import argparse
import io
import os
import sys
from functools import partial

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

sys.path.append(".")
import calibration as cal
import wandb as wb

from algorithms import (generate_label_mapping_by_frequency, get_dist_matrix,
                        label_mapping_base)
from data import IMAGENETCLASSES, IMAGENETNORMALIZE, prepare_expansive_data
from models import ExpansiveVisualPrompt
from tools.mapping_visualization import plot_mapping
from tools.misc import gen_folder_name, set_seed

# from cfg import *


def wandb_setup(args):
    return wb.init(
        config=args, name=args.run_name, project="Reprogram-Sparse", entity="landskape"
    )


def check_sparsity(model, conv1=True):
    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == "conv1":
                if conv1:
                    sum_list = sum_list + float(m.weight.nelement())
                    zero_sum = zero_sum + float(torch.sum(m.weight == 0))
                else:
                    print("skip conv1 for sparsity checking")
            else:
                sum_list = sum_list + float(m.weight.nelement())
                zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    print("* remain weight = ", 100 * (1 - zero_sum / sum_list), "%")

    return 100 * (1 - zero_sum / sum_list)


def get_pruned_model(args):
    # pretrained = args.pretrained

    # mask_dir = args.mask_dir

    # -------------------------------------------
    # works only for lottery ticket
    pretrained = os.path.join(
        args.pretrained_dir, f"resnet50_dyn4_{args.sparsity}_checkpoint.pth"
    )
    mask_dir = os.path.join(
        args.pretrained_dir, f"resnet50_dyn4_{args.sparsity}_mask.pth"
    )
    # -------------------------------------------

    current_mask_weight = torch.load(mask_dir)
    curr_weight = torch.load(pretrained)

    new_weights = {}
    for name in current_mask_weight.keys():
        name_ = name.replace("model.", "")
        new_weights[str(name_)] = (
            current_mask_weight[str(name)] * curr_weight[str(name)]
        )

    for k in curr_weight.keys():
        if str(k) not in new_weights.keys():
            # print(k)
            k_ = k.replace("model.", "")

            new_weights[k_] = curr_weight[k]

    return new_weights


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--network", choices=["LT", "dense"], default="LT")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument(
        "--dataset",
        choices=[
            "cifar10",
            "cifar100",
            "abide",
            "dtd",
            "flowers102",
            "ucf101",
            "food101",
            "gtsrb",
            "svhn",
            "eurosat",
            "oxfordpets",
            "stanfordcars",
            "sun397",
        ],
        required=True,
    )
    p.add_argument("--mapping-interval", type=int, default=1)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--results_path", type=str, default="/data/jaygala/ILM-VP/results")
    p.add_argument(
        "--pretrained_dir",
        type=str,
        default="/data/jaygala/ILM-VP/artifacts/ImageNetCheckpoint_LT",
    )
    p.add_argument("--sparsity", type=int, default=9)
    p.add_argument("--train_data_fraction", type=float, default=1.0)
    p.add_argument("--n_shot", type=float, default=-1.0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run_name", type=str, default="exp")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    if args.wandb:
        wb_logger = wandb_setup(args)

    # Misc
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    exp = f"cnn/ilm_vp"

    save_path = os.path.join(args.results_path, exp, gen_folder_name(args))
    data_path = os.path.join(args.results_path, "data")

    # Data
    loaders, configs = prepare_expansive_data(args, args.dataset, data_path=data_path)
    normalize = transforms.Normalize(
        IMAGENETNORMALIZE["mean"], IMAGENETNORMALIZE["std"]
    )

    # Network
    if args.network == "dense":
        from torchvision.models import ResNet50_Weights, resnet50

        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    elif args.network == "LT":
        network = torchvision.models.__dict__["resnet50"](pretrained=(False))
        new_dict = get_pruned_model(args)
        network = network.to(device)
        network.load_state_dict(new_dict)
    else:
        raise NotImplementedError(f"{args.network} is not supported")
    network.requires_grad_(False)
    network.eval()

    if args.wandb:
        wb_logger.log({"Sparsity": check_sparsity(network, False)})
    # Visual Prompt
    visual_prompt = ExpansiveVisualPrompt(
        224, mask=configs["mask"], normalize=normalize
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(visual_prompt.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.5 * args.epoch), int(0.72 * args.epoch)], gamma=0.1
    )

    # Make dir
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, "tensorboard"))

    # Train
    best_acc = 0.0
    scaler = GradScaler()
    for epoch in range(args.epoch):
        if epoch % args.mapping_interval == 0:
            mapping_sequence = generate_label_mapping_by_frequency(
                visual_prompt, network, loaders["train"]
            )
            label_mapping = partial(
                label_mapping_base, mapping_sequence=mapping_sequence
            )
        visual_prompt.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(
            loaders["train"],
            total=len(loaders["train"]),
            desc=f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}",
            ncols=100,
        )
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            pbar.set_description_str(
                f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}",
                refresh=True,
            )
            optimizer.zero_grad()
            with autocast():
                fx = label_mapping(network(visual_prompt(x)))
                loss = F.cross_entropy(fx, y, reduction="mean")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Acc {100*true_num/total_num:.2f}%")
        scheduler.step()
        logger.add_scalar("train/acc", true_num / total_num, epoch)
        logger.add_scalar("train/loss", loss_sum / total_num, epoch)
        if args.wandb:
            wb_logger.log(
                {
                    "Train/Train-Loss": loss_sum / total_num,
                    "Train/Train-ACC": true_num / total_num,
                    "Epoch": epoch,
                    "LR": scheduler.get_last_lr()[0],
                }
            )

        # Test
        visual_prompt.eval()
        total_num = 0
        true_num = 0
        calibration_error = 0
        pbar = tqdm(
            loaders["test"],
            total=len(loaders["test"]),
            desc=f"Epo {epoch} Testing",
            ncols=100,
        )
        fx0s = []
        ys = []
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            ys.append(y)
            with torch.no_grad():
                fx0 = network(visual_prompt(x))
                fx = label_mapping(fx0)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            calibration_error += cal.get_ece(fx.cpu().numpy(), y.cpu().numpy())
            acc = true_num / total_num
            fx0s.append(fx0)
            pbar.set_postfix_str(f"Acc {100*acc:.2f}%")
        fx0s = torch.cat(fx0s).cpu()
        ys = torch.cat(ys).cpu()
        mapping_matrix = get_dist_matrix(fx0s, ys)
        with io.BytesIO() as buf:
            plot_mapping(
                mapping_matrix,
                mapping_sequence,
                buf,
                row_names=configs["class_names"],
                col_names=np.array(IMAGENETCLASSES),
            )
            buf.seek(0)
            im = transforms.ToTensor()(Image.open(buf))
        logger.add_image("mapping-matrix", im, epoch)
        logger.add_scalar("test/acc", acc, epoch)
        if args.wandb:
            wb_logger.log(
                {"Test/Test-ACC": acc, "Test/ECE": calibration_error / total_num}
            )

        # Save CKPT
        state_dict = {
            "visual_prompt_dict": visual_prompt.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "mapping_sequence": mapping_sequence,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict["best_acc"] = best_acc
            torch.save(state_dict, os.path.join(save_path, "best.pth"))
        torch.save(state_dict, os.path.join(save_path, "ckpt.pth"))
