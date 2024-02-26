import argparse
import io
import os
import sys
from functools import partial

import numpy as np
import timm
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
        config=args,
        name=args.run_name,
        project="model-transferability",
        entity="landskape",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=[
            "deit_small",
            "deit_base",
            "deit3_small",
            "deit3_base",
            "vit_base",
            "moco_base",
            "moco_small",
        ],
        default="deit_base",
    )
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
            "caltech101",
        ],
        required=True,
    )
    p.add_argument("--mapping-interval", type=int, default=1)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--results_path", type=str, default="results")
    p.add_argument("--n_shot", type=float, default=-1.0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run_name", type=str, default="exp")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--caltech_path", type=str, default="/data/caltech101_data.npz")
    args = p.parse_args()

    args.run_name = f"{args.model}-{args.dataset}-{args.n_shot}shot-seed{args.seed}-ilm"

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
    choices = (
        [
            "deit_base_patch16_224.fb_in1k",
            "deit3_base_patch16_224.fb_in1k",
            "vit_base_patch16_224.orig_in21k_ft_in1k",
        ],
    )
    if args.model == "deit_small":
        network = timm.create_model("deit_small_patch16_224.fb_in1k", pretrained=True)
    elif args.model == "deit_base":
        network = timm.create_model("deit_base_patch16_224.fb_in1k", pretrained=True)
    elif args.model == "deit3_small":
        network = timm.create_model("deit3_small_patch16_224.fb_in1k", pretrained=True)
    elif args.model == "deit3_base":
        network = timm.create_model("deit3_base_patch16_224.fb_in1k", pretrained=True)
    elif args.model == "vit_base":
        network = timm.create_model(
            "vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True
        )
    elif args.model == "moco_small":
        network = timm.create_model(
            "vit_small_patch16_224.augreg_in1k", pretrained=False
        )
        url = "https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/linear-vit-s-300ep.pth.tar"
        state_dict = torch.hub.load_state_dict_from_url(url)["state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            new_state_dict[key.replace("module.", "")] = value
        network.load_state_dict(new_state_dict)
    elif args.model == "moco_base":
        network = timm.create_model(
            "vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False
        )
        url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar"
        state_dict = torch.hub.load_state_dict_from_url(url)["state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            new_state_dict[key.replace("module.", "")] = value
        network.load_state_dict(new_state_dict)
    else:
        raise NotImplementedError(f"{args.model} is not supported")
    network.requires_grad_(False)
    network.eval()
    network = network.to(device)

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

        # # Save CKPT
        # state_dict = {
        #     "visual_prompt_dict": visual_prompt.state_dict(),
        #     "optimizer_dict": optimizer.state_dict(),
        #     "epoch": epoch,
        #     "best_acc": best_acc,
        #     "mapping_sequence": mapping_sequence,
        # }
        # if acc > best_acc:
        #     best_acc = acc
        #     state_dict["best_acc"] = best_acc
        #     torch.save(state_dict, os.path.join(save_path, "best.pth"))
        # torch.save(state_dict, os.path.join(save_path, "ckpt.pth"))
