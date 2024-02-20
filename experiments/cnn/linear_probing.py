import argparse
import os
import sys

import timm
import torch
import torchvision
from timm.models import create_model
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

sys.path.append(".")
import calibration as cal
import wandb as wb

sys.path.append("VMamba")
from functools import partial

from classification.models.vmamba import VSSM

from data import IMAGENETNORMALIZE, prepare_additive_data
from models import models_mamba
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
            "vim_tiny",
            "vim_small",
            "vssm_tiny",
            "vssm_small",
            "vssm_base",
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
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--results_path", type=str, default="results")
    p.add_argument("--n_shot", type=float, default=-1.0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run_name", type=str, default="exp")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--caltech_path", type=str, default="/data/caltech101_data.npz")
    args = p.parse_args()

    args.run_name = f"{args.model}-{args.dataset}-{args.n_shot}shot-seed{args.seed}-lp"

    if args.wandb:
        wb_logger = wandb_setup(args)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    exp = f"cnn/linear_probing"

    save_path = os.path.join(args.results_path, exp, gen_folder_name(args))
    data_path = os.path.join(args.results_path, "data")

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Lambda(
                lambda x: x.convert("RGB") if hasattr(x, "convert") else x
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENETNORMALIZE["mean"], IMAGENETNORMALIZE["std"]),
        ]
    )
    if args.dataset == "caltech101":
        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.Lambda(
                    lambda x: x.convert("RGB") if hasattr(x, "convert") else x
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    IMAGENETNORMALIZE["mean"], IMAGENETNORMALIZE["std"]
                ),
            ]
        )

    loaders, class_names = prepare_additive_data(
        args, args.dataset, data_path=data_path, preprocess=preprocess
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
    elif args.model in ["vim_tiny", "vim_small"]:
        if args.model == "vim_tiny":
            network = create_model(
                "vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2",
                pretrained=False,
                num_classes=1000,
                drop_rate=0.0,
                drop_path_rate=0.1,
                drop_block_rate=None,
                img_size=224,
            )

            checkpoint = torch.load(
                "/home/mila/d/diganta.misra/scratch/mamba_weights/vim_t_midclstok_76p1acc.pth", map_location="cpu"
            )

        else:
            network = create_model(
                "vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2",
                pretrained=False,
                num_classes=1000,
                drop_rate=0.0,
                drop_path_rate=0.1,
                drop_block_rate=None,
                img_size=224,
            )

            checkpoint = torch.load(
                "/home/mila/d/diganta.misra/scratch/mamba_weights/vim_s_midclstok_80p5acc.pth", map_location="cpu"
            )

        checkpoint_model = checkpoint["model"]
        state_dict = network.state_dict()
        for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = network.patch_embed.num_patches
        num_extra_tokens = network.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model["pos_embed"] = new_pos_embed

        network.load_state_dict(checkpoint_model, strict=False)
    elif args.model in ["vssm_tiny", "vssm_small", "vssm_base"]:
        if args.model == "vssm_tiny":
            network = partial(
                VSSM,
                patch_size=16,
                in_chans=3,
                num_classes=1000,
                downsample_version="v1",
                patchembed_version="v1",
                dims=96,
                depths=[2, 2, 9, 2],
                mlp_ratio=0.0,
                ssm_d_state=16,
                ssm_ratio=2.0,
                ssm_dt_rank="auto",
                drop_path_rate=0.2,
            )()

            checkpoint = torch.load(
                "/home/mila/d/diganta.misra/scratch/mamba_weights/vssmtiny_dp01_ckpt_epoch_292.pth", map_location="cpu"
            )

        elif args.model == "vssm_small":
            network = partial(
                VSSM,
                patch_size=16,
                in_chans=3,
                num_classes=1000,
                downsample_version="v1",
                patchembed_version="v1",
                dims=96,
                depths=[2, 2, 27, 2],
                mlp_ratio=0.0,
                ssm_d_state=16,
                ssm_ratio=2.0,
                ssm_dt_rank="auto",
                drop_path_rate=0.3,
            )()

            checkpoint = torch.load(
                "/home/mila/d/diganta.misra/scratch/mamba_weights/vssmsmall_dp03_ckpt_epoch_238.pth",
                map_location="cpu",
            )

        elif args.model == "vssm_base":
            network = partial(
                VSSM,
                patch_size=16,
                in_chans=3,
                num_classes=1000,
                downsample_version="v1",
                patchembed_version="v1",
                dims=128,
                depths=[2, 2, 27, 2],
                mlp_ratio=0.0,
                ssm_d_state=16,
                ssm_ratio=2.0,
                ssm_dt_rank="auto",
                drop_path_rate=0.5,
            )()

            checkpoint = torch.load(
                "/home/mila/d/diganta.misra/scratch/mamba_weights/vssmbase_dp05_ckpt_epoch_260.pth", map_location="cpu"
            )

        checkpoint_model = checkpoint["model"]
        state_dict = network.state_dict()
        for k in ["classifier.head.weight", "classifier.head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
    else:
        raise NotImplementedError(f"{args.model} is not supported")
    network.requires_grad_(False)
    network.eval()
    if args.model in args.model in ["vssm_tiny", "vssm_small", "vssm_base"]:
        network.classifier.head = torch.nn.Linear(
            network.classifier.head.in_features, len(class_names)
        )
        network.classifier.head.requires_grad_(True)
    else:
        network.head = torch.nn.Linear(network.head.in_features, len(class_names))
        network.head.requires_grad_(True)
    network = network.to(device)

    # Optimizer
    if args.model in args.model in ["vssm_tiny", "vssm_small", "vssm_base"]:
        optimizer = torch.optim.Adam(network.classifier.head.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(network.head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.5 * args.epoch), int(0.72 * args.epoch)], gamma=0.1
    )

    # Make Dir
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, "tensorboard"))

    # Train
    best_acc = 0.0
    scaler = GradScaler()
    for epoch in range(args.epoch):
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
            if x.get_device() == -1:
                x, y = x.to(device), y.to(device)
            pbar.set_description_str(
                f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}",
                refresh=True,
            )
            optimizer.zero_grad()
            with autocast():
                fx = network(x)
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
        total_num = 0
        true_num = 0
        calibration_error = 0
        pbar = tqdm(
            loaders["test"],
            total=len(loaders["test"]),
            desc=f"Epo {epoch} Testing",
            ncols=100,
        )
        for x, y in pbar:
            if x.get_device() == -1:
                x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx = network(x)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            calibration_error += cal.get_ece(fx.cpu().numpy(), y.cpu().numpy())
            acc = true_num / total_num
            pbar.set_postfix_str(f"Acc {100*acc:.2f}%")
        logger.add_scalar("test/acc", acc, epoch)
        if args.wandb:
            wb_logger.log(
                {"Test/Test-ACC": acc, "Test/ECE": calibration_error / total_num}
            )

        # Save CKPT
        if args.model in args.model in ["vssm_tiny", "vssm_small", "vssm_base"]:
            fc_dict = network.classifier.head.state_dict()
        else:
            fc_dict = network.head.state_dict()
        state_dict = {
            "fc_dict": fc_dict,
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict["best_acc"] = best_acc
            torch.save(state_dict, os.path.join(save_path, "best.pth"))
        torch.save(state_dict, os.path.join(save_path, "ckpt.pth"))
