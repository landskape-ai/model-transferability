import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

sys.path.append(".")

import wandb as wb

from data import IMAGENETNORMALIZE, prepare_additive_data
from tools.misc import gen_folder_name, set_seed


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
    pretrained = args.pretrained

    mask_dir = args.mask_dir

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
    p.add_argument(
        "--network", choices=["LT", "rigL", "acdc", "STR", "dense"], default="LT"
    )
    p.add_argument("--seed", type=int, default=7)
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
    p.add_argument("--results_path", type=str, default="/reprogram_new")
    p.add_argument(
        "--mask_dir", type=str, default="/ImageNetCheckpoint/resnet50_dyn4_9_mask.pth"
    )
    p.add_argument(
        "--pretrained",
        type=str,
        default="/ImageNetCheckpoint/resnet50_dyn4_9_checkpoint.pth",
    )
    p.add_argument("--epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_shot", type=float, default=-1.0)
    args = p.parse_args()

    wb_logger = wandb_setup(args)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    exp = f"cnn/full_finetuning"
    save_path = os.path.join(args.results_path, exp, gen_folder_name(args))
    data_path = os.path.join(args.results_path, "data")
    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(
                lambda x: x.convert("RGB") if hasattr(x, "convert") else x
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENETNORMALIZE["mean"], IMAGENETNORMALIZE["std"]),
        ]
    )
    loaders, class_names = prepare_additive_data(
        args.dataset, data_path=data_path, preprocess=preprocess
    )

    # Network
    if args.network == "dense":
        from torchvision.models import ResNet50_Weights, resnet50

        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    elif args.network == "LT":
        network = torchvision.models.__dict__["resnet50"](pretrained=(False)).to("cpu")
        new_dict = get_pruned_model(args)
        # network = network.to(device)
        network.load_state_dict(new_dict)
    elif args.network == "rigL":
        network = torchvision.models.__dict__["resnet50"](pretrained=(False))
        new_dict = get_pruned_model(args)
        network = network.to(device)
        network.load_state_dict(new_dict)
    elif args.network == "acdc":
        network = torchvision.models.__dict__["resnet50"](pretrained=(False))
        new_dict = get_pruned_model(args)
        network = network.to(device)
        network.load_state_dict(new_dict)
    elif args.network == "STR":
        network = torchvision.models.__dict__["resnet50"](pretrained=(False))
        new_dict = get_pruned_model(args)
        network = network.to(device)
        network.load_state_dict(new_dict)
    else:
        raise NotImplementedError(f"{args.network} is not supported")

    mask_dir = args.mask_dir
    current_mask_weight = torch.load(mask_dir)

    print("Check Sparsity Before Training : ")
    check_sparsity(network)
    network.requires_grad_(True)

    network = network.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, "tensorboard"))

    # Train
    best_acc = 0.0
    scaler = GradScaler()
    for epoch in range(args.epoch):
        network.train()
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
                fx = network(x)
                loss = F.cross_entropy(fx, y, reduction="mean")
            scaler.scale(loss).backward()

            # Apply mask to param_data and gradients to preserve pruned connections
            for name, param in network.named_parameters():
                name = "model." + name
                if name in current_mask_weight:
                    param.data *= current_mask_weight[name].to(device)
                    param.grad *= current_mask_weight[name].to(device)
            #  END OF APPLYING MASKS

            scaler.step(optimizer)
            scaler.update()
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Acc {100*true_num/total_num:.2f}%")

        scheduler.step()
        logger.add_scalar("train/acc", true_num / total_num, epoch)
        logger.add_scalar("train/loss", loss_sum / total_num, epoch)
        wb_logger.log({"Train-Loss": loss_sum / total_num})
        wb_logger.log({"Train-ACC": true_num / total_num})
        wb_logger.log({"Sparsity": check_sparsity(network, False)})

        # Test
        network.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(
            loaders["test"],
            total=len(loaders["test"]),
            desc=f"Epo {epoch} Testing",
            ncols=100,
        )
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx = network(x)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = true_num / total_num
            pbar.set_postfix_str(f"Acc {100*acc:.2f}%")
        logger.add_scalar("test/acc", acc, epoch)
        wb_logger.log({"Test-ACC": acc})
        # Save CKPT
        state_dict = {
            "network_dict": network.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict["best_acc"] = best_acc
            torch.save(state_dict, os.path.join(save_path, "best.pth"))
        check_sparsity(network)
        torch.save(state_dict, os.path.join(save_path, "ckpt.pth"))
