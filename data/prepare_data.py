import os
import json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from .dataset_lmdb import COOPLMDBDataset
from .abide import ABIDE
from .const import GTSRB_LABEL_MAP, IMAGENETNORMALIZE


def refine_classnames(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def get_class_names_from_split(root):
    with open(os.path.join(root, "split.json")) as f:
        split = json.load(f)["test"]
    idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split}.items()))
    return list(idx_to_class.values())


def prepare_expansive_data(args, dataset, data_path):
    data_path = os.path.join(data_path, dataset)
    if dataset == "cifar10":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = True, transform = preprocess)
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "cifar100":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR100(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = True, transform = preprocess)
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "gtsrb":
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(list(GTSRB_LABEL_MAP.values())),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "svhn":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.SVHN(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = True, transform = preprocess)
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': [f'{i}' for i in range(10)],
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "abide":
        preprocess = transforms.ToTensor()
        D = ABIDE(root = data_path)
        X_train, X_test, y_train, y_test = train_test_split(D.data, D.targets, test_size=0.1, stratify=D.targets, random_state=1)
        train_data = ABIDE(root = data_path, transform = preprocess)
        train_data.data = X_train
        train_data.targets = y_train
        test_data = ABIDE(root = data_path, transform = preprocess)
        test_data.data = X_test
        test_data.targets = y_test
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': ["non ASD", "ASD"],
            'mask': D.get_mask(),
        }
    elif dataset in ["food101", "eurosat", "sun397", "ucf101", "stanfordcars", "flowers102"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=8),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")
    return loaders, configs


def get_data_splits(args, train_data):
    # subsetting with train_data_fraction from args
    # uniform between classes
    if args.train_data_fraction < 1.0:
        # make map of class and indices of the samples
        target_to_sample_idxs = [[] for _ in range(len(train_data.classes))]
        for i, (_, target) in enumerate(train_data):
            target_to_sample_idxs[target].append(i)

        # shuffle each class
        for sample_idxs in target_to_sample_idxs:
            np.random.shuffle(sample_idxs)
            # only keep fraction
            sample_idxs[:] = sample_idxs[
                : int(len(sample_idxs) * args.train_data_fraction)
            ]

        # concat
        subsampled_idxs = np.concatenate(target_to_sample_idxs)
        # shuffle again
        np.random.shuffle(subsampled_idxs)
        # make subset
        trainset = Subset(train_data, subsampled_idxs)

    return train_data


def prepare_additive_data(args, dataset, data_path, preprocess):
    data_path = os.path.join(data_path, dataset)
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = False, transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = False, transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset == "svhn":
        train_data = datasets.SVHN(root = data_path, split="train", download = False, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = False, transform = preprocess)
        class_names = [f'{i}' for i in range(10)]
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset in ["food101", "sun397", "eurosat", "ucf101", "stanfordcars", "flowers102"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=8),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
    elif dataset == "gtsrb":
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset == "abide":         
        D = ABIDE(root = data_path)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
        ])
        X_train, X_test, y_train, y_test = train_test_split(D.data, D.targets, test_size=0.1, stratify=D.targets, random_state=1)
        train_data = ABIDE(root = data_path, transform = preprocess)
        train_data.data = X_train
        train_data.targets = y_train
        test_data = ABIDE(root = data_path, transform = preprocess)
        test_data.data = X_test
        test_data.targets = y_test
        train_data = get_data_splits(args, train_data)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=2),
        }
        class_names = ["non ASD", "ASD"]
    else:
        raise NotImplementedError(f"{dataset} not supported")

    return loaders, class_names


def prepare_gtsrb_fraction_data(data_path, fraction, preprocess=None):
    data_path = os.path.join(data_path, "gtsrb")
    assert 0 < fraction <= 1
    new_length = int(fraction*26640)
    indices = torch.randperm(26640)[:new_length]
    sampler = SubsetRandomSampler(indices)
    if preprocess == None:
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, sampler=sampler, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(list(GTSRB_LABEL_MAP.values())),
            'mask': np.zeros((32, 32)),
        }
        return loaders, configs
    else:
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            'train': DataLoader(train_data, 128, sampler=sampler, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        return loaders, class_names