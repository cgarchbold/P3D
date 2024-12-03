import os
import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from .ucla_protest import ProtestDataset


def get_dataset(config):
    if config["dataset"] == "MNIST":
        train_transform = transforms.Compose(
            [
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5])
            ]
        )
        train_loader = torch.utils.data.DataLoader(
            MNIST("data", train=True, download=True, transform=train_transform),
            batch_size=config["batch_size"],
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            MNIST("data", train=False, download=True, transform=test_transform),
            batch_size=config["batch_size"],
            shuffle=False,
        )
    if config["dataset"] == "CIFAR10":
        train_transform = transforms.Compose(
            [
                transforms.Resize(size=(config["size"], config["size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(size=(config["size"], config["size"])),
                transforms.ToTensor(),
            ]
        )
        train_loader = torch.utils.data.DataLoader(
            CIFAR10("data", train=True, download=True, transform=train_transform),
            batch_size=config["batch_size"],
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            CIFAR10("data", train=False, download=True, transform=test_transform),
            batch_size=config["batch_size"],
            shuffle=False,
        )
    if config["dataset"] == "UCLA":
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

        train_loader = torch.utils.data.DataLoader(
            ProtestDataset(
                txt_file=os.path.join(config["root_dir"], "annot_train.txt"),
                img_dir=os.path.join(config["root_dir"], "img", "train"),
                multi_label=config["multi_label"],
                transform=train_transform,
            ),
            batch_size=config["batch_size"],
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            ProtestDataset(
                txt_file=os.path.join(config["root_dir"], "annot_test.txt"),
                img_dir=os.path.join(config["root_dir"], "img", "test"),
                multi_label=config["multi_label"],
                transform=test_transform,
            ),
            batch_size=config["batch_size"],
            shuffle=False,
        )

    if config["dataset"] == "small_UCLA":
        train_transform = transforms.Compose(
            [
                transforms.Resize(size=(config["size"], config["size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(size=(config["size"], config["size"])),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

        train_loader = torch.utils.data.DataLoader(
            ProtestDataset(
                txt_file=os.path.join(config["root_dir"], "annot_train.txt"),
                img_dir=os.path.join(config["root_dir"], "img", "train"),
                transform=train_transform,
            ),
            batch_size=config["batch_size"],
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            ProtestDataset(
                txt_file=os.path.join(config["root_dir"], "annot_test.txt"),
                img_dir=os.path.join(config["root_dir"], "img", "test"),
                transform=test_transform,
            ),
            batch_size=config["batch_size"],
            shuffle=False,
        )

    return train_loader, test_loader
