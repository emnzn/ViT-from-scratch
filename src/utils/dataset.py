import torch
from typing import Tuple
from torchvision import transforms
from torchvision.datasets import CIFAR10

def get_dataset(data_dir: str) -> Tuple[CIFAR10]:
    """
    Applies transformation and returns
    the training and test dataset classes for CIFAR-10.

    Parameters
    ----------
    data_dir: str
        The path to where the datasets will be downloaded.

    Returns
    -------
    train_dataset: CIFAR10
        The train set of CIFAR10.
    
    val_dataset: CIFAR10
        The test set of CIFAR10.
    """

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transforms)
    val_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=val_transforms)

    return train_dataset, val_dataset