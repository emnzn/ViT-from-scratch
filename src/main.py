import os
import torch
from tqdm import tqdm
from typing import Tuple
import torch.nn.functional as F
from utils import VisionTransformer
from sklearn.metrics import f1_score
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split

def train(
        dataloader: DataLoader, 
        criterion: torch.nn, 
        optimizer: torch.optim, 
        model: VisionTransformer, 
        device: str
        ) -> Tuple[float]:
    """
    Trains the model for a single epoch.

    Parameters
    ----------
    dataloader: DataLoader
        The dataloader to parse.

    criterion: torch.nn
        The loss function to be used.

    optimizer: torch.optim
        The optimizer for gradient descent.

    model: VisionTransformer
        The implemented ViT

    device: str
        The device to train the model in.

    Returns
    -------
    epoch_loss: float
        The loss for the entire epoch.

    epoch_f1: float
        The f1 score for the epoch.
    """

    metrics = {
        "running_loss": 0,
        "running_f1": 0
    }

    model.train()
    for img, target in tqdm(dataloader, desc="Training in progress"):
        img = img.to(device)
        target = target.to(device)

        logits = model(img)
        confidence = F.softmax(img, dim=1)
        pred = torch.argmax(confidence, dim=1)

        loss = criterion(logits, target)
        f1 = f1_score(target, pred, average="macro")

        metrics["running_loss"] += loss
        metrics["running_f1"] += f1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = metrics["running_loss"] / len(dataloader)
    epoch_f1 = metrics["running_f1"] / len(dataloader)

    return epoch_loss, epoch_f1

def validate(dataloader, criterion, model, device):
    """
    Validates the model for a given epoch.
    """

    metrics = {
        "running_loss": 0,
        "running_f1": 0
    }

    model.eval()
    for img, target in tqdm(dataloader, "Validation in Progress"):
        img = img.to(device)
        target = target.to(device)

        logits = model(img)
        confidence = F.softmax(logits, dim=1)
        pred = torch.argmax(confidence, dim=1)

        loss = criterion(pred, target)
        f1 = f1_score(target, pred, average="macro")

        metrics["running_loss"] += loss
        metrics["running_f1"] += f1

    epoch_loss = metrics["running_loss"] / len(dataloader)
    epoch_f1 = metrics["running_f1"] / len(dataloader)

    return epoch_loss, epoch_f1

def main():
    dest = os.path.join("..", "data")

    transform = transforms.compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CIFAR10(root=dest, train=True, download=True, transform=transform)

    train_size = len(dataset) * 0.8
    val_size = len(dataset) - train_size

    train, val = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val, batch_size=args["batch_size"], shuffle=False)
