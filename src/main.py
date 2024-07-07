import os
import torch
from tqdm import tqdm
from typing import Tuple
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from utils import VisionTransformer, get_args, save_args, \
    get_model, get_checkpoint

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

        metrics["running_loss"] += loss.detach().cpu().item()
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

        metrics["running_loss"] += loss.detach().cpu().item()
        metrics["running_f1"] += f1

    epoch_loss = metrics["running_loss"] / len(dataloader)
    epoch_f1 = metrics["running_f1"] / len(dataloader)

    return epoch_loss, epoch_f1

def main():
    data_dir = os.path.join("..", "data")
    arg_path = os.path.join("config", "train_config.yaml")
    args = get_args(arg_path)
    id = datetime.now().strftime("%m-%d-%Y-%H:%M") 

    model_dir = os.path.join("..", "assets", "models", id)
    log_dir = os.path.join("runs", id)
    writer = SummaryWriter(log_dir)
    save_args(log_dir, args)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    transform = transforms.compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    train_size = len(dataset) * 0.8
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)

    model = get_model(
        args["img_size"], patch_size=4, variant=args["variant"], 
        dropout_probability=args["dropout_probability"],
        num_classes=args["num_classes"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args["patience"])

    running_val_loss, running_val_f1 = [], []

    for epoch in range(1, args["epochs"] + 1):
        print(f"Epoch [{epoch}/{args['epochs']}]")

        train_loss, train_f1 = train(train_loader, criterion, optimizer, model, device)

        print("\nTrain Statistics:")
        print(f"Loss: {train_loss:.4f} | F1: {train_f1:.4f}")

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/F1", train_f1, epoch)

        val_loss, val_f1 = validate(val_loader, criterion, model, device)

        print("\nValidation Statistics:")
        print(f"Loss: {val_loss:.4f} | F1: {val_f1:.4f}")

        writer.add_scalar("Validation/Loss", val_loss)
        writer.add_scalar("Validation/F1", val_f1, epoch)

        if len(running_val_loss) > 0 and val_loss < running_val_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, f"vit-{args["variant"]}-lowest-loss.pth"))
            print("New minimum loss — model saved.")

        if len(running_val_f1) > 0 and val_f1 > max(running_val_f1):
            torch.save(model.state_dict(), os.path.join(model_dir, f"vit-{args["variant"]}-highest-f1.pth"))
            print("New maximum F1 - model saved.")

        if epoch % 5 == 0:
            checkpoint = get_checkpoint(epoch, model, optimizer, scheduler)
            torch.save(checkpoint, os.path.join(model_dir, f"vit-{args["variant"]}-latest-checkpoint.pth"))
            print("Checkpoint saved.")

        running_val_loss.append(val_loss)
        running_val_f1.append(val_f1)

        scheduler.step(val_loss)

        print("___________________________________________________________________\n")

    torch.save(model.state_dict(), os.path.join(model_dir, f"vit-{args["variant"]}-latest-model.pth"))

if __name__ == "__main__":
    main()