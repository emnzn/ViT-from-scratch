import os
import torch
from tqdm import tqdm
from typing import Tuple
from datetime import datetime
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from sklearn.metrics import accuracy_score
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from utils import VisionTransformer, get_args, save_args, \
    get_model, get_checkpoint, set_seed

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

    epoch_accuracy: float
        The accuracy for the epoch.
    """

    metrics = {
        "running_loss": 0,
        "running_accuracy": 0
    }

    model.train()
    for img, target in tqdm(dataloader, desc="Training in progress"):
        img = img.to(device)
        target = target.to(device)

        logits = model(img)
        confidence = F.softmax(logits, dim=1)
        pred = torch.argmax(confidence, dim=1)

        loss = criterion(logits, target)
        accuracy = accuracy_score(target.cpu(), pred.cpu())

        metrics["running_loss"] += loss.detach().cpu().item()
        metrics["running_accuracy"] += accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = metrics["running_loss"] / len(dataloader)
    epoch_accuracy = metrics["running_accuracy"] / len(dataloader)

    return epoch_loss, epoch_accuracy

def validate(dataloader, criterion, model, device):
    """
    Validates the model for a given epoch.
    """

    metrics = {
        "running_loss": 0,
        "running_accuracy": 0
    }

    model.eval()
    for img, target in tqdm(dataloader, "Validation in Progress"):
        img = img.to(device)
        target = target.to(device)

        logits = model(img)
        confidence = F.softmax(logits, dim=1)
        pred = torch.argmax(confidence, dim=1)

        loss = criterion(logits, target)
        accuracy = accuracy_score(target.cpu(), pred.cpu())

        metrics["running_loss"] += loss.detach().cpu().item()
        metrics["running_accuracy"] += accuracy

    epoch_loss = metrics["running_loss"] / len(dataloader)
    epoch_accuracy = metrics["running_accuracy"] / len(dataloader)

    return epoch_loss, epoch_accuracy

def main():
    data_dir = os.path.join("..", "data")
    arg_path = os.path.join("config", "train_config.yaml")
    args = get_args(arg_path)
    id = datetime.now().strftime("%m-%d-%Y_%H-hrs") 

    model_dir = os.path.join("..", "assets", "models", id)
    log_dir = os.path.join("runs", id)
    writer = SummaryWriter(log_dir)
    save_args(log_dir, args)
    set_seed(args["seed"])

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) - train_size)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)

    model = get_model(
        args["img_size"], patch_size=args["patch_size"], variant=args["variant"], 
        dropout_probability=args["dropout_probability"],
        num_classes=args["num_classes"], learnable_pe=args["learnable_pe"]
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args["patience"])

    running_val_loss, running_val_accuracy = [], []

    for epoch in range(1, args["epochs"] + 1):
        writer.add_scalar("Learning Rate", scheduler.optimizer.param_groups[0]["lr"], epoch)
        print(f"Epoch [{epoch}/{args['epochs']}]")

        train_loss, train_accuracy = train(train_loader, criterion, optimizer, model, device)

        print("Train Statistics:")
        print(f"Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f}\n")

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_accuracy, epoch)

        val_loss, val_accuracy = validate(val_loader, criterion, model, device)

        print("Validation Statistics:")
        print(f"Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.4f}\n")

        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)

        if len(running_val_loss) > 0 and val_loss < min(running_val_loss):
            torch.save(model.state_dict(), os.path.join(model_dir, f"vit-{args['variant']}-lowest-loss.pth"))
            print("New minimum loss — model saved.")

        if len(running_val_accuracy) > 0 and val_accuracy > max(running_val_accuracy):
            torch.save(model.state_dict(), os.path.join(model_dir, f"vit-{args['variant']}-{args["patch_size"]}-highest-accuracy.pth"))
            print("New maximum Accuracy - model saved.")

        if epoch % 5 == 0:
            checkpoint = get_checkpoint(epoch, model, optimizer, scheduler)
            torch.save(checkpoint, os.path.join(model_dir, f"vit-{args['variant']}-latest-checkpoint.pth"))
            print("Checkpoint saved.")

        running_val_loss.append(val_loss)
        running_val_accuracy.append(val_accuracy)

        scheduler.step(val_loss)

        print("-------------------------------------------------------------------\n")

    torch.save(model.state_dict(), os.path.join(model_dir, f"vit-{args['variant']}-latest-model.pth"))

if __name__ == "__main__":
    main()