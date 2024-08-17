import os
import torch
from tqdm import tqdm
from typing import Tuple
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from utils import VisionTransformer, get_args, save_args, \
    get_model, get_checkpoint, set_seed, get_dataset, get_torch_model

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

@torch.no_grad()
def validate(
    dataloader: DataLoader,
    criterion: torch.nn,
    model: VisionTransformer, 
    device: str
    ) -> Tuple[float]:
    
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset, val_dataset = get_dataset(data_dir, args["apply_augmentation"])

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)

    if args["torch_implementation"]:
        model = get_torch_model(
            img_size=args["img_size"], patch_size=args["patch_size"], variant=args["variant"],
            dropout_probability=args["dropout_probability"], 
            num_classes=args["num_classes"]
        ).to(device)

    else:
        model = get_model(
            img_size=args["img_size"], patch_size=args["patch_size"], variant=args["variant"], 
            dropout_probability=args["dropout_probability"],
            num_classes=args["num_classes"], learnable_pe=args["learnable_pe"]
        ).to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args["label_smoothing"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args["epochs"], eta_min=args["eta_min"])

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
            torch.save(model.state_dict(), os.path.join(model_dir, f"vit-{args['variant']}-highest-accuracy.pth"))
            print("New maximum accuracy — model saved.")

        if epoch % 5 == 0:
            checkpoint = get_checkpoint(epoch, model, optimizer, scheduler)
            torch.save(checkpoint, os.path.join(model_dir, f"vit-{args['variant']}-latest-checkpoint.pth"))
            print("Checkpoint saved.")

        running_val_loss.append(val_loss)
        running_val_accuracy.append(val_accuracy)

        scheduler.step()

        print("-------------------------------------------------------------------\n")

    torch.save(checkpoint, os.path.join(model_dir, f"vit-{args['variant']}-latest-checkpoint.pth"))

if __name__ == "__main__":
    main()