import torch
from tqdm import tqdm
from torchvison import transforms

def train(dataloader, criterion, optimizer, model, device):
    model.train()

    for img, label in tqdm(dataloader, desc="Training Model"):
        logits = model(img)
        
        
        optimizer.zero_grad()



def validate(dataloader, optimizer, model, device):
    pass

def main():
    pass
