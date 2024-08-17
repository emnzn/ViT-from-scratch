from torch import optim
from typing import Dict, Any
from .vit import VisionTransformer

def get_checkpoint(
    epoch: int,
    model: VisionTransformer, 
    optimizer: optim, 
    scheduler: optim.lr_scheduler
    ) -> Dict[str, Any]:
    
    """
    Creates a checkpoint to be saved.

    Parameters
    ----------
    epoch: int
        The current epoch.

    model: VisionTransformer
        The model to be saved for the checkpoint.

    optimizer: optim
        The optimizer being used.

    scheduler: optim.lr_scheduler
        the scheduler being used.

    Returns
    -------
    checkpoint: Dict[str, Any]
        The checkpoint containing the current epoch, model, 
        optimizer, and learning rate scheduler.
    """

    checkpoint = {
        "epoch": epoch,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler
    }

    return checkpoint
