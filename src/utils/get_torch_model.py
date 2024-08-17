import torch
from torchvision import models
from torchvision.models import VisionTransformer

def get_torch_model(
    img_size: int,
    patch_size: int,
    variant: str,
    dropout_probability: float,
    num_classes: int
    ) -> VisionTransformer:
    
    """
    Initializes a Vision Transformer from PyTorch to serve as a baseline for validating current implementation.

    Parameters
    ----------
    img_size: int
        The expected size of the input image.
        This implementation requires the input image shape to be (N x N).

    patch_size: int
        The desired patch size for the flattened image.

    variant: str
        The ViT variant to use.
        One of [base, large, huge].

    dropout_probability: float
        The probability of dropout in the output of each dense layer.
        (Excludes the dense layers within the query, key, and value projections).

    num_classes: int
        The number of target classes.
    """
    
    patch_lookup = {
        "base": [16, 32],
        "large": [16, 32],
        "huge": [14]
    }

    valid_patches = patch_lookup[variant]
    
    assert patch_size in valid_patches, f"Patch sizes must be one of {valid_patches} for selected Vit variant."

    if variant == "base":

        if patch_size == 16:
            model = models.vit_b_16(
                weights=None, image_size=img_size, dropout=dropout_probability, 
                attention_dropout=dropout_probability, num_classes=num_classes
                )
            
        elif patch_size == 32:
            model = models.vit_b_32(
                weights=None, image_size=img_size, dropout=dropout_probability, 
                attention_dropout=dropout_probability, num_classes=num_classes
                )
    
    if variant == "large":

        if patch_size == 16:
            model = models.vit_l_16(
                weights=None, image_size=img_size, dropout=dropout_probability, 
                attention_dropout=dropout_probability, num_classes=num_classes
                )
            
        elif patch_size == 32:
            model = models.vit_l_32(
                weights=None, image_size=img_size, dropout=dropout_probability, 
                attention_dropout=dropout_probability, num_classes=num_classes
                )
            
    if variant == "huge":
        model = models.vit_h_14(
            weights=None, image_size=img_size, dropout=dropout_probability, 
            attention_dropout=dropout_probability, num_classes=num_classes
            )
        
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model