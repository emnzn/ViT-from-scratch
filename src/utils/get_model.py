from .ViT import VisionTransformer

def get_model(
        img_size: int,
        patch_size: int, 
        variant: str, 
        dropout_probability: float,
        num_classes: int
        ) -> VisionTransformer:

    num_channels = 3
    sequence_len = ((img_size // patch_size) ** 2) + 1

    if variant == "base":
        hidden_size = 768
        num_heads = 12
        num_layers = 12
        mlp_size = 3072

    elif variant == "large":
        hidden_size = 1024
        num_heads = 16
        num_layers = 24
        mlp_size = 4098

    elif variant == "huge":
        hidden_size = 1280
        num_heads = 16
        num_layers = 32
        mlp_size = 5120


    model = VisionTransformer(
                patch_size=patch_size, num_channels=num_channels, hidden_size=hidden_size,
                sequence_len=sequence_len, num_heads=num_heads, num_layers=num_layers,
                mlp_size=mlp_size, dropout_probability=dropout_probability, num_classes=num_classes
            )
    
    return model