from .ViT import VisionTransformer

def get_model(
        img_size: int,
        patch_size: int, 
        variant: str, 
        dropout_probability: float,
        num_classes: int,
        learnable_pe: bool
        ) -> VisionTransformer:
    
    """
    Initializes a ViT of the desired variant.

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

    learnable_pe: bool
        Whether to initialize a learnable position embedding.
        If false returns static Sinusoidal position embedding from `Attention Is All You Need`. 
        (https://arxiv.org/abs/1706.03762)

    Returns
    -------
    model: VisionTransformer
        The ViT variant to be used.
    """

    num_channels = 3
    sequence_len = ((img_size // patch_size) ** 2) + 1

    if variant == "small":
        hidden_size = 64
        num_heads = 4
        num_layers = 4
        mlp_size = 256      

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
                mlp_size=mlp_size, dropout_probability=dropout_probability, 
                num_classes=num_classes, learnable_pe=learnable_pe
            )
    
    return model