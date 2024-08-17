import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    """
    An implementation of the Vision Transformer from 
    `AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE`.(https://arxiv.org/abs/2010.11929)

    Parameters
    ----------
    patch_size: int
        The size of each image patch.

    num_channels: int
        The number of input channels.

    hidden_size: int
        The dimension of the output embedding.

    sequence_len: int
        The length of the output sequence

    num_heads: int
        The number of self attention heads.

    num_layers: int
        The number of transformer encoder layers.
    
    mlp_size: int
        The hidden size of the MLP heads.

    dropout_probability: float
        The probability of dropouts within each dense layer, together with the 
        combined embedding from position encoding 
        (excludes the dense layers within the query, key, value projections).

    num_classes: int
        The number of classes to be predicted.

    learnable_pe: bool
        Whether to initialize a learnable position embedding.
        If false returns static Sinusoidal position embedding from `Attention Is All You Need`. 
        (https://arxiv.org/abs/1706.03762)

    Returns
    -------
    logits: torch.Tensor
        The raw logits of the classification head,
        projected from the final hidden state of the cls token.
    """

    def __init__(
        self, patch_size: int, num_channels: int, hidden_size: int, 
        sequence_len: int, num_heads: int, num_layers: int, mlp_size: int,
        dropout_probability: int, num_classes: int, learnable_pe: bool
        ) -> torch.Tensor:

        super(VisionTransformer, self).__init__()

        self.patchifier = Patchifier(patch_size, num_channels, hidden_size)
        self.position_encoder = PositionEncoder(sequence_len, hidden_size, dropout_probability, learnable_pe)
        
        self.encoder = Encoder(*[
            TransformerBlock(
                num_heads, sequence_len, hidden_size, mlp_size, dropout_probability
                ) for _ in range(num_layers)
                ])

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.classification_head = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        patches = self.patchifier(x)
        x = self.position_encoder(patches)
        z = self.encoder(x)
        cls_token = self.layer_norm(z[:,0,:])

        logits = self.classification_head(cls_token)

        return logits

class Patchifier(nn.Module):
    """
    A class to patchify the input image.

    Parameters
    ----------
    patch_size: int
        The size of each patch.

    num_channels: int
        The number of channels in the input image.

    hidden_size: int
        The dimensions of the output embedding

    Returns
    -------
    x: torch.Tensor
        The flattened sequence of patches projected through 
        a linear transformation combined with the special CLS token.
    """

    def __init__(self, patch_size: int, num_channels: int, hidden_size: int) -> torch.Tensor:
        super(Patchifier, self).__init__()

        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.proj = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, img):
        batch_size = img.shape[0]

        linear_projection = self.proj(img)
        linear_projection = linear_projection.flatten(-2).permute(0, 2, 1)
        
        cls_token = self.cls_token.expand(batch_size, -1, -1) 
        x = torch.cat([cls_token, linear_projection], dim=1)

        return x

class PositionEncoder(nn.Module):
    """
    The position encoder to combine a position embedding to an input sequence.

    Parameters
    ----------
    sequence_len: int
        The length of the input sequence, i.e., the number of image patches.

    hidden_size: int
        The dimension of the output embedding.

    dropout_probability: float
        The probability of dropout for the combined representation from the initial
        linear projection and positional encoding.

    learnable: bool
        Whether to initialize a learnable position embedding.
        If false returns static Sinusoidal position embedding from `Attention Is All You Need`. 
        (https://arxiv.org/abs/1706.03762)

    Returns
    -------
    combined_embedding: torch.Tensor
        The input embedding combined with the positional embedding.
    """

    def __init__(
        self, sequence_len: int, hidden_size: int, 
        dropout_probability: float, learnable: bool
        ) -> torch.Tensor:
        
        super(PositionEncoder, self).__init__()

        assert hidden_size % 2 == 0, "Ensure embedding length is even for convenience and efficiency"

        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.learnable = learnable
        self.position_encoding = nn.Parameter(torch.zeros(sequence_len, hidden_size)) if learnable else self.sinusoidal_pe()

        self.dropout = nn.Dropout(p=dropout_probability)

    def sinusoidal_pe(self):
        max_iter = self.hidden_size // 2
        pos = torch.arange(self.sequence_len, dtype=torch.float).unsqueeze(1)

        i = torch.arange(max_iter, dtype=torch.float)

        angle_rates = 1 / (torch.pow(10_000, (2 * i) / self.hidden_size)).unsqueeze(0)
        x = pos * angle_rates

        sin_pe = torch.sin(x)
        cos_pe = torch.cos(x)

        position_encoding = torch.zeros(self.sequence_len, self.hidden_size)
        position_encoding[:, 0::2] = sin_pe
        position_encoding[:, 1::2] = cos_pe

        return position_encoding

    def forward(self, x):
        device = x.device
        position_encoding = self.position_encoding.unsqueeze(0)
        
        if not self.learnable:
            position_encoding = position_encoding.to(device)

        combined_embedding = x + position_encoding
        combined_embedding = self.dropout(combined_embedding)

        return combined_embedding

class MultiHeadAttention(nn.Module):
    """
    Takes in a tensor of shape (batch size, sequence length, embedding dimension) and returns the attention-informed embedding.

    Parameters
    ----------
    num_heads: int
        The number of attention heads to employ.

    sequence_length: int
        The length of the sequence, i.e., the number of image patches.

    hidden_size: int
        The output embedding dimension of the attention module.
    
    dropout_probability: float
        The probability of dropout being applied to the attention probabilities.

    Returns
    -------
    final_representation: torch.Tensor
        The attention weighted representation of the model for each sequence/image patch.

    Attributes
    ----------
    head_dim: int
        The output dimension of each attention head.

    q_w: nn.Linear
        The linear transformation weights for the query vector.

    k_w: nn.Linear
        The linear transformation weights for the key vector.

    v_w: nn.Linear
        The linear transformation weights for the value vector.
    """

    def __init__(self, num_heads: int, sequence_length: int, hidden_size: int, dropout_probability: float) -> torch.Tensor:
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_size % num_heads == 0, "embedding dimension must be divisible by the number of attention heads"

        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.embed_dim = hidden_size
        self.head_dim = hidden_size // num_heads

        self.q_w = nn.Linear(hidden_size, hidden_size)
        self.k_w = nn.Linear(hidden_size, hidden_size)
        self.v_w = nn.Linear(hidden_size, hidden_size)
    
        self.final_projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, x):
        batch_size = x.shape[0]

        v = self.v_w(x).reshape(batch_size, self.sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_w(x).reshape(batch_size, self.sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_w(x).reshape(batch_size, self.sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_embedding = torch.matmul(attention_weights, v)

        combined_heads = attention_embedding.transpose(1, 2).reshape(batch_size, self.sequence_length, self.embed_dim)
        final_representation = self.final_projection(combined_heads)
        final_representation = self.dropout(final_representation)

        return final_representation

class TransformerBlock(nn.Module):
    """
    Transformer Encoder

    Parameters
    ----------
    num_heads: int
        The number of attention heads to employ.

    sequence_length: int
        The length of the sequence, i.e., the number of image patches.

    hidden_size: int
        The output embedding dimension of the attention module.

    mlp_size: int
        The hidden size of the MLP layer.

    Returns
    -------
    z_l: torch.Tensor
        The normalized output of the final dense layer in the transformer encoder.
    """
    
    def __init__(
        self, num_heads: int, sequence_length: int, hidden_size: int, 
        mlp_size: int, dropout_probability: float
        ) -> torch.Tensor:

        super(TransformerBlock, self).__init__()

        self.multihead_attention = MultiHeadAttention(
            num_heads, sequence_length, hidden_size, dropout_probability
            )

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.mlp = MLP(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(mlp_size, hidden_size),
            nn.Dropout(p=dropout_probability)
        )

    def forward(self, x):
        normalized_x = self.layer_norm1(x)
        z = self.multihead_attention(normalized_x) + x

        normalized_z = self.layer_norm2(z)
        z = self.mlp(normalized_z) + z

        return z 

class MLP(nn.Sequential):
    """
    A wrapper for the sequential module. Done for the purpose of better 
    labelling within the visualization of the model's architecture.
    """
    
    def __init__(self, *args):
        super(MLP, self).__init__(*args)

class Encoder(nn.Sequential):
    """
    A wrapper for the sequential module. Done for the purpose of better 
    labelling within the visualization of the model's architecture.
    """

    def __init__(self, *args):
        super(Encoder, self).__init__(*args)