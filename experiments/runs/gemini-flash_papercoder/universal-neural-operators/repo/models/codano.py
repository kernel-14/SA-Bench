import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, Any, Union, List, Tuple, Optional

from models.base_operator import CoreOperator
from utils import get_activation_fn


class CodomainAttention(nn.Module):
    """
    Implements a multi-head codomain attention mechanism.
    In codomain attention, similarity is computed between feature dimensions
    (the 'codomain') rather than sequence elements.
    """

    def __init__(self, dim: int, heads: int):
        """
        Initializes the CodomainAttention module.

        Args:
            dim (int): The input and output feature dimension of the attention module.
                       This is the dimension across which attention will be computed
                       when reshaped.
            heads (int): The number of attention heads.
        """
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if not isinstance(heads, int) or heads <= 0:
            raise ValueError(f"heads must be a positive integer, got {heads}")
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")

        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        # Linear layer to project input to Query, Key, and Value vectors
        # For self-attention, a single input 'x' is projected into Q, K, V.
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        # Output linear layer to combine results from all heads
        self.to_out = nn.Linear(dim, dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for codomain attention.

        In the context of `CoDANoBlock`, `queries`, `keys`, and `values` will
        typically be the same input tensor (self-attention).

        Args:
            queries (torch.Tensor): Query tensor of shape (batch_size, sequence_length, dim).
            keys (torch.Tensor): Key tensor of shape (batch_size, sequence_length, dim).
            values (torch.Tensor): Value tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim).
        """
        B, L, D = queries.shape
        H = self.heads
        Dh = self.head_dim

        # Project input to Q, K, V and split for multi-head attention
        # In this self-attention context, we project queries into QKV.
        # qkv shape: (B, L, D*3) -> split into (B, L, D) for q, k, v
        qkv = self.to_qkv(queries).chunk(3, dim=-1) # Returns tuple of 3 tensors
        
        # Rearrange to (B, H, L, Dh) for standard multi-head attention processing
        q, k, v = map(lambda t: rearrange(t, 'b l (h dh) -> b h l dh', h=H, dh=Dh), qkv)

        # Codomain Transposition: Transpose 'L' (sequence length) and 'Dh' (head dimension)
        # to compute attention across features (codomain) rather than sequence positions.
        # Shape becomes (B, H, Dh, L)
        q_c = rearrange(q, 'b h l dh -> b h dh l')
        k_c = rearrange(k, 'b h l dh -> b h dh l')
        v_c = rearrange(v, 'b h l dh -> b h dh l')

        # Compute attention scores: (B, H, Dh, L) @ (B, H, L, Dh) -> (B, H, Dh, Dh)
        # This computes similarity between feature dimensions within each head.
        attn_scores = (q_c @ k_c.transpose(-2, -1)) * self.scale

        # Apply softmax to get attention weights over the feature dimension
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to values: (B, H, Dh, Dh) @ (B, H, Dh, L) -> (B, H, Dh, L)
        out = attn_weights @ v_c

        # Transpose back and combine heads: (B, H, Dh, L) -> (B, L, H, Dh) -> (B, L, D)
        out = rearrange(out, 'b h dh l -> b l h dh') # (B, L, H, Dh)
        out = rearrange(out, 'b l h dh -> b l (h dh)') # (B, L, D)

        # Apply output linear projection
        return self.to_out(out)


class CoDANoBlock(nn.Module):
    """
    Represents a single block of the Codomain Attention Neural Operator (CoDANo).
    It combines a CodomainAttention layer, layer normalization, and a Feed-Forward Network (FFN).
    """

    def __init__(self, dim: int, num_attention_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 activation: str = 'gelu'):
        """
        Initializes a single CoDANoBlock.

        Args:
            dim (int): The feature dimension for the block (i.e., `hidden_dim`).
            num_attention_heads (int): Number of heads for the `CodomainAttention`.
            mlp_ratio (float): Ratio to determine the hidden dimension of the FFN
                               (e.g., `dim * mlp_ratio`). Defaults to 4.0.
            dropout (float): Dropout rate applied after attention and within the FFN.
                             Defaults to 0.0.
            activation (str): Name of the activation function to use in the FFN.
        """
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if not isinstance(num_attention_heads, int) or num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be a positive integer, got {num_attention_heads}")
        if not isinstance(mlp_ratio, (int, float)) or mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be a positive number, got {mlp_ratio}")
        if not isinstance(dropout, (int, float)) or not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be a float between 0 and 1, got {dropout}")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = CodomainAttention(dim, heads=num_attention_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through one CoDANoBlock.
        Applies layer normalization, codomain attention, residual connection,
        another layer normalization, FFN, and a second residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of the same shape (batch_size, sequence_length, dim).
        """
        # Attention Sub-layer with residual connection
        norm_x_for_attn = self.norm1(x)
        attn_output = self.attn(norm_x_for_attn, norm_x_for_attn, norm_x_for_attn)
        x = x + self.dropout1(attn_output)

        # MLP Sub-layer with residual connection
        norm_x_for_mlp = self.norm2(x)
        mlp_output = self.mlp(norm_x_for_mlp)
        x = x + mlp_output # Dropout is internal to self.mlp

        return x


class CoDANo(CoreOperator):
    """
    The main Codomain Attention Neural Operator (CoDANo) model.
    It acts as a CoreOperator within the overall Lifting-Operator-Projection framework.
    It processes data using stacked CoDANoBlocks after flattening spatial dimensions,
    and reshapes it back for the ProjectionAdapter.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 num_attention_heads: int, num_layers: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, activation: str = 'gelu'):
        """
        Initializes the CoDANo model.

        Args:
            input_dim (int): The feature dimension of the input tensor received
                             from the LiftingAdapter (e.g., config.model.hidden_dim).
            output_dim (int): The feature dimension of the output tensor produced
                              for the ProjectionAdapter (e.g., config.model.hidden_dim).
            hidden_dim (int): The internal feature dimension used throughout the CoDANoBlocks.
            num_attention_heads (int): Number of attention heads for CodomainAttention
                                       within each block.
            num_layers (int): Number of stacked CoDANoBlock modules.
            mlp_ratio (float): Ratio for the hidden dimension of FFNs in CoDANoBlocks.
                               Defaults to 4.0 if not specified in config.
            dropout (float): Dropout rate for CoDANoBlocks. Defaults to 0.0 if not
                             specified in config.
            activation (str): Name of the activation function to use in CoDANoBlocks.
        """
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(num_attention_heads, int) or num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be a positive integer, got {num_attention_heads}")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers}")
        if not isinstance(mlp_ratio, (int, float)) or mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be a positive number, got {mlp_ratio}")
        if not isinstance(dropout, (int, float)) or not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be a float between 0 and 1, got {dropout}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Initial projection to map input_dim to hidden_dim if they differ.
        # Input from LiftingAdapter has shape (B, H, W, input_dim).
        # We process after flattening to (B, L, input_dim), so the projection
        # acts on the last dimension (feature dim).
        self.input_projection = nn.Identity()
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Stack multiple CoDANoBlocks
        self.blocks = nn.ModuleList([
            CoDANoBlock(
                dim=hidden_dim,
                num_attention_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])

        # Final projection to map hidden_dim to output_dim if they differ.
        self.output_projection = nn.Identity()
        if hidden_dim != output_dim:
            self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the entire CoDANo model.

        Args:
            x (torch.Tensor): Input tensor from the LiftingAdapter.
                              Expected shape: (batch_size, spatial_h, spatial_w, input_dim).

        Returns:
            torch.Tensor: Output tensor for the ProjectionAdapter.
                          Expected shape: (batch_size, spatial_h, spatial_w, output_dim).
        """
        # Ensure input dimensions are as expected
        if x.dim() != 4 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input to CoDANo to have shape (batch_size, H, W, input_dim={self.input_dim}), "
                f"but got {x.shape}"
            )

        B, H, W, C_in = x.shape

        # Flatten spatial dimensions: (B, H, W, C_in) -> (B, H*W, C_in)
        # This creates a 'sequence' of spatial points for the transformer-like blocks.
        x_flat = x.view(B, H * W, C_in)

        # Apply initial projection
        x_processed = self.input_projection(x_flat)

        # Pass through CoDANoBlocks
        for block in self.blocks:
            x_processed = block(x_processed)

        # Apply final projection
        x_final_flat = self.output_projection(x_processed)

        # Reshape back to original spatial dimensions for ProjectionAdapter
        # (B, H*W, output_dim) -> (B, H, W, output_dim)
        output = x_final_flat.view(B, H, W, self.output_dim)

        return output
