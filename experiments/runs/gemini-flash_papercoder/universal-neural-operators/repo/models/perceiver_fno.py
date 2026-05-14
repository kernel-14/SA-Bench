import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Dict, Any, Union, List, Tuple

from models.base_operator import CoreOperator
from models.fno import FNO
from utils import get_activation_fn


class PerceiverAttention(nn.Module):
    """
    Implements a standard multi-head attention (MHA) block, adaptable for
    self-attention and cross-attention operations.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        """
        Initializes the PerceiverAttention module.

        Args:
            dim (int): The input and output feature dimension.
            heads (int): The number of attention heads. Defaults to 8.
            dim_head (int): The dimension of each attention head. Defaults to 64.
        """
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if not isinstance(heads, int) or heads <= 0:
            raise ValueError(f"heads must be a positive integer, got {heads}")
        if not isinstance(dim_head, int) or dim_head <= 0:
            raise ValueError(f"dim_head must be a positive integer, got {dim_head}")

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        # Linear layers for queries, keys, and values. Bias is typically False for QKV in transformers.
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        # Output linear layer
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the attention mechanism.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, query_seq_len, dim).
            key (torch.Tensor): Key tensor of shape (batch_size, key_seq_len, dim).
            value (torch.Tensor): Value tensor of shape (batch_size, value_seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, query_seq_len, dim).
        """
        h = self.heads

        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)

        # Reshape for multi-head attention: (batch, seq_len, inner_dim) -> (batch, heads, seq_len, dim_head)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        # Compute attention scores
        # scores: (batch, heads, query_seq_len, key_seq_len)
        scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # Apply softmax to get attention weights
        attn = scores.softmax(dim=-1)

        # Compute weighted sum of values
        # out: (batch, heads, query_seq_len, dim_head)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        # Reshape back and apply output linear layer
        # (batch, heads, query_seq_len, dim_head) -> (batch, query_seq_len, inner_dim) -> (batch, query_seq_len, dim)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PerceiverIOModule(nn.Module):
    """
    Implements the core logic of a Perceiver IO block as described in the paper.
    It uses cross-attention to map input features to a set of latent vectors,
    self-attention on these latents, and then cross-attention to map latents
    back to the output feature space. FNOs are used to generate keys and values
    from the input for the initial cross-attention.
    """

    def __init__(
        self,
        input_dim: int,          # Feature dimension of the input from LiftingAdapter
        latent_dim: int,         # Feature dimension of the latent vectors (should be input_dim for consistency)
        num_latents: int,        # Number of learnable latent vectors
        num_blocks: int,         # Number of self-attention blocks applied to the latent array
        num_heads: int,          # Number of attention heads for all attention modules
        inner_fno_config: Dict[str, Any], # Config for FNOs generating K/V from input
        activation: str = 'gelu',# Activation function name
        dim_head: Optional[int] = None # Dimension per attention head, if None, calculated as latent_dim // num_heads
    ):
        """
        Initializes the PerceiverIOModule.

        Args:
            input_dim (int): Dimensionality of the input features coming from the LiftingAdapter.
                             This should typically match `config.model.hidden_dim`.
            latent_dim (int): Dimensionality of the latent vectors. This should also typically
                              match `config.model.hidden_dim` for channel consistency.
            num_latents (int): The number of latent vectors to use in the Perceiver.
            num_blocks (int): The number of self-attention blocks to apply to the latent array.
            num_heads (int): The number of attention heads to use in all attention mechanisms.
            inner_fno_config (Dict[str, Any]): Configuration dictionary for the FNOs used to
                                                generate keys and values from the input.
            activation (str): Name of the activation function to use in FFNs.
            dim_head (Optional[int]): The dimension of each attention head. If None, it defaults to
                                      `latent_dim // num_heads`.
        """
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError(f"latent_dim must be a positive integer, got {latent_dim}")
        if not isinstance(num_latents, int) or num_latents <= 0:
            raise ValueError(f"num_latents must be a positive integer, got {num_latents}")
        if not isinstance(num_blocks, int) or num_blocks <= 0:
            raise ValueError(f"num_blocks must be a positive integer, got {num_blocks}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads}")
        if not isinstance(inner_fno_config, dict):
            raise TypeError(f"inner_fno_config must be a dictionary, got {type(inner_fno_config)}")
        if not isinstance(activation, str):
            raise TypeError(f"activation must be a string, got {type(activation)}")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        
        self.dim_head = dim_head if dim_head is not None else (latent_dim // num_heads)
        if self.dim_head == 0:
            raise ValueError("dim_head calculated as 0. latent_dim must be divisible by num_heads.")

        # Learnable latent vectors
        self.latent_queries = nn.Parameter(torch.randn(num_latents, latent_dim))

        # FNOs to generate keys and values from the input for cross-attention
        # These FNOs will operate on input_dim features and output latent_dim features.
        # Assuming FNO's hidden_dim is consistent with the latent_dim.
        # The inner_fno_config provides other parameters like modes, layers, mlp_width.
        self.fno_k = FNO(hidden_dim=input_dim, **inner_fno_config)
        self.fno_v = FNO(hidden_dim=input_dim, **inner_fno_config)

        # Helper for FFNs
        def _build_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                get_activation_fn(activation),
                nn.Linear(out_dim, out_dim)
            )

        # Input-to-Latent Cross-Attention (I->L CA)
        self.norm_input_to_latent = nn.LayerNorm(latent_dim)
        self.attn_input_to_latent = PerceiverAttention(latent_dim, num_heads, self.dim_head)
        self.ffn_input_to_latent = _build_mlp(latent_dim, latent_dim)

        # Latent Self-Attention (L SA) Blocks
        self.latent_self_attention_blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.latent_self_attention_blocks.append(nn.ModuleList([
                nn.LayerNorm(latent_dim),
                PerceiverAttention(latent_dim, num_heads, self.dim_head),
                nn.LayerNorm(latent_dim),
                _build_mlp(latent_dim, latent_dim)
            ]))

        # Projection for output queries (if input_dim != latent_dim, need to project input for query)
        # However, typically input_dim == latent_dim based on the overall hidden_dim
        self.query_proj_for_output = nn.Identity() if input_dim == latent_dim else nn.Linear(input_dim, latent_dim)

        # Latent-to-Output Cross-Attention (L->O CA)
        self.norm_latent_to_output_query = nn.LayerNorm(latent_dim)
        self.norm_latent_to_output_kv = nn.LayerNorm(latent_dim)
        self.attn_latent_to_output = PerceiverAttention(latent_dim, num_heads, self.dim_head)
        self.ffn_latent_to_output = _build_mlp(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the PerceiverIOModule.

        Args:
            x (torch.Tensor): Input tensor from the LiftingAdapter.
                              Expected shape: (batch_size, spatial_h, spatial_w, input_dim).

        Returns:
            torch.Tensor: Output tensor representing the transformed hidden features.
                          Expected shape: (batch_size, spatial_h, spatial_w, latent_dim).
        """
        batch_size, H, W, _ = x.shape
        
        # 1. Prepare Input for K/V Generation using FNOs
        # FNO expects input (B, C, H, W), so permute from (B, H, W, C)
        x_permuted = x.permute(0, 3, 1, 2)
        
        # Apply FNOs and flatten spatial dimensions for attention
        # FNO output is (B, H, W, latent_dim), then permute to (B, H, W, C) then flatten to (B, H*W, C)
        k_from_input = self.fno_k(x_permuted).permute(0, 2, 3, 1).flatten(1, 2) # (B, H*W, latent_dim)
        v_from_input = self.fno_v(x_permuted).permute(0, 2, 3, 1).flatten(1, 2) # (B, H*W, latent_dim)

        # Flatten original input for later use as queries in L->O CA
        input_flat_for_output_queries = x.flatten(1, 2) # (B, H*W, input_dim)

        # 2. Initialize Latents: (num_latents, latent_dim) -> (batch_size, num_latents, latent_dim)
        latents = repeat(self.latent_queries, 'n d -> b n d', b=batch_size)

        # 3. Input-to-Latent Cross-Attention (I->L CA)
        # Query: latents, Key/Value: from input (k_from_input, v_from_input)
        normed_latents = self.norm_input_to_latent(latents)
        attended_latents = self.attn_input_to_latent(query=normed_latents, key=k_from_input, value=v_from_input)
        latents = latents + attended_latents
        latents = self.norm_input_to_latent(latents + self.ffn_input_to_latent(latents))

        # 4. Latent Self-Attention (L SA) Blocks
        for norm1, attn, norm2, ffn in self.latent_self_attention_blocks:
            normed_latents = norm1(latents)
            attended_latents = attn(query=normed_latents, key=normed_latents, value=normed_latents)
            latents = latents + attended_latents
            latents = norm2(latents + ffn(latents))

        # 5. Latent-to-Output Cross-Attention (L->O CA)
        # Query: projected original input, Key/Value: processed latents
        projected_input_queries = self.query_proj_for_output(input_flat_for_output_queries) # (B, H*W, latent_dim)
        normed_queries = self.norm_latent_to_output_query(projected_input_queries)
        normed_latents_for_kv = self.norm_latent_to_output_kv(latents)

        output = self.attn_latent_to_output(
            query=normed_queries,
            key=normed_latents_for_kv,
            value=normed_latents_for_kv
        )
        output = projected_input_queries + output # Residual connection
        output = self.norm_latent_to_output_query(output + self.ffn_latent_to_output(output))

        # Reshape output back to (batch_size, H, W, latent_dim)
        output = output.view(batch_size, H, W, self.latent_dim)
        return output


class PerceiverFNO(CoreOperator):
    """
    Implements the Perceiver IO-based Neural Operator as a concrete CoreOperator.
    This class wraps the `PerceiverIOModule` to fit into the overall Lifting-Operator-Projection
    framework.
    """

    def __init__(self, hidden_dim: int, perceiver_config: Dict[str, Any]):
        """
        Initializes the PerceiverFNO model.

        Args:
            hidden_dim (int): The dimensionality of the hidden feature representation
                              from the LiftingAdapter and expected by the ProjectionAdapter.
                              This value comes from `config.model.hidden_dim` and is used
                              as both `input_dim` and `latent_dim` for the `PerceiverIOModule`.
            perceiver_config (Dict[str, Any]): Configuration dictionary for the Perceiver.
                                                Expected keys: 'num_latents', 'num_blocks',
                                                'num_heads', 'inner_fno_config'.
        """
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(perceiver_config, dict):
            raise TypeError(f"perceiver_config must be a dictionary, got {type(perceiver_config)}")

        # Extract parameters from perceiver_config with default values
        num_latents = perceiver_config.get('num_latents', 64)
        num_blocks = perceiver_config.get('num_blocks', 2)
        num_heads = perceiver_config.get('num_heads', 8)
        inner_fno_config = perceiver_config.get('inner_fno_config', {}) # Default to empty dict

        # The 'dim_head' for PerceiverAttention is often derived.
        # It's not explicitly in perceiver_config but can be inferred or set as default.
        dim_head = perceiver_config.get('dim_head', None) # Pass None to let PerceiverAttention calculate it

        # Assuming activation is consistent across the model, pull from a general model config if available
        # Here we hardcode 'gelu' as a reasonable default for Perceiver's internal FFNs
        activation = perceiver_config.get('activation', 'gelu') 

        # Initialize the PerceiverIOModule
        self.perceiver_module = PerceiverIOModule(
            input_dim=hidden_dim,
            latent_dim=hidden_dim, # For consistency, latent_dim matches the overall hidden_dim
            num_latents=num_latents,
            num_blocks=num_blocks,
            num_heads=num_heads,
            inner_fno_config=inner_fno_config,
            activation=activation,
            dim_head=dim_head
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the PerceiverFNO.

        Args:
            x (torch.Tensor): Input tensor from the LiftingAdapter.
                              Expected shape: (batch_size, spatial_h, spatial_w, hidden_dim).

        Returns:
            torch.Tensor: Output tensor from the PerceiverIOModule.
                          Expected shape: (batch_size, spatial_h, spatial_w, hidden_dim).
        """
        return self.perceiver_module(x)

