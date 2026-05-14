## utils/positional_encodings.py
"""
Fixed sinusoidal temporal positional encodings and cyclic indexing.

These functions implement the Cyclic‑TPE mechanism described in the
Ca2‑VDM paper: a base sinusoidal table is generated once, and temporal
position embeddings are retrieved via cyclic look‑up using modulo
arithmetic. This enables training/inference alignment when the temporal
KV‑cache queue exceeds the training sequence length.
"""

import math
from typing import List, Union

import torch


def get_sinusoidal_encoding(seq_len: int, dim: int) -> torch.Tensor:
    """
    Generate a sinusoidal positional encoding table of shape (seq_len, dim).

    The encoding follows the original Transformer scheme:

        PE_(pos, 2i)   = sin(pos / 10000^(2i / dim))
        PE_(pos, 2i+1) = cos(pos / 10000^(2i / dim))

    Args:
        seq_len: Maximum sequence length for which encodings are prepared.
        dim: Encoding dimension (must be even).

    Returns:
        Float tensor of shape ``(seq_len, dim)`` containing the positional
        encodings (float32 by default).
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )  # (dim//2,)

    angles = position * div_term  # (seq_len, dim//2)

    # interleave sin and cos
    encodings = torch.empty(seq_len, dim, dtype=torch.float32)
    encodings[:, 0::2] = torch.sin(angles)
    encodings[:, 1::2] = torch.cos(angles)

    return encodings


def cyclic_temporal_pos_embed(
    positions: Union[int, torch.Tensor, List[int]],
    base_table: torch.Tensor,
    L_train_max: int,
) -> torch.Tensor:
    """
    Retrieve temporal positional embeddings via cyclic look‑up.

    For each requested position ``p``, the function computes
    ``idx = p % L_train_max`` and returns ``base_table[idx]``.
    This implements the Cyclic‑TPE scheme that allows the model to
    correctly assign TPEs when the autoregressive generation exceeds
    the training length.

    Args:
        positions: A single integer or a 1‑D list/tensor of position
            indices (non‑negative).  If a scalar is given, the output
            will be squeezed to a 1‑D tensor of shape ``(dim,)``.
        base_table: Pre‑computed sinusoidal encoding table of shape
            ``(L_train_max, dim)``.
        L_train_max: The number of unique positions in ``base_table``
            (must equal ``base_table.shape[0]``).

    Returns:
        Tensor of shape ``(len(positions), dim)`` (or ``(dim,)`` if
        ``positions`` is a scalar) with the corresponding positional
        embeddings.  The dtype matches ``base_table.dtype``.

    Examples:
        >>> base = get_sinusoidal_encoding(65, 512)
        >>> cyclic_temporal_pos_embed([0, 1, 2], base, 65).shape
        torch.Size([3, 512])
        >>> cyclic_temporal_pos_embed(66, base, 65).shape   # uses index 1
        torch.Size([512])
    """
    scalar_input = isinstance(positions, (int, float))

    if scalar_input:
        positions = [int(positions)]

    # Convert to 1‑D LongTensor
    if not isinstance(positions, torch.Tensor):
        positions = torch.tensor(positions, dtype=torch.long)
    else:
        positions = positions.long()

    if positions.dim() != 1:
        raise ValueError(f"positions must be 1‑D, got shape {positions.shape}")

    indices = positions % L_train_max
    embeddings = base_table[indices]  # shape (num_positions, dim)

    if scalar_input:
        embeddings = embeddings.squeeze(0)  # (dim,)

    return embeddings

