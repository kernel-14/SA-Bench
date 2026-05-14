"""
utils.py

Utility functions for the nGPT reproduction project.

Contains:
- set_seed : set random seeds for reproducibility.
- normalize_weights : enforce hyperspherical constraints on model parameters.
- lerp_update : compute normalized spherical interpolation update step.
- init_weights : initialise weights according to the paper's configuration.
"""

import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
    Also configures cuDNN deterministic mode.

    Args:
        seed: Integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For full reproducibility, we enable deterministic algorithms.
        # This may impact performance but is recommended for exact reproduction.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_weights(model: nn.Module) -> None:
    """
    Enforce that all matrix weights and embedding tables reside on the unit hypersphere.

    This function iterates over all named parameters of the model and normalises
    those that correspond to the weight matrices listed in Section 2.6 of the paper:
        - Input & output embeddings (normalised along dimension 1, i.e. each row)
        - Attention Q, K, V, O projections (normalised along dimension 0, i.e. each column)
        - MLP U, V, O_MLP projections (normalised along dimension 0)

    The operation is performed **in place** on the parameter data (no gradient tracking).
    It is meant to be called after every optimizer step in the nGPT variant.

    Args:
        model: The nGPT model whose weights should be projected onto the sphere.
    """
    # Suffixes that identify matrices to normalise along the column (dim=0)
    linear_targets = {"w_q", "w_k", "w_v", "w_o", "w_u", "w_v", "w_o_mlp"}
    # Suffixes for embeddings (dim=1)
    emb_targets = {"input_embed", "output_embed"}

    for name, param in model.named_parameters():
        # Linear weights: shape (out_features, in_features) -> normalise columns (dim=0)
        if any(name.endswith(f"{t}.weight") for t in linear_targets):
            # Use float32 for numerical stability and cast back to original dtype
            param.data = F.normalize(param.data.float(), dim=0, p=2).to(param.dtype)
        # Embedding weights: shape (vocab_size, d_model) -> normalise rows (dim=1)
        elif any(name.endswith(f"{t}.weight") for t in emb_targets):
            param.data = F.normalize(param.data.float(), dim=1, p=2).to(param.dtype)
        # All other parameters (biases, scaling factors, etc.) are left unchanged.


def lerp_update(
    h: torch.Tensor,
    h_block: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Perform the normalised update step that approximates SLERP (Spherical Linear
    Interpolation).  Implements Equation 7 / 10 / 11 from the paper:

        h_new = Norm( h + α ⊙ (h_block − h) )

    where `h` and `h_block` are points on the unit hypersphere, `α` is a vector of
    per‑dimension eigen learning rates (non‑negative), and `Norm` is L2 normalisation
    along the last dimension.

    Args:
        h: Current hidden state on the sphere, shape (..., d_model).
        h_block: Output of attention or MLP block, already normalised.
        alpha: Eigen learning rates, shape (d_model,) broadcastable to h.

    Returns:
        Updated hidden state, still on the sphere, same shape as h.
    """
    delta = h_block - h                          # update direction
    scaled_delta = alpha * delta                 # element‑wise scaling per dimension
    new_h = h + scaled_delta                     # unnormalised point
    return F.normalize(new_h, dim=-1, p=2)       # retract back to the sphere


def init_weights(module: nn.Module, config: Config) -> None:
    """
    Initialize weights of linear and embedding layers according to the paper.

    The initialisation depends on whether we are training the baseline GPT or the
    normalised Transformer (nGPT).  Output projection matrices of residual blocks
    receive an additional scaling factor of ``1 / sqrt(2 * n_layers)``, as described
    in Radford et al. (2018) and followed in the nGPT paper.

    Args:
        module: A PyTorch module (either nn.Linear or nn.Embedding).
        config: The global configuration object.
    """
    # Only process modules that have a weight attribute (Linear / Embedding)
    if not hasattr(module, "weight") or module.weight is None:
        return

    # Determine the base standard deviation
    if config.model.use_ngpt:
        std = config.model.init_scale_ngpt
    else:
        std = config.model.init_scale_norm

    if isinstance(module, nn.Linear):
        # Output projections get an extra factor
        is_output = getattr(module, "is_output_projection", False)
        if is_output:
            std *= 1.0 / math.sqrt(2.0 * config.model.n_layers)

        # Initialise weight with a normal distribution
        module.weight.data.normal_(mean=0.0, std=std)

        # Bias is initialised to zero (paper omits bias, but we handle it if present)
        if module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, nn.Embedding):
        # Embedding weight (vocab_size × d_model)
        module.weight.data.normal_(mean=0.0, std=std)
        # No bias for nn.Embedding
    # Other module types (e.g., ScaledParam) are ignored here.

