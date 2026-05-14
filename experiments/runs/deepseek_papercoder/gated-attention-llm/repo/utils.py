## utils.py

import yaml
import random
import numpy as np
import torch
from typing import Dict, Optional


def load_config(path: str) -> Dict:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration sections (model, training, data, etc.).
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (including CUDA) to ensure reproducibility.

    Args:
        seed: The integer seed to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adjust_intermediate_size(config: Dict) -> int:
    """
    Compensate for added gate parameters by reducing the FFN intermediate size so that
    the total number of model parameters remains identical to the baseline (non‑gated) model.

    The formula follows the paper's approach: when gating is enabled, the FFN width is shrunk
    to keep the parameter count unchanged. This function calculates the new 'intermediate_size'
    that should be used in place of the original one.

    Args:
        config: The full configuration dictionary (must contain a 'model' sub‑dictionary).

    Returns:
        The new intermediate_size for the SwiGLU FFN layers. If gating is not enabled,
        the original intermediate_size is returned unchanged.
    """
    model_cfg = config["model"]
    if not model_cfg.get("use_gated_attention", False):
        return model_cfg["intermediate_size"]

    hidden_size = model_cfg["hidden_size"]
    num_layers = model_cfg["num_layers"]
    original_intermediate = model_cfg["intermediate_size"]

    # Compute the number of parameters required by the gate per layer
    score_dim = _score_dim_for_gate(model_cfg)
    gate_params_per_layer = hidden_size * score_dim
    total_gate_params = num_layers * gate_params_per_layer

    # Each FFN layer has 3 linear weights of shape (hidden_size x intermediate) without bias.
    # The total FFN parameters per layer is therefore 3 * hidden_size * intermediate.
    # To compensate for the gate parameters, we reduce the FFN parameter budget accordingly.
    # new_intermediate = old_intermediate - gate_params_total / (3 * num_layers * hidden_size)
    intermediate_new = original_intermediate - total_gate_params / (3 * num_layers * hidden_size)
    intermediate_new = int(intermediate_new)

    if intermediate_new < 1:
        raise ValueError(
            f"Gate parameters ({total_gate_params}) exceed the original FFN parameters "
            f"({3 * num_layers * hidden_size * original_intermediate}). "
            "Unable to match parameter count; consider increasing the original intermediate size."
        )

    # Warn if integer truncation leads to a noticeable parameter mismatch (> 0.01% of one FFN layer)
    param_reduction = 3 * num_layers * hidden_size * (original_intermediate - intermediate_new)
    if abs(total_gate_params - param_reduction) > 0.01 * (3 * hidden_size):
        print(
            f"Warning: Small parameter mismatch due to integer rounding. "
            f"Gate params = {total_gate_params}, actual reduction = {param_reduction}"
        )

    return intermediate_new


# ----------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------

def _score_dim_for_gate(model_cfg: Dict) -> int:
    """
    Compute the output dimension (number of gating scores) of the gate's linear projection
    based on the gate configuration.

    The mapping follows Table 1 and the descriptions in Section 2.2.
    """
    pos = model_cfg["gate_position"]
    granularity = model_cfg["gate_granularity"]
    head_specific = model_cfg.get("gate_head_specific", True)

    # Special case: gating after the final dense output layer
    if pos == "dense_output":
        # Score shape is n × d_model for elementwise, 1 for headwise (not used in paper)
        return model_cfg["hidden_size"] if granularity == "elementwise" else 1

    # Determine the number of heads relevant for the gate position
    if pos in ("query", "SDPA_output"):
        n_heads = model_cfg["num_attention_heads"]
    elif pos in ("key", "value"):
        n_heads = model_cfg["num_key_value_heads"]
    else:
        raise ValueError(f"Unknown gate_position: '{pos}'")

    head_dim = model_cfg["head_dim"]

    if granularity == "elementwise":
        return n_heads * head_dim if head_specific else head_dim
    else:  # headwise
        return n_heads if head_specific else 1
