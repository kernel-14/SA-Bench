"""
Macro Action Value Function Estimation for MA-RLHF.

Implements three σ assignments for estimating macro action values
from token-level value functions:

1. Equal assignment: Each token contributes equally (σ_i = 1/|ω_τ|).
2. Unit assignment: Only the last token contributes (σ = [0,...,0,1]).
3. Position decayed assignment: Earlier tokens contribute more,
   with weights decaying by position.

Reference: Appendix D.1 of the MA-RLHF paper.
"""

import torch
from typing import List, Literal


def compute_macro_action_values_equal(
    values: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    sequence: List[int],
) -> torch.Tensor:
    """
    Equal assignment: each token in the macro action contributes equally.
    
    σ_τ = {1/|ω_τ|}_{i=1}^{τ}
    
    Args:
        values: Token-level values of shape (batch, seq_len).
        mask: Attention mask of shape (batch, seq_len).
        start: Starting index (prompt length - 1).
        sequence: List of macro action boundary positions.
    
    Returns:
        Macro action values of shape (batch, num_macro_actions).
    
    Reference: Appendix D.1, Section 1 (default assignment).
    """
    split_list = torch.diff(torch.tensor(sequence)).tolist()
    splited_values = torch.split(values[:, start:], split_list, dim=-1)
    splited_mask = torch.split(mask[:, start:], split_list, dim=-1)
    
    batch_size = values.size(0)
    inplace_values = torch.zeros(batch_size, len(split_list), 
                                  dtype=values.dtype, device=values.device)
    
    for idx, (value_i, mask_i) in enumerate(zip(splited_values, splited_mask)):
        masked_values = value_i[mask_i != 0]
        inplace_values[:, idx] = (
            torch.mean(masked_values) if masked_values.numel() > 0 else 0.0
        )
    
    return inplace_values


def compute_macro_action_values_unit(
    values: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    sequence: List[int],
) -> torch.Tensor:
    """
    Unit assignment: only the last token of the macro action contributes.
    
    σ_τ = {0, 0, ..., 0, 1}
    
    Args:
        values: Token-level values of shape (batch, seq_len).
        mask: Attention mask of shape (batch, seq_len).
        start: Starting index (prompt length - 1).
        sequence: List of macro action boundary positions.
    
    Returns:
        Macro action values of shape (batch, num_macro_actions).
    
    Reference: Appendix D.1, Section 2.
    """
    split_list = torch.diff(torch.tensor(sequence)).tolist()
    splited_values = torch.split(values[:, start:], split_list, dim=-1)
    splited_mask = torch.split(mask[:, start:], split_list, dim=-1)
    
    batch_size = values.size(0)
    inplace_values = torch.zeros(batch_size, len(split_list), 
                                  dtype=values.dtype, device=values.device)
    
    for idx, (value_i, mask_i) in enumerate(zip(splited_values, splited_mask)):
        # Get the last non-masked value
        if mask_i.sum() > 0:
            # Find last valid position
            last_valid_idx = mask_i.sum().int().item() - 1
            flat_idx = last_valid_idx
            inplace_values[:, idx] = value_i[:, flat_idx]
        else:
            inplace_values[:, idx] = 0.0
    
    return inplace_values


def compute_macro_action_values_position_decayed(
    values: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    sequence: List[int],
) -> torch.Tensor:
    """
    Position decayed assignment: weights decay by position.
    
    σ_τ = {1 / ((|ω_τ| - i) * H)}_{i=0}^{|ω_τ|-1}
    where H = Σ_{i=0}^{|ω_τ|-1} 1 / (|ω_τ| - i)
    
    Earlier tokens in the macro action have higher weight.
    
    Args:
        values: Token-level values of shape (batch, seq_len).
        mask: Attention mask of shape (batch, seq_len).
        start: Starting index (prompt length - 1).
        sequence: List of macro action boundary positions.
    
    Returns:
        Macro action values of shape (batch, num_macro_actions).
    
    Reference: Appendix D.1, Section 3.
    """
    split_list = torch.diff(torch.tensor(sequence)).tolist()
    splited_values = torch.split(values[:, start:], split_list, dim=-1)
    splited_mask = torch.split(mask[:, start:], split_list, dim=-1)
    
    batch_size = values.size(0)
    inplace_values = torch.zeros(batch_size, len(split_list), 
                                  dtype=values.dtype, device=values.device)
    
    for idx, (value_i, mask_i) in enumerate(zip(splited_values, splited_mask)):
        macro_len = split_list[idx]
        if macro_len == 0 or mask_i.sum() == 0:
            inplace_values[:, idx] = 0.0
            continue
        
        # Compute position decayed weights
        positions = torch.arange(macro_len, dtype=values.dtype, device=values.device)
        weights = 1.0 / (macro_len - positions)
        H = weights.sum()
        weights = weights / H  # Normalize so sum = 1
        
        # Apply mask and compute weighted mean
        weights = weights.unsqueeze(0) * mask_i.float()
        weighted_sum = (value_i * weights).sum(dim=-1)
        weight_sum = weights.sum(dim=-1)
        inplace_values[:, idx] = torch.where(
            weight_sum > 0, weighted_sum / weight_sum, torch.zeros_like(weighted_sum)
        )
    
    return inplace_values


def get_macro_action_values(
    values: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    sequence: List[int],
    value_assignment: Literal['equal', 'unit', 'position_decayed'] = 'equal',
) -> torch.Tensor:
    """
    Compute macro action values using the specified value function estimation.
    
    Args:
        values: Token-level values of shape (batch, seq_len).
        mask: Attention mask of shape (batch, seq_len).
        start: Starting index (prompt length - 1).
        sequence: List of macro action boundary positions.
        value_assignment: Type of σ assignment ('equal', 'unit', 'position_decayed').
    
    Returns:
        Macro action values of shape (batch, num_macro_actions).
    
    Reference: Appendix D.1 and Algorithm 1.
    """
    if value_assignment == 'equal':
        return compute_macro_action_values_equal(values, mask, start, sequence)
    elif value_assignment == 'unit':
        return compute_macro_action_values_unit(values, mask, start, sequence)
    elif value_assignment == 'position_decayed':
        return compute_macro_action_values_position_decayed(values, mask, start, sequence)
    else:
        raise ValueError(f"Unknown value assignment: {value_assignment}")
