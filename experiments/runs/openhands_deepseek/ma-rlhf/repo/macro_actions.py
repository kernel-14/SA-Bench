"""Macro action termination strategies for MA-RLHF.

Implements four termination conditions described in §3.2.1:
- Fixed n-gram based termination
- Randomized n-gram based termination
- Parsing-based termination (using constituent trees)
- Perplexity-based termination

Also implements the value function estimation methods from Appendix D.1:
- Equal assignment (default)
- Unit assignment
- Position decayed assignment
"""
import torch
from typing import List, Optional, Tuple, Literal
import numpy as np


def get_macro_action_positions(
    start: int,
    mask: torch.Tensor,
    termination: Literal["ngram", "randomized_ngram", "ppl", "parser"] = "ngram",
    n_gram: int = 5,
    n_gram_list: List[int] = None,
    repeat_times: int = 3,
    ppl_values: Optional[List[float]] = None,
    cutoff: int = 5,
    parse_tree_root=None,
) -> List[int]:
    """Determine macro action boundaries based on termination condition.

    Args:
        start: Starting position (prompt length - 1)
        mask: Attention mask for generated tokens
        termination: Type of termination condition
        n_gram: Fixed n-gram size
        n_gram_list: List of n-gram sizes for randomized strategy
        repeat_times: Number of times to repeat the n_gram_list
        ppl_values: Per-token perplexity values for PPL-based termination
        cutoff: Maximum leaf tokens threshold for parsing termination
        parse_tree_root: Root node of constituent parse tree

    Returns:
        List of boundary indices defining macro actions
    """
    sequence = [start]

    if termination == "ngram":
        sequence = _fixed_ngram_positions(start, mask, n_gram)

    elif termination == "randomized_ngram":
        sequence = _randomized_ngram_positions(start, mask, n_gram_list, repeat_times)

    elif termination == "ppl":
        assert ppl_values is not None, "PPL values required for perplexity-based termination"
        sequence = _ppl_positions(start, ppl_values)

    elif termination == "parser":
        sequence = _parser_positions(start, mask, parse_tree_root, cutoff)

    # Ensure last boundary covers remaining tokens
    total_len = mask.size(1)
    if sequence[-1] < total_len - 1:
        sequence.append(total_len - 1)

    return sequence


def _fixed_ngram_positions(
    start: int,
    mask: torch.Tensor,
    n_gram: int,
) -> List[int]:
    """Fixed n-gram macro action boundaries. §3.2.1"""
    sequence = [start]
    current_count = 0
    for i in range(mask[:, start:].size(1) - 1):
        current_count += int(mask[0, start + i].item())
        if current_count == n_gram:
            sequence.append(start + i + 1)
            current_count = 0
    return sequence


def _randomized_ngram_positions(
    start: int,
    mask: torch.Tensor,
    n_gram_list: Optional[List[int]] = None,
    repeat_times: int = 3,
) -> List[int]:
    """Randomized n-gram macro action boundaries. §3.2.1"""
    if n_gram_list is None:
        n_gram_list = [2, 3, 5, 10]

    k_list = torch.tensor(n_gram_list, dtype=torch.int)
    k_list = torch.repeat_interleave(k_list, repeat_times)
    k_list = k_list[torch.randperm(k_list.size(-1))]

    indexed_k_list = torch.cumsum(k_list, dim=-1)
    total_positions = mask[:, start:].size(1) - 1

    sequence = [start]
    for x in indexed_k_list:
        pos = x.item()
        if pos < total_positions:
            sequence.append(start + pos)
        else:
            break

    return sequence


def _ppl_positions(
    start: int,
    ppl_values: List[float],
) -> List[int]:
    """Perplexity-based macro action boundaries. §3.2.1

    A macro action terminates when the next token would increase
    the perplexity of the current macro action.
    """
    sequence = [start]
    for i in range(1, len(ppl_values)):
        if ppl_values[i] > ppl_values[i - 1]:
            sequence.append(start + i)
    return sequence


def _parser_positions(
    start: int,
    mask: torch.Tensor,
    parse_tree_root,
    cutoff: int = 5,
) -> List[int]:
    """Parsing-based macro action boundaries using DFS on constituent tree. §3.2.1

    Traverses the constituent tree, expanding non-terminal nodes until
    the current node contains <= cutoff leaf tokens.
    """
    sequence = [start]

    if parse_tree_root is None:
        return sequence

    def dfs(node, current_ma_length: int) -> Tuple[bool, int]:
        nonlocal sequence
        if len(node.leaves()) < 1:
            return False, current_ma_length + 1
        if len(node.leaves()) < cutoff:
            sequence.append(current_ma_length + len(node.leaves()))
            return True, current_ma_length + len(node.leaves())
        for child in node.childs() if hasattr(node, 'childs') else node:
            state, current_ma_length = dfs(child, current_ma_length)
            if not state:
                if sequence:
                    sequence[-1] = current_ma_length
        return True, current_ma_length

    try:
        dfs(parse_tree_root, 0)
    except Exception:
        pass

    return sequence


def get_macro_action_values(
    values: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    sequence: List[int],
    value_estimation: Literal["equal", "unit", "position_decayed"] = "equal",
) -> torch.Tensor:
    """Compute macro-action-level value estimates from token-level values.

    Args:
        values: Token-level value estimates, shape (batch, seq_len)
        mask: Attention mask for values
        start: Starting position offset
        sequence: Macro action boundary indices
        value_estimation: Method for aggregating token values

    Returns:
        Macro-action-level values, shape (batch, num_macro_actions)
    """
    split_list = torch.diff(torch.tensor(sequence)).tolist()
    splited_values = torch.split(values[:, start:], split_list, dim=-1)
    splited_mask = torch.split(mask[:, start:], split_list, dim=-1)

    num_macros = len(split_list)
    inplace_values = torch.zeros(
        1, num_macros, dtype=values.dtype, device=values.device
    )

    for idx, (value_i, mask_i) in enumerate(zip(splited_values, splited_mask)):
        if value_estimation == "equal":
            masked_values = value_i[mask_i != 0]
            inplace_values[0, idx] = (
                torch.mean(masked_values) if masked_values.numel() > 0 else 0.0
            )
        elif value_estimation == "unit":
            # Use last token's value
            if mask_i.sum() > 0:
                last_idx = mask_i.nonzero()[-1]
                inplace_values[0, idx] = value_i[last_idx]
            else:
                inplace_values[0, idx] = 0.0
        elif value_estimation == "position_decayed":
            macro_len = len(value_i)
            weights = torch.tensor(
                [1.0 / (macro_len - i) for i in range(macro_len)],
                device=values.device,
            )
            weights = weights / weights.sum()
            inplace_values[0, idx] = (value_i * weights * mask_i).sum() / (
                (weights * mask_i).sum() + 1e-8
            )

    return inplace_values


def get_macro_action_rewards(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    sequence: List[int],
) -> torch.Tensor:
    """Compute macro-action-level rewards from token-level rewards.

    The macro reward R_τ = sum of token rewards within the macro action.
    According to §3.2.2, discount factor ρ = 1.
    """
    split_list = torch.diff(torch.tensor(sequence)).tolist()
    splited_rewards = torch.split(rewards[:, start:], split_list, dim=-1)
    splited_mask = torch.split(mask[:, start:], split_list, dim=-1)

    num_macros = len(split_list)
    inplace_rewards = torch.zeros(
        1, num_macros, dtype=rewards.dtype, device=rewards.device
    )

    for idx, (reward_i, mask_i) in enumerate(zip(splited_rewards, splited_mask)):
        masked_rewards = reward_i[mask_i != 0]
        inplace_rewards[0, idx] = (
            masked_rewards.sum() if masked_rewards.numel() > 0 else 0.0
        )

    return inplace_rewards


def compute_perplexity_values(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> List[float]:
    """Compute per-token perplexity for PPL-based termination.

    Uses reference model logits (no additional forward passes needed).

    ppl(ω_τ) ∝ -1/|ω_τ| * Σ log(P(a_t | a_<t))

    Args:
        logits: Logits from reference model, shape (seq_len, vocab_size)
        token_ids: Token IDs of generated sequence, shape (seq_len,)

    Returns:
        List of per-token perplexity values
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[range(len(token_ids)), token_ids]

    ppl_values = []
    cumulative_log_prob = 0.0
    for i, lp in enumerate(token_log_probs):
        cumulative_log_prob += lp.item()
        avg_log_prob = cumulative_log_prob / (i + 1)
        ppl = np.exp(-avg_log_prob)
        ppl_values.append(ppl)

    return ppl_values
