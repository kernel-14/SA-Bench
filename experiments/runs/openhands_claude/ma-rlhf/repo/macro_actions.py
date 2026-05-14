"""
Macro action utilities for MA-RLHF.

Implements:
  - Four termination strategies (§3.2.1, §B.4):
      * Fixed n-gram
      * Randomized n-gram
      * Parsing-based (constituent tree DFS)
      * Perplexity-based (PPL monotone)
  - Macro action value function estimation (§D.1):
      * Equal assignment (default)
      * Unit assignment (last token)
      * Position-decayed assignment
  - Policy loss with macro actions (MA-PPO, Eq. 4)
  - Critic loss with macro actions
"""

import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Termination strategies
# ---------------------------------------------------------------------------

def get_macro_action_positions_ngram(
    start: int,
    seq_len: int,
    mask: torch.Tensor,
    n_gram: int,
) -> List[int]:
    """Fixed n-gram termination: group every n valid (unmasked) tokens.

    Args:
        start: index of the first response token in the full sequence.
        seq_len: total length of the full sequence (prompt + response).
        mask: attention/action mask of shape (1, seq_len-1) or (seq_len-1,).
        n_gram: fixed macro action length.

    Returns:
        List of boundary indices (absolute positions in the sequence).
        The first element is `start`, the last is the final token index.
    """
    mask_1d = mask.view(-1)
    sequence = [start]
    current_count = 0
    response_mask = mask_1d[start:]
    for i in range(len(response_mask) - 1):
        if response_mask[i].item() != 0:
            current_count += 1
        if current_count == n_gram:
            sequence.append(start + i + 1)
            current_count = 0
    sequence.append(int(mask_1d.size(0)))
    return sequence


def get_macro_action_positions_randomized_ngram(
    start: int,
    seq_len: int,
    mask: torch.Tensor,
    lengths: List[int] = None,
    repeat_times: int = 3,
) -> List[int]:
    """Randomized n-gram termination: lengths drawn from {2,3,5,10} (§3.2.1, §B.4).

    The list [2,3,5,10] is repeated `repeat_times` times, shuffled, then used
    as cumulative boundaries.  Any remainder is absorbed into a final large
    macro action (equivalent to n=∞ for the tail).
    """
    if lengths is None:
        lengths = [2, 3, 5, 10]

    k_list = lengths * repeat_times
    random.shuffle(k_list)

    mask_1d = mask.view(-1)
    response_mask = mask_1d[start:]
    total_valid = int(response_mask.sum().item())

    sequence = [start]
    cumulative = 0
    pos = 0  # position within response_mask
    for k in k_list:
        cumulative += k
        # advance pos until we have consumed `cumulative` valid tokens
        count = 0
        while pos < len(response_mask) - 1:
            if response_mask[pos].item() != 0:
                count += 1
            pos += 1
            if count == k:
                break
        if pos < len(response_mask):
            sequence.append(start + pos)
        if cumulative >= total_valid:
            break

    last = int(mask_1d.size(0))
    if sequence[-1] != last:
        sequence.append(last)
    return sequence


def _dfs_constituent(node, cutoff: int, current_pos: int) -> Tuple[List[int], int]:
    """Depth-first search on a constituent tree node.

    Returns (boundary_positions, updated_current_pos).
    A macro action boundary is placed after a node whose leaf count < cutoff,
    or after single-token nodes (to avoid punctuation-only macro actions).
    """
    boundaries = []
    try:
        leaves = node.leaves()
    except AttributeError:
        # Terminal node
        return boundaries, current_pos + 1

    n_leaves = len(leaves)
    if n_leaves == 0:
        return boundaries, current_pos

    if n_leaves < cutoff:
        # This node terminates a macro action
        current_pos += n_leaves
        boundaries.append(current_pos)
        return boundaries, current_pos

    # Recurse into children
    for child in node:
        child_boundaries, current_pos = _dfs_constituent(child, cutoff, current_pos)
        boundaries.extend(child_boundaries)

    return boundaries, current_pos


def get_macro_action_positions_parser(
    start: int,
    seq_len: int,
    mask: torch.Tensor,
    parse_tree,
    cutoff: int = 5,
) -> List[int]:
    """Parsing-based termination using constituent tree DFS (§3.2.1, §B.4).

    Args:
        parse_tree: a nltk.Tree or benepar parse tree for the response.
        cutoff: nodes with fewer than `cutoff` leaf tokens end a macro action.

    Returns:
        List of boundary indices.
    """
    if parse_tree is None:
        # Fall back to fixed 5-gram when parsing fails (paper §B.4)
        return get_macro_action_positions_ngram(start, seq_len, mask, n_gram=5)

    relative_boundaries, _ = _dfs_constituent(parse_tree, cutoff, 0)

    # Convert relative positions to absolute sequence positions
    sequence = [start]
    for rel in relative_boundaries:
        abs_pos = start + rel
        if abs_pos < seq_len and abs_pos not in sequence:
            sequence.append(abs_pos)

    last = int(mask.view(-1).size(0))
    if not sequence or sequence[-1] != last:
        sequence.append(last)
    return sequence


def get_macro_action_positions_ppl(
    start: int,
    seq_len: int,
    mask: torch.Tensor,
    ppl_values: List[float],
) -> List[int]:
    """Perplexity-based termination (§3.2.1, §B.4).

    A macro action ω_τ = {a_{t_τ}, ..., a_{t_{τ+1}-1}} is constructed such
    that ppl(ω_τ ∪ a_{t_{τ+1}}) > ppl(ω_τ) and ppl is monotonically
    non-increasing within the macro action.

    ppl_values[i] is the perplexity of the partial response up to token i
    (computed from reference model logits, §B.4).
    """
    mask_1d = mask.view(-1)
    sequence = [start]
    response_ppl = ppl_values  # length = number of response tokens

    i = 1
    while i < len(response_ppl):
        # Terminate when perplexity increases (next token hurts perplexity)
        if response_ppl[i] > response_ppl[i - 1]:
            sequence.append(start + i)
        i += 1

    last = int(mask_1d.size(0))
    if not sequence or sequence[-1] != last:
        sequence.append(last)
    return sequence


def compute_token_perplexities(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    start: int,
) -> List[float]:
    """Compute per-step perplexity for the response portion.

    ppl at step t = exp(-log p(a_t | a_{<t}))

    Args:
        logits: shape (1, seq_len, vocab_size) from the reference model.
        input_ids: shape (1, seq_len).
        start: index of the first response token.

    Returns:
        List of perplexity values, one per response token.
    """
    log_probs = F.log_softmax(logits[0, start - 1 : -1], dim=-1)  # (T, V)
    target_ids = input_ids[0, start:]  # (T,)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)  # (T,)

    ppl_values = []
    cumulative_neg_log_prob = 0.0
    for i, lp in enumerate(token_log_probs.tolist()):
        cumulative_neg_log_prob += -lp
        ppl = math.exp(cumulative_neg_log_prob / (i + 1))
        ppl_values.append(ppl)
    return ppl_values


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def get_macro_action_positions(
    start: int,
    seq_len: int,
    mask: torch.Tensor,
    termination: str = "ngram",
    n_gram: int = 5,
    randomized_lengths: Optional[List[int]] = None,
    repeat_times: int = 3,
    parse_tree=None,
    parser_cutoff: int = 5,
    ppl_values: Optional[List[float]] = None,
) -> List[int]:
    """Unified dispatcher for all termination strategies.

    Returns a list of boundary indices (absolute positions in the full
    prompt+response sequence).  The list always starts with `start` and ends
    with `seq_len` (exclusive upper bound, i.e., one past the last token).
    """
    if termination == "ngram":
        return get_macro_action_positions_ngram(start, seq_len, mask, n_gram)
    elif termination == "randomized_ngram":
        lengths = randomized_lengths or [2, 3, 5, 10]
        return get_macro_action_positions_randomized_ngram(
            start, seq_len, mask, lengths, repeat_times
        )
    elif termination == "parser":
        return get_macro_action_positions_parser(
            start, seq_len, mask, parse_tree, parser_cutoff
        )
    elif termination == "ppl":
        if ppl_values is None:
            raise ValueError("ppl_values must be provided for PPL termination.")
        return get_macro_action_positions_ppl(start, seq_len, mask, ppl_values)
    else:
        raise ValueError(f"Unknown termination strategy: {termination}")


# ---------------------------------------------------------------------------
# Value function estimation (§D.1)
# ---------------------------------------------------------------------------

def _harmonic_weights(n: int) -> List[float]:
    """Position-decayed weights: σ_i = 1/((n-i) * H) where H = Σ 1/(n-i)."""
    raw = [1.0 / (n - i) for i in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


def get_macro_action_values(
    values: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    sequence: List[int],
    sigma_assignment: str = "equal",
) -> torch.Tensor:
    """Aggregate token-level values into macro action values.

    V^π(s_τ, ω_τ) = Σ_{i=0}^{|ω_τ|} σ_{t_τ+i} · V^π(s_{t_τ+i}, a_{t_τ+i})

    Args:
        values: token-level value estimates, shape (1, seq_len) or (seq_len,).
        mask: action mask, shape (1, seq_len) or (seq_len,).
        start: index of the first response token.
        sequence: boundary list from get_macro_action_positions.
        sigma_assignment: 'equal' | 'unit' | 'position_decayed'.

    Returns:
        Macro action values, shape (1, num_macro_actions).
    """
    values_1d = values.view(-1)
    mask_1d = mask.view(-1)

    split_list = [sequence[i + 1] - sequence[i] for i in range(len(sequence) - 1)]
    num_macro = len(split_list)

    response_values = values_1d[start:]
    response_mask = mask_1d[start:]

    splited_values = torch.split(response_values, split_list, dim=0)
    splited_mask = torch.split(response_mask, split_list, dim=0)

    inplace_values = torch.zeros(1, num_macro, dtype=values.dtype, device=values.device)

    for idx, (val_i, mask_i) in enumerate(zip(splited_values, splited_mask)):
        valid = val_i[mask_i != 0]
        if valid.numel() == 0:
            inplace_values[0, idx] = 0.0
            continue

        n = valid.numel()
        if sigma_assignment == "equal":
            inplace_values[0, idx] = valid.mean()
        elif sigma_assignment == "unit":
            inplace_values[0, idx] = valid[-1]
        elif sigma_assignment == "position_decayed":
            weights = torch.tensor(
                _harmonic_weights(n), dtype=values.dtype, device=values.device
            )
            inplace_values[0, idx] = (valid * weights).sum()
        else:
            raise ValueError(f"Unknown sigma_assignment: {sigma_assignment}")

    return inplace_values


# ---------------------------------------------------------------------------
# Policy loss (MA-PPO, Eq. 4)
# ---------------------------------------------------------------------------

def policy_loss_macro_action(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    sequence: List[int],
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    """Clipped PPO policy loss at the macro action level (Eq. 4).

    The importance ratio for macro action ω_τ is the product of per-token
    ratios: π_θ(ω_τ|s_τ) / π_θ_old(ω_τ|s_τ) = Π_t exp(logp_t - logp_old_t).

    Args:
        logprobs: current log-probs, shape (1, response_len) or (response_len,).
        old_logprobs: old log-probs, same shape.
        advantages: macro-level advantages, shape (1, num_macro_actions).
        mask: action mask for response tokens, same shape as logprobs.
        sequence: boundary list (relative to response start, i.e., starting at 0).
        clip_ratio: ε in Eq. 4.

    Returns:
        Scalar policy loss.
    """
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)

    # Adjust sequence to be relative to the start of logprobs
    rel_sequence = [s - sequence[0] for s in sequence]
    split_list = [rel_sequence[i + 1] - rel_sequence[i] for i in range(len(rel_sequence) - 1)]

    split_ratio = torch.split(ratio.view(-1), split_list, dim=0)
    split_mask = torch.split(mask.view(-1), split_list, dim=0)

    pg_loss = 0.0
    total_mask_sum = 0.0

    for i, (ratio_i, mask_i) in enumerate(zip(split_ratio, split_mask)):
        adv_i = advantages.view(-1)[i]
        pg_loss1 = -adv_i * ratio_i
        pg_loss2 = -adv_i * torch.clamp(ratio_i, 1.0 - clip_ratio, 1.0 + clip_ratio)
        pg_loss = pg_loss + torch.sum(torch.max(pg_loss1, pg_loss2) * mask_i)
        total_mask_sum = total_mask_sum + mask_i.sum()

    if total_mask_sum > 0:
        pg_loss = pg_loss / total_mask_sum
    return pg_loss


# ---------------------------------------------------------------------------
# Critic loss (MA-PPO)
# ---------------------------------------------------------------------------

def critic_loss_macro_action(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    sequence: List[int],
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    """Clipped value function loss at the macro action level.

    Uses the same clipping trick as PPO for the value function.

    Args:
        values: current value estimates for response tokens, shape (1, response_len).
        old_values: old value estimates, same shape.
        returns: macro-level returns (targets), shape (1, num_macro_actions).
        mask: action mask, same shape as values.
        sequence: boundary list.
        clip_ratio: clipping range.

    Returns:
        Scalar critic loss.
    """
    rel_sequence = [s - sequence[0] for s in sequence]
    split_list = [rel_sequence[i + 1] - rel_sequence[i] for i in range(len(rel_sequence) - 1)]

    # Aggregate current and old values to macro level
    macro_values = _aggregate_to_macro(values.view(-1), mask.view(-1), split_list)
    macro_old_values = _aggregate_to_macro(old_values.view(-1), mask.view(-1), split_list)

    returns_flat = returns.view(-1)

    # Clipped value loss
    vf_loss1 = (macro_values - returns_flat) ** 2
    clipped = macro_old_values + torch.clamp(
        macro_values - macro_old_values, -clip_ratio, clip_ratio
    )
    vf_loss2 = (clipped - returns_flat) ** 2
    vf_loss = 0.5 * torch.mean(torch.max(vf_loss1, vf_loss2))
    return vf_loss


def _aggregate_to_macro(
    token_values: torch.Tensor,
    token_mask: torch.Tensor,
    split_list: List[int],
) -> torch.Tensor:
    """Mean-pool token values within each macro action segment."""
    splits_v = torch.split(token_values, split_list, dim=0)
    splits_m = torch.split(token_mask, split_list, dim=0)
    macro_vals = []
    for v, m in zip(splits_v, splits_m):
        valid = v[m != 0]
        macro_vals.append(valid.mean() if valid.numel() > 0 else torch.tensor(0.0, device=v.device))
    return torch.stack(macro_vals)


# ---------------------------------------------------------------------------
# Macro action log-probability (joint probability of the macro action)
# ---------------------------------------------------------------------------

def macro_action_log_prob(
    token_log_probs: torch.Tensor,
    mask: torch.Tensor,
    sequence: List[int],
) -> torch.Tensor:
    """Compute log π_θ(ω_τ | s_τ) = Σ_t log π_θ(a_t | a_{<t}) for each macro action.

    Args:
        token_log_probs: shape (response_len,) or (1, response_len).
        mask: action mask, same shape.
        sequence: boundary list.

    Returns:
        Macro action log-probs, shape (num_macro_actions,).
    """
    lp = token_log_probs.view(-1)
    m = mask.view(-1)

    rel_sequence = [s - sequence[0] for s in sequence]
    split_list = [rel_sequence[i + 1] - rel_sequence[i] for i in range(len(rel_sequence) - 1)]

    splits_lp = torch.split(lp, split_list, dim=0)
    splits_m = torch.split(m, split_list, dim=0)

    macro_lp = []
    for lp_i, m_i in zip(splits_lp, splits_m):
        valid = lp_i[m_i != 0]
        macro_lp.append(valid.sum() if valid.numel() > 0 else torch.tensor(0.0, device=lp.device))
    return torch.stack(macro_lp)
