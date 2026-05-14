"""
MA-RLHF: Macro Action definitions and termination strategies.

Implements the macro action framework from:
  "MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions"

Macro actions are sequences of tokens that serve as higher-level constructs
in the RLHF framework, reducing the temporal distance between actions and
rewards to improve credit assignment.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


def get_macro_action_positions(
    start: int,
    mask: torch.Tensor,
    termination: str = "ngram",
    n_gram: Optional[int] = None,
    ppl: Optional[List[float]] = None,
    repeat_times: Optional[int] = None,
    cutoff: Optional[int] = None,
    parse_tree=None,
) -> List[int]:
    """
    Compute the boundary positions of macro actions given a termination strategy.

    Args:
        start: The starting token index (typically prompt length - 1).
        mask: Attention mask of shape (1, seq_len). Non-zero entries indicate
              valid (non-padding) tokens.
        termination: One of 'ngram', 'randomized_ngram', 'ppl', 'parser'.
        n_gram: Fixed n-gram length (used when termination='ngram').
        ppl: Per-token perplexity values (used when termination='ppl').
        repeat_times: Number of times to repeat the randomized n-gram list.
        cutoff: Threshold for parsing-based termination (default C=5).
        parse_tree: A constituency parse tree object (used when termination='parser').

    Returns:
        A list of integer positions marking the start of each macro action
        boundary, including the final position (end of sequence).
    """
    sequence = [start]

    if termination == "ngram":
        assert n_gram is not None, "n_gram must be specified for 'ngram' termination"
        current_count = 0
        for i in range(mask[:, start:].size(1) - 1):
            current_count += mask[0, start + i].item()
            if current_count == n_gram:
                sequence.append(start + i + 1)
                current_count = 0

    elif termination == "randomized_ngram":
        # Build a shuffled list of lengths from {2, 3, 5, 10}, repeated 3 times
        k_list = torch.tensor([2, 3, 5, 10], dtype=torch.long)
        k_list = torch.repeat_interleave(k_list, 3)  # length 12
        k_list = k_list[torch.randperm(k_list.size(-1))]
        indexed_k_list = torch.cumsum(k_list, dim=-1)

        # All valid token positions in the response
        all_positions = list(range(start, mask[:, start:].size(1) - 1))
        valid_indices = [x.item() for x in indexed_k_list if x.item() < len(all_positions)]
        sequence = [start] + [all_positions[i] for i in valid_indices]

    elif termination == "ppl":
        assert ppl is not None, "ppl must be provided for 'ppl' termination"
        # A macro action ends when perplexity increases (monotonic decrease broken)
        for i in range(1, len(ppl)):
            if ppl[i] > ppl[i - 1]:
                sequence.append(start + i)

    elif termination == "parser":
        # Parsing-based termination using constituency parse tree (DFS)
        if parse_tree is not None and cutoff is not None:
            _dfs_parse(parse_tree, start, sequence, cutoff)
        # Fall back to standard token-level if parse tree unavailable

    # Always append the final position
    sequence.append(int(mask.size(1) - 1))
    return sequence


def _dfs_parse(node, ma_length: int, sequence: List[int], cutoff: int) -> Tuple[bool, int]:
    """
    Depth-first search over a constituency parse tree to determine macro action
    boundaries. Nodes with fewer than `cutoff` leaf tokens mark the end of a
    macro action.

    Args:
        node: Current parse tree node (e.g., nltk.Tree).
        ma_length: Current accumulated macro action length.
        sequence: List being built with boundary positions.
        cutoff: Maximum number of leaf tokens for a node to be a macro action.

    Returns:
        (terminated, updated_ma_length)
    """
    leaves = node.leaves() if hasattr(node, "leaves") else []
    if len(leaves) < 1:
        return False, ma_length + 1
    if len(leaves) < cutoff:
        sequence.append(ma_length + len(leaves))
        return True, ma_length + len(leaves)
    for child in (node if hasattr(node, "__iter__") else []):
        state, ma_length = _dfs_parse(child, ma_length, sequence, cutoff)
        if not state:
            if sequence:
                sequence[-1] = ma_length
    return True, ma_length


def get_macro_action_values(
    values: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    sequence: List[int],
) -> torch.Tensor:
    """
    Aggregate token-level value estimates into macro-action-level values using
    equal weighting (mean over valid tokens within each macro action).

    This implements the "equal assignment" strategy from Section D.1:
        V^pi(s_tau, omega_tau) = (1/|omega_tau|) * sum V^pi(s_t, a_t)

    Args:
        values: Token-level value estimates, shape (1, seq_len).
        mask: Attention mask, shape (1, seq_len).
        start: Starting index of the response tokens.
        sequence: List of macro action boundary positions.

    Returns:
        Macro-action-level values, shape (1, num_macro_actions).
    """
    split_list = [int(s) for s in torch.diff(torch.tensor(sequence)).tolist()]

    splited_values = torch.split(values[:, start:], split_list, dim=-1)
    splited_mask = torch.split(mask[:, start:], split_list, dim=-1)

    inplace_values = torch.zeros(
        1, len(split_list), dtype=values.dtype, device=values.device
    )
    for idx, (value_i, mask_i) in enumerate(zip(splited_values, splited_mask)):
        masked_values = value_i[mask_i != 0]
        inplace_values[0, idx] = torch.mean(masked_values) if masked_values.numel() > 0 else 0.0

    return inplace_values


def get_macro_action_values_unit(
    values: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    sequence: List[int],
) -> torch.Tensor:
    """
    Unit assignment: use the value of the last token in each macro action.
    sigma_tau = {0, 0, ..., 0, 1}
    """
    split_list = [int(s) for s in torch.diff(torch.tensor(sequence)).tolist()]
    splited_values = torch.split(values[:, start:], split_list, dim=-1)
    splited_mask = torch.split(mask[:, start:], split_list, dim=-1)

    inplace_values = torch.zeros(
        1, len(split_list), dtype=values.dtype, device=values.device
    )
    for idx, (value_i, mask_i) in enumerate(zip(splited_values, splited_mask)):
        masked_values = value_i[mask_i != 0]
        inplace_values[0, idx] = masked_values[-1] if masked_values.numel() > 0 else 0.0

    return inplace_values


def get_macro_action_values_position_decayed(
    values: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    sequence: List[int],
) -> torch.Tensor:
    """
    Position-decayed assignment: sigma_i = 1 / ((|omega_tau| - i) * H)
    where H = sum_{i=0}^{|omega_tau|-1} 1/(|omega_tau| - i) is the harmonic normalizer.
    """
    split_list = [int(s) for s in torch.diff(torch.tensor(sequence)).tolist()]
    splited_values = torch.split(values[:, start:], split_list, dim=-1)
    splited_mask = torch.split(mask[:, start:], split_list, dim=-1)

    inplace_values = torch.zeros(
        1, len(split_list), dtype=values.dtype, device=values.device
    )
    for idx, (value_i, mask_i) in enumerate(zip(splited_values, splited_mask)):
        masked_values = value_i[mask_i != 0]
        n = masked_values.numel()
        if n == 0:
            continue
        # Harmonic weights: w_i = 1/(n - i), normalized
        weights = torch.tensor(
            [1.0 / (n - i) for i in range(n)], dtype=values.dtype, device=values.device
        )
        H = weights.sum()
        weights = weights / H
        inplace_values[0, idx] = (masked_values * weights).sum()

    return inplace_values


def policy_loss_macro_action(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    sequence: List[int],
    cliprange: float = 0.2,
) -> torch.Tensor:
    """
    Compute the MA-PPO clipped policy gradient loss (Equation 3 in the paper).

    The importance sampling ratio is computed at the macro-action level:
        r_tau = pi_theta(omega_tau | s_tau) / pi_theta_old(omega_tau | s_tau)
              = exp(sum_{t in omega_tau} (log pi_theta - log pi_theta_old))

    The clipped PPO objective is applied at the macro-action level, and the
    loss is broadcast back to all tokens within each macro action.

    Args:
        logprobs: Current policy log-probabilities, shape (batch, seq_len).
        old_logprobs: Old policy log-probabilities, shape (batch, seq_len).
        advantages: Macro-action advantages, shape (batch, num_macro_actions).
        mask: Token-level action mask, shape (batch, seq_len).
        sequence: Macro action boundary positions.
        cliprange: PPO clipping parameter epsilon.

    Returns:
        Scalar policy loss.
    """
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)

    split_list = [int(s) for s in torch.diff(torch.tensor(sequence)).tolist()]
    split_ratio = torch.split(ratio, split_list, dim=-1)
    split_mask = torch.split(mask, split_list, dim=-1)

    pg_loss = 0.0
    total_mask_sum = 0.0

    for i in range(len(split_list)):
        ratio_i = split_ratio[i]          # (batch, macro_len)
        mask_i = split_mask[i]            # (batch, macro_len)
        advantages_i = advantages[:, i]   # (batch,)

        # Broadcast advantage to token dimension
        adv_i = advantages_i.unsqueeze(-1)  # (batch, 1)

        pg_loss1 = -adv_i * ratio_i
        pg_loss2 = -adv_i * torch.clamp(ratio_i, 1.0 - cliprange, 1.0 + cliprange)
        pg_loss += torch.sum(torch.max(pg_loss1, pg_loss2) * mask_i)
        total_mask_sum += mask_i.sum()

    pg_loss = pg_loss / (total_mask_sum + 1e-8)
    return pg_loss


def critic_loss_macro_action(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    sequence: List[int],
    cliprange_value: float = 0.2,
) -> torch.Tensor:
    """
    Compute the MA-PPO critic (value function) loss.

    The value loss is computed at the macro-action level and broadcast back
    to all tokens within each macro action.

    Args:
        values: Current value estimates, shape (batch, seq_len).
        old_values: Old value estimates, shape (batch, seq_len).
        returns: Target returns (macro-action level), shape (batch, num_macro_actions).
        mask: Token-level action mask, shape (batch, seq_len).
        sequence: Macro action boundary positions.
        cliprange_value: Clipping range for value function.

    Returns:
        Scalar critic loss.
    """
    split_list = [int(s) for s in torch.diff(torch.tensor(sequence)).tolist()]
    split_values = torch.split(values, split_list, dim=-1)
    split_old_values = torch.split(old_values, split_list, dim=-1)
    split_mask = torch.split(mask, split_list, dim=-1)

    vf_loss = 0.0
    total_mask_sum = 0.0

    for i in range(len(split_list)):
        values_i = split_values[i]
        old_values_i = split_old_values[i]
        mask_i = split_mask[i]
        returns_i = returns[:, i].unsqueeze(-1)  # (batch, 1)

        # Clipped value loss
        values_clipped = old_values_i + torch.clamp(
            values_i - old_values_i, -cliprange_value, cliprange_value
        )
        vf_loss1 = (values_i - returns_i) ** 2
        vf_loss2 = (values_clipped - returns_i) ** 2
        vf_loss += torch.sum(torch.max(vf_loss1, vf_loss2) * mask_i)
        total_mask_sum += mask_i.sum()

    vf_loss = vf_loss / (total_mask_sum + 1e-8)
    return vf_loss


def compute_perplexity_sequence(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    start: int,
) -> List[float]:
    """
    Compute per-token perplexity values for the response portion of a sequence.

    The perplexity at position t is defined as the exponential of the average
    negative log-likelihood of tokens up to position t:
        ppl(y_{<=t}) proportional to -1/t * sum_{i=1}^{t} log pi(a_i | a_{<i})

    Args:
        logits: Model logits, shape (1, seq_len, vocab_size).
        input_ids: Token IDs, shape (1, seq_len).
        start: Starting index of the response.

    Returns:
        List of perplexity values, one per response token.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log-probs of the actual tokens
    token_log_probs = log_probs[0, :, :].gather(
        dim=-1, index=input_ids[0, :].unsqueeze(-1)
    ).squeeze(-1)  # (seq_len,)

    response_log_probs = token_log_probs[start:]
    ppl_values = []
    cumsum = 0.0
    for i, lp in enumerate(response_log_probs.tolist()):
        cumsum += lp
        avg_nll = -cumsum / (i + 1)
        ppl_values.append(float(torch.exp(torch.tensor(avg_nll)).item()))

    return ppl_values
