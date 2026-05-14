"""
MDM inference strategies: vanilla and adaptive (Section 4).

Vanilla MDM inference (Algorithm 1 in paper):
  At each step t → s:
  (a) Randomly select a set S of masked positions, P(i ∈ S) = (α_s - α_t) / (1 - α_t)
  (b) For each i ∈ S, sample x_s^i ~ p_θ(x^i | x_t)

Adaptive MDM inference (Section 4.1):
  Replace step (a) with an oracle F(θ, x_t) that selects positions strategically.

  Two oracle strategies:
  1. Top Probability: select K positions with highest max_j p_θ(x^i=j | x_t)
  2. Top Probability Margin: select K positions with highest |p1 - p2|
     where p1, p2 are the top-2 probabilities at position i.

  For text generation (Section D.1.2), Gaussian noise is added to the oracle scores.
  For logic puzzles (Section D.2), Gumbel noise is added.
"""

import math
from typing import Optional, Callable

import torch
import torch.nn.functional as F

from mdm import MDM, NoiseSchedule


# ---------------------------------------------------------------------------
# Oracle functions F(θ, x_t) → scores for each masked position
# ---------------------------------------------------------------------------

def oracle_top_probability(
    probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Top Probability oracle (Section 4.1, Zheng et al. 2023).

    Certainty at position i = max_j p_θ(x^i = j | x_t)

    Args:
        probs: (B, L, vocab_size) predicted probabilities
        mask:  (B, L) boolean mask — True where token is masked

    Returns:
        scores: (B, L) certainty scores (0 for unmasked positions)
    """
    max_probs = probs.max(dim=-1).values  # (B, L)
    scores = max_probs * mask.float()
    return scores


def oracle_top_probability_margin(
    probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Top Probability Margin oracle (Section 4.1, proposed in this paper).

    Certainty at position i = |p_θ(x^i=j1|x_t) - p_θ(x^i=j2|x_t)|
    where j1, j2 are the top-2 most probable values.

    This is more robust than Top Probability when multiple values have
    similar high probabilities (common in Sudoku).

    Args:
        probs: (B, L, vocab_size) predicted probabilities
        mask:  (B, L) boolean mask — True where token is masked

    Returns:
        scores: (B, L) certainty scores (0 for unmasked positions)
    """
    # Get top-2 probabilities
    top2 = torch.topk(probs, k=2, dim=-1).values  # (B, L, 2)
    margin = (top2[..., 0] - top2[..., 1]).abs()   # (B, L)
    scores = margin * mask.float()
    return scores


def oracle_random(
    probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Random oracle — equivalent to vanilla MDM inference.
    Returns uniform random scores for masked positions.
    """
    scores = torch.rand_like(probs[..., 0]) * mask.float()
    return scores


ORACLE_REGISTRY = {
    "vanilla": oracle_random,
    "top_prob": oracle_top_probability,
    "top_prob_margin": oracle_top_probability_margin,
}


# ---------------------------------------------------------------------------
# Number of tokens to unmask at each step (Section D.1.2)
# ---------------------------------------------------------------------------

def compute_num_to_unmask(
    x_t: torch.Tensor,
    alpha_s: float,
    alpha_t: float,
    mask_token_id: int = 0,
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Compute K = number of tokens to unmask at this step.

    K = (# masked tokens) × (α_s - α_t) / (1 - α_t)

    This keeps the number of revealed tokens balanced throughout inference,
    matching the marginal distribution seen during training (Section D.1.2).

    Args:
        x_t:           (B, L) current sequence
        alpha_s:       noise level at target step s < t
        alpha_t:       noise level at current step t
        mask_token_id: mask token id
        stochastic:    if True, sample K ~ Binomial(n_masked, p); else use floor(K)

    Returns:
        K: (B,) number of tokens to unmask per sequence
    """
    n_masked = (x_t == mask_token_id).float().sum(dim=1)  # (B,)
    p = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-8)
    p = max(0.0, min(1.0, p))

    if stochastic:
        K = torch.distributions.Binomial(
            total_count=n_masked, probs=torch.tensor(p)
        ).sample()
    else:
        # Use round instead of floor to avoid floating-point edge cases at the
        # final step where p ≈ 1.0 but floor(n * 0.9999...) = n - 1.
        K = (n_masked * p).round().long()
        K = K.clamp(min=1)

    return K.long()


# ---------------------------------------------------------------------------
# Single denoising step
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoising_step(
    model: MDM,
    x_t: torch.Tensor,
    alpha_s: float,
    alpha_t: float,
    strategy: str = "top_prob_margin",
    gumbel_noise_coeff: float = 0.0,
    oracle_noise_std: float = 0.0,
    temperature: float = 1.0,
    mask_token_id: int = 0,
) -> torch.Tensor:
    """
    Perform one step of the MDM reverse process: x_t → x_s.

    Args:
        model:              MDM model
        x_t:                (B, L) current partially masked sequence
        alpha_s:            noise level at target step s
        alpha_t:            noise level at current step t
        strategy:           oracle strategy ("vanilla", "top_prob", "top_prob_margin")
        gumbel_noise_coeff: coefficient for Gumbel noise added to oracle scores (Section D.2)
        oracle_noise_std:   std of Gaussian noise added to oracle scores (Section D.1.2)
        temperature:        sampling temperature for token values
        mask_token_id:      mask token id

    Returns:
        x_s: (B, L) sequence with some tokens unmasked
    """
    B, L = x_t.shape
    device = x_t.device

    mask = (x_t == mask_token_id)  # (B, L)
    if not mask.any():
        return x_t

    # Get predicted probabilities from denoising network
    logits = model(x_t)  # (B, L, vocab_size)
    if temperature != 1.0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)  # (B, L, vocab_size)

    # Compute oracle scores
    oracle_fn = ORACLE_REGISTRY[strategy]
    scores = oracle_fn(probs, mask)  # (B, L)

    # Add noise to oracle scores for diversity
    if gumbel_noise_coeff > 0.0:
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(scores) + 1e-10
        ) + 1e-10)
        scores = scores + gumbel_noise_coeff * gumbel_noise * mask.float()

    if oracle_noise_std > 0.0:
        gaussian_noise = torch.randn_like(scores)
        scores = scores + oracle_noise_std * gaussian_noise * mask.float()

    # Determine how many tokens to unmask
    K = compute_num_to_unmask(x_t, alpha_s, alpha_t, mask_token_id)  # (B,)

    # Select top-K positions per sequence
    x_s = x_t.clone()
    for b in range(B):
        k = K[b].item()
        if k <= 0:
            continue

        masked_indices = mask[b].nonzero(as_tuple=True)[0]
        n_masked = len(masked_indices)
        if n_masked == 0:
            continue

        k = min(k, n_masked)

        # Get scores for masked positions only
        masked_scores = scores[b, masked_indices]

        # Select top-k positions
        _, top_k_local = torch.topk(masked_scores, k=k)
        selected_indices = masked_indices[top_k_local]

        # Sample token values for selected positions.
        # Zero out the mask token probability so the model never predicts mask.
        selected_probs = probs[b, selected_indices].clone()  # (k, vocab_size)
        selected_probs[:, mask_token_id] = 0.0
        # Re-normalize (handles the case where mask token had non-zero prob)
        row_sums = selected_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        selected_probs = selected_probs / row_sums
        sampled_tokens = torch.multinomial(selected_probs, num_samples=1).squeeze(-1)

        x_s[b, selected_indices] = sampled_tokens

    return x_s


# ---------------------------------------------------------------------------
# Full MDM sampling (reverse process)
# ---------------------------------------------------------------------------

@torch.no_grad()
def mdm_sample(
    model: MDM,
    batch_size: int,
    seq_len: int,
    num_steps: int = 50,
    strategy: str = "top_prob_margin",
    gumbel_noise_coeff: float = 0.0,
    oracle_noise_std: float = 0.0,
    temperature: float = 1.0,
    mask_token_id: int = 0,
    device: str = "cuda",
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Full MDM reverse process: sample from p_θ(x_0).

    Starts from fully masked sequence x_1 = (0, ..., 0) and iteratively
    unmasks tokens using the specified strategy.

    Args:
        model:              MDM model
        batch_size:         number of sequences to generate
        seq_len:            sequence length
        num_steps:          number of reverse diffusion steps
        strategy:           oracle strategy
        gumbel_noise_coeff: Gumbel noise coefficient
        oracle_noise_std:   Gaussian noise std for oracle
        temperature:        sampling temperature
        mask_token_id:      mask token id
        device:             device string
        condition:          (B, L) optional conditioning — non-mask tokens are fixed

    Returns:
        x_0: (B, L) generated sequences
    """
    model.eval()

    # Start from fully masked sequence
    x_t = torch.full((batch_size, seq_len), mask_token_id, dtype=torch.long, device=device)

    # Apply conditioning (fix given tokens)
    if condition is not None:
        given_mask = (condition != mask_token_id)
        x_t[given_mask] = condition[given_mask]

    # Build time schedule: t from 1 to 0 in num_steps steps
    t_schedule = torch.linspace(1.0, 0.0, num_steps + 1)

    noise_schedule = model.noise_schedule

    for step in range(num_steps):
        t = t_schedule[step].item()
        s = t_schedule[step + 1].item()

        alpha_t = noise_schedule.alpha(torch.tensor(t)).item()
        alpha_s = noise_schedule.alpha(torch.tensor(s)).item()

        x_t = denoising_step(
            model=model,
            x_t=x_t,
            alpha_s=alpha_s,
            alpha_t=alpha_t,
            strategy=strategy,
            gumbel_noise_coeff=gumbel_noise_coeff,
            oracle_noise_std=oracle_noise_std,
            temperature=temperature,
            mask_token_id=mask_token_id,
        )

        # Re-apply conditioning at each step
        if condition is not None:
            x_t[given_mask] = condition[given_mask]

    return x_t


@torch.no_grad()
def mdm_solve_puzzle(
    model: MDM,
    puzzle: torch.Tensor,
    num_steps: int = 50,
    strategy: str = "top_prob_margin",
    gumbel_noise_coeff: float = 0.5,
    temperature: float = 1.0,
    mask_token_id: int = 0,
) -> torch.Tensor:
    """
    Solve a logic puzzle (Sudoku/Zebra) using MDM with adaptive inference.

    Args:
        model:              MDM model
        puzzle:             (B, L) partially filled puzzle (0 = empty/mask)
        num_steps:          number of reverse diffusion steps
        strategy:           oracle strategy
        gumbel_noise_coeff: Gumbel noise coefficient (0.5 per Section D.2)
        temperature:        sampling temperature
        mask_token_id:      mask token id

    Returns:
        solution: (B, L) completed puzzle
    """
    B, L = puzzle.shape
    device = puzzle.device

    return mdm_sample(
        model=model,
        batch_size=B,
        seq_len=L,
        num_steps=num_steps,
        strategy=strategy,
        gumbel_noise_coeff=gumbel_noise_coeff,
        temperature=temperature,
        mask_token_id=mask_token_id,
        device=str(device),
        condition=puzzle,
    )


# ---------------------------------------------------------------------------
# Adaptive inference for infilling tasks (Section D.3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def mdm_infill(
    model: MDM,
    context: torch.Tensor,
    mask_positions: torch.Tensor,
    num_steps: int = 50,
    strategy: str = "top_prob_margin",
    oracle_noise_std: float = 0.0,
    temperature: float = 1.0,
    mask_token_id: int = 0,
) -> torch.Tensor:
    """
    MDM infilling: fill in masked positions given context.

    Args:
        model:          MDM model
        context:        (B, L) sequence with mask_token_id at positions to fill
        mask_positions: (B, L) boolean tensor — True where tokens should be filled
        num_steps:      number of reverse diffusion steps
        strategy:       oracle strategy
        oracle_noise_std: Gaussian noise std for oracle
        temperature:    sampling temperature
        mask_token_id:  mask token id

    Returns:
        filled: (B, L) completed sequence
    """
    return mdm_sample(
        model=model,
        batch_size=context.shape[0],
        seq_len=context.shape[1],
        num_steps=num_steps,
        strategy=strategy,
        oracle_noise_std=oracle_noise_std,
        temperature=temperature,
        mask_token_id=mask_token_id,
        device=str(context.device),
        condition=context,
    )


# ---------------------------------------------------------------------------
# Semi-autoregressive sampling for instruction-following tasks (Section D.3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def mdm_semi_autoregressive_sample(
    model: MDM,
    prompt: torch.Tensor,
    max_new_tokens: int,
    num_steps: int = 50,
    strategy: str = "top_prob_margin",
    temperature: float = 1.0,
    mask_token_id: int = 0,
) -> torch.Tensor:
    """
    Semi-autoregressive sampling for instruction-answering tasks (Section D.3).

    The prompt is fixed; the response tokens are generated using MDM inference.
    This follows the sampling configuration of LLaDA (Nie et al., 2025).

    Args:
        model:          MDM model
        prompt:         (B, T_prompt) prompt tokens
        max_new_tokens: number of response tokens to generate
        num_steps:      number of reverse diffusion steps
        strategy:       oracle strategy
        temperature:    sampling temperature
        mask_token_id:  mask token id

    Returns:
        output: (B, T_prompt + max_new_tokens) complete sequence
    """
    B, T_prompt = prompt.shape
    device = prompt.device
    L = T_prompt + max_new_tokens

    # Build condition: prompt tokens fixed, response tokens masked
    condition = torch.full((B, L), mask_token_id, dtype=torch.long, device=device)
    condition[:, :T_prompt] = prompt

    result = mdm_sample(
        model=model,
        batch_size=B,
        seq_len=L,
        num_steps=num_steps,
        strategy=strategy,
        temperature=temperature,
        mask_token_id=mask_token_id,
        device=str(device),
        condition=condition,
    )
    return result
