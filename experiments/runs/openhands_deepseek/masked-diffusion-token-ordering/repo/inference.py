import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Callable, Literal
from config import InferenceConfig, DiffusionConfig, ModelConfig


def reverse_process_step(
    model: nn.Module,
    x_t: torch.Tensor,
    alpha_s: float,
    alpha_t: float,
    mask_token: int,
    strategy: Literal["vanilla", "top_probability", "top_probability_margin"],
    gumbel_noise: float = 0.0,
    top_k: Optional[int] = None,
    temperature: float = 0.0,
) -> torch.Tensor:
    """
    One step of the reverse process: from time t to time s < t.
    Unmasks a subset of currently masked tokens.
    Returns x_s (partially unmasked).
    """
    B, L = x_t.shape
    device = x_t.device
    masked = (x_t == mask_token)  # (B, L)
    num_masked = masked.sum(dim=1)  # (B,)

    # Get model predictions
    with torch.no_grad():
        logits = model(x_t)  # (B, L, V+1)
        vocab_size = logits.shape[-1] - 1
        clean_logits = logits[:, :, :vocab_size]  # (B, L, V)
        # Mask out logits for already-unmasked positions
        clean_logits = clean_logits + masked.float().unsqueeze(-1) * 1e9

    probs = F.softmax(clean_logits, dim=-1)  # (B, L, V)

    # Number of tokens to unmask per sample
    # Following: |S| = num_masked * (α_s - α_t) / (1 - α_t)
    mask_ratio = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-8)
    k_per_sample = (num_masked.float() * mask_ratio).round().long()
    k_per_sample = k_per_sample.clamp(min=1)  # at least 1 if any masked

    if top_k is not None:
        k_per_sample = torch.full_like(k_per_sample, top_k)

    # Select tokens to unmask
    if strategy == "vanilla":
        # Random selection with probability (α_s - α_t) / (1 - α_t)
        rand = torch.rand(B, L, device=device)
        threshold = torch.full((B, L), mask_ratio, device=device)
        selected = (rand < threshold) & masked

    elif strategy == "top_probability":
        # Select by max probability
        max_prob, _ = probs.max(dim=-1)  # (B, L)
        max_prob = max_prob.masked_fill(~masked, -1.0)
        selected = _select_top_k_per_sample(max_prob, k_per_sample, device)

    elif strategy == "top_probability_margin":
        # Select by difference between top-2 probabilities
        top2 = probs.topk(2, dim=-1).values  # (B, L, 2)
        margin = top2[:, :, 0] - top2[:, :, 1]  # (B, L)
        margin = margin.masked_fill(~masked, -1.0)

        if gumbel_noise > 0:
            gumbel = _sample_gumbel(B, L, device) * gumbel_noise
            margin = margin + gumbel

        if temperature > 0:
            margin = margin / temperature

        selected = _select_top_k_per_sample(margin, k_per_sample, device)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Unmask selected positions by sampling from model predictions
    x_s = x_t.clone()
    for b in range(B):
        sel = selected[b]
        if sel.any():
            logits_b = logits[b, sel, :vocab_size]
            sampled = torch.multinomial(F.softmax(logits_b, dim=-1), 1).squeeze(-1)
            x_s[b, sel] = sampled

    return x_s


def _select_top_k_per_sample(
    scores: torch.Tensor,
    k_per_sample: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Select top-k masked positions per sample. Returns bool mask (B, L)."""
    B, L = scores.shape
    selected = torch.zeros(B, L, dtype=torch.bool, device=device)
    for b in range(B):
        k = k_per_sample[b].item()
        if k > 0:
            _, indices = torch.topk(scores[b], k, dim=0)
            selected[b, indices] = True
    return selected


def _sample_gumbel(B: int, L: int, device: torch.device) -> torch.Tensor:
    """Sample from Gumbel(0, 1) distribution."""
    u = torch.rand(B, L, device=device).clamp(min=1e-8)
    return -torch.log(-torch.log(u))


@torch.no_grad()
def sample_mdm(
    model: nn.Module,
    seq_len: int,
    mask_token: int,
    alpha_schedule: List[float],
    strategy: Literal["vanilla", "top_probability", "top_probability_margin"],
    batch_size: int = 1,
    gumbel_noise: float = 0.0,
    top_k: Optional[int] = None,
    temperature: float = 0.0,
    x_1: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Full reverse sampling from fully masked to clean.
    alpha_schedule: list of α values from 1 (fully masked) down to 0 (clean).
    """
    device = next(model.parameters()).device

    if x_1 is not None:
        x_t = x_1
    else:
        x_t = torch.full((batch_size, seq_len), mask_token, dtype=torch.long, device=device)

    for step_idx in range(len(alpha_schedule) - 1):
        alpha_t = alpha_schedule[step_idx]
        alpha_s = alpha_schedule[step_idx + 1]
        x_t = reverse_process_step(
            model=model,
            x_t=x_t,
            alpha_s=alpha_s,
            alpha_t=alpha_t,
            mask_token=mask_token,
            strategy=strategy,
            gumbel_noise=gumbel_noise,
            top_k=top_k,
            temperature=temperature,
        )
    return x_t


def get_alpha_schedule(
    num_steps: int,
    noise_schedule_fn: Callable,
) -> List[float]:
    """Discretize the noise schedule into num_steps+1 points from 1→0."""
    t_values = np.linspace(1.0, 0.0, num_steps + 1)
    with torch.no_grad():
        alphas = noise_schedule_fn(torch.tensor(t_values, dtype=torch.float32))
    return alphas.tolist()


@torch.no_grad()
def sample_autoregressive(
    model: nn.Module,
    seq_len: int,
    mask_token: int,
    pad_token: Optional[int],
    batch_size: int = 1,
    temperature: float = 1.0,
    x_prefix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Autoregressive (left-to-right) generation using the same model in causal mode.
    """
    device = next(model.parameters()).device
    vocab_size = model.cfg.vocab_size

    if x_prefix is not None:
        prefix_len = x_prefix.shape[1]
        x = x_prefix
    else:
        prefix_len = 0
        x = torch.full((batch_size, 1), pad_token if pad_token is not None else 0,
                        dtype=torch.long, device=device)

    for i in range(prefix_len, seq_len):
        logits = model(x, causal=True)  # (B, len, V+1)
        next_logits = logits[:, -1, :vocab_size] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        x = torch.cat([x, next_token], dim=1)

    return x


@torch.no_grad()
def compute_generative_perplexity(
    generated_samples: torch.Tensor,
    eval_model,
    eval_tokenizer=None,
) -> float:
    """
    Compute generative perplexity using a pretrained evaluation model (e.g., LLaMA-7B).
    For simplicity, we compute NLL on the generated samples.
    """
    # Placeholder: in practice this uses an external LLM
    log_probs = []
    for x in generated_samples:
        with torch.no_grad():
            outputs = eval_model(x.unsqueeze(0))
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = x[1:].unsqueeze(0).contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                reduction='mean'
            )
            log_probs.append(-loss.item())
    ppl = np.exp(-np.mean(log_probs))
    return float(ppl)
