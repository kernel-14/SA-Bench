"""Utility functions for MA-RLHF."""
import torch
import numpy as np
from typing import Dict, List, Optional


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(
    p_log_probs: torch.Tensor,
    q_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence D_KL(p || q) per token.

    D_KL(p || q) = p * (log p - log q) = p_log_probs - q_log_probs (in expectation)
    When summed: E_p[log p - log q]
    """
    return p_log_probs - q_log_probs


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Compute mean over masked values."""
    return (tensor * mask).sum(dim=dim) / (mask.sum(dim=dim) + 1e-8)


def masked_sum(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Compute sum over masked values."""
    return (tensor * mask).sum(dim=dim)


def compute_l2_norm(tensor: torch.Tensor) -> float:
    """Compute L2 norm of a tensor (for advantage/Q-value analysis, Fig 11)."""
    return torch.norm(tensor, p=2).item()


def compute_advantage_q_norms(
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
) -> Dict[str, float]:
    """Compute L2 norms of advantages and Q-values.

    Used for Figure 11 analysis.
    """
    q_values = advantages + values
    return {
        "advantage_l2": compute_l2_norm(advantages),
        "q_value_l2": compute_l2_norm(q_values),
        "value_l2": compute_l2_norm(values),
    }


def best_of_n_sampling(
    policy_model,
    tokenizer,
    prompt: str,
    reward_model,
    n: int = 8,
    temperature: float = 1.0,
    max_length: int = 512,
    device: torch.device = torch.device("cuda"),
) -> str:
    """Best-of-N (rejection) sampling.

    Generate N candidates and select the one with highest RM score.
    Used in §4.4 for robustness analysis.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(device)

    best_response = ""
    best_score = float("-inf")

    for _ in range(n):
        with torch.no_grad():
            generated = policy_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        full_mask = torch.ones_like(generated, dtype=torch.float32).to(device)
        score = reward_model(generated, full_mask).item()

        if score > best_score:
            best_score = score
            best_response = tokenizer.decode(
                generated[0, inputs["input_ids"].size(1):],
                skip_special_tokens=True,
            )

    return best_response


def compute_rm_score_distribution(
    rm_scores: List[float],
    bins: int = 50,
) -> Dict[str, np.ndarray]:
    """Compute RM score distribution histogram.

    Used for Figure 3, 10, 14, 16 analysis.
    """
    hist, bin_edges = np.histogram(rm_scores, bins=bins)
    return {"histogram": hist.tolist(), "bin_edges": bin_edges.tolist()}


def format_prompt(
    task: str,
    item: Dict,
) -> str:
    """Format a prompt for the given task."""
    from data import PROMPT_TEMPLATES
    if task == "tldr":
        return PROMPT_TEMPLATES["tldr"].format(
            subreddit=item.get("subreddit", ""),
            title=item.get("title", ""),
            post=item.get("post", item.get("content", "")),
        )
    elif task == "hh-rlhf":
        return PROMPT_TEMPLATES["hh-rlhf"].format(query=item.get("prompt", ""))
    elif task == "webgpt":
        return PROMPT_TEMPLATES["webgpt"].format(question=item.get("question", ""))
    elif task == "apps":
        return PROMPT_TEMPLATES["apps"].format(question=item.get("question", ""))
    return ""
