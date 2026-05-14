## utils/metrics.py
"""Shared metric utilities for scaling law experiments, error imbalance analysis,
and IsoFLOP curve generation.

This module provides pure utility functions consumed by trainers, evaluators,
and the main experiment runner. It has no dependencies on trainers or
evaluators to avoid circular imports.

Functions:
    compute_nll: Computes NLL of a sequence under an MDM or π-learner.
    compute_pi_learner_loss: Computes π-learner loss for an ARM model.
    sample_permutation: Generates permutations for scaling law experiments.
    compute_isoflop_point: Returns (log_flops, val_loss) for IsoFLOP curves.
    count_flops: Estimates FLOPs for one forward pass.
"""

import logging
import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    # Avoid circular imports at runtime; used only for type annotations.
    from models.arm_transformer import ARMTransformer
    from models.mdm_transformer import MDMTransformer

logger = logging.getLogger(__name__)

# Default mask token ID, aligned with config.yaml noise_schedule.mask_token_id
_DEFAULT_MASK_TOKEN_ID: int = 0

# Number of Monte Carlo samples for MDM NLL estimation (no-pi case)
_NLL_MC_SAMPLES: int = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_nll(
    model: nn.Module,
    x0: torch.Tensor,
    pi: Optional[torch.Tensor] = None,
    mask_token_id: int = _DEFAULT_MASK_TOKEN_ID,
    n_mc_samples: int = _NLL_MC_SAMPLES,
) -> float:
    """Computes the negative log-likelihood of sequence(s) under an MDM model.

    Two evaluation modes are supported:

    **Without permutation** (``pi=None``):
        Estimates the MDM ELBO via Monte Carlo sampling of random masks at
        uniformly sampled noise levels.  Returns the average NLL estimate
        over ``n_mc_samples`` random masks.

    **With permutation** (``pi`` provided):
        Evaluates the π-learner factorisation (Equation 3 in the paper):

        .. math::

            -\\log p_\\theta(x_0) = -\\sum_{i=0}^{L-1}
                \\log p_\\theta\\!\\left(x_0^{\\pi(i)} \\mid
                x_0[\\pi\\{i,\\ldots,L-1\\}]\\right)

        Constructs a batch of ``L`` masked sequences and performs a single
        batched forward pass for efficiency.

    Args:
        model: A bidirectional MDM transformer.  Must expose a ``forward``
            method that accepts integer token tensors of shape ``[B, L]`` and
            returns logits of shape ``[B, L, V]``.
        x0: Clean token sequence(s).  Shape ``[L]`` or ``[B, L]``.
        pi: Optional permutation tensor of shape ``[L]`` (dtype ``torch.long``)
            specifying the π-learner evaluation order.  When provided, ``x0``
            must be a single sequence (shape ``[L]`` or ``[1, L]``).
        mask_token_id: Integer ID of the ``[MASK]`` token.  Defaults to ``0``
            (aligned with ``config.yaml`` ``noise_schedule.mask_token_id``).
        n_mc_samples: Number of Monte Carlo mask samples used when ``pi`` is
            ``None``.  Higher values give a lower-variance NLL estimate.

    Returns:
        Scalar NLL estimate (float), averaged over the batch dimension when
        ``x0`` is batched.

    Raises:
        ValueError: If ``pi`` is provided but ``x0`` contains more than one
            sequence.
    """
    # ------------------------------------------------------------------ #
    # Normalise input shape to [B, L]                                     #
    # ------------------------------------------------------------------ #
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)  # [1, L]

    batch_size: int = x0.shape[0]
    seq_len: int = x0.shape[1]
    device: torch.device = x0.device

    if pi is not None and batch_size > 1:
        raise ValueError(
            "compute_nll with a permutation (pi) only supports single sequences "
            f"(batch_size=1), but received batch_size={batch_size}."
        )

    model.eval()
    with torch.no_grad():
        if pi is None:
            return _compute_nll_mc(
                model=model,
                x0=x0,
                mask_token_id=mask_token_id,
                n_mc_samples=n_mc_samples,
                device=device,
            )
        else:
            return _compute_nll_pi_learner(
                model=model,
                x0=x0.squeeze(0),  # [L]
                pi=pi,
                mask_token_id=mask_token_id,
                device=device,
            )


def compute_pi_learner_loss(
    model: nn.Module,
    x0: torch.Tensor,
    pi: torch.Tensor,
) -> float:
    """Computes the π-learner loss for a causal ARM model.

    The ARM π-learner is trained on permuted sequences ``pi(x0)`` with
    standard left-to-right causal attention.  This function evaluates the
    resulting loss:

    .. math::

        \\mathcal{L}_\\pi = -\\sum_{i=0}^{L-1}
            \\log p_\\theta\\!\\left(x_0^{\\pi(i)} \\mid
            x_0^{\\pi(0)}, \\ldots, x_0^{\\pi(i-1)}\\right)

    which equals the standard causal LM cross-entropy on the permuted
    sequence ``x_permuted = x0[:, pi]``.

    Args:
        model: A causal ARM transformer.  Must expose a ``forward`` method
            accepting integer token tensors of shape ``[B, L]`` and returning
            logits of shape ``[B, L, V]``.
        x0: Clean token sequence(s).  Shape ``[L]`` or ``[B, L]``.
        pi: Permutation tensor of shape ``[L]``, dtype ``torch.long``.

    Returns:
        Mean π-learner loss over the batch (float).
    """
    # Normalise to [B, L]
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)

    batch_size: int = x0.shape[0]
    seq_len: int = x0.shape[1]
    device: torch.device = x0.device

    # Permute each sequence in the batch: x_permuted[b, i] = x0[b, pi[i]]
    pi_expanded: torch.Tensor = pi.unsqueeze(0).expand(batch_size, -1)  # [B, L]
    x_permuted: torch.Tensor = torch.gather(x0, dim=1, index=pi_expanded)  # [B, L]

    model.eval()
    with torch.no_grad():
        logits: torch.Tensor = model(x_permuted)  # [B, L, V]

    vocab_size: int = logits.shape[-1]

    # Standard causal LM loss: predict position i+1 from positions 0..i.
    # logits[:, :-1, :] predicts x_permuted[:, 1:].
    logits_shifted: torch.Tensor = logits[:, :-1, :].contiguous().view(
        -1, vocab_size
    )  # [(B*(L-1)), V]
    targets_shifted: torch.Tensor = x_permuted[:, 1:].contiguous().view(-1)  # [B*(L-1)]

    loss: torch.Tensor = F.cross_entropy(
        logits_shifted,
        targets_shifted,
        reduction="mean",
    )
    return loss.item()


def sample_permutation(
    L: int,
    mode: str,
    n_swaps: Optional[Union[int, str]] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generates a permutation of ``[0, L-1]`` for scaling law experiments.

    Supports four modes that interpolate between the ARM baseline (identity)
    and the MDM-equivalent (uniform random), as described in Appendix C.1 of
    the paper.

    Args:
        L: Sequence length.  The returned permutation is over ``[0, L-1]``.
        mode: One of:

            * ``'identity'`` — returns ``[0, 1, ..., L-1]`` (ARM baseline).
            * ``'uniform'`` — returns a fully random permutation (MDM-equivalent).
            * ``'closer'`` — starts from identity and applies ``L // 10``
              random transpositions.
            * ``'much_closer'`` — starts from identity and applies
              ``int(sqrt(L))`` random transpositions.

        n_swaps: Optional override for the number of random transpositions.
            Accepts an integer or one of the symbolic strings ``'sqrt_L'``
            (resolved to ``int(sqrt(L))``) and ``'L_over_10'`` (resolved to
            ``L // 10``).  When provided, overrides the swap count implied by
            ``mode``.
        seed: Optional random seed for reproducibility.  When ``None``, the
            global PyTorch RNG state is used.

    Returns:
        A ``torch.long`` tensor of shape ``[L]`` containing a permutation of
        ``[0, L-1]``.

    Raises:
        ValueError: If ``mode`` is not one of the four recognised values.
    """
    valid_modes: List[str] = ["identity", "uniform", "closer", "much_closer"]
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown permutation mode '{mode}'.  "
            f"Expected one of {valid_modes}."
        )

    # Resolve symbolic n_swaps strings from config.
    resolved_swaps: Optional[int] = _resolve_n_swaps(n_swaps, L)

    # Optionally seed a local generator for reproducibility without
    # disturbing the global RNG.
    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    if mode == "identity":
        return torch.arange(L, dtype=torch.long)

    if mode == "uniform":
        return torch.randperm(L, generator=generator)

    # Modes 'closer' and 'much_closer': start from identity, apply swaps.
    if resolved_swaps is not None:
        n_swap_steps: int = resolved_swaps
    elif mode == "closer":
        n_swap_steps = L // 10
    else:  # much_closer
        n_swap_steps = int(math.sqrt(L))

    perm: torch.Tensor = torch.arange(L, dtype=torch.long)
    perm = _apply_random_swaps(perm, n_swap_steps, generator=generator)
    return perm


def compute_isoflop_point(
    model: nn.Module,
    n_tokens_seen: int,
    val_loss: float = 0.0,
) -> Tuple[float, float]:
    """Returns ``(log_flops, val_loss)`` for an IsoFLOP scaling law curve.

    Computes total training FLOPs as:

    .. math::

        C = 6 \\times N_{\\text{params}} \\times n_{\\text{tokens\\_seen}}

    following Hoffmann et al. (2022) and the paper's Appendix C.1.  The
    x-axis of Fig. 2 (left) is ``log(FLOPs)`` (natural logarithm).

    Args:
        model: The trained model.  Non-embedding parameters are counted
            automatically.
        n_tokens_seen: Total number of tokens processed during training
            (``= C / (6 * N_params)`` when solving for the optimal token
            count given a FLOP budget ``C``).
        val_loss: Validation loss to associate with this data point.  The
            function packages it with the computed ``log_flops`` value.

    Returns:
        A tuple ``(log_flops, val_loss)`` where ``log_flops = log(C)``
        (natural logarithm) and ``val_loss`` is the value passed in.
    """
    n_params: int = count_non_embedding_params(model)
    total_flops: float = 6.0 * n_params * n_tokens_seen
    # Guard against log(0) for degenerate cases (e.g. empty models in tests).
    log_flops: float = math.log(max(total_flops, 1.0))
    return log_flops, val_loss


def count_flops(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
) -> float:
    """Estimates training FLOPs for one forward+backward pass.

    Uses the standard approximation from Kaplan et al. (2020) and Hoffmann
    et al. (2022):

    .. math::

        \\text{FLOPs} \\approx 6 \\times N_{\\text{params}} \\times
        \\text{seq\\_len} \\times \\text{batch\\_size}

    where ``N_params`` counts only non-embedding parameters.

    Args:
        model: The model to profile.
        seq_len: Sequence length (number of tokens per sample).
        batch_size: Number of sequences in the batch.

    Returns:
        Estimated FLOPs as a float.
    """
    n_params: int = count_non_embedding_params(model)
    tokens_per_batch: int = seq_len * batch_size
    return 6.0 * n_params * tokens_per_batch


def count_non_embedding_params(model: nn.Module) -> int:
    """Counts trainable non-embedding parameters in a model.

    Embedding layers (``token_emb``, ``pos_emb``, and any parameter whose
    name contains ``'embed'``) are excluded, following the IsoFLOP convention
    of Hoffmann et al. (2022) and Kaplan et al. (2020).

    Args:
        model: The model to inspect.

    Returns:
        Total number of non-embedding trainable parameters.
    """
    _EMBEDDING_KEYWORDS: Tuple[str, ...] = ("token_emb", "pos_emb", "embed")

    total: int = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in _EMBEDDING_KEYWORDS):
            logger.debug("Excluding embedding parameter '%s' (%d).", name, param.numel())
            continue
        total += param.numel()
    return total


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_nll_mc(
    model: nn.Module,
    x0: torch.Tensor,
    mask_token_id: int,
    n_mc_samples: int,
    device: torch.device,
) -> float:
    """Monte Carlo estimate of MDM NLL without a fixed permutation.

    Samples ``n_mc_samples`` random noise levels ``t ~ Uniform(0, 1)`` and
    corresponding random masks, computes cross-entropy over masked positions,
    and returns the average as the NLL estimate.

    Args:
        model: Bidirectional MDM transformer.
        x0: Clean sequences, shape ``[B, L]``.
        mask_token_id: ID of the ``[MASK]`` token.
        n_mc_samples: Number of Monte Carlo samples.
        device: Device for tensor construction.

    Returns:
        Mean NLL estimate (float) over the batch.
    """
    batch_size: int = x0.shape[0]
    seq_len: int = x0.shape[1]
    total_nll: float = 0.0

    for _ in range(n_mc_samples):
        # Sample noise level t ~ Uniform(0, 1) per sequence.
        t: torch.Tensor = torch.rand(batch_size, device=device)  # [B]

        # Linear schedule: alpha_t = 1 - t  →  mask prob = t.
        mask_prob: torch.Tensor = t.unsqueeze(1).expand(batch_size, seq_len)  # [B, L]

        # Bernoulli masking: 1 = masked, 0 = revealed.
        is_masked: torch.Tensor = torch.bernoulli(mask_prob).bool()  # [B, L]

        # Construct x_t: replace masked positions with mask_token_id.
        x_t: torch.Tensor = x0.clone()
        x_t[is_masked] = mask_token_id

        # Forward pass.
        logits: torch.Tensor = model(x_t)  # [B, L, V]
        vocab_size: int = logits.shape[-1]

        # Cross-entropy only over masked positions.
        # Flatten and apply ignore_index for unmasked positions.
        targets: torch.Tensor = x0.clone()
        targets[~is_masked] = -100  # ignore_index for F.cross_entropy

        loss: torch.Tensor = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        total_nll += loss.item()

    return total_nll / n_mc_samples


def _compute_nll_pi_learner(
    model: nn.Module,
    x0: torch.Tensor,
    pi: torch.Tensor,
    mask_token_id: int,
    device: torch.device,
) -> float:
    """Evaluates the π-learner NLL for a single sequence using a batched pass.

    Constructs a batch of ``L`` masked sequences where row ``i`` has positions
    ``pi[i], pi[i+1], ..., pi[L-1]`` masked, then performs a single batched
    forward pass.

    Args:
        model: Bidirectional MDM transformer.
        x0: Single clean sequence, shape ``[L]``.
        pi: Permutation tensor, shape ``[L]``, dtype ``torch.long``.
        mask_token_id: ID of the ``[MASK]`` token.
        device: Device for tensor construction.

    Returns:
        NLL of ``x0`` under the π-learner factorisation (float).
    """
    seq_len: int = x0.shape[0]

    # Build x_batch[i] = x0 with positions pi[i], pi[i+1], ..., pi[L-1] masked.
    # Shape: [L, L]
    x_batch: torch.Tensor = x0.unsqueeze(0).expand(seq_len, -1).clone()  # [L, L]

    # For row i, mask positions pi[i:].
    # Build a boolean mask matrix: mask_matrix[i, j] = True iff j is in pi[i:].
    # Efficient construction: for each row i, the masked positions are pi[i:].
    mask_matrix: torch.Tensor = torch.zeros(
        seq_len, seq_len, dtype=torch.bool, device=device
    )
    for i in range(seq_len):
        # Positions to mask in row i: pi[i], pi[i+1], ..., pi[L-1]
        mask_matrix[i, pi[i:]] = True

    x_batch[mask_matrix] = mask_token_id  # [L, L]

    # Single batched forward pass.
    logits: torch.Tensor = model(x_batch)  # [L, L, V]
    vocab_size: int = logits.shape[-1]

    # For row i, extract logit at position pi[i] and compute log-prob of x0[pi[i]].
    # logits[i, pi[i], :] → log-prob of true token x0[pi[i]].
    row_indices: torch.Tensor = torch.arange(seq_len, device=device)  # [L]
    position_indices: torch.Tensor = pi  # [L]

    # Gather logits at the target positions: shape [L, V]
    target_logits: torch.Tensor = logits[row_indices, position_indices, :]  # [L, V]

    # True token values at each target position: x0[pi[i]] for each i.
    true_tokens: torch.Tensor = x0[pi]  # [L]

    # Log-probabilities via log-softmax.
    log_probs: torch.Tensor = F.log_softmax(target_logits, dim=-1)  # [L, V]

    # Gather log-prob of the true token at each step.
    true_log_probs: torch.Tensor = log_probs[
        row_indices, true_tokens
    ]  # [L]

    # NLL = -sum of log-probs over all L steps.
    nll: float = -true_log_probs.sum().item()
    return nll


def _resolve_n_swaps(
    n_swaps: Optional[Union[int, str]],
    L: int,
) -> Optional[int]:
    """Resolves symbolic ``n_swaps`` strings from the config to integers.

    Handles the config values ``'sqrt_L'`` and ``'L_over_10'`` as described
    in ``config.yaml`` ``scaling_law.permutations.distributions``.

    Args:
        n_swaps: Integer, symbolic string, or ``None``.
        L: Sequence length used to resolve symbolic values.

    Returns:
        Resolved integer number of swaps, or ``None`` if ``n_swaps`` is
        ``None``.

    Raises:
        ValueError: If ``n_swaps`` is a string but not a recognised symbol.
    """
    if n_swaps is None:
        return None
    if isinstance(n_swaps, int):
        return n_swaps
    if isinstance(n_swaps, str):
        if n_swaps == "sqrt_L":
            return int(math.sqrt(L))
        if n_swaps == "L_over_10":
            return L // 10
        # Try parsing as a plain integer string.
        try:
            return int(n_swaps)
        except ValueError:
            pass
        raise ValueError(
            f"Unrecognised symbolic n_swaps value '{n_swaps}'.  "
            "Expected an integer, 'sqrt_L', or 'L_over_10'."
        )
    # Fallback: attempt int conversion for numeric types (e.g. float from YAML).
    return int(n_swaps)


def _apply_random_swaps(
    perm: torch.Tensor,
    n_swaps: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Applies ``n_swaps`` random transpositions to a permutation tensor.

    Each transposition picks two distinct positions uniformly at random and
    swaps the values at those positions.

    Args:
        perm: Permutation tensor of shape ``[L]``, dtype ``torch.long``.
            Modified in-place and returned.
        n_swaps: Number of random transpositions to apply.
        generator: Optional ``torch.Generator`` for reproducibility.

    Returns:
        The modified permutation tensor (same object as ``perm``).
    """
    L: int = perm.shape[0]
    if L < 2 or n_swaps <= 0:
        return perm

    for _ in range(n_swaps):
        # Sample two distinct indices uniformly from [0, L).
        idx: torch.Tensor = torch.randperm(L, generator=generator)[:2]
        i: int = idx[0].item()  # type: ignore[assignment]
        j: int = idx[1].item()  # type: ignore[assignment]
        perm[i], perm[j] = perm[j].clone(), perm[i].clone()

    return perm
