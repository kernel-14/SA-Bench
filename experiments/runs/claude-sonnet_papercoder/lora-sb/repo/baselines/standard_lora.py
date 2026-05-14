## baselines/standard_lora.py
"""Standard LoRA-family baseline configurations for LoRA-SB reproduction.

This module provides thin factory functions that translate a Config object into
PEFT LoraConfig instances for the four standard LoRA-family baselines evaluated
in the paper: LoRA, rsLoRA, PiSSA, and DoRA.

These baselines are compared against LoRA-SB in Tables 1-3 of the paper across
arithmetic reasoning (Mistral-7B, Gemma-2 9B), commonsense reasoning
(Llama-3.2 3B), and natural language understanding (RoBERTa-large) benchmarks.

All four methods share the same core PEFT LoraConfig structure with
alpha = rank (config.yaml: baselines.*.alpha_equals_rank: true), differing
only in their initialization strategy or scaling mechanism:

    - LoRA (ref 17): Standard Gaussian/zeros initialization, s = alpha/rank
    - rsLoRA (ref 20): Rank-stabilized scaling s = alpha/sqrt(rank)
    - PiSSA (ref 30): Principal singular vector initialization from W₀
    - DoRA (ref 26): Weight-decomposed low-rank adaptation

This module is consumed exclusively by ModelBuilder.build_lora(),
build_rslora(), build_pissa(), and build_dora() in lora_sb/model_builder.py.
The returned LoraConfig is passed directly to peft.get_peft_model().

What this module does NOT handle:
    - LoRA-XS: custom nn.Module in baselines/lora_xs.py (PEFT lacks frozen R)
    - LoRA-Pro: custom gradient hooks in baselines/lora_pro.py
    - LoRA-SB: handled entirely in lora_sb/ modules
    - Full FT: ModelBuilder.build_full_ft() unfreezes all parameters directly

References:
    Paper Section 3 (Tables 1-3): Baseline comparison results
    config.yaml: baselines.lora, baselines.rslora, baselines.pissa, baselines.dora
    PEFT documentation: https://huggingface.co/docs/peft/conceptual_guides/lora

Typical usage:
    from baselines.standard_lora import get_lora_config, get_pissa_config
    from peft import get_peft_model

    peft_config = get_lora_config(config)
    model = get_peft_model(base_model, peft_config)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Union

from peft import LoraConfig, TaskType

from config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid task strings from Config (sourced from config.py _VALID_TASKS)
# ---------------------------------------------------------------------------
_TASK_TO_PEFT_TASK_TYPE: Dict[str, TaskType] = {
    "math": TaskType.CAUSAL_LM,
    "commonsense": TaskType.CAUSAL_LM,
    "glue": TaskType.SEQ_CLS,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_task_type(config: Config) -> TaskType:
    """Map config.task string to the corresponding PEFT TaskType enum value.

    The task type is required by LoraConfig to correctly handle weight shapes
    and forward pass behavior in PEFT's model wrapping logic.

    Mapping (sourced from config.py _VALID_TASKS and PEFT TaskType enum):
        "math"         → TaskType.CAUSAL_LM  (Mistral-7B, Gemma-2 9B)
        "commonsense"  → TaskType.CAUSAL_LM  (Llama-3.2 3B)
        "glue"         → TaskType.SEQ_CLS    (RoBERTa-large)

    Note: For TaskType.SEQ_CLS (RoBERTa/GLUE), the number of labels is set
    on the base model config before loading, not in LoraConfig. ModelBuilder
    handles num_labels at model loading time.

    Args:
        config: Experiment configuration. Uses config.task field.

    Returns:
        The PEFT TaskType enum value corresponding to config.task.

    Raises:
        ValueError: If config.task is not one of the recognized task strings.
            This should not occur if Config._validate() has been called, but
            is included as a defensive check.
    """
    task_type: TaskType | None = _TASK_TO_PEFT_TASK_TYPE.get(config.task)
    if task_type is None:
        raise ValueError(
            f"Unrecognized task '{config.task}' for PEFT TaskType mapping. "
            f"Expected one of: {list(_TASK_TO_PEFT_TASK_TYPE.keys())}. "
            f"Ensure Config._validate() was called before this function."
        )
    return task_type


def _get_base_lora_kwargs(config: Config) -> Dict[str, Any]:
    """Build the shared LoraConfig keyword arguments for all LoRA-family baselines.

    Extracts and derives the parameters that are identical across LoRA, rsLoRA,
    PiSSA, and DoRA. Method-specific flags (use_rslora, use_dora,
    init_lora_weights) are added by each public factory function.

    Parameter derivation (all sourced from config.yaml):

    rank:
        config.rank directly. Values: {32, 64, 96} for LLMs (Table 8),
        {8, 16, 24} for RoBERTa (Table 9). For LoRA baselines in Tables 1-3,
        rank=32 is the primary comparison point.

    lora_alpha:
        config.rank (alpha = rank for all LoRA-family baselines per
        config.yaml: baselines.lora.alpha_equals_rank: true). This gives
        effective scaling s = alpha/rank = 1.0 for standard LoRA. For rsLoRA,
        PEFT uses alpha/sqrt(rank) internally when use_rslora=True.

    target_modules:
        config.target_modules. Set per model family in YAML configs:
        - LLMs: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj',
                 'up_proj', 'down_proj'] (all attention + MLP, Section 3.1)
        - RoBERTa: ['query', 'key', 'value', 'dense'] (attention only, Section 3.3)

    lora_dropout:
        config.dropout. Values: 0.0 for Mistral/Gemma/RoBERTa, 0.05 for
        Llama-3.2 3B (config.yaml: llama_commonsense.training.dropout: 0.05).

    bias:
        "none" — standard practice for LoRA; the paper does not mention bias
        adaptation for any baseline. Fixed to "none" regardless of config.

    task_type:
        Derived from config.task via _get_task_type().

    modules_to_save:
        None — no additional modules need saving beyond the LoRA adapters.
        The classifier head for RoBERTa is handled by PEFT automatically
        when task_type=TaskType.SEQ_CLS.

    Args:
        config: Experiment configuration. Key fields: rank, target_modules,
            dropout, task.

    Returns:
        A dict of keyword arguments suitable for passing to LoraConfig(**kwargs)
        after adding the method-specific flag. Contains keys: 'r', 'lora_alpha',
        'target_modules', 'lora_dropout', 'bias', 'task_type', 'modules_to_save'.
    """
    task_type: TaskType = _get_task_type(config)

    # alpha = rank for all LoRA-family baselines (config.yaml: alpha_equals_rank: true)
    # This gives s = alpha/rank = 1.0 for standard LoRA.
    # For rsLoRA: PEFT uses alpha/sqrt(rank) when use_rslora=True.
    lora_alpha: int = config.rank

    logger.debug(
        "Building base LoRA kwargs: rank=%d, lora_alpha=%d, "
        "target_modules=%s, dropout=%.3f, task_type=%s",
        config.rank,
        lora_alpha,
        config.target_modules,
        config.dropout,
        task_type.value,
    )

    return {
        "r": config.rank,
        "lora_alpha": lora_alpha,
        "target_modules": config.target_modules,
        "lora_dropout": config.dropout,
        # "none": do not adapt bias parameters. Standard LoRA practice.
        # The paper does not mention bias adaptation for any baseline.
        "bias": "none",
        "task_type": task_type,
        # No additional modules to save beyond LoRA adapters.
        # For RoBERTa SEQ_CLS, PEFT handles the classifier head automatically.
        "modules_to_save": None,
    }


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def get_lora_config(config: Config) -> LoraConfig:
    """Build a LoraConfig for standard LoRA (ref 17).

    Standard LoRA parameterizes the weight update as:
        W = W₀ + (alpha/rank) * B @ A

    where B ∈ R^{m×r} is initialized with Kaiming uniform (non-zero) and
    A ∈ R^{r×n} is initialized to zeros. This ensures B @ A = 0 at
    initialization, so the model starts identical to the pre-trained model.

    Scaling: s = alpha/rank = rank/rank = 1.0 (since alpha = rank per
    config.yaml: baselines.lora.alpha_equals_rank: true).

    Trainable parameters per layer: r * (m + n)
    Example (Mistral-7B, r=32): 32 * (4096 + 4096) = 262,144 per layer
    Total (Mistral-7B, r=32): ~83.88M (Table 1)

    This is the primary LoRA baseline in Tables 1-3. LoRA-SB outperforms
    this baseline while using 27-90x fewer trainable parameters.

    Args:
        config: Experiment configuration. Key fields used:
            - config.rank: LoRA rank r (32 for LLM comparisons in Tables 1-3)
            - config.target_modules: Layers to apply LoRA to
            - config.dropout: LoRA dropout (0.0 for most tasks, 0.05 for Llama)
            - config.task: Used to determine PEFT TaskType

    Returns:
        A LoraConfig instance ready for use with peft.get_peft_model().
        Key settings: init_lora_weights=True (Gaussian B, zeros A),
        use_rslora=False, use_dora=False.

    Example:
        >>> peft_config = get_lora_config(config)
        >>> model = get_peft_model(base_model, peft_config)
        >>> # model now has LoRA adapters on target_modules
        >>> # Only lora_A and lora_B are trainable
    """
    kwargs: Dict[str, Any] = _get_base_lora_kwargs(config)

    # Standard LoRA initialization: Gaussian for B, zeros for A.
    # True is the PEFT default but we set it explicitly for clarity.
    # config.yaml: baselines.lora.init_lora_weights: true
    kwargs["init_lora_weights"] = True

    # Standard LoRA scaling: s = alpha / rank (not rank-stabilized).
    # use_rslora=False is the PEFT default but set explicitly for clarity.
    kwargs["use_rslora"] = False

    # No weight decomposition (DoRA disabled).
    # use_dora=False is the PEFT default but set explicitly for clarity.
    kwargs["use_dora"] = False

    lora_config: LoraConfig = LoraConfig(**kwargs)

    logger.info(
        "Created standard LoRA config: rank=%d, alpha=%d, "
        "target_modules=%s, dropout=%.3f, task_type=%s | "
        "effective_scaling=%.4f",
        config.rank,
        config.rank,  # lora_alpha = rank
        config.target_modules,
        config.dropout,
        _get_task_type(config).value,
        1.0,  # alpha/rank = rank/rank = 1.0
    )

    return lora_config


def get_rslora_config(config: Config) -> LoraConfig:
    """Build a LoraConfig for rsLoRA (ref 20) with rank-stabilized scaling.

    rsLoRA modifies the LoRA scaling factor from s = alpha/rank to
    s = alpha/sqrt(rank), which provides more stable training at higher ranks
    by preventing the update magnitude from shrinking as rank increases.

    With alpha = rank (config.yaml: baselines.rslora.alpha_equals_rank: true):
        Standard LoRA: s = rank/rank = 1.0
        rsLoRA:        s = rank/sqrt(rank) = sqrt(rank)

    For rank=32: rsLoRA effective scale = sqrt(32) ≈ 5.66
    For rank=64: rsLoRA effective scale = sqrt(64) = 8.0
    For rank=96: rsLoRA effective scale = sqrt(96) ≈ 9.80

    In PEFT, rsLoRA is activated by setting use_rslora=True in LoraConfig.
    PEFT internally computes the scaling as lora_alpha / sqrt(r) when this
    flag is set.

    Trainable parameters per layer: r * (m + n) — identical to standard LoRA.
    Total (Mistral-7B, r=32): ~83.88M (Table 1, same as LoRA).

    Args:
        config: Experiment configuration. Key fields used:
            - config.rank: LoRA rank r
            - config.target_modules: Layers to apply rsLoRA to
            - config.dropout: LoRA dropout
            - config.task: Used to determine PEFT TaskType

    Returns:
        A LoraConfig instance with use_rslora=True, ready for use with
        peft.get_peft_model(). Key settings: init_lora_weights=True
        (Gaussian B, zeros A), use_rslora=True, use_dora=False.

    Example:
        >>> peft_config = get_rslora_config(config)
        >>> model = get_peft_model(base_model, peft_config)
        >>> # model now has rsLoRA adapters with sqrt(rank) scaling
    """
    kwargs: Dict[str, Any] = _get_base_lora_kwargs(config)

    # Standard initialization: Gaussian for B, zeros for A.
    # Same as standard LoRA — rsLoRA only changes the scaling factor.
    kwargs["init_lora_weights"] = True

    # Rank-stabilized scaling: s = alpha / sqrt(rank).
    # config.yaml: baselines.rslora.use_rslora: true
    kwargs["use_rslora"] = True

    # No weight decomposition.
    kwargs["use_dora"] = False

    lora_config: LoraConfig = LoraConfig(**kwargs)

    import math
    effective_scale: float = config.rank / math.sqrt(config.rank)  # = sqrt(rank)

    logger.info(
        "Created rsLoRA config: rank=%d, alpha=%d, "
        "target_modules=%s, dropout=%.3f, task_type=%s | "
        "effective_scaling=%.4f (= sqrt(%d))",
        config.rank,
        config.rank,  # lora_alpha = rank
        config.target_modules,
        config.dropout,
        _get_task_type(config).value,
        effective_scale,
        config.rank,
    )

    return lora_config


def get_pissa_config(config: Config) -> LoraConfig:
    """Build a LoraConfig for PiSSA (ref 30) with principal singular vector init.

    PiSSA (Principal Singular values and Singular vectors Adaptation) initializes
    the LoRA matrices B and A using the principal singular vectors of the
    pre-trained weight matrix W₀, rather than the standard Gaussian/zeros
    initialization. This captures the most significant subspaces of the
    pre-trained model.

    PiSSA initialization (from ref 30):
        U, S, Vt = SVD(W₀)
        B_init = U[:, :r] * sqrt(S[:r])    shape (m, r)
        A_init = sqrt(S[:r]) * Vt[:r, :]   shape (r, n)

    This ensures B @ A ≈ W₀ (low-rank approximation of the pre-trained weight),
    and the residual W₀ - B @ A is stored as the frozen base weight.

    Key distinction from LoRA-SB:
        PiSSA: initializes from W₀ (pre-trained weight subspace)
        LoRA-SB: initializes from ΔW_avg (task-relevant subspace from first FT step)

    The paper (Section 2.2) notes that PiSSA-style initialization "fails to
    capture the specific subspaces relevant to the FT task" — this is the
    core motivation for LoRA-SB's initialization strategy.

    In PEFT, PiSSA is activated by setting init_lora_weights='pissa'.
    PEFT handles the SVD computation and weight residual storage internally.

    Trainable parameters per layer: r * (m + n) — identical to standard LoRA.
    Total (Mistral-7B, r=32): ~83.88M (Table 1, same as LoRA).

    Args:
        config: Experiment configuration. Key fields used:
            - config.rank: LoRA rank r
            - config.target_modules: Layers to apply PiSSA to
            - config.dropout: LoRA dropout
            - config.task: Used to determine PEFT TaskType

    Returns:
        A LoraConfig instance with init_lora_weights='pissa', ready for use
        with peft.get_peft_model(). Key settings: init_lora_weights='pissa',
        use_rslora=False, use_dora=False.

    Example:
        >>> peft_config = get_pissa_config(config)
        >>> model = get_peft_model(base_model, peft_config)
        >>> # model now has PiSSA adapters initialized from SVD of W₀
    """
    kwargs: Dict[str, Any] = _get_base_lora_kwargs(config)

    # PiSSA initialization: principal singular vectors of W₀.
    # PEFT computes SVD of W₀ and initializes B, A from top-r singular vectors.
    # config.yaml: baselines.pissa.init_lora_weights: pissa
    kwargs["init_lora_weights"] = "pissa"

    # Standard LoRA scaling (not rank-stabilized).
    kwargs["use_rslora"] = False

    # No weight decomposition.
    kwargs["use_dora"] = False

    lora_config: LoraConfig = LoraConfig(**kwargs)

    logger.info(
        "Created PiSSA config: rank=%d, alpha=%d, "
        "target_modules=%s, dropout=%.3f, task_type=%s | "
        "init=pissa (SVD of W₀), effective_scaling=%.4f",
        config.rank,
        config.rank,  # lora_alpha = rank
        config.target_modules,
        config.dropout,
        _get_task_type(config).value,
        1.0,  # alpha/rank = rank/rank = 1.0
    )

    return lora_config


def get_dora_config(config: Config) -> LoraConfig:
    """Build a LoraConfig for DoRA (ref 26) with weight-decomposed adaptation.

    DoRA (Weight-Decomposed Low-Rank Adaptation) decomposes the pre-trained
    weight W₀ into magnitude and direction components, then applies LoRA-style
    adaptation to the direction component while learning a separate magnitude
    scaling vector.

    DoRA decomposition:
        W = m * (W₀ + B @ A) / ||W₀ + B @ A||_c

    where m is a learnable magnitude vector (one scalar per output feature)
    and ||·||_c denotes column-wise normalization. This allows DoRA to
    independently control the magnitude and direction of weight updates.

    The magnitude vector adds a small number of additional parameters compared
    to standard LoRA. For Mistral-7B with r=32, DoRA has ~85.26M trainable
    parameters vs LoRA's ~83.88M (Table 1 in the paper).

    In PEFT, DoRA is activated by setting use_dora=True in LoraConfig.
    PEFT handles the weight decomposition and magnitude vector internally.

    Initialization: Standard LoRA initialization (Gaussian B, zeros A) for
    the direction component. The magnitude vector is initialized from the
    column norms of W₀.
    config.yaml: baselines.dora.use_dora: true, init_lora_weights: true (implied).

    Args:
        config: Experiment configuration. Key fields used:
            - config.rank: LoRA rank r
            - config.target_modules: Layers to apply DoRA to
            - config.dropout: LoRA dropout
            - config.task: Used to determine PEFT TaskType

    Returns:
        A LoraConfig instance with use_dora=True, ready for use with
        peft.get_peft_model(). Key settings: init_lora_weights=True,
        use_rslora=False, use_dora=True.

    Example:
        >>> peft_config = get_dora_config(config)
        >>> model = get_peft_model(base_model, peft_config)
        >>> # model now has DoRA adapters with magnitude+direction decomposition
        >>> # Slightly more parameters than standard LoRA due to magnitude vectors
    """
    kwargs: Dict[str, Any] = _get_base_lora_kwargs(config)

    # Standard initialization for the direction component.
    # DoRA initializes the magnitude vector from column norms of W₀ internally.
    # config.yaml: baselines.dora.use_dora: true (init_lora_weights=True implied)
    kwargs["init_lora_weights"] = True

    # Standard LoRA scaling (not rank-stabilized).
    kwargs["use_rslora"] = False

    # Weight-decomposed adaptation: magnitude + direction components.
    # config.yaml: baselines.dora.use_dora: true
    kwargs["use_dora"] = True

    lora_config: LoraConfig = LoraConfig(**kwargs)

    logger.info(
        "Created DoRA config: rank=%d, alpha=%d, "
        "target_modules=%s, dropout=%.3f, task_type=%s | "
        "use_dora=True (magnitude+direction decomposition), effective_scaling=%.4f",
        config.rank,
        config.rank,  # lora_alpha = rank
        config.target_modules,
        config.dropout,
        _get_task_type(config).value,
        1.0,  # alpha/rank = rank/rank = 1.0
    )

    return lora_config
