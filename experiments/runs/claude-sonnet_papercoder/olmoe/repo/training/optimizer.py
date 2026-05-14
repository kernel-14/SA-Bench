## Code: training/optimizer.py

```python
## training/optimizer.py
"""Optimizer creation for OLMoE pretraining and adaptation.

Implements optimizer creation for all training phases of OLMoE-1B-7B.
The most critical design decision is that weight decay is applied to ALL
parameters — including embeddings and RMSNorm weights — which is explicitly
stated in the paper (Sections 4.2.3 and 4.2.4).

Key design decisions from the paper:
  1. Universal weight decay (Sections 4.2.3, 4.2.4):
     "for simplicity, we weight decay all parameters in OLMoE-1B-7B
     including embedding and RMSNorm."
     -> Single parameter group, no no-decay group, no name-based filtering.

  2. AdamW epsilon = 1e-8 (Section 4.2.6):
     Changed from OLMo's 1e-5 to 1e-8, "significantly improves performance."

  3. beta2 = 0.95 (Table 10):
     Non-standard (common default is 0.999), must be set explicitly.

  4. KTO uses RMSProp (Appendix F, Table 14):
     "we use the RMSProp optimizer instead of Adam" for KTO adaptation.

Phase-specific configurations (from config.yaml):
  pretrain: AdamW, lr=4e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
  sft:      AdamW, lr=2e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
  dpo:      AdamW, lr=5e-7, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
  kto:      RMSProp, lr=5e-7, weight_decay=0.1

Configuration values used (from config.yaml):
  pretraining.learning_rate: 4.0e-04
  pretraining.adam_beta1: 0.9
  pretraining.adam_beta2: 0.95
  pretraining.adam_eps: 1.0e-08
  pretraining.weight_decay: 0.1
  sft.learning_rate: 2.0e-05
  sft.adam_beta1: 0.9
  sft.adam_beta2: 0.95
  sft.adam_eps: 1.0e-08
  sft.weight_decay: 0.1
  dpo.learning_rate: 5.0e-07
  dpo.adam_beta1: 0.9
  dpo.adam_beta2: 0.95
  dpo.adam_eps: 1.0e-08
  dpo.weight_decay: 0.1
  kto.learning_rate: 5.0e-07
"""

import logging
from typing import List, Union

import torch
import torch.nn as nn
from torch.optim import AdamW, RMSprop, Optimizer

from config import TrainingConfig, SFTConfig, DPOConfig, KTOConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for supported optimizer types returned by this module.
# ---------------------------------------------------------------------------
SupportedOptimizer = Union[AdamW, RMSprop]


def _collect_trainable_params(model: nn.Module) -> List[torch.nn.Parameter]:
    """Collect all trainable parameters from a model.

    Returns only parameters where requires_grad=True. This correctly handles:
      - Frozen reference models in DPO (all params have requires_grad=False)
      - FSDP-wrapped models (model.parameters() returns sharded params)
      - Any partially frozen models used in ablation experiments

    IMPORTANT: No filtering by parameter name, type, or dimensionality.
    All trainable parameters go into a single group with the same weight decay.
    This enforces the paper's design decision (Sections 4.2.3, 4.2.4):
    "weight decay all parameters including embedding and RMSNorm."

    Args:
        model: The nn.Module to collect parameters from. May be an OLMoEModel,
               an FSDP-wrapped OLMoEModel, or any other nn.Module.

    Returns:
        List of all Parameter objects where requires_grad=True.
        Empty list if all parameters are frozen.
    """
    params: List[torch.nn.Parameter] = [
        p for p in model.parameters() if p.requires_grad
    ]
    return params


def _verify_optimizer(
    optimizer: Optimizer,
    expected_param_count: int,
    expected_weight_decay: float,
) -> None:
    """Verify optimizer invariants after creation.

    Checks that:
      1. The total number of parameters in the optimizer matches the expected count.
         This catches accidental parameter exclusions.
      2. All parameter groups have the expected weight_decay value.
         This catches accidental no-decay groups.

    Logs a warning (rather than raising) for weight_decay mismatches to allow
    flexibility in edge cases (e.g., ablation experiments with wd=0).

    Args:
        optimizer: The optimizer to verify.
        expected_param_count: Expected total number of trainable parameters.
                              Should match len(_collect_trainable_params(model)).
        expected_weight_decay: Expected weight_decay value for all param groups.
                               For OLMoE-1B-7B: 0.1 (all phases).

    Raises:
        AssertionError: If the total parameter count in the optimizer does not
                        match expected_param_count. This is a hard error because
                        missing parameters means they won't be updated during training.
    """
    # Count total parameters across all optimizer param groups.
    total_optimizer_params: int = sum(
        len(group["params"]) for group in optimizer.param_groups
    )

    # Hard assertion: all trainable parameters must be in the optimizer.
    # A mismatch means some parameters will never be updated — a silent bug.
    assert total_optimizer_params == expected_param_count, (
        f"Optimizer parameter count mismatch: "
        f"optimizer has {total_optimizer_params} parameters but "
        f"model has {expected_param_count} trainable parameters. "
        f"This indicates some parameters were accidentally excluded from the optimizer. "
        f"OLMoE requires ALL parameters (including embeddings and RMSNorm) to be "
        f"optimized with weight_decay={expected_weight_decay}."
    )

    # Soft check: warn if any param group has unexpected weight_decay.
    # This is a warning (not an error) to allow ablation experiments.
    for i, group in enumerate(optimizer.param_groups):
        group_wd: float = group.get("weight_decay", 0.0)
        if abs(group_wd - expected_weight_decay) > 1e-9:
            logger.warning(
                f"Optimizer param group {i} has weight_decay={group_wd}, "
                f"expected {expected_weight_decay}. "
                f"The paper (Sections 4.2.3, 4.2.4) specifies weight_decay=0.1 "
                f"for ALL parameters including embeddings and RMSNorm. "
                f"This may be intentional for an ablation experiment."
            )


def _log_optimizer_info(
    optimizer: Optimizer,
    phase: str,
    num_params: int,
) -> None:
    """Log optimizer configuration and parameter statistics.

    Logs the optimizer type, hyperparameters, number of parameters being
    optimized, and total parameter size in MB. This is useful for verifying
    the setup is correct and for debugging.

    Args:
        optimizer: The created optimizer.
        phase: Training phase string (e.g., 'pretrain', 'sft', 'dpo', 'kto').
        num_params: Number of trainable parameters in the optimizer.
    """
    # Compute total parameter size in MB (assuming float32 = 4 bytes).
    # In BF16 training, actual memory is ~half this, but we report float32
    # equivalent for consistency with parameter count reporting.
    total_param_elements: int = sum(
        p.numel()
        for group in optimizer.param_groups
        for p in group["params"]
    )
    param_size_mb: float = total_param_elements * 4 / (1024 ** 2)  # float32 bytes

    # Extract key hyperparameters from the first param group for logging.
    first_group = optimizer.param_groups[0]
    lr: float = first_group.get("lr", 0.0)
    weight_decay: float = first_group.get("weight_decay", 0.0)

    # Build optimizer-specific hyperparameter string.
    opt_type: str = type(optimizer).__name__
    if isinstance(optimizer, AdamW):
        betas = first_group.get("betas", (0.0, 0.0))
        eps: float = first_group.get("eps", 0.0)
        hparam_str: str = (
            f"lr={lr:.2e}, betas=({betas[0]}, {betas[1]}), "
            f"eps={eps:.2e}, weight_decay={weight_decay}"
        )
    elif isinstance(optimizer, RMSprop):
        alpha: float = first_group.get("alpha", 0.0)
        eps = first_group.get("eps", 0.0)
        momentum: float = first_group.get("momentum", 0.0)
        hparam_str = (
            f"lr={lr:.2e}, alpha={alpha}, eps={eps:.2e}, "
            f"momentum={momentum}, weight_decay={weight_decay}"
        )
    else:
        hparam_str = f"lr={lr:.2e}, weight_decay={weight_decay}"

    logger.info(
        f"[{phase.upper()}] Optimizer created: {opt_type}({hparam_str}), "
        f"num_param_tensors={num_params:,}, "
        f"total_elements={total_param_elements:,}, "
        f"param_size_fp32={param_size_mb:.1f}MB"
    )


def create_pretrain_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> AdamW:
    """Create AdamW optimizer for OLMoE pretraining.

    Uses the exact hyperparameters from Table 10 and Appendix B:
      - Optimizer: AdamW (Loshchilov & Hutter 2019)
      - Learning rate: 4e-4 (peak, managed by LRScheduler)
      - beta1: 0.9
      - beta2: 0.95 (non-standard, explicitly set per Table 10)
      - epsilon: 1e-8 (Section 4.2.6; changed from OLMo's 1e-5)
      - weight_decay: 0.1 (applied to ALL parameters, Sections 4.2.3, 4.2.4)

    All parameters (including embeddings and RMSNorm weights) are placed in
    a single parameter group with the same weight_decay=0.1. This is the
    key non-standard design decision from the paper.

    The initial LR is set to config.learning_rate (4e-4). The LRScheduler
    in training/lr_scheduler.py will update this on every step according to
    the warmup → cosine → annealing schedule.

    Args:
        model: The OLMoEModel (or FSDP-wrapped OLMoEModel) to optimize.
               Only parameters with requires_grad=True are included.
        config: TrainingConfig instance. Key fields used:
                - learning_rate (4e-4): initial/peak LR
                - adam_beta1 (0.9): AdamW beta1
                - adam_beta2 (0.95): AdamW beta2
                - adam_eps (1e-8): AdamW epsilon
                - weight_decay (0.1): applied to ALL parameters

    Returns:
        AdamW optimizer with all trainable parameters in a single group.

    Raises:
        AssertionError: If the optimizer parameter count doesn't match the
                        model's trainable parameter count.

    Example:
        >>> config = TrainingConfig()
        >>> model = OLMoEModel(OLMoEConfig())
        >>> optimizer = create_pretrain_optimizer(model, config)
        >>> len(optimizer.param_groups)
        1
        >>> optimizer.param_groups[0]['weight_decay']
        0.1
        >>> optimizer.param_groups[0]['betas']
        (0.9, 0.95)
        >>> optimizer.param_groups[0]['eps']
        1e-08
    """
    # Collect all trainable parameters — no filtering, no grouping.
    trainable_params: List[torch.nn.Parameter] = _collect_trainable_params(model)

    if len(trainable_params) == 0:
        raise ValueError(
            "No trainable parameters found in model for pretraining. "
            "Ensure the model has parameters with requires_grad=True."
        )

    # Create AdamW with a single parameter group containing ALL trainable params.
    # CRITICAL: Do NOT split into decay/no-decay groups.
    # The paper explicitly applies weight_decay=0.1 to all parameters.
    optimizer: AdamW = AdamW(
        params=trainable_params,
        lr=config.learning_rate,          # 4e-4 (config.yaml: pretraining.learning_rate)
        betas=(config.adam_beta1, config.adam_beta2),  # (0.9, 0.95) (Table 10)
        eps=config.adam_eps,              # 1e-8 (config.yaml: pretraining.adam_eps)
        weight_decay=config.weight_decay, # 0.1 (config.yaml: pretraining.weight_decay)
        fused=False,  # Disable fused kernel for FSDP compatibility
    )

    # Verify invariants: all params included, correct weight_decay.
    _verify_optimizer(
        optimizer=optimizer,
        expected_param_count=len(trainable_params),
        expected_weight_decay=config.weight_decay,
    )

    # Log optimizer configuration for debugging and reproducibility.
    _log_optimizer_info(
        optimizer=optimizer,
        phase="pretrain",
        num_params=len(trainable_params),
    )

    return optimizer


def create_sft_optimizer(
    model: nn.Module,
    config: SFTConfig,
) -> AdamW:
    """Create AdamW optimizer for OLMoE instruction tuning (SFT).

    Uses the SFT-specific hyperparameters from Appendix B:
      - Optimizer: AdamW
      - Learning rate: 2e-5 (constant, managed by ConstantLRScheduler)
      - beta1: 0.9
      - beta2: 0.95
      - epsilon: 1e-8
      - weight_decay: 0.1 (applied to ALL parameters)

    Same universal weight decay policy as pretraining — all parameters
    including embeddings and RMSNorm are in a single group with wd=0.1.

    Args:
        model: The OLMoEModel to fine-tune. Should be initialized from the
               post-annealing pretraining checkpoint (Section 4.3).
               Only parameters with requires_grad=True are included.
        config: SFTConfig instance. Key fields used:
                - learning_rate (2e-5): constant LR for SFT
                - adam_beta1 (0.9): AdamW beta1
                - adam_beta2 (0.95): AdamW beta2
                - adam_eps (1e-8): AdamW epsilon
                - weight_decay (0.1): applied to ALL parameters

    Returns:
        AdamW optimizer with all trainable parameters in a single group.

    Raises:
        AssertionError: If the optimizer parameter count doesn't match the
                        model's trainable parameter count.

    Example:
        >>> sft_config = SFTConfig()
        >>> optimizer = create_sft_optimizer(model, sft_config)
        >>> optimizer.param_groups[0]['lr']
        2e-05
        >>> optimizer.param_groups[0]['weight_decay']
        0.1
    """
    trainable_params: List[torch.nn.Parameter] = _collect_trainable_params(model)

    if len(trainable_params) == 0:
        raise ValueError(
            "No trainable parameters found in model for SFT. "
            "Ensure the model has parameters with requires_grad=True."
        )

    optimizer: AdamW = AdamW(
        params=trainable_params,
        lr=config.learning_rate,          # 2e-5 (config.yaml: sft.learning_rate)
        betas=(config.adam_beta1, config.adam_beta2),  # (0.9, 0.95)
        eps=config.adam_eps,              # 1e-8 (config.yaml: sft.adam_eps)
        weight_decay=config.weight_decay, # 0.1 (config.yaml: sft.weight_decay)
        fused=False,
    )

    _verify_optimizer(
        optimizer=optimizer,
        expected_param_count=len(trainable_params),
        expected_weight_decay=config.weight_decay,
    )

    _log_optimizer_info(
        optimizer=optimizer,
        phase="sft",
        num_params=len(trainable_params),
    )

    return optimizer


def create_dpo_optimizer(
    model: nn.Module,
    config: DPOConfig,
) -> AdamW:
    """Create AdamW optimizer for OLMoE preference tuning (DPO).

    Uses the DPO-specific hyperparameters from Appendix B:
      - Optimizer: AdamW
      - Learning rate: 5e-7 (constant, much smaller than SFT)
      - beta1: 0.9
      - beta2: 0.95
      - epsilon: 1e-8
      - weight_decay: 0.1 (applied to ALL parameters)

    Only the policy model (the model being trained) needs an optimizer.
    The reference model (frozen SFT checkpoint) should have all parameters
    set to requires_grad=False before calling this function, so it won't
    be included even if passed accidentally.

    Args:
        model: The policy OLMoEModel to optimize via DPO. Should be
               initialized from the SFT checkpoint (config.dpo.base_model).
               Only parameters with requires_grad=True are included.
        config: DPOConfig instance. Key fields used:
                - learning_rate (5e-7): constant LR for DPO
                - adam_beta1 (0.9): AdamW beta1
                - adam_beta2 (0.95): AdamW beta2
                - adam_eps (1e-8): AdamW epsilon
                - weight_decay (0.1): applied to ALL parameters

    Returns:
        AdamW optimizer with all trainable parameters in a single group.

    Raises:
        AssertionError: If the optimizer parameter count doesn't match the
                        model's trainable parameter count.

    Example:
        >>> dpo_config = DPOConfig()
        >>> optimizer = create_dpo_optimizer(policy_model, dpo_config)
        >>> optimizer.param_groups[0]['lr']
        5e-07
        >>> optimizer.param_groups[0]['betas']
        (0.9, 0.95)
    """
    trainable_params: List[torch.nn.Parameter] = _collect_trainable_params(model)

    if len(trainable_params) == 0:
        raise ValueError(
            "No trainable parameters found in model for DPO. "
            "Ensure the policy model has parameters with requires_grad=True. "
            "The reference model should have requires_grad=False for all params."
        )

    optimizer: AdamW = AdamW(
        params=trainable_params,
        lr=config.learning_rate,          # 5e-7 (config.yaml: dpo.learning_rate)
        betas=(config.adam_beta1, config.adam_beta2),  # (0.9, 0.95)
        eps=config.adam_eps,              # 1e-8 (config.yaml: dpo.adam_eps)
        weight_decay=config.weight_decay, # 0.1 (config.yaml: dpo.weight_decay)
        fused=False,
    )

    _verify_optimizer(
        optimizer=optimizer,
        expected_param_count=len(trainable_params),
        expected_weight_decay=config.weight_decay,
    )

    _log_optimizer_info(
        optimizer=optimizer,
        phase="dpo",
        num_params=len(trainable_params),
    )

    return optimizer


def create_kto_optimizer(
    model: nn.Module,
    config: KTOConfig,
) -> RMSprop:
    """Create RMSProp optimizer for OLMoE KTO adaptation.

    Uses RMSProp instead of AdamW, as specified in Appendix F and Table 14:
    "we use the RMSProp optimizer instead of Adam" for KTO.

    The paper does not specify RMSProp-specific hyperparameters (alpha, momentum)
    beyond the learning rate and weight decay. We use standard defaults:
      - alpha (smoothing constant): 0.99
      - momentum: 0.0
      - epsilon: 1e-8 (consistent with AdamW epsilon choice from Section 4.2.6)

    KTO hyperparameters from Appendix F and Table 14:
      - Optimizer: RMSProp
      - Learning rate: 5e-7 (same as DPO, config.yaml: kto.learning_rate)
      - weight_decay: 0.1 (applied to ALL parameters)
      - Steps: 5000 (1.3 epochs, best configuration from Table 14)

    Args:
        model: The OLMoEModel to optimize via KTO. Should be initialized
               from the SFT checkpoint.
               Only parameters with requires_grad=True are included.
        config: KTOConfig instance. Key fields used:
                - learning_rate (5e-7): constant LR for KTO
                - weight_decay (0.1): applied to ALL parameters

    Returns:
        RMSprop optimizer with all trainable parameters in a single group.

    Raises:
        AssertionError: If the optimizer parameter count doesn't match the
                        model's trainable parameter count.

    Example:
        >>> kto_config = KTOConfig()
        >>> optimizer = create_kto_optimizer(model, kto_config)
        >>> type(optimizer).__name__
        'RMSprop'
        >>> optimizer.param_groups[0]['lr']
        5e-07
        >>> optimizer.param_groups[0]['weight_decay']
        0.1
    """
    trainable_params: List[torch.nn.Parameter] = _collect_trainable_params(model)

    if len(trainable_params) == 0:
        raise ValueError(
            "No trainable parameters found in model for KTO. "
            "Ensure the model has parameters with requires_grad=True."
        )

    # RMSProp hyperparameters.
    # The paper specifies only the optimizer type and LR; we use standard
    # defaults for alpha, momentum, and eps.
    rmsprop_alpha: float = 0.99    # Smoothing constant (standard default)
    rmsprop_momentum: float = 0.0  # No momentum (standard default)
    rmsprop_eps: float = 1e-8      # Consistent with AdamW eps choice (Section 4.2.6)

    optimizer: RMSprop = RMSprop(
        params=trainable_params,
        lr=config.learning_rate,          # 5e-7 (config.yaml: kto.learning_rate)
        alpha=rmsprop_alpha,
        eps=rmsprop_eps,
        weight_decay=config.weight_decay, # 0.1 (config.yaml: kto.weight_decay)
        momentum=rmsprop_momentum,
        centered=False,  # Standard RMSProp (not centered variant)
    )

    _verify_optimizer(
        optimizer=optimizer,
        expected_param_count=len(trainable_params),
        expected_weight_decay=config.weight_decay,
    )

    _log_optimizer_info(
        optimizer=optimizer,
        phase="kto",
        num_params=len(trainable_params),
    )

    return optimizer


def create_optimizer(
    model: nn.Module,
    config: Union[TrainingConfig, SFTConfig, DPOConfig, KTOConfig],
    phase: str = "pretrain",
) -> SupportedOptimizer:
    """Factory function to create the appropriate optimizer for a training phase.

    Dispatches to the phase-specific optimizer creation function based on the
    `phase` argument. Provides a uniform interface for all trainer classes.

    All phases enforce the paper's universal weight decay policy:
    weight_decay=0.1 applied to ALL parameters (including embeddings and
    RMSNorm weights), with no no-decay parameter groups.

    Supported phases and their optimizers (from config.yaml):
        'pretrain': AdamW, lr=4e-4, betas=(0.9, 0.95), eps=1e-8, wd=0.1
                    (config.yaml: pretraining section)
        'sft':      AdamW, lr=2e-5, betas=(0.9, 0.95), eps=1e-8, wd=0.1
                    (config.yaml: sft section)
        'dpo':      AdamW, lr=5e-7, betas=(0.9, 0.95), eps=1e-8, wd=0.1
                    (config.yaml: dpo section)
        'kto':      RMSProp, lr=5e-7, alpha=0.99, eps=1e-8, wd=0.1
                    (config.yaml: kto section)

    Args:
        model: The nn.Module to optimize. May be an OLMoEModel or an
               FSDP-wrapped OLMoEModel. Only parameters with requires_grad=True
               are included in the optimizer.
        config: Configuration instance for the training phase. The type must
                match the phase:
                - 'pretrain': TrainingConfig
                - 'sft': SFTConfig
                - 'dpo': DPOConfig
                - 'kto': KTOConfig
                If a TrainingConfig is passed for non-pretrain phases, the
                function will use the appropriate default hyperparameters
                from the config.yaml values embedded in the phase-specific
                create_*_optimizer functions.
        phase: Training phase string. One of: 'pretrain', 'sft', 'dpo', 'kto'.
               Case-insensitive. Defaults to 'pretrain'.

    Returns:
        The appropriate optimizer (AdamW or RMSprop) with all trainable
        parameters in a single group with weight_decay=0.1.

    Raises:
        ValueError: If phase is not one of the supported values, or if the
                    config type doesn't match the phase.
        AssertionError: If the optimizer parameter count doesn't match the
                        model's trainable parameter count.

    Example:
        >>> # Pretraining
        >>> config = TrainingConfig()
        >>> optimizer = create_optimizer(model, config, phase='pretrain')
        >>> type(optimizer).__name__
        'AdamW'
        >>> optimizer.param_groups[0]['lr']
        0.0004

        >>> # SFT
        >>> sft_config = SFTConfig()
        >>> optimizer = create_optimizer(model, sft_config, phase='sft')
        >>> optimizer.param_groups[0]['lr']
        2e-05

        >>> # DPO (policy model only; reference model has requires_grad=False)
        >>> dpo_config = DPOConfig()
        >>> optimizer = create_optimizer(policy_model, dpo_config, phase='dpo')
        >>> optimizer.param_groups[0]['lr']
        5e-07

        >>> # KTO
        >>> kto_config = KTOConfig()
        >>> optimizer = create_optimizer(model, kto_config, phase='kto')
        >>> type(optimizer).__name__
        'RMSprop'
    """
    phase_lower: str = phase.lower()

    if phase_lower == "pretrain":
        # Validate config type for pretraining.
        if not isinstance(config, TrainingConfig):
            raise ValueError(
                f"phase='pretrain' requires a TrainingConfig instance, "
                f"got {type(config).__name__}. "
                f"Pass a TrainingConfig with pretraining hyperparameters."
            )
        return create_pretrain_optimizer(model=model, config=config)

    elif phase_lower == "sft":
        # Validate config type for SFT.
        if not isinstance(config, SFTConfig):
            raise ValueError(
                f"phase='sft' requires a SFTConfig instance, "
                f"got {type(config).__name__}. "
                f"Pass a SFTConfig with SFT hyperparameters."
            )
        return create_sft_optimizer(model=model, config=config)

    elif phase_lower == "dpo":
        # Validate config type for DPO.
        if not isinstance(config, DPOConfig):
            raise ValueError(
                f"phase='dpo' requires a DPOConfig instance, "
                f"got {type(config).__name__}. "
                f"Pass a DPOConfig with DPO hyperparameters."
            )
        return create_dpo_optimizer(model=model, config=config)

    elif phase_lower == "kto":
        # Validate config type for KTO.
        if not isinstance(config, KTOConfig):
            raise ValueError(
                f"phase='kto' requires a KTOConfig instance, "
                f"got {type(config).__name__}. "
                f"Pass a KTOConfig with KTO hyperparameters."
            )
        return create_kto_optimizer(model=model, config=config)

    else:
        raise ValueError(
            f"Unsupported optimizer phase: '{phase}'. "
            f"Must be one of: 'pretrain', 'sft', 'dpo', 'kto'. "
            f"Got: '{phase}'."
        )


def get_optimizer_state_summary(optimizer: Optimizer) -> dict:
    """Return a summary of the optimizer state for logging and debugging.

    Extracts key statistics from the optimizer's param groups and state dict
    for logging to wandb or console. Useful for verifying the optimizer is
    configured correctly and for monitoring training dynamics.

    Args:
        optimizer: Any optimizer (AdamW or RMSprop) created by this module.

    Returns:
        Dictionary with the following keys:
            - 'optimizer_type': Class name of the optimizer (e.g., 'AdamW')
            - 'num_param_groups': Number of parameter groups
            - 'num_param_tensors': Total number of parameter tensors
            - 'total_param_elements': Total number of scalar parameters
            - 'param_size_mb': Total parameter size in MB (float32 equivalent)
            - 'lr': Learning rate of the first param group
            - 'weight_decay': Weight decay of the first param group
            - 'has_state': Whether the optimizer has accumulated state (True
              after at least one optimizer.step() call)

    Example:
        >>> summary = get_optimizer_state_summary(optimizer)
        >>> summary['optimizer_type']
        'AdamW'
        >>> summary['weight_decay']
        0.1
    """
    first_group = optimizer.param_groups[0] if optimizer.param_groups else {}

    # Count total parameter tensors and elements across all groups.
    num_param_tensors: int = sum(
        len(group["params"]) for group in optimizer.param_groups
    )
    total_elements: int = sum(