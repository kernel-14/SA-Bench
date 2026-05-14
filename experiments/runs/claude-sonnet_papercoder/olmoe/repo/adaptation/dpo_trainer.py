```python
## adaptation/dpo_trainer.py
"""Direct Preference Optimization (DPO) and KTO trainer for OLMoE-1B-7B.

Implements preference tuning as described in Section 4.3 and Appendix B/F of
the OLMoE paper. Builds on the SFT checkpoint and trains with a frozen reference
model using the DPO objective (Rafailov et al. 2023).

Key design decisions from the paper:
  1. NO auxiliary losses during DPO (Section 4.3, Table 7):
     - use_lb_loss=False improves avg score from 57.1 to 57.7
     - Routing patterns are already established during pretraining (Section 5.1)

  2. DPO β = 0.1 (Appendix B):
     - Controls deviation from reference model
     - Standard DPO loss: -log(sigmoid(β * ((chosen_logps - ref_chosen_logps)
       - (rejected_logps - ref_rejected_logps))))

  3. Constant LR = 5e-7 for 3 epochs (Appendix B):
     - No warmup, no decay — model is already well-initialized from SFT
     - Post-SFT checkpoint is the starting point

  4. KTO variant uses RMSProp optimizer (Appendix F, Table 14):
     - 5,000 steps (1.3 epochs), same LR = 5e-7
     - Matches DPO on average (57.7) but lower on AlpacaEval (81.6 vs 84.0)

  5. Sum (not mean) of log-probs over response tokens:
     - DPO theory uses sequence-level log-probabilities
     - Mean would introduce length bias

Configuration values used (from config.yaml):
  dpo.learning_rate: 5.0e-07
  dpo.lr_schedule: "constant"
  dpo.adam_beta1: 0.9
  dpo.adam_beta2: 0.95
  dpo.adam_eps: 1.0e-08
  dpo.weight_decay: 0.1
  dpo.num_epochs: 3
  dpo.global_batch_size: 32
  dpo.per_device_batch_size: 1
  dpo.gradient_accumulation_steps: 1
  dpo.dpo_beta: 0.1
  dpo.use_lb_loss: false
  dpo.use_router_z_loss: false
  kto.optimizer: "rmsprop"
  kto.learning_rate: 5.0e-07
  kto.num_steps: 5000
  kto.global_batch_size: 32
  kto.per_device_batch_size: 1
  kto.use_lb_loss: false
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer, RMSprop
from torch.utils.data import DataLoader

from config import DPOConfig, KTOConfig, OLMoEConfig, TrainingConfig
from model.olmoe_model import OLMoEModel, OLMoEOutput
from utils.checkpoint import CheckpointManager
from utils.distributed import DistributedUtils
from utils.logging_utils import WandbLogger, get_logger

logger: logging.Logger = get_logger("olmoe.dpo")

# ---------------------------------------------------------------------------
# Optional FSDP import for distributed training support.
# ---------------------------------------------------------------------------
try:
    from torch.distributed.fsdp import FullyShardedDataParallel
    FSDP_AVAILABLE: bool = True
except ImportError:
    FSDP_AVAILABLE = False
    FullyShardedDataParallel = None  # type: ignore[assignment,misc]


class DPOTrainer:
    """Direct Preference Optimization trainer for OLMoE-1B-7B.

    Implements the DPO preference tuning stage from Section 4.3 and Appendix B
    of the paper. Trains the policy model to prefer chosen responses over
    rejected ones relative to a frozen reference model.

    DPO loss (Rafailov et al. 2023):
        L_DPO = -log(sigmoid(β * ((log π(y_w|x) - log π_ref(y_w|x))
                                  - (log π(y_l|x) - log π_ref(y_l|x)))))

    where:
        π = policy model (being trained)
        π_ref = reference model (frozen SFT checkpoint)
        y_w = chosen (preferred) response
        y_l = rejected (dispreferred) response
        β = 0.1 (temperature controlling deviation from reference)

    Key paper findings (Section 4.3, Table 7):
        - NOT using load balancing loss during DPO: avg 57.7 (vs 57.1 with LBL)
        - Post-annealing checkpoint: avg 57.7 (vs 56.3 for pre-annealing)
        - DPO preferred over KTO for AlpacaEval: 84.0% (vs 81.6% for KTO)

    Attributes:
        model: The trainable policy OLMoEModel (loaded from SFT checkpoint).
        ref_model: The frozen reference OLMoEModel (same SFT checkpoint, no gradients).
        train_loader: DataLoader yielding DPO batches with chosen/rejected pairs.
        config: DPOConfig with all DPO hyperparameters from config.yaml.
        beta: DPO temperature = 0.1 (config.yaml: dpo.dpo_beta).
        use_lb_loss: Whether to use load balancing loss (always False for DPO).
        use_router_z_loss: Whether to use router z-loss (always False for DPO).
        optimizer: AdamW optimizer with all parameters and weight_decay=0.1.
        checkpoint_manager: CheckpointManager for saving DPO checkpoints.
        wandb_logger: WandbLogger for experiment tracking (rank 0 only).
        global_step: Current optimizer step count (0-indexed).
        current_epoch: Current training epoch (0-indexed).
        total_steps: Total optimizer steps across all epochs.
        device: CUDA device for this process.
        world_size: Total number of processes in the distributed group.
        use_bf16: Whether to use BF16 autocast for the forward pass.

    Example:
        >>> dpo_config = DPOConfig()
        >>> policy_model = OLMoEModel(OLMoEConfig())
        >>> ref_model = OLMoEModel(OLMoEConfig())
        >>> # Load SFT checkpoint into both models
        >>> trainer = DPOTrainer(
        ...     model=policy_model,
        ...     ref_model=ref_model,
        ...     train_loader=dpo_dataloader,
        ...     config=dpo_config,
        ...     beta=0.1,
        ...     use_lb_loss=False,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: OLMoEModel,
        ref_model: OLMoEModel,
        train_loader: DataLoader,
        config: DPOConfig,
        beta: float = 0.1,
        use_lb_loss: bool = False,
        wandb_logger: Optional[WandbLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        """Initialize DPOTrainer.

        Args:
            model: The trainable policy OLMoEModel. Must be loaded from the
                   SFT checkpoint (config.yaml: dpo.base_model="sft_checkpoint").
                   All parameters should have requires_grad=True.
            ref_model: The frozen reference OLMoEModel. Must be loaded from the
                       same SFT checkpoint as model. All parameters will be
                       frozen (requires_grad=False) in this constructor.
                       Must be on the same device as model.
            train_loader: DataLoader yielding DPO batches. Each batch must contain:
                          - "chosen_input_ids": (batch, seq_len) token IDs
                          - "chosen_labels": (batch, seq_len) with -100 for prompt tokens
                          - "chosen_attention_mask": (batch, seq_len) attention mask
                          - "rejected_input_ids": (batch, seq_len) token IDs
                          - "rejected_labels": (batch, seq_len) with -100 for prompt tokens
                          - "rejected_attention_mask": (batch, seq_len) attention mask
                          Produced by DataCollator with mode="dpo".
            config: DPOConfig instance with all DPO hyperparameters.
                    Key fields: learning_rate=5e-7, num_epochs=3, dpo_beta=0.1,
                    use_lb_loss=False, global_batch_size=32.
            beta: DPO temperature parameter controlling deviation from reference.
                  Default: 0.1 (config.yaml: dpo.dpo_beta).
                  Higher β → policy stays closer to reference.
                  Lower β → policy can deviate more from reference.
            use_lb_loss: Whether to use load balancing loss during DPO.
                         Default: False (paper's recommended setting, Section 4.3).
                         Set to True only for the ablation experiment in Table 7.
                         WARNING: Setting to True is expected to HURT performance
                         (avg drops from 57.7 to 57.1 per Table 7).
            wandb_logger: Optional WandbLogger for experiment tracking.
                          If None, a new logger is created on rank 0.
            checkpoint_manager: Optional CheckpointManager for saving checkpoints.
                                 If None, a new manager is created using config.output_dir.

        Raises:
            ValueError: If beta <= 0.
            RuntimeError: If the model has no trainable parameters.
        """
        if beta <= 0:
            raise ValueError(
                f"beta must be > 0, got {beta}. "
                f"DPO beta controls deviation from reference model. "
                f"Paper uses beta=0.1 (config.yaml: dpo.dpo_beta)."
            )

        # Warn if load balancing loss is enabled — this is the ablation path.
        if use_lb_loss:
            logger.warning(
                "use_lb_loss=True for DPO. "
                "Per paper Section 4.3 and Table 7, this is expected to HURT performance "
                "(avg drops from 57.7 to 57.1). "
                "This setting is only for reproducing the ablation experiment in Table 7. "
                "For the paper's recommended setup, use use_lb_loss=False."
            )

        self.model: OLMoEModel = model
        """The trainable policy OLMoEModel (loaded from SFT checkpoint)."""

        self.ref_model: OLMoEModel = ref_model
        """The frozen reference OLMoEModel (same SFT checkpoint, no gradients)."""

        self.train_loader: DataLoader = train_loader
        """DataLoader yielding DPO batches with chosen/rejected pairs."""

        self.config: DPOConfig = config
        """DPOConfig with all DPO hyperparameters from config.yaml."""

        self.beta: float = beta
        """DPO temperature = 0.1 (config.yaml: dpo.dpo_beta)."""

        self.use_lb_loss: bool = use_lb_loss
        """Whether to use load balancing loss (False for paper's setup)."""

        self.use_router_z_loss: bool = False
        """Whether to use router z-loss (always False for DPO per config.yaml)."""

        # -----------------------------------------------------------------------
        # Freeze the reference model completely.
        # The reference model must not be updated during DPO training.
        # Setting requires_grad_(False) prevents gradient computation.
        # Setting eval() disables any dropout (though OLMoE has none per config).
        # -----------------------------------------------------------------------
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()
        logger.info(
            "Reference model frozen: requires_grad_(False) and eval() applied. "
            "Reference model will not be updated during DPO training."
        )

        # -----------------------------------------------------------------------
        # Device and distributed setup.
        # -----------------------------------------------------------------------
        self.device: torch.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        """CUDA device for this process."""

        self.world_size: int = DistributedUtils.get_world_size()
        """Total number of processes in the distributed group."""

        # -----------------------------------------------------------------------
        # BF16 mixed precision configuration.
        # config.yaml: dpo.bf16=true
        # BF16 does not need GradScaler (wider dynamic range than FP16).
        # -----------------------------------------------------------------------
        self.use_bf16: bool = config.bf16
        """Whether to use BF16 autocast for the forward pass."""

        # -----------------------------------------------------------------------
        # Gradient accumulation configuration.
        # config.yaml: dpo.gradient_accumulation_steps=1
        # With per_device_batch_size=1 and 32 GPUs, effective global batch = 32.
        # -----------------------------------------------------------------------
        self.gradient_accumulation_steps: int = config.gradient_accumulation_steps
        """Number of micro-steps per optimizer step = 1 (config.yaml: dpo.gradient_accumulation_steps)."""

        # -----------------------------------------------------------------------
        # Create optimizer: AdamW with ALL parameters and weight_decay=0.1.
        # Universal weight decay — no exclusion groups (Sections 4.2.3, 4.2.4).
        # config.yaml: dpo.learning_rate=5e-7, dpo.adam_beta1=0.9,
        #              dpo.adam_beta2=0.95, dpo.adam_eps=1e-8, dpo.weight_decay=0.1
        # -----------------------------------------------------------------------
        self.optimizer: Optimizer = self._create_optimizer()
        """AdamW optimizer with all policy model parameters and weight_decay=0.1."""

        # -----------------------------------------------------------------------
        # Training step counters.
        # global_step: optimizer steps (incremented after each optimizer.step())
        # current_epoch: current epoch (0-indexed)
        # -----------------------------------------------------------------------
        self.global_step: int = 0
        """Current optimizer step count (0-indexed)."""

        self.current_epoch: int = 0
        """Current training epoch (0-indexed)."""

        # -----------------------------------------------------------------------
        # Compute total optimizer steps across all epochs.
        # total_steps = num_epochs * (num_batches // gradient_accumulation_steps)
        # For DPO: 3 epochs × (60800 / 32) ≈ 5700 steps
        # -----------------------------------------------------------------------
        num_batches_per_epoch: int = len(train_loader)
        self.total_steps: int = (
            config.num_epochs
            * max(1, num_batches_per_epoch // self.gradient_accumulation_steps)
        )
        """Total optimizer steps across all epochs."""

        # -----------------------------------------------------------------------
        # Checkpoint manager for saving DPO checkpoints.
        # -----------------------------------------------------------------------
        if checkpoint_manager is not None:
            self.checkpoint_manager: CheckpointManager = checkpoint_manager
        else:
            self.checkpoint_manager = CheckpointManager(
                output_dir=config.output_dir,
                max_checkpoints=3,  # Keep last 3 DPO checkpoints
            )
        """CheckpointManager for saving DPO checkpoints."""

        # -----------------------------------------------------------------------
        # Wandb logger for experiment tracking (rank 0 only).
        # -----------------------------------------------------------------------
        if wandb_logger is not None:
            self.wandb_logger: WandbLogger = wandb_logger
        else:
            self.wandb_logger = WandbLogger(
                project=config.wandb_project,
                run_name=config.run_name,
                config_dict=config.to_dict(),
            )
        """WandbLogger for experiment tracking (rank 0 only)."""

        # -----------------------------------------------------------------------
        # Last metrics dict for checkpoint metadata.
        # -----------------------------------------------------------------------
        self._last_metrics: Dict[str, float] = {}
        """Most recent training metrics. Stored in checkpoint metadata."""

        # -----------------------------------------------------------------------
        # Log initialization summary.
        # -----------------------------------------------------------------------
        logger.info(
            f"DPOTrainer initialized: "
            f"beta={beta}, "
            f"use_lb_loss={use_lb_loss}, "
            f"learning_rate={config.learning_rate:.2e}, "
            f"num_epochs={config.num_epochs}, "
            f"gradient_accumulation_steps={self.gradient_accumulation_steps}, "
            f"total_steps={self.total_steps:,}, "
            f"global_batch_size={config.global_batch_size}, "
            f"use_bf16={self.use_bf16}, "
            f"device={self.device}, "
            f"world_size={self.world_size}"
        )

        # Verify effective global batch size matches config.
        effective_global_batch: int = (
            config.per_device_batch_size
            * self.world_size
            * self.gradient_accumulation_steps
        )
        if effective_global_batch != config.global_batch_size:
            logger.warning(
                f"Effective global batch size mismatch: "
                f"per_device_batch_size={config.per_device_batch_size} × "
                f"world_size={self.world_size} × "
                f"gradient_accumulation_steps={self.gradient_accumulation_steps} = "
                f"{effective_global_batch}, "
                f"but config.global_batch_size={config.global_batch_size}. "
                f"Expected: 1 × 32 × 1 = 32 for the paper's setup."
            )

    def _create_optimizer(self) -> Optimizer:
        """Create AdamW optimizer for DPO with universal weight decay.

        Uses the paper's DPO hyperparameters from config.yaml (dpo section):
          - lr=5e-7 (constant throughout training)
          - betas=(0.9, 0.95)
          - eps=1e-8
          - weight_decay=0.1 applied to ALL parameters

        Only the policy model's parameters are optimized. The reference model
        is frozen and excluded automatically (requires_grad=False).

        Returns:
            AdamW optimizer with all trainable policy model parameters in a
            single group with weight_decay=0.1.

        Raises:
            RuntimeError: If the policy model has no trainable parameters.
        """
        # Collect all trainable parameters from the policy model only.
        # Reference model parameters have requires_grad=False and are excluded.
        trainable_params: List[nn.Parameter] = [
            p for p in self.model.parameters() if p.requires_grad
        ]

        if len(trainable_params) == 0:
            raise RuntimeError(
                "No trainable parameters found in policy model for DPO. "
                "Ensure the policy model has parameters with requires_grad=True. "
                "The reference model should have requires_grad=False for all params."
            )

        # Create AdamW with a single parameter group containing ALL trainable params.
        # CRITICAL: Do NOT split into decay/no-decay groups.
        # The paper explicitly applies weight_decay=0.1 to all parameters
        # (Sections 4.2.3, 4.2.4).
        optimizer: AdamW = AdamW(
            params=trainable_params,
            lr=self.config.learning_rate,          # 5e-7 (config.yaml: dpo.learning_rate)
            betas=(self.config.adam_beta1, self.config.adam_beta2),  # (0.9, 0.95)
            eps=self.config.adam_eps,              # 1e-8 (config.yaml: dpo.adam_eps)
            weight_decay=self.config.weight_decay, # 0.1 (config.yaml: dpo.weight_decay)
            fused=False,  # Disable fused kernel for FSDP compatibility
        )

        # Verify all trainable parameters are included.
        total_optimizer_params: int = sum(
            len(group["params"]) for group in optimizer.param_groups
        )
        assert total_optimizer_params == len(trainable_params), (
            f"DPO optimizer parameter count mismatch: "
            f"optimizer has {total_optimizer_params} parameters but "
            f"policy model has {len(trainable_params)} trainable parameters."
        )

        # Count total parameter elements for logging.
        total_elements: int = sum(p.numel() for p in trainable_params)
        param_size_mb: float = total_elements * 4 / (1024 ** 2)  # float32 equivalent

        logger.info(
            f"DPO optimizer created: AdamW("
            f"lr={self.config.learning_rate:.2e}, "
            f"betas=({self.config.adam_beta1}, {self.config.adam_beta2}), "
            f"eps={self.config.adam_eps:.2e}, "
            f"weight_decay={self.config.weight_decay}), "
            f"num_param_tensors={len(trainable_params):,}, "
            f"total_elements={total_elements:,}, "
            f"param_size_fp32={param_size_mb:.1f}MB"
        )

        return optimizer

    def _get_log_probs(
        self,
        model: OLMoEModel,
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute sum of log-probabilities of response tokens given context.

        Implements the sequence-level log-probability computation for DPO.
        Uses the standard causal LM shift (logits[:-1] vs labels[1:]) and
        sums log-probs over response tokens only (prompt tokens are masked
        with -100 in labels).

        Why sum (not mean)?
            DPO theory uses sequence-level log-probabilities which are sums
            of token log-probs. Using mean would introduce length bias where
            shorter responses are artificially preferred. The paper follows
            standard DPO (Rafailov et al. 2023) which uses sum.

        Args:
            model: The OLMoEModel to compute log-probs with. Can be either
                   the policy model (with gradients) or the reference model
                   (without gradients — caller must use torch.no_grad()).
            input_ids: Token IDs of shape (batch_size, seq_len).
                       Contains the full sequence: prompt + response tokens.
            labels: Target token IDs of shape (batch_size, seq_len).
                    Prompt tokens are masked with -100 (set by DataCollator).
                    Response tokens contain the actual token IDs.
                    The response mask is derived as: labels != -100.
            attention_mask: Optional attention mask of shape (batch_size, seq_len).
                            If None, no mask is applied (causal masking is
                            applied automatically inside OLMoEAttention).

        Returns:
            Tensor of shape (batch_size,) containing the sum of log-probabilities
            over response tokens for each sequence in the batch.
            Values are negative floats (log-probs are <= 0).
            Sequences with no response tokens (all labels == -100) return 0.0.

        Shape example:
            input_ids: (4, 4096)
            labels: (4, 4096) with -100 for prompt positions
            -> returns: (4,) — one scalar per sequence
        """
        # -----------------------------------------------------------------------
        # Forward pass to get logits.
        # We pass labels=None to skip the model's internal CE loss computation —
        # we compute our own log-probs below.
        # -----------------------------------------------------------------------
        output: OLMoEOutput = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # Don't compute CE loss inside model
        )
        logits: Tensor = output.logits  # (batch, seq_len, vocab_size)

        # -----------------------------------------------------------------------
        # Apply causal LM shift.
        # At position t, the model predicts token t+1.
        # shift_logits[t] predicts shift_labels[t] = input_ids[t+1].
        #
        # shift_logits: (batch, seq_len-1, vocab_size)
        # shift_labels: (batch, seq_len-1)
        # -----------------------------------------------------------------------
        shift_logits: Tensor = logits[:, :-1, :].contiguous()
        shift_labels: Tensor = labels[:, 1:].contiguous()
        # shift_logits: (batch, seq_len-1, vocab_size)
        # shift_labels: (batch, seq_len-1)

        # -----------------------------------------------------------------------
        # Compute per-token log-probabilities.
        # Use float32 for numerical stability in the softmax computation.
        # BF16 has limited precision that can cause issues with log_softmax
        # over large vocabulary (50304 tokens).
        # -----------------------------------------------------------------------
        log_probs: Tensor = F.log_softmax(
            shift_logits.float(), dim=-1
        )  # (batch, seq_len-1, vocab_size)

        # -----------------------------------------------------------------------
        # Gather log-prob of the actual label token at each position.
        # For positions where shift_labels == -100 (prompt tokens), we'll
        # mask them out below, so we clamp to 0 to avoid invalid gather indices.
        # -----------------------------------------------------------------------
        # Clamp -100 labels to 0 for safe gather (will be masked out anyway).
        safe_labels: Tensor = shift_labels.clone()
        safe_labels[safe_labels == -100] = 0

        # Gather: token_logps[b, t] = log_probs[b, t, shift_labels[b, t]]
        # token_logps: (batch, seq_len-1)
        token_logps: Tensor = log_probs.gather(
            dim=2,
            index=safe_labels.unsqueeze(2),
        ).squeeze(2)
        # token_logps: (batch, seq_len-1)

        # -----------------------------------------------------------------------
        # Create response token mask.
        # response_mask[b, t] = 1.0 if shift_labels[b, t] != -100 (response token)
        #                      = 0.0 if shift_labels[b, t] == -100 (prompt token)
        # -----------------------------------------------------------------------
        response_mask: Tensor = (shift_labels != -100).float()
        # response_mask: (batch, seq_len-1)

        # -----------------------------------------------------------------------
        # Sum log-probs over response tokens only.
        # Multiply by mask to zero out prompt token contributions.
        # Sum (not mean) to avoid length bias in DPO.
        # -----------------------------------------------------------------------
        # Cast token_logps back to the input dtype for consistency.
        token_logps = token_logps.to(logits.dtype)

        # Sum over sequence dimension: (batch, seq_len-1) -> (batch,)
        sequence_logps: Tensor = (token_logps * response_mask).sum(dim=-1)
        # sequence_logps: (batch,) — sum of response token log-probs per sequence

        return sequence_logps

    def _dpo_loss(
        self,
        chosen_logps: Tensor,
        rejected_logps: Tensor,
        ref_chosen_logps: Tensor,
        ref_rejected_logps: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute the DPO loss and reward statistics.

        Implements the standard DPO objective from Rafailov et al. (2023):
            L_DPO = -E[log(sigmoid(β * ((log π(y_w|x) - log π_ref(y_w|x))
                                        - (log π(y_l|x) - log π_ref(y_l|x)))))]

        which simplifies to:
            reward_margin = (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)
            loss = -log(sigmoid(β * reward_margin)).mean()

        The reward margin measures how much more the policy prefers the chosen
        response over the rejected response, relative to the reference model.
        A positive margin means the policy correctly prefers the chosen response.

        Args:
            chosen_logps: Sum of log-probs for chosen responses under policy model.
                          Shape: (batch_size,). From _get_log_probs(policy, chosen).
            rejected_logps: Sum of log-probs for rejected responses under policy model.
                            Shape: (batch_size,). From _get_log_probs(policy, rejected).
            ref_chosen_logps: Sum of log-probs for chosen responses under reference model.
                              Shape: (batch_size,). From _get_log_probs(ref, chosen).
            ref_rejected_logps: Sum of log-probs for rejected responses under reference model.
                                Shape: (batch_size,). From _get_log_probs(ref, rejected).

        Returns:
            Tuple of four scalar tensors:
                - loss: DPO loss scalar (differentiable, for backward()).
                - chosen_reward_mean: Mean implicit reward for chosen responses.
                  = mean(chosen_logps - ref_chosen_logps). Detached, for logging.
                  Higher is better — policy prefers chosen more than reference does.
                - rejected_reward_mean: Mean implicit reward for rejected responses.
                  = mean(rejected_logps - ref_rejected_logps). Detached, for logging.
                  Lower is better — policy prefers rejected less than reference does.
                - reward_accuracy: Fraction of pairs where chosen reward > rejected reward.
                  = mean(chosen_rewards > rejected_rewards). Detached, for logging.
                  Should approach 1.0 as training progresses.

        Shape example:
            chosen_logps: (4,)
            rejected_logps: (4,)
            ref_chosen_logps: (4,)
            ref_rejected_logps: (4,)
            -> loss: scalar
            -> chosen_reward_mean: scalar
            -> rejected_reward_mean: scalar
            -> reward_accuracy: scalar
        """
        # -----------------------------------------------------------------------
        # Compute implicit rewards (log-ratio of policy to reference).
        # chosen_rewards[i] = log π(y_w_i|x_i) - log π_ref(y_w_i|x_i)
        # rejected_rewards[i] = log π(y_l_i|x_i) - log π_ref(y_l_i|x_i)
        # -----------------------------------------------------------------------
        chosen_rewards: Tensor = chosen_logps - ref_chosen_logps
        # chosen_rewards: (batch,)

        rejected_rewards: Tensor = rejected_logps - ref_rejected_logps
        # rejected_rewards: (batch,)

        # -----------------------------------------------------------------------
        # Compute reward margin: how much more the policy prefers chosen over rejected
        # relative to the reference model.
        # reward_margin[i] = chosen_rewards[i] - rejected_rewards[i]
        # -----------------------------------------------------------------------
        reward_margin: Tensor = chosen_rewards - rejected_rewards
        # reward_margin: (batch,)

        # -----------------------------------------------------------------------
        # DPO loss: -log(sigmoid(β * reward_margin)).mean()
        # F.logsigmoid is numerically stable: logsigmoid(x) = -softplus(-x)
        # -----------------------------------------------------------------------
        loss: Tensor = -F.logsigmoid(self.beta * reward_margin).mean()
        # loss: scalar

        # -----------------------------------------------------------------------
        # Compute monitoring statistics (detached — for logging only).
        # These are not used in the backward pass.