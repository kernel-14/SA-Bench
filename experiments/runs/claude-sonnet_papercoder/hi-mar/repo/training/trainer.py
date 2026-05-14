```python
## training/trainer.py
"""Training orchestrator for Hi-MAR hierarchical masked autoregressive models.

This module implements the ``Trainer`` class that manages the full training loop
for both ImageNet class-conditional generation and MS-COCO text-to-image
generation tasks. It handles mixed precision training, EMA updates, distributed
training via DDP, gradient clipping, checkpointing, and TensorBoard logging.

Configuration alignment (config.yaml):
    training_imagenet.optimizer.beta1/2      = 0.9, 0.95
    training_imagenet.lr_schedule.base_lr    = 1e-4
    training_imagenet.lr_schedule.warmup_epochs = 100
    training_imagenet.optimizer.weight_decay = 0.02
    training_imagenet.epochs                 = 800
    training_imagenet.grad_clip_norm         = 1.0
    training_imagenet.ema.enabled            = false
    training_coco.optimizer.beta1/2          = 0.9, 0.999
    training_coco.lr_schedule.base_lr        = 8e-4
    training_coco.lr_schedule.warmup_steps   = 8000
    training_coco.optimizer.weight_decay     = 0.03
    training_coco.ema.enabled                = true
    training_coco.ema.momentum               = 0.9999
    training_coco.grad_clip_norm             = 1.0
    mixed_precision                          = true
    output.log_every_steps                   = 100
    output.save_every_epochs                 = 50
    seed                                     = 42

Paper reference: Section 4.2 (Training Setup).
"""

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.himar import HiMAR, HiMARConfig
from models.vae_tokenizer import VAETokenizer
from utils.misc import AverageMeter, setup_logger


# ---------------------------------------------------------------------------
# Trainer Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Flat configuration consumed by Trainer.

    All fields have defaults matching the Hi-MAR-Base ImageNet configuration.
    Values are sourced from the active training section of config.yaml.

    Attributes:
        dataset: Task identifier. 'imagenet' for class-conditional generation,
            'coco' for text-to-image generation.
            Config: training_imagenet.dataset / training_coco.dataset.
        lr: Base learning rate after warmup.
            Config: training_imagenet.lr_schedule.base_lr = 1e-4
                    training_coco.lr_schedule.base_lr = 8e-4.
        weight_decay: AdamW weight decay.
            Config: training_imagenet.optimizer.weight_decay = 0.02
                    training_coco.optimizer.weight_decay = 0.03.
        beta1: AdamW beta1 parameter.
            Config: training_imagenet.optimizer.beta1 = 0.9
                    training_coco.optimizer.beta1 = 0.9.
        beta2: AdamW beta2 parameter.
            Config: training_imagenet.optimizer.beta2 = 0.95 (ImageNet)
                    training_coco.optimizer.beta2 = 0.999 (COCO).
        warmup_epochs: Number of linear warmup epochs (ImageNet only).
            Config: training_imagenet.lr_schedule.warmup_epochs = 100.
        warmup_steps: Number of linear warmup steps (COCO only).
            Config: training_coco.lr_schedule.warmup_steps = 8000.
        epochs: Total training epochs. None for COCO (not specified in paper).
            Config: training_imagenet.epochs = 800.
        grad_clip_norm: Maximum gradient norm for clipping.
            Config: training_imagenet.grad_clip_norm = 1.0
                    training_coco.grad_clip_norm = 1.0.
        ema_enabled: Whether to maintain an EMA copy of the model.
            Config: training_imagenet.ema.enabled = false
                    training_coco.ema.enabled = true.
        ema_momentum: EMA decay factor.
            Config: training_coco.ema.momentum = 0.9999.
        mixed_precision: Whether to use torch.cuda.amp for mixed precision.
            Config: mixed_precision = true.
        log_every_steps: TensorBoard logging frequency in optimizer steps.
            Config: output.log_every_steps = 100.
        save_every_epochs: Checkpoint saving frequency in epochs.
            Config: output.save_every_epochs = 50.
        n_classes: Number of ImageNet classes for null class CFG index.
            Config: training_imagenet.n_classes = 1000.
        hidden_size: Transformer hidden size for context construction.
            Config: models.himar_b.transformer.hidden_size = 768.
        cfg_dropout: Probability of replacing conditioning with null context
            during training (classifier-free guidance dropout).
            Not specified in paper; standard practice is 0.1.
        accumulation_steps: Gradient accumulation steps. 1 = no accumulation.
            Not specified in paper; default 1 for standard training.
        clip_dim: CLIP text embedding dimension for COCO context.
            Derived from openai/clip-vit-large-patch14 (768-dim).
    """

    dataset: str = "imagenet"
    lr: float = 1e-4
    weight_decay: float = 0.02
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_epochs: int = 100
    warmup_steps: int = 8000
    epochs: Optional[int] = 800
    grad_clip_norm: float = 1.0
    ema_enabled: bool = False
    ema_momentum: float = 0.9999
    mixed_precision: bool = True
    log_every_steps: int = 100
    save_every_epochs: int = 50
    n_classes: int = 1000
    hidden_size: int = 768
    cfg_dropout: float = 0.1
    accumulation_steps: int = 1
    clip_dim: int = 768


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Training orchestrator for Hi-MAR two-phase hierarchical generation.

    Manages the full training loop for both ImageNet class-conditional and
    MS-COCO text-to-image generation tasks. Handles:
    - Mixed precision training via torch.cuda.amp
    - Exponential moving average (EMA) of model parameters
    - Distributed training via DistributedDataParallel (DDP)
    - Gradient clipping to norm 1.0
    - Linear warmup LR scheduling (constant post-warmup)
    - TensorBoard logging of losses and learning rate
    - Checkpoint saving and loading with full state restoration

    The Trainer is task-agnostic: it detects the task from ``config.dataset``
    and builds context tensors accordingly (class embeddings for ImageNet,
    projected CLIP embeddings for COCO).

    Attributes:
        config: TrainerConfig with all training hyperparameters.
        model: HiMAR model (may be DDP-wrapped after construction).
        vae: Frozen VAETokenizer for dual-resolution encoding.
        device: Target compute device.
        output_dir: Root directory for checkpoints and logs.
        optimizer: AdamW optimizer with parameter group weight decay separation.
        scheduler: LambdaLR scheduler with linear warmup + constant LR.
        scaler: GradScaler for mixed precision training.
        ema_model: EMA copy of the model (frozen parameters).
        writer: TensorBoard SummaryWriter.
        global_step: Total optimizer steps across all epochs.
        logger: Python logger for console and file output.
        task: Task identifier ('imagenet' or 'coco').
        _is_distributed: Whether DDP is active.
        _null_text_embed: Pre-computed null CLIP embedding for CFG (COCO only).
    """

    def __init__(
        self,
        config: TrainerConfig,
        model: HiMAR,
        vae: VAETokenizer,
        device: torch.device,
        output_dir: str = "outputs",
        steps_per_epoch: int = 0,
    ) -> None:
        """Initialises the Trainer.

        Args:
            config: TrainerConfig with all training hyperparameters. All fields
                have defaults matching Hi-MAR-Base ImageNet configuration.
            model: Instantiated HiMAR model. Will be wrapped in DDP if
                distributed training is detected.
            vae: Frozen VAETokenizer. Never trained; used only for encoding.
            device: Target compute device. Should match the device on which
                ``model`` and ``vae`` reside.
            output_dir: Root directory for checkpoints and TensorBoard logs.
                Config: output.dir = "outputs". Default: "outputs".
            steps_per_epoch: Number of optimizer steps per epoch. Required for
                ImageNet warmup schedule computation (warmup_steps =
                warmup_epochs * steps_per_epoch). Pass 0 if unknown at
                construction time; the scheduler will use warmup_steps from
                config directly in that case.
        """
        self.config: TrainerConfig = config
        self.device: torch.device = device
        self.output_dir: str = output_dir
        self.global_step: int = 0
        self.task: str = config.dataset  # 'imagenet' or 'coco'

        # ------------------------------------------------------------------
        # Setup logger.
        # ------------------------------------------------------------------
        log_dir: str = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file: str = os.path.join(log_dir, f"trainer_{self.task}.log")
        self.logger: logging.Logger = setup_logger("trainer", log_file)

        # ------------------------------------------------------------------
        # Detect distributed training.
        # ------------------------------------------------------------------
        self._is_distributed: bool = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        self._local_rank: int = 0
        if self._is_distributed:
            self._local_rank = torch.distributed.get_rank()
            self.logger.info(
                f"Distributed training active. Rank {self._local_rank} / "
                f"{torch.distributed.get_world_size()}."
            )

        # ------------------------------------------------------------------
        # Store VAE (frozen, never trained).
        # ------------------------------------------------------------------
        self.vae: VAETokenizer = vae

        # ------------------------------------------------------------------
        # Wrap model in DDP if distributed.
        # The EMA model is never wrapped in DDP — it runs on rank 0 only.
        # ------------------------------------------------------------------
        self.model: Union[HiMAR, nn.parallel.DistributedDataParallel] = model
        if self._is_distributed and torch.cuda.device_count() > 1:
            self.model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
                find_unused_parameters=False,
            )
            self.logger.info(
                f"Model wrapped in DistributedDataParallel on device "
                f"{self._local_rank}."
            )

        # ------------------------------------------------------------------
        # EMA model: deep copy of the unwrapped model with frozen parameters.
        # Config: training_coco.ema.enabled = true, momentum = 0.9999.
        #         training_imagenet.ema.enabled = false.
        # ------------------------------------------------------------------
        unwrapped: HiMAR = self._unwrap_model()
        self.ema_model: HiMAR = copy.deepcopy(unwrapped)
        self.ema_model.to(device)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self.ema_model.eval()

        # ------------------------------------------------------------------
        # Build optimizer with parameter group weight decay separation.
        # ------------------------------------------------------------------
        self.optimizer: AdamW = self._build_optimizer()

        # ------------------------------------------------------------------
        # Compute warmup steps for the LR scheduler.
        # ImageNet: warmup_steps = warmup_epochs * steps_per_epoch.
        # COCO: warmup_steps = 8000 (from config, step-based).
        # ------------------------------------------------------------------
        if self.task == "imagenet" and steps_per_epoch > 0:
            warmup_steps: int = config.warmup_epochs * steps_per_epoch
        elif self.task == "imagenet" and steps_per_epoch == 0:
            # Fallback: use warmup_steps directly if steps_per_epoch unknown.
            # This is a best-effort approximation; callers should provide
            # steps_per_epoch for accurate ImageNet warmup.
            warmup_steps = config.warmup_steps
            self.logger.warning(
                "steps_per_epoch=0 for ImageNet task. Using config.warmup_steps "
                f"({warmup_steps}) as warmup duration. Pass steps_per_epoch to "
                "Trainer.__init__ for accurate epoch-based warmup."
            )
        else:
            # COCO: 8K-step linear warmup from config.
            # Config: training_coco.lr_schedule.warmup_steps = 8000.
            warmup_steps = config.warmup_steps

        self.scheduler: LRScheduler = self._build_scheduler(warmup_steps)

        # ------------------------------------------------------------------
        # Mixed precision GradScaler.
        # Config: mixed_precision = true.
        # ------------------------------------------------------------------
        self.scaler: torch.cuda.amp.GradScaler = torch.cuda.amp.GradScaler(
            enabled=config.mixed_precision
        )

        # ------------------------------------------------------------------
        # TensorBoard SummaryWriter.
        # Config: output.log_dir = "outputs/logs".
        # ------------------------------------------------------------------
        tb_log_dir: str = os.path.join(output_dir, "logs", "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer: SummaryWriter = SummaryWriter(log_dir=tb_log_dir)

        # ------------------------------------------------------------------
        # Pre-compute null text embedding for COCO CFG dropout.
        # The null embedding is the CLIP encoding of an empty string ''.
        # Shape: [1, 77, clip_dim] on CPU; moved to device in train_step.
        # For ImageNet, this is unused (class-based CFG uses null class index).
        # ------------------------------------------------------------------
        self._null_text_embed: Optional[torch.Tensor] = None
        # Callers should set this via set_null_text_embed() for COCO training.

        self.logger.info(
            f"Trainer initialised. Task: {self.task}, "
            f"LR: {config.lr}, WD: {config.weight_decay}, "
            f"Warmup steps: {warmup_steps}, "
            f"EMA: {config.ema_enabled} (momentum={config.ema_momentum}), "
            f"Mixed precision: {config.mixed_precision}."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unwrap_model(self) -> HiMAR:
        """Returns the underlying HiMAR model, unwrapping DDP if necessary.

        In distributed training, ``self.model`` is a
        ``DistributedDataParallel`` wrapper. Accessing sub-modules like
        ``model.transformer`` requires unwrapping via ``model.module``.

        Returns:
            The underlying ``HiMAR`` instance.
        """
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module  # type: ignore[return-value]
        return self.model  # type: ignore[return-value]

    def _build_optimizer(self) -> AdamW:
        """Constructs AdamW with parameter group weight decay separation.

        Separates parameters into two groups:
        1. Parameters with weight decay (all non-bias, non-norm, non-embedding
           parameters): weight_decay = config.weight_decay.
        2. Parameters without weight decay (biases, LayerNorm weights,
           positional embeddings, mask token, scale embeddings):
           weight_decay = 0.0.

        This is standard practice for Transformer training and matches MAR's
        convention. The VAE is frozen (requires_grad=False) and is excluded
        from the optimizer automatically.

        Paper reference (Section 4.2):
            ImageNet: "AdamW optimizer (β₁=0.9, β₂=0.95) with 0.02 weight decay"
            COCO: "AdamW optimizer with an 8e-4 learning rate, 0.03 weight decay"

        Config alignment:
            training_imagenet.optimizer.beta1 = 0.9
            training_imagenet.optimizer.beta2 = 0.95
            training_imagenet.optimizer.weight_decay = 0.02
            training_coco.optimizer.beta1 = 0.9
            training_coco.optimizer.beta2 = 0.999
            training_coco.optimizer.weight_decay = 0.03

        Returns:
            Configured AdamW optimizer instance.
        """
        # Parameter names that should NOT receive weight decay.
        # Includes biases, LayerNorm/norm parameters, positional embeddings,
        # mask tokens, and scale embeddings.
        no_decay_names: Tuple[str, ...] = (
            "bias",
            "norm",          # LayerNorm weights (e.g., norm.weight, norm1.weight)
            "pos_embed",     # Learnable positional embeddings
            "mask_token",    # Learnable mask token embedding
            "scale_embed",   # Scale ID embedding
            "class_embed",   # Class conditioning embedding
        )

        unwrapped: HiMAR = self._unwrap_model()

        # Separate parameters into decay and no-decay groups.
        decay_params: List[torch.Tensor] = []
        no_decay_params: List[torch.Tensor] = []

        for name, param in unwrapped.named_parameters():
            if not param.requires_grad:
                # Skip frozen parameters (e.g., EMA model params if any leaked).
                continue
            if any(nd in name for nd in no_decay_names):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups: List[Dict[str, Any]] = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        optimizer: AdamW = AdamW(
            param_groups,
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            eps=1e-8,
        )

        self.logger.info(
            f"AdamW optimizer: lr={self.config.lr}, "
            f"betas=({self.config.beta1}, {self.config.beta2}), "
            f"weight_decay={self.config.weight_decay}. "
            f"Decay params: {sum(p.numel() for p in decay_params):,}, "
            f"No-decay params: {sum(p.numel() for p in no_decay_params):,}."
        )

        return optimizer

    def _build_scheduler(self, warmup_steps: int) -> LRScheduler:
        """Builds a LambdaLR scheduler with linear warmup and constant LR.

        Implements the paper's training schedule:
        - Steps 0 … warmup_steps-1: LR scales linearly from 0 → base_lr.
        - Steps warmup_steps … ∞: LR stays constant at base_lr.

        This is a "constant LR after warmup" schedule, matching the paper's
        description for both ImageNet ("constant lr schedule with a 1e-4
        learning rate and 100-epoch linear warmup") and COCO ("8K-step linear
        warmup").

        The scheduler steps once per optimizer step (not per epoch), so the
        warmup duration is specified in steps, not epochs.

        Paper reference (Section 4.2):
            ImageNet: "constant lr schedule with a 1e-4 learning rate and
                       100-epoch linear warmup"
            COCO: "8K-step linear warmup"

        Args:
            warmup_steps: Number of linear warmup steps. For ImageNet:
                warmup_epochs * steps_per_epoch. For COCO: 8000.

        Returns:
            LambdaLR scheduler that steps once per optimizer step.
        """
        def lr_lambda(current_step: int) -> float:
            """Computes the LR multiplier for the given optimizer step."""
            if current_step < warmup_steps:
                # Linear ramp: 0 → 1 over warmup_steps.
                return float(current_step) / float(max(1, warmup_steps))
            # Constant LR after warmup — the paper's default for both tasks.
            return 1.0

        scheduler: LambdaLR = LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda,
        )

        self.logger.info(
            f"LR scheduler: linear warmup for {warmup_steps} steps, "
            f"then constant at {self.config.lr}."
        )

        return scheduler

    def _build_context(
        self,
        batch: Tuple,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Constructs context tensors for the Transformer backbone.

        Handles both ImageNet (class embeddings) and COCO (CLIP text
        embeddings) conditioning. Also applies CFG dropout: with probability
        ``config.cfg_dropout``, replaces the conditioning with the null
        context to train the model unconditionally.

        For ImageNet:
            - Conditioned: class_embed(class_id) → [B, 1, hidden_size]
            - Null (CFG): class_embed(n_classes) → [B, 1, hidden_size]
              where n_classes=1000 is the extra null class index.

        For COCO:
            - Conditioned: text_proj(text_emb) → [B, 77, hidden_size]
            - Null (CFG): text_proj(null_text_embed) → [B, 77, hidden_size]
              where null_text_embed is the CLIP embedding of empty string ''.

        CFG dropout is applied per-sample (not per-batch) to allow the model
        to learn both conditional and unconditional generation simultaneously.

        Args:
            batch: Tuple from the DataLoader.
                ImageNet: (img_256, img_128, class_id)
                COCO: (img_256, img_128, text_emb)

        Returns:
            Tuple of:
                - ``context``: Conditioned context tensor with CFG dropout
                  applied, shape [B, C, hidden_size]. C=1 for ImageNet,
                  C=77 for COCO.
                - ``null_context``: Null context tensor for CFG inference,
                  shape [B, C, hidden_size]. Same shape as context.
        """
        unwrapped: HiMAR = self._unwrap_model()
        batch_size: int = batch[0].shape[0]

        if self.task == "imagenet":
            # ------------------------------------------------------------------
            # ImageNet class-conditional context.
            # batch[2]: class_id tensor [B], values in {0, ..., 999}.
            # ------------------------------------------------------------------
            class_ids: torch.Tensor = batch[2].to(self.device)  # [B]

            # Null class index = n_classes (extra entry in class_embed).
            # Config: training_imagenet.n_classes = 1000.
            null_class_ids: torch.Tensor = torch.full(
                (batch_size,),
                fill_value=self.config.n_classes,
                dtype=torch.long,
                device=self.device,
            )

            # Apply CFG dropout: replace some samples' class_ids with null.
            # Standard practice: 10% unconditional training (cfg_dropout=0.1).
            if self.model.training and self.config.cfg_dropout > 0.0:
                dropout_mask: torch.Tensor = (
                    torch.rand(batch_size, device=self.device)
                    < self.config.cfg_dropout
                )  # [B], True where we apply null conditioning
                class_ids = torch.where(dropout_mask, null_class_ids, class_ids)

            # Encode class IDs to context tokens: [B] → [B, 1, hidden_size].
            context: torch.Tensor = unwrapped.transformer.encode_class_context(
                class_ids
            )  # [B, 1, hidden_size]

            # Null context for CFG inference (all null class).
            null_context: torch.Tensor = unwrapped.transformer.encode_class_context(
                null_class_ids
            )  # [B, 1, hidden_size]

        else:
            # ------------------------------------------------------------------
            # COCO text-to-image context.
            # batch[2]: text_emb tensor [B, 77, 768] (CLIP last_hidden_state).
            # ------------------------------------------------------------------
            text_emb: torch.Tensor = batch[2].to(self.device)  # [B, 77, 768]

            # Project CLIP embeddings to hidden_size: [B, 77, 768] → [B, 77, H].
            context = unwrapped.transformer.encode_text_context(
                text_emb
            )  # [B, 77, hidden_size]

            # Null context: CLIP embedding of empty string ''.
            # Use pre-computed null_text_embed if available, else zeros.
            if self._null_text_embed is not None:
                null_text: torch.Tensor = self._null_text_embed.to(self.device)
                # Expand to batch size: [1, 77, 768] → [B, 77, 768].
                null_text_expanded: torch.Tensor = null_text.expand(
                    batch_size, -1, -1
                )
                null_context = unwrapped.transformer.encode_text_context(
                    null_text_expanded
                )  # [B, 77, hidden_size]
            else:
                # Fallback: zero null context if not set.
                null_context = torch.zeros_like(context)

            # Apply CFG dropout: replace some samples' context with null.
            if self.model.training and self.config.cfg_dropout > 0.0:
                dropout_mask = (
                    torch.rand(batch_size, device=self.device)
                    < self.config.cfg_dropout
                )  # [B]
                # Expand dropout_mask to [B, 1, 1] for broadcasting over [B, 77, H].
                dropout_mask_expanded: torch.Tensor = dropout_mask.view(
                    batch_size, 1, 1
                )
                context = torch.where(
                    dropout_mask_expanded, null_context, context
                )

        return context, null_context

    def _strip_ddp_prefix(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Strips the 'module.' prefix from DDP-wrapped model state dicts.

        When a model is saved from a DDP-wrapped instance, all parameter keys
        are prefixed with 'module.' (e.g., 'module.transformer.blocks.0.norm1.weight').
        This helper strips that prefix so the state dict can be loaded into an
        unwrapped model.

        Args:
            state_dict: State dict potentially containing 'module.' prefixes.

        Returns:
            State dict with 'module.' prefixes removed from all keys.
        """
        cleaned: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                cleaned[key[len("module."):]] = value
            else:
                cleaned[key] = value
        return cleaned

    # ------------------------------------------------------------------
    # Public API: null text embedding setter
    # ------------------------------------------------------------------

    def set_null_text_embed(self, null_text_embed: torch.Tensor) -> None:
        """Sets the pre-computed null CLIP text embedding for CFG dropout.

        Must be called before training on COCO to enable CFG dropout.
        The null embedding is the CLIP encoding of an empty string ''.

        Typically called as:
            null_embed = coco_dataset.null_text_embedding  # [1, 77, 768]
            trainer.set_null_text_embed(null_embed)

        Args:
            null_text_embed: Float tensor of shape [1, 77, 768] on CPU.
                The CLIP last_hidden_state for an empty string caption.
        """
        self._null_text_embed = null_text_embed.cpu()
        self.logger.info(
            f"Null text embedding set. Shape: {null_text_embed.shape}."
        )

    # ------------------------------------------------------------------
    # Public API: training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader: DataLoader,
        n_epochs: int,
        start_epoch: int = 0,
    ) -> None:
        """Runs the full training loop for n_epochs epochs.

        Outer loop from start_epoch to n_epochs. For each epoch:
        1. Sets DistributedSampler epoch for correct shuffling (if distributed).
        2. Calls train_epoch() to run all batches.
        3. Logs epoch-level average loss to TensorBoard.
        4. Saves checkpoint every save_every_epochs epochs.

        Paper reference (Section 4.2):
            ImageNet: "train the models ... for 800 epochs"
            COCO: training duration not specified.

        Config alignment:
            training_imagenet.epochs = 800
            output.save_every_epochs = 50
            output.log_every_steps = 100

        Args:
            dataloader: Training DataLoader. For distributed training, must
                use DistributedSampler (set_epoch is called automatically).
            n_epochs: Total number of training epochs.
            start_epoch: Epoch to start from (0 for fresh training, or
                loaded from checkpoint for resumption). Default: 0.
        """
        self.logger.info(
            f"Starting training. Task: {self.task}, "
            f"Epochs: {start_epoch} → {n_epochs}, "
            f"Steps per epoch: {len(dataloader)}."
        )

        for epoch in range(start_epoch, n_epochs):
            # ------------------------------------------------------------------
            # Set DistributedSampler epoch for correct per-epoch shuffling.
            # Without this, all epochs see the same data order in DDP.
            # ------------------------------------------------------------------
            if (
                self._is_distributed
                and hasattr(dataloader, "sampler")
                and hasattr(dataloader.sampler, "set_epoch")
            ):
                dataloader.sampler.set_epoch(epoch)

            # ------------------------------------------------------------------
            # Run one epoch of training.
            # ------------------------------------------------------------------
            avg_loss: float = self.train_epoch(dataloader, epoch)

            # ------------------------------------------------------------------
            # Log epoch-level metrics to TensorBoard.
            # ------------------------------------------------------------------
            self.writer.add_scalar("loss/epoch_avg", avg_loss, epoch)
            self.logger.info(
                f"Epoch {epoch + 1}/{n_epochs} complete. "
                f"Avg loss: {avg_loss:.4f}. "
                f"Global step: {self.global_step}."
            )

            