"""
trainer.py

Trainer class for Hi‑MAR – orchestrates the two‑phase masked autoregressive
training loop using diffusion losses.  Integrates with HuggingFace Accelerate
for mixed precision, distributed training, and automatic device placement.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
from accelerate import Accelerator
from torch import Tensor

# These imports are assumed to be available from the project structure.
from config import TrainingConfig   # We use the full TrainingConfig for clarity
from masking import TokenMasker
from model import HiMARTransformer
from diffusion_heads import MLPDiffusionHead, DiffusionTransformerHead
from vae_tokenizer import VAETokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Exponential Moving Average helper
# ---------------------------------------------------------------------------

class EMA:
    """
    Exponential moving average for model parameters.

    Maintains a shadow copy of the parameters of the provided modules,
    updated with decay ``momentum`` at each call to ``update``.
    """

    def __init__(self, modules: list[torch.nn.Module], momentum: float = 0.9999) -> None:
        self.momentum = momentum
        self.shadow = {}
        # Collect all trainable parameters from the modules
        for mod in modules:
            for name, param in mod.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone().detach()

    def update(self, modules: list[torch.nn.Module]) -> None:
        """Update the shadow parameters with the current parameter values."""
        with torch.no_grad():
            for mod in modules:
                for name, param in mod.named_parameters():
                    if param.requires_grad and name in self.shadow:
                        shadow_p = self.shadow[name]
                        shadow_p.mul_(self.momentum).add_(param.data, alpha=1.0 - self.momentum)

    def apply(self, modules: list[torch.nn.Module]) -> None:
        """Copy shadow parameters back into the live modules (e.g., for evaluation)."""
        with torch.no_grad():
            for mod in modules:
                for name, param in mod.named_parameters():
                    if param.requires_grad and name in self.shadow:
                        param.data.copy_(self.shadow[name])


# ---------------------------------------------------------------------------
#  Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Hi‑MAR Trainer implementing the two‑phase hierarchical masked autoregressive
    training protocol described in Section 3 and 4 of the paper.

    Args:
        model:         The scale‑aware Hi‑MAR Transformer backbone.
        head1:         MLP‑based diffusion head for low‑resolution (Phase 1).
        head2:         Diffusion Transformer head for high‑resolution (Phase 2).
        vae:           Frozen KL‑16 VAE tokenizer (continuous latents, dim 16).
        dataloader:    Training DataLoader.
        train_config:  Training hyper‑parameter configuration (subset of Config.training.imagenet
                       or Config.training.coco).
        token_masker:  Masking utility configured for the dataset.
        accelerator:   HuggingFace Accelerator instance.
        dataset_type:  Either ``"imagenet"`` or ``"coco"``.
        epochs:        Number of training epochs.  Overrides any implicit epoch count from config.
        ema_momentum:  EMA momentum (default 0.9999).  Pass ``None`` to disable EMA.
        gradient_clip: Max gradient norm (default None, i.e., disabled).
    """

    def __init__(
        self,
        model: HiMARTransformer,
        head1: MLPDiffusionHead,
        head2: DiffusionTransformerHead,
        vae: VAETokenizer,
        dataloader: torch.utils.data.DataLoader,
        train_config: Union[TrainingConfig, Dict[str, Any]],   # can be dict for flexibility
        token_masker: TokenMasker,
        accelerator: Accelerator,
        dataset_type: str,
        epochs: int,
        ema_momentum: Optional[float] = 0.9999,
        gradient_clip: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Basic fields
        self.model = model
        self.head1 = head1
        self.head2 = head2
        self.vae = vae
        self.train_loader = dataloader
        self.token_masker = token_masker
        self.accelerator = accelerator
        self.dataset_type = dataset_type
        self.epochs = epochs
        self.gradient_clip = gradient_clip

        # Extract hyper‑parameters from config (supports dataclass or dict)
        if isinstance(train_config, dict):
            self.lr = train_config["learning_rate"]
            self.weight_decay = train_config["weight_decay"]
            self.beta1 = train_config["beta1"]
            self.beta2 = train_config["beta2"]
            self.warmup_steps = train_config.get("warmup_steps", 0)
            self.warmup_epochs = train_config.get("warmup_epochs", 0)
            self.total_steps_config = train_config.get("total_steps", None)
            self.mixed_precision = train_config.get("mixed_precision", "bf16")
        else:
            self.lr = train_config.learning_rate
            self.weight_decay = train_config.weight_decay
            self.beta1 = train_config.beta1
            self.beta2 = train_config.beta2
            self.warmup_steps = getattr(train_config, "warmup_steps", 0)
            self.warmup_epochs = getattr(train_config, "warmup_epochs", 0)
            self.total_steps_config = getattr(train_config, "total_steps", None)
            self.mixed_precision = getattr(train_config, "mixed_precision", "bf16")

        # ------------------------------------------------------------------
        # Optimizer – include all trainable parameters from model + heads
        # ------------------------------------------------------------------
        params = list(self.model.parameters()) + list(self.head1.parameters()) + list(self.head2.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )

        # ------------------------------------------------------------------
        # Learning‑rate scheduler (linear warmup + constant)
        # ------------------------------------------------------------------
        total_steps = self._compute_total_steps()
        if self.dataset_type == "imagenet" and self.warmup_epochs > 0:
            warmup_steps = self.warmup_epochs * len(self.train_loader)
        else:
            warmup_steps = self.warmup_steps  # for COCO, it's a step count
        warmup_steps = max(1, warmup_steps)  # avoid zero

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # ------------------------------------------------------------------
        # EMA (optional)
        # ------------------------------------------------------------------
        self.ema: Optional[EMA] = None
        if ema_momentum is not None and ema_momentum > 0:
            self.ema = EMA([self.model, self.head1, self.head2], momentum=ema_momentum)

        # ------------------------------------------------------------------
        # Prepare everything with Accelerator (mixed precision, device, etc.)
        # ------------------------------------------------------------------
        (self.model, self.head1, self.head2, self.optimizer, self.train_loader) = \
            self.accelerator.prepare(self.model, self.head1, self.head2, self.optimizer, self.train_loader)

        # Total steps for logging / schedule
        self.total_steps = total_steps
        self.current_step = 0

        # Diffuson timestep count – reuse schedule from heads
        self.num_timesteps = self.head1.betas.shape[0]

        logger.info(
            "Trainer initialized. Total steps: %d, Warmup steps: %d, Epochs: %d, "
            "Batch size: %d, EMA: %s, Mixed precision: %s",
            self.total_steps, warmup_steps, self.epochs,
            self.train_loader.batch_size,
            "enabled" if self.ema else "disabled",
            self.mixed_precision,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_total_steps(self) -> int:
        """Return total number of training steps."""
        steps_per_epoch = len(self.train_loader)
        if self.total_steps_config is not None:
            return self.total_steps_config
        return self.epochs * steps_per_epoch

    def _vae_encode(self, images_high: Tensor, images_low: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode both resolutions to continuous latents under ``torch.no_grad()``.
        Returns (low_latent, high_latent), both of shape ``(B, N, 16)``.
        """
        with torch.no_grad():
            # images are already on device and normalised to [-1, 1]
            low_latent = self.vae.encode(images_low)
            high_latent = self.vae.encode(images_high)
        return low_latent.detach(), high_latent.detach()

    def _prepare_context(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Create context tokens from a batch.

        - ImageNet: expects ``class_id`` (LongTensor, (B,))
          → uses model’s class embedding.
        - COCO:     expects ``text_emb`` (FloatTensor, (B, L, d_text))
          → projects through model’s text_proj.

        Returns:
            context tensor of shape ``(B, L_ctx, hidden_size)``.
        """
        if self.dataset_type == "imagenet":
            class_ids = batch["class_id"]
            # Handle possible distributed gather: accelerator may automatically
            # broadcast; we just use the tensor as is.
            if self.model.class_embedding is None:
                raise RuntimeError("Model has no class_embedding for ImageNet.")
            ctx = self.model.class_embedding(class_ids)              # (B, hidden_size)
            return ctx.unsqueeze(1)                                   # (B, 1, hidden)
        else:
            text_emb = batch["text_emb"]                              # (B, L, d_text)
            if self.model.text_proj is None:
                raise RuntimeError("Model has no text_proj for COCO.")
            return self.model.text_proj(text_emb)                     # (B, L, hidden)

    def _sample_phase2_ratio(self, step: int) -> float:
        """
        Determine the masking ratio for high‑resolution tokens during training.

        - ImageNet: cosine schedule decreasing from 1 to 0 over total steps
          (MaskGIT‑style).
        - COCO:     random Beta(α=4, β=1) for each batch.
        """
        if self.dataset_type == "imagenet":
            # Cosine schedule: r = cos( (step / total_steps) * (π/2) )
            # Note: we want ratio ~ 1 at start, ~ 0 at end.
            progress = min(step / max(1, self.total_steps), 1.0)
            ratio = math.cos(progress * math.pi / 2.0)
            return float(ratio)
        else:
            # COCO uses random Beta(4,1)
            return self.token_masker.sample_mask_ratio(phase=2, dataset_type="coco")

    @staticmethod
    def _q_sample(
        x_start: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
        sqrt_alphas_cumprod: Tensor = None,
        sqrt_one_minus_alphas_cumprod: Tensor = None,
    ) -> Tensor:
        """
        Forward diffusion process: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε.

        Args:
            x_start: Clean latent, shape (B, N, D).
            t: Integer timesteps, shape (B,); each in [0, T-1].
            noise: Optional noise tensor of same shape as x_start.
            sqrt_alphas_cumprod: pre‑computed schedule buffer (from head).
            sqrt_one_minus_alphas_cumprod: pre‑computed schedule buffer.
        Returns:
            Noisy latent, shape (B, N, D).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        # Gather the schedule values for each timestep
        sqrt_alpha_bar = sqrt_alphas_cumprod[t]               # (B,)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod[t]  # (B,)
        # Reshape for broadcasting over N and D
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    # ------------------------------------------------------------------
    # Per‑step training logic
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Execute a single training step (forward + loss + backward).

        Returns a dictionary with loss values for logging.
        """
        # 1. Encode images to latents
        low_latent, high_latent = self._vae_encode(batch["high_res"], batch["low_res"])
        B = low_latent.shape[0]
        device = low_latent.device

        # 2. Context tokens
        ctx = self._prepare_context(batch)                      # (B, L_ctx, hidden)

        # 3. Sample masking ratios
        r_s = self.token_masker.sample_mask_ratio(phase=1)      # low‑res
        r_l = self._sample_phase2_ratio(self.current_step)      # high‑res

        # 4. Apply masks (uses the model’s learnable mask token)
        mask_token = self.model.get_mask_token()                # (1, 1, latent_dim)
        masked_low, mask_low = self.token_masker.apply_masks(low_latent, r_s, mask_token)
        # mask_low: (B, N_s) boolean
        masked_high, mask_high = self.token_masker.apply_masks(high_latent, r_l, mask_token)

        # 5. Embed latent tokens → hidden_size (same linear projection for both)
        #    We embed even before phase concatenation for simplicity.
        emb_low = self.model.input_proj(masked_low)             # (B, N_s, H)
        emb_high = self.model.input_proj(masked_high)           # (B, N_l, H)

        # 6. Forward Phase 1 – low‑res transformer
        #    Input: context + embedded low‑res tokens
        Z_s = self.model(ctx, emb_low, scale_id=0)              # (B, L_ctx+N_s, H)
        # Extract image‑part conditional tokens
        ctx_len = ctx.shape[1]
        Z_s_img = Z_s[:, ctx_len:, :]                           # (B, N_s, H)

        # 7. Forward Phase 2 – high‑res transformer
        #    Input: context + pivots (Z_s_img) + embedded high‑res tokens
        image_tokens_phase2 = torch.cat([Z_s_img, emb_high], dim=1)  # (B, N_s+N_l, H)
        Z_l = self.model(ctx, image_tokens_phase2, scale_id=1)       # (B, L_ctx+N_s+N_l, H)
        # Extract high‑res conditional tokens
        Z_l_img = Z_l[:, ctx_len + Z_s_img.shape[1]:, :]             # (B, N_l, H)

        # 8. Diffusion timesteps (one per image)
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # 9. Phase 1 loss (MLP‑based head)
        noise_low = torch.randn_like(low_latent)
        x_t_low = self._q_sample(
            low_latent, t, noise_low,
            sqrt_alphas_cumprod=self.head1.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.head1.sqrt_one_minus_alphas_cumprod,
        )
        noise_pred_low = self.head1(z_cond=Z_s_img, x_noisy=x_t_low, t=t.float()/1000.0)  # head expects normalized t
        loss1 = F.mse_loss(noise_pred_low[mask_low], noise_low[mask_low])

        # 10. Phase 2 loss (Diffusion Transformer head)
        noise_high = torch.randn_like(high_latent)
        x_t_high = self._q_sample(
            high_latent, t, noise_high,
            sqrt_alphas_cumprod=self.head2.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.head2.sqrt_one_minus_alphas_cumprod,
        )
        noise_pred_high = self.head2(z_cond_all=Z_l_img, x_noisy_all=x_t_high, t=t.float()/1000.0)
        loss2 = F.mse_loss(noise_pred_high[mask_high], noise_high[mask_high])

        total_loss = loss1 + loss2

        # 11. Backward & optimizer step (accelerator handles scaling)
        self.accelerator.backward(total_loss)
        if self.gradient_clip is not None:
            self.accelerator.clip_grad_norm_(
                list(self.model.parameters()) + list(self.head1.parameters()) + list(self.head2.parameters()),
                self.gradient_clip,
            )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # 12. EMA update (after optimizer step)
        if self.ema is not None:
            # Accelerator may unwrap models; we need the unwrapped versions for EMA
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_head1 = self.accelerator.unwrap_model(self.head1)
            unwrapped_head2 = self.accelerator.unwrap_model(self.head2)
            self.ema.update([unwrapped_model, unwrapped_head1, unwrapped_head2])

        self.current_step += 1

        return {
            "loss": total_loss.item(),
            "loss1": loss1.item(),
            "loss2": loss2.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "ratio_s": r_s,
            "ratio_l": r_l,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training procedure."""
        logger.info("Starting training for %d epochs.", self.epochs)
        self.model.train()
        self.head1.train()
        self.head2.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_loss1 = 0.0
            epoch_loss2 = 0.0
            num_batches = 0

            progress_bar = self.accelerator.gather is None  # only master shows progress
            if progress_bar and self.accelerator.is_main_process:
                from tqdm import tqdm
                bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            else:
                bar = self.train_loader

            for batch in bar:
                with self.accelerator.autocast():
                    metrics = self.training_step(batch)

                # Accumulate
                epoch_loss += metrics["loss"]
                epoch_loss1 += metrics["loss1"]
                epoch_loss2 += metrics["loss2"]
                num_batches += 1

                if progress_bar:
                    bar.set_postfix(
                        loss=metrics["loss"],
                        l1=metrics["loss1"],
                        l2=metrics["loss2"],
                        lr=metrics["lr"],
                        r_s=metrics["ratio_s"],
                        r_l=metrics["ratio_l"],
                    )

            # End of epoch
            avg_loss = epoch_loss / max(1, num_batches)
            avg_loss1 = epoch_loss1 / max(1, num_batches)
            avg_loss2 = epoch_loss2 / max(1, num_batches)
            logger.info(
                "Epoch %d/%d finished. Avg loss: %.4f (L1: %.4f, L2: %.4f), LR: %e",
                epoch + 1, self.epochs, avg_loss, avg_loss1, avg_loss2,
                self.optimizer.param_groups[0]["lr"],
            )

        # Save final EMA parameters if needed (handled by main script)
        logger.info("Training completed.")

