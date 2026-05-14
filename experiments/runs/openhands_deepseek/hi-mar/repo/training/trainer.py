import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import copy


def cosine_scheduler(base_lr: float, total_steps: int, warmup_steps: int):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


class HiMARTrainer:
    """Trainer for Hi-MAR model."""

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['lr'],
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.95)),
            weight_decay=config.get('weight_decay', 0.02),
        )

        # EMA
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_decay = config.get('ema_decay', 0.9999)

        # LR scheduler
        self.lr_lambda = cosine_scheduler(
            base_lr=config['lr'],
            total_steps=config['total_steps'],
            warmup_steps=config.get('warmup_steps', 0),
        )

        self.global_step = 0

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        phase1_mask_fn,
        phase2_mask_fn,
    ) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_p1_loss = 0.0
        total_p2_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            self.global_step += 1

            # Update LR
            lr_scale = self.lr_lambda(self.global_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config['lr'] * lr_scale

            # Move data to device
            high_res = batch['high_res_tokens'].to(self.device)
            low_res = batch['low_res_tokens'].to(self.device)
            class_ids = batch.get('class_ids')
            if class_ids is not None:
                class_ids = class_ids.to(self.device)

            B = high_res.shape[0]
            N_low = low_res.shape[1]
            N_high = high_res.shape[1]

            # Phase 1 mask
            p1_mask = phase1_mask_fn(B, N_low, self.device)
            # Phase 2 mask
            p2_mask = phase2_mask_fn(B, N_high, self.device)

            # Forward
            output = self.model(low_res, high_res, p1_mask, p2_mask, class_ids)

            loss = output['loss']

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
            self.optimizer.step()

            # EMA update
            self._update_ema()

            total_loss += loss.item()
            total_p1_loss += output['phase1_loss'].item()
            total_p2_loss += output['phase2_loss'].item()

        num_batches = len(dataloader) if len(dataloader) > 0 else 1
        return {
            'loss': total_loss / num_batches,
            'phase1_loss': total_p1_loss / num_batches,
            'phase2_loss': total_p2_loss / num_batches,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

    def _update_ema(self):
        """Update EMA model parameters."""
        for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.lerp_(model_p.data, 1 - self.ema_decay)


class MaskSampler:
    """Sampling masking ratios for training."""

    @staticmethod
    def uniform_mask_ratio(B: int, N: int, device: torch.device, min_ratio: float = 0.7, max_ratio: float = 1.0) -> torch.Tensor:
        """Phase 1 masking: uniform random ratio in [min_ratio, max_ratio]."""
        ratios = torch.rand(B, device=device) * (max_ratio - min_ratio) + min_ratio
        num_mask = (ratios * N).ceil().long().clamp(0, N)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(N, device=device)[:num_mask[b].item()]
            mask[b, idx] = True
        return mask

    @staticmethod
    def cosine_mask_ratio(B: int, N: int, device: torch.device) -> torch.Tensor:
        """Phase 2 masking: cosine schedule (MaskGIT-style)."""
        r = torch.rand(1, device=device)  # random mask ratio
        threshold = torch.cos(math.pi / 2 * r)
        rand = torch.rand(B, N, device=device)
        mask = rand > threshold
        return mask

    @staticmethod
    def beta_mask_ratio(B: int, N: int, device: torch.device, alpha: float = 4.0, beta: float = 1.0) -> torch.Tensor:
        """Phase 2 masking: Beta distribution (used for MS-COCO text-to-image)."""
        dist = torch.distributions.Beta(alpha, beta)
        ratios = dist.sample((B,)).to(device)
        num_mask = (ratios * N).ceil().long().clamp(0, N)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(N, device=device)[:num_mask[b].item()]
            mask[b, idx] = True
        return mask
