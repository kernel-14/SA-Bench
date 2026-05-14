"""
Hi-MAR: Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots.

Main model implementation combining:
1. Scale-aware Transformer backbone (shared for both phases)
2. MLP-based diffusion head for phase 1 (low-resolution)
3. Diffusion Transformer head for phase 2 (high-resolution)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from .transformer import (
    MaskedAutoregressiveTransformer,
    ScaleAwareTransformerBlock,
    ScaleEmbedder,
    ClassEmbedder,
    get_2d_sincos_pos_embed,
)
from .diffusion_loss import DiffusionLoss, MLPDiffusionHead, DiffusionTransformerHead


def cosine_mask_schedule(t, total_steps):
    """
    Cosine masking schedule from MaskGIT.
    Returns the fraction of tokens to mask at step t.
    """
    return torch.cos(t / total_steps * math.pi / 2)


class HiMAR(nn.Module):
    """
    Hierarchical Masked Autoregressive Model (Hi-MAR).

    Architecture:
    - Phase 1: Masked autoregressive modeling over low-resolution tokens
      - Scale-aware Transformer backbone
      - MLP-based diffusion head
    - Phase 2: Masked autoregressive modeling over high-resolution tokens
      - Same scale-aware Transformer backbone (with phase 1 conditional tokens as pivots)
      - Diffusion Transformer head

    Key design choices:
    1. Uses conditional tokens (not visual tokens) from phase 1 as pivots for phase 2,
       avoiding training-inference discrepancy.
    2. Scale-aware Transformer blocks with AdaLN-Zero conditioning on scale vectors.
    3. Diffusion Transformer head in phase 2 for global context among all tokens.
    """

    def __init__(
        self,
        # Image settings
        img_size=256,
        low_res_img_size=128,
        patch_size=16,
        in_channels=16,
        # Transformer settings
        hidden_size=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        # Diffusion head settings (phase 1 - MLP)
        diff_head1_depth=8,
        diff_head1_hidden=1280,
        # Diffusion head settings (phase 2 - Transformer)
        diff_head2_depth=8,
        diff_head2_hidden=512,
        diff_head2_num_heads=8,
        # Conditioning
        num_classes=1000,
        class_dropout_prob=0.1,
        # Masking
        mask_ratio_min=0.7,
        mask_ratio_max=1.0,
        # Diffusion
        num_sampling_steps=100,
        # Other
        grad_checkpointing=False,
        label_drop_prob=0.1,
    ):
        super().__init__()

        self.img_size = img_size
        self.low_res_img_size = low_res_img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.num_sampling_steps = num_sampling_steps

        # Token counts
        self.high_res_seq_len = (img_size // patch_size) ** 2  # e.g., 256 for 256x256 with patch 16
        self.low_res_seq_len = (low_res_img_size // patch_size) ** 2  # e.g., 64 for 128x128 with patch 16

        # Shared Transformer backbone
        self.transformer = MaskedAutoregressiveTransformer(
            img_size=img_size,  # Use high-res size for positional embeddings
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            dropout=0.0,
            class_dropout_prob=class_dropout_prob,
            grad_checkpointing=grad_checkpointing,
        )

        # Phase 1: MLP-based diffusion head for low-resolution tokens
        self.diff_head1 = DiffusionLoss(
            target_channels=in_channels,
            hidden_size=diff_head1_hidden,
            depth=diff_head1_depth,
            token_dim=hidden_size,
            use_transformer_head=False,
            num_sampling_steps=num_sampling_steps,
        )

        # Phase 2: Diffusion Transformer head for high-resolution tokens
        self.diff_head2 = DiffusionLoss(
            target_channels=in_channels,
            hidden_size=diff_head2_hidden,
            depth=diff_head2_depth,
            token_dim=hidden_size,
            num_heads=diff_head2_num_heads,
            use_transformer_head=True,
            num_sampling_steps=num_sampling_steps,
        )

        # Projection for low-res conditional tokens to be used as pivots in phase 2
        # The low-res conditional tokens need to be projected to match high-res positional space
        self.pivot_proj = nn.Linear(hidden_size, hidden_size)

        # Separate positional embeddings for low-res tokens
        self.low_res_pos_embed = nn.Parameter(
            torch.zeros(1, self.low_res_seq_len, hidden_size), requires_grad=False
        )
        self._init_low_res_pos_embed()

    def _init_low_res_pos_embed(self):
        """Initialize low-resolution positional embeddings."""
        low_res_grid = int(self.low_res_seq_len ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, low_res_grid)
        self.low_res_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def sample_mask(self, batch_size, seq_len, mask_ratio_min, mask_ratio_max, device):
        """
        Sample random masks for masked autoregressive training.
        Returns binary mask where 1 = masked, 0 = unmasked.
        """
        # Sample masking ratio uniformly from [mask_ratio_min, mask_ratio_max]
        mask_ratio = torch.rand(batch_size, device=device) * (mask_ratio_max - mask_ratio_min) + mask_ratio_min

        # For each sample, randomly mask tokens
        masks = []
        for i in range(batch_size):
            num_masked = int(math.ceil(mask_ratio[i].item() * seq_len))
            perm = torch.randperm(seq_len, device=device)
            mask = torch.zeros(seq_len, device=device)
            mask[perm[:num_masked]] = 1
            masks.append(mask)

        return torch.stack(masks)  # [B, N]

    def sample_cosine_mask(self, batch_size, seq_len, step, total_steps, device):
        """
        Sample mask using cosine schedule (for phase 2 during training).
        """
        ratio = math.cos(step / total_steps * math.pi / 2)
        num_masked = int(math.ceil(ratio * seq_len))

        masks = []
        for _ in range(batch_size):
            perm = torch.randperm(seq_len, device=device)
            mask = torch.zeros(seq_len, device=device)
            mask[perm[:num_masked]] = 1
            masks.append(mask)

        return torch.stack(masks)

    def forward_phase1(self, low_res_tokens, class_labels, train=True):
        """
        Phase 1: Process low-resolution tokens.

        Args:
            low_res_tokens: [B, N_s, C] low-resolution visual tokens
            class_labels: [B] class labels
            train: whether in training mode

        Returns:
            z_s: conditional tokens [B, N_s, hidden_size]
            mask_s: binary mask [B, N_s]
            loss_s: diffusion loss for phase 1 (only during training)
        """
        B, N_s, C = low_res_tokens.shape
        device = low_res_tokens.device

        # Sample mask for phase 1 (ratio in [0.7, 1.0] as in MAR)
        mask_s = self.sample_mask(B, N_s, self.mask_ratio_min, self.mask_ratio_max, device)

        # Scale ID for phase 1 (low-resolution = 0)
        scale_id = torch.zeros(B, dtype=torch.long, device=device)

        # Forward through Transformer
        # Use low-res positional embeddings
        z_s = self._forward_transformer_low_res(low_res_tokens, mask_s, class_labels, scale_id, train)

        # Compute diffusion loss for phase 1
        if train:
            loss_s = self.diff_head1(low_res_tokens, z_s, mask=mask_s)
        else:
            loss_s = None

        return z_s, mask_s, loss_s

    def _forward_transformer_low_res(self, tokens, mask, class_labels, scale_id, train=True):
        """Forward pass through Transformer for low-resolution tokens."""
        B, N, C = tokens.shape

        # Embed tokens
        x = self.transformer.token_embed(tokens)

        # Add low-res positional embedding
        x = x + self.low_res_pos_embed[:, :N, :]

        # Replace masked tokens with mask token
        mask_tokens = self.transformer.mask_token.expand(B, N, -1)
        mask_expanded = mask.unsqueeze(-1).float()
        x = x * (1 - mask_expanded) + mask_tokens * mask_expanded

        # Get conditioning
        class_emb = self.transformer.class_embed(class_labels, train=train)
        scale_emb = self.transformer.scale_embed(scale_id)
        c = class_emb + scale_emb

        # Process through Transformer blocks
        for block in self.transformer.blocks:
            if self.transformer.grad_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x, c)
            else:
                x = block(x, c)

        x = self.transformer.norm(x)
        return x

    def forward_phase2(self, high_res_tokens, z_s, class_labels, train=True):
        """
        Phase 2: Process high-resolution tokens conditioned on phase 1 pivots.

        Args:
            high_res_tokens: [B, N_l, C] high-resolution visual tokens
            z_s: conditional tokens from phase 1 [B, N_s, hidden_size]
            class_labels: [B] class labels
            train: whether in training mode

        Returns:
            z_l: conditional tokens [B, N_l, hidden_size]
            mask_l: binary mask [B, N_l]
            loss_l: diffusion loss for phase 2 (only during training)
        """
        B, N_l, C = high_res_tokens.shape
        device = high_res_tokens.device

        # Sample mask for phase 2 (cosine schedule from MaskGIT)
        # During training, sample a random step
        if train:
            step = torch.randint(0, 32, (1,)).item()
            mask_l = self.sample_cosine_mask(B, N_l, step, 32, device)
        else:
            mask_l = torch.ones(B, N_l, device=device)  # All masked at inference start

        # Scale ID for phase 2 (high-resolution = 1)
        scale_id = torch.ones(B, dtype=torch.long, device=device)

        # Project phase 1 conditional tokens (pivots)
        pivots = self.pivot_proj(z_s)  # [B, N_s, hidden_size]

        # Forward through Transformer with pivots
        z_l = self._forward_transformer_high_res(
            high_res_tokens, mask_l, pivots, class_labels, scale_id, train
        )

        # Compute diffusion loss for phase 2
        if train:
            loss_l = self.diff_head2(high_res_tokens, z_l, mask=mask_l)
        else:
            loss_l = None

        return z_l, mask_l, loss_l

    def _forward_transformer_high_res(self, tokens, mask, pivots, class_labels, scale_id, train=True):
        """
        Forward pass through Transformer for high-resolution tokens.
        Concatenates pivot tokens from phase 1 as additional context.
        """
        B, N_l, C = tokens.shape
        N_s = pivots.shape[1]

        # Embed high-res tokens
        x = self.transformer.token_embed(tokens)

        # Add high-res positional embedding
        x = x + self.transformer.pos_embed[:, :N_l, :]

        # Replace masked tokens with mask token
        mask_tokens = self.transformer.mask_token.expand(B, N_l, -1)
        mask_expanded = mask.unsqueeze(-1).float()
        x = x * (1 - mask_expanded) + mask_tokens * mask_expanded

        # Concatenate pivot tokens at the beginning
        # [B, N_s + N_l, hidden_size]
        x_with_pivots = torch.cat([pivots, x], dim=1)

        # Get conditioning
        class_emb = self.transformer.class_embed(class_labels, train=train)
        scale_emb = self.transformer.scale_embed(scale_id)
        c = class_emb + scale_emb

        # Process through Transformer blocks
        for block in self.transformer.blocks:
            if self.transformer.grad_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x_with_pivots = checkpoint(block, x_with_pivots, c)
            else:
                x_with_pivots = block(x_with_pivots, c)

        x_with_pivots = self.transformer.norm(x_with_pivots)

        # Extract only the high-res token outputs (skip pivot tokens)
        z_l = x_with_pivots[:, N_s:, :]  # [B, N_l, hidden_size]

        return z_l

    def forward(self, high_res_tokens, low_res_tokens, class_labels):
        """
        Full forward pass for training.

        Args:
            high_res_tokens: [B, N_l, C] high-resolution visual tokens
            low_res_tokens: [B, N_s, C] low-resolution visual tokens
            class_labels: [B] class labels

        Returns:
            total_loss: combined loss from both phases
            loss_dict: dictionary with individual losses
        """
        # Phase 1: Low-resolution
        z_s, mask_s, loss_s = self.forward_phase1(low_res_tokens, class_labels, train=True)

        # Phase 2: High-resolution (conditioned on phase 1 pivots)
        z_l, mask_l, loss_l = self.forward_phase2(high_res_tokens, z_s, class_labels, train=True)

        # Total loss
        total_loss = loss_s + loss_l

        loss_dict = {
            'loss_phase1': loss_s.item(),
            'loss_phase2': loss_l.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def generate(
        self,
        class_labels,
        num_steps_phase1=32,
        num_steps_phase2=4,
        cfg_scale=1.5,
        temperature=1.0,
        diff_temperature=1.0,
    ):
        """
        Generate images using hierarchical masked autoregressive sampling.

        Args:
            class_labels: [B] class labels
            num_steps_phase1: number of autoregressive steps for phase 1
            num_steps_phase2: number of autoregressive steps for phase 2
            cfg_scale: classifier-free guidance scale
            temperature: sampling temperature for mask scheduling
            diff_temperature: temperature for diffusion sampling

        Returns:
            generated tokens [B, N_l, C]
        """
        B = class_labels.shape[0]
        device = class_labels.device

        # Phase 1: Generate low-resolution tokens
        low_res_tokens = self._generate_phase1(
            class_labels, num_steps_phase1, cfg_scale, diff_temperature
        )

        # Get conditional tokens from phase 1 (pivots)
        scale_id = torch.zeros(B, dtype=torch.long, device=device)
        mask_s = torch.zeros(B, self.low_res_seq_len, device=device)  # All unmasked
        z_s = self._forward_transformer_low_res(low_res_tokens, mask_s, class_labels, scale_id, train=False)

        # Phase 2: Generate high-resolution tokens conditioned on pivots
        high_res_tokens = self._generate_phase2(
            class_labels, z_s, num_steps_phase2, cfg_scale, diff_temperature
        )

        return high_res_tokens

    def _generate_phase1(self, class_labels, num_steps, cfg_scale, temperature):
        """Generate low-resolution tokens using masked autoregressive sampling."""
        B = class_labels.shape[0]
        device = class_labels.device
        N_s = self.low_res_seq_len

        # Start with all tokens masked
        tokens = torch.zeros(B, N_s, self.in_channels, device=device)
        mask = torch.ones(B, N_s, device=device)  # All masked

        scale_id = torch.zeros(B, dtype=torch.long, device=device)

        for step in range(num_steps):
            # Compute number of tokens to unmask at this step (cosine schedule)
            ratio = math.cos((step + 1) / num_steps * math.pi / 2)
            num_to_keep_masked = int(math.ceil(ratio * N_s))
            num_to_unmask = int(mask.sum(dim=1).max().item()) - num_to_keep_masked
            num_to_unmask = max(1, num_to_unmask)

            # Forward pass through Transformer
            z_s = self._forward_transformer_low_res(tokens, mask, class_labels, scale_id, train=False)

            # Apply CFG
            if cfg_scale > 1.0:
                # Unconditional forward pass
                uncond_labels = torch.full_like(class_labels, self.num_classes)
                z_s_uncond = self._forward_transformer_low_res(
                    tokens, mask, uncond_labels, scale_id, train=False
                )
                z_s = z_s_uncond + cfg_scale * (z_s - z_s_uncond)

            # Sample tokens for masked positions using diffusion head
            sampled = self.diff_head1.sample(z_s, temperature=temperature, num_steps=10)

            # Update tokens at masked positions
            tokens = tokens * (1 - mask.unsqueeze(-1)) + sampled * mask.unsqueeze(-1)

            # Update mask: unmask the most confident tokens
            if step < num_steps - 1:
                # Compute confidence scores (use L2 norm of sampled tokens as proxy)
                confidence = sampled.norm(dim=-1)  # [B, N_s]
                confidence = confidence * mask  # Only consider masked tokens

                # Unmask top-k most confident tokens
                _, indices = confidence.topk(num_to_unmask, dim=1)
                for b in range(B):
                    mask[b, indices[b]] = 0
            else:
                mask = torch.zeros_like(mask)

        return tokens

    def _generate_phase2(self, class_labels, z_s, num_steps, cfg_scale, temperature):
        """Generate high-resolution tokens using masked autoregressive sampling with pivots."""
        B = class_labels.shape[0]
        device = class_labels.device
        N_l = self.high_res_seq_len

        # Start with all tokens masked
        tokens = torch.zeros(B, N_l, self.in_channels, device=device)
        mask = torch.ones(B, N_l, device=device)  # All masked

        scale_id = torch.ones(B, dtype=torch.long, device=device)
        pivots = self.pivot_proj(z_s)

        for step in range(num_steps):
            # Compute number of tokens to unmask at this step (cosine schedule)
            ratio = math.cos((step + 1) / num_steps * math.pi / 2)
            num_to_keep_masked = int(math.ceil(ratio * N_l))
            num_to_unmask = int(mask.sum(dim=1).max().item()) - num_to_keep_masked
            num_to_unmask = max(1, num_to_unmask)

            # Forward pass through Transformer with pivots
            z_l = self._forward_transformer_high_res(
                tokens, mask, pivots, class_labels, scale_id, train=False
            )

            # Apply CFG (only for high-res phase as per paper)
            if cfg_scale > 1.0:
                uncond_labels = torch.full_like(class_labels, self.num_classes)
                z_l_uncond = self._forward_transformer_high_res(
                    tokens, mask, pivots, uncond_labels, scale_id, train=False
                )
                z_l = z_l_uncond + cfg_scale * (z_l - z_l_uncond)

            # Sample tokens for masked positions using Diffusion Transformer head
            # Only sample for masked positions
            sampled = self.diff_head2.sample(z_l, temperature=temperature, num_steps=10)

            # Update tokens at masked positions
            tokens = tokens * (1 - mask.unsqueeze(-1)) + sampled * mask.unsqueeze(-1)

            # Update mask
            if step < num_steps - 1:
                confidence = sampled.norm(dim=-1) * mask
                _, indices = confidence.topk(num_to_unmask, dim=1)
                for b in range(B):
                    mask[b, indices[b]] = 0
            else:
                mask = torch.zeros_like(mask)

        return tokens


# Model factory functions
def HiMAR_B(**kwargs):
    """Hi-MAR Base: 244M parameters"""
    return HiMAR(
        hidden_size=768,
        depth=24,
        num_heads=12,
        diff_head1_depth=6,
        diff_head1_hidden=1024,
        diff_head2_depth=6,
        diff_head2_hidden=512,
        diff_head2_num_heads=8,
        **kwargs
    )


def HiMAR_L(**kwargs):
    """Hi-MAR Large: 529M parameters"""
    return HiMAR(
        hidden_size=1024,
        depth=32,
        num_heads=16,
        diff_head1_depth=8,
        diff_head1_hidden=1280,
        diff_head2_depth=8,
        diff_head2_hidden=512,
        diff_head2_num_heads=8,
        **kwargs
    )


def HiMAR_H(**kwargs):
    """Hi-MAR Huge: 1090M parameters"""
    return HiMAR(
        hidden_size=1280,
        depth=40,
        num_heads=16,
        diff_head1_depth=12,
        diff_head1_hidden=1536,
        diff_head2_depth=12,
        diff_head2_hidden=768,
        diff_head2_num_heads=12,
        **kwargs
    )


def HiMAR_S(**kwargs):
    """Hi-MAR Small: lightweight version for text-to-image (MS-COCO)"""
    return HiMAR(
        hidden_size=512,
        depth=16,
        num_heads=8,
        diff_head1_depth=4,
        diff_head1_hidden=768,
        diff_head2_depth=4,
        diff_head2_hidden=384,
        diff_head2_num_heads=6,
        **kwargs
    )
