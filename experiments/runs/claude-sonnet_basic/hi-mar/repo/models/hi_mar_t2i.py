"""
Hi-MAR for Text-to-Image generation.
Extends Hi-MAR with CLIP text conditioning for MS-COCO text-to-image generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .transformer import (
    ScaleAwareTransformerBlock,
    ScaleEmbedder,
    get_2d_sincos_pos_embed,
)
from .diffusion_loss import DiffusionLoss
from .hi_mar import HiMAR


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for conditioning on text embeddings.
    """

    def __init__(self, hidden_size, text_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_text = nn.LayerNorm(text_dim, eps=1e-6)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # Cross-attention with text
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True,
            kdim=text_dim, vdim=text_dim
        )

        # FFN
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # AdaLN for scale conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, text_emb):
        """
        Args:
            x: input tokens [B, N, hidden_size]
            c: scale conditioning vector [B, hidden_size]
            text_emb: text embeddings [B, L, text_dim]
        """
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.adaLN_modulation(c).chunk(6, dim=-1)

        # Self-attention
        x_norm = alpha1.unsqueeze(1) * self.norm1(x) + beta1.unsqueeze(1)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + gamma1.unsqueeze(1) * attn_out

        # Cross-attention with text
        text_norm = self.norm_text(text_emb)
        x_norm2 = self.norm2(x)
        cross_out, _ = self.cross_attn(x_norm2, text_norm, text_norm)
        x = x + cross_out

        # FFN
        x_norm = alpha2.unsqueeze(1) * self.norm2(x) + beta2.unsqueeze(1)
        x = x + gamma2.unsqueeze(1) * self.ff(x_norm)

        return x


class HiMARText(nn.Module):
    """
    Hi-MAR for text-to-image generation.
    Uses CLIP text embeddings as conditioning instead of class labels.

    Following the paper's setup for MS-COCO text-to-image generation.
    """

    def __init__(
        self,
        # Image settings
        img_size=256,
        low_res_img_size=128,
        patch_size=16,
        in_channels=16,
        # Transformer settings
        hidden_size=512,
        depth=16,
        num_heads=8,
        mlp_ratio=4.0,
        # Diffusion head settings (phase 1 - MLP)
        diff_head1_depth=4,
        diff_head1_hidden=768,
        # Diffusion head settings (phase 2 - Transformer)
        diff_head2_depth=4,
        diff_head2_hidden=384,
        diff_head2_num_heads=6,
        # Text conditioning
        text_dim=768,  # CLIP text embedding dimension
        text_seq_len=77,  # CLIP max sequence length
        # Masking
        mask_ratio_min=0.0,
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
        self.text_dim = text_dim
        self.num_sampling_steps = num_sampling_steps
        self.label_drop_prob = label_drop_prob

        # Token counts
        self.high_res_seq_len = (img_size // patch_size) ** 2
        self.low_res_seq_len = (low_res_img_size // patch_size) ** 2

        # Token embedding
        self.token_embed = nn.Linear(in_channels, hidden_size)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Positional embeddings
        self.high_res_pos_embed = nn.Parameter(
            torch.zeros(1, self.high_res_seq_len, hidden_size), requires_grad=False
        )
        self.low_res_pos_embed = nn.Parameter(
            torch.zeros(1, self.low_res_seq_len, hidden_size), requires_grad=False
        )

        # Scale embedding
        self.scale_embed = ScaleEmbedder(hidden_size)

        # Text projection
        self.text_proj = nn.Linear(text_dim, hidden_size)

        # Null text embedding for CFG
        self.null_text_embed = nn.Parameter(torch.zeros(1, text_seq_len, text_dim))

        # Transformer blocks (scale-aware)
        self.blocks = nn.ModuleList([
            ScaleAwareTransformerBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Output norm
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # Phase 1: MLP-based diffusion head
        self.diff_head1 = DiffusionLoss(
            target_channels=in_channels,
            hidden_size=diff_head1_hidden,
            depth=diff_head1_depth,
            token_dim=hidden_size,
            use_transformer_head=False,
            num_sampling_steps=num_sampling_steps,
        )

        # Phase 2: Diffusion Transformer head
        self.diff_head2 = DiffusionLoss(
            target_channels=in_channels,
            hidden_size=diff_head2_hidden,
            depth=diff_head2_depth,
            token_dim=hidden_size,
            num_heads=diff_head2_num_heads,
            use_transformer_head=True,
            num_sampling_steps=num_sampling_steps,
        )

        # Pivot projection
        self.pivot_proj = nn.Linear(hidden_size, hidden_size)

        self._init_weights()

    def _init_weights(self):
        # Initialize positional embeddings
        high_res_grid = int(self.high_res_seq_len ** 0.5)
        pos_embed_high = get_2d_sincos_pos_embed(self.hidden_size, high_res_grid)
        self.high_res_pos_embed.data.copy_(torch.from_numpy(pos_embed_high).float().unsqueeze(0))

        low_res_grid = int(self.low_res_seq_len ** 0.5)
        pos_embed_low = get_2d_sincos_pos_embed(self.hidden_size, low_res_grid)
        self.low_res_pos_embed.data.copy_(torch.from_numpy(pos_embed_low).float().unsqueeze(0))

        # Initialize mask token
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.null_text_embed, std=0.02)

        # Initialize linear layers
        self.apply(self._init_linear)

    def _init_linear(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _get_text_conditioning(self, text_emb, scale_id, drop_text=False):
        """
        Get conditioning vector from text embeddings and scale.
        """
        B = text_emb.shape[0]

        # Project text to hidden size and take mean as global condition
        text_proj = self.text_proj(text_emb)  # [B, L, hidden_size]
        text_global = text_proj.mean(dim=1)  # [B, hidden_size]

        # Scale conditioning
        scale_emb = self.scale_embed(scale_id)  # [B, hidden_size]

        # Combined conditioning
        c = text_global + scale_emb
        return c

    def _forward_transformer(self, tokens, mask, text_emb, scale_id, pos_embed, train=True):
        """
        Forward pass through Transformer.
        """
        B, N, C = tokens.shape

        # Embed tokens
        x = self.token_embed(tokens)

        # Add positional embedding
        x = x + pos_embed[:, :N, :]

        # Replace masked tokens
        mask_tokens = self.mask_token.expand(B, N, -1)
        mask_expanded = mask.unsqueeze(-1).float()
        x = x * (1 - mask_expanded) + mask_tokens * mask_expanded

        # Get conditioning
        if train and self.label_drop_prob > 0:
            # Randomly drop text conditioning for CFG training
            drop_mask = torch.rand(B, device=tokens.device) < self.label_drop_prob
            null_text = self.null_text_embed.expand(B, -1, -1)
            text_emb_used = torch.where(
                drop_mask[:, None, None].expand_as(text_emb),
                null_text,
                text_emb
            )
        else:
            text_emb_used = text_emb

        c = self._get_text_conditioning(text_emb_used, scale_id)

        # Process through Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        x = self.norm(x)
        return x

    def forward(self, high_res_tokens, low_res_tokens, text_emb):
        """
        Training forward pass.

        Args:
            high_res_tokens: [B, N_l, C]
            low_res_tokens: [B, N_s, C]
            text_emb: [B, L, text_dim] CLIP text embeddings

        Returns:
            total_loss, loss_dict
        """
        B = high_res_tokens.shape[0]
        device = high_res_tokens.device

        # Phase 1: Low-resolution
        N_s = self.low_res_seq_len
        mask_s = self._sample_mask_beta(B, N_s, device)
        scale_id_s = torch.zeros(B, dtype=torch.long, device=device)

        z_s = self._forward_transformer(
            low_res_tokens, mask_s, text_emb, scale_id_s, self.low_res_pos_embed
        )
        loss_s = self.diff_head1(low_res_tokens, z_s, mask=mask_s)

        # Phase 2: High-resolution with pivots
        N_l = self.high_res_seq_len
        step = torch.randint(0, 32, (1,)).item()
        mask_l = self._sample_cosine_mask(B, N_l, step, 32, device)
        scale_id_l = torch.ones(B, dtype=torch.long, device=device)

        pivots = self.pivot_proj(z_s)

        # Concatenate pivots with high-res tokens
        high_res_embedded = self.token_embed(high_res_tokens)
        high_res_embedded = high_res_embedded + self.high_res_pos_embed[:, :N_l, :]
        mask_tokens = self.mask_token.expand(B, N_l, -1)
        mask_expanded = mask_l.unsqueeze(-1).float()
        high_res_embedded = high_res_embedded * (1 - mask_expanded) + mask_tokens * mask_expanded

        x_with_pivots = torch.cat([pivots, high_res_embedded], dim=1)

        # Get conditioning
        c = self._get_text_conditioning(text_emb, scale_id_l)

        for block in self.blocks:
            x_with_pivots = block(x_with_pivots, c)

        x_with_pivots = self.norm(x_with_pivots)
        z_l = x_with_pivots[:, N_s:, :]

        loss_l = self.diff_head2(high_res_tokens, z_l, mask=mask_l)

        total_loss = loss_s + loss_l
        loss_dict = {
            'loss_phase1': loss_s.item(),
            'loss_phase2': loss_l.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict

    def _sample_mask_beta(self, batch_size, seq_len, device, alpha=4, beta=1):
        """
        Sample masking ratio from Beta distribution (for text-to-image as per paper).
        """
        import torch.distributions as dist
        beta_dist = dist.Beta(torch.tensor(float(alpha)), torch.tensor(float(beta)))
        mask_ratios = beta_dist.sample((batch_size,)).to(device)

        masks = []
        for i in range(batch_size):
            num_masked = int(math.ceil(mask_ratios[i].item() * seq_len))
            perm = torch.randperm(seq_len, device=device)
            mask = torch.zeros(seq_len, device=device)
            mask[perm[:num_masked]] = 1
            masks.append(mask)

        return torch.stack(masks)

    def _sample_cosine_mask(self, batch_size, seq_len, step, total_steps, device):
        """Sample mask using cosine schedule."""
        ratio = math.cos(step / total_steps * math.pi / 2)
        num_masked = int(math.ceil(ratio * seq_len))

        masks = []
        for _ in range(batch_size):
            perm = torch.randperm(seq_len, device=device)
            mask = torch.zeros(seq_len, device=device)
            mask[perm[:num_masked]] = 1
            masks.append(mask)

        return torch.stack(masks)

    @torch.no_grad()
    def generate(
        self,
        text_emb,
        num_steps_phase1=32,
        num_steps_phase2=4,
        cfg_scale=1.5,
        temperature=1.0,
        diff_temperature=1.0,
    ):
        """
        Generate images from text embeddings.
        """
        B = text_emb.shape[0]
        device = text_emb.device

        # Phase 1: Generate low-resolution tokens
        low_res_tokens = self._generate_phase1(
            text_emb, num_steps_phase1, cfg_scale, diff_temperature
        )

        # Get pivots from phase 1
        scale_id_s = torch.zeros(B, dtype=torch.long, device=device)
        mask_s = torch.zeros(B, self.low_res_seq_len, device=device)
        z_s = self._forward_transformer(
            low_res_tokens, mask_s, text_emb, scale_id_s, self.low_res_pos_embed, train=False
        )

        # Phase 2: Generate high-resolution tokens
        high_res_tokens = self._generate_phase2(
            text_emb, z_s, num_steps_phase2, cfg_scale, diff_temperature
        )

        return high_res_tokens

    def _generate_phase1(self, text_emb, num_steps, cfg_scale, temperature):
        """Generate low-resolution tokens."""
        B = text_emb.shape[0]
        device = text_emb.device
        N_s = self.low_res_seq_len

        tokens = torch.zeros(B, N_s, self.in_channels, device=device)
        mask = torch.ones(B, N_s, device=device)
        scale_id = torch.zeros(B, dtype=torch.long, device=device)

        for step in range(num_steps):
            ratio = math.cos((step + 1) / num_steps * math.pi / 2)
            num_to_keep_masked = int(math.ceil(ratio * N_s))
            num_to_unmask = max(1, int(mask.sum(dim=1).max().item()) - num_to_keep_masked)

            z_s = self._forward_transformer(tokens, mask, text_emb, scale_id, self.low_res_pos_embed, train=False)

            if cfg_scale > 1.0:
                null_text = self.null_text_embed.expand(B, -1, -1)
                z_s_uncond = self._forward_transformer(tokens, mask, null_text, scale_id, self.low_res_pos_embed, train=False)
                z_s = z_s_uncond + cfg_scale * (z_s - z_s_uncond)

            sampled = self.diff_head1.sample(z_s, temperature=temperature, num_steps=10)
            tokens = tokens * (1 - mask.unsqueeze(-1)) + sampled * mask.unsqueeze(-1)

            if step < num_steps - 1:
                confidence = sampled.norm(dim=-1) * mask
                _, indices = confidence.topk(num_to_unmask, dim=1)
                for b in range(B):
                    mask[b, indices[b]] = 0
            else:
                mask = torch.zeros_like(mask)

        return tokens

    def _generate_phase2(self, text_emb, z_s, num_steps, cfg_scale, temperature):
        """Generate high-resolution tokens with pivots."""
        B = text_emb.shape[0]
        device = text_emb.device
        N_l = self.high_res_seq_len
        N_s = self.low_res_seq_len

        tokens = torch.zeros(B, N_l, self.in_channels, device=device)
        mask = torch.ones(B, N_l, device=device)
        scale_id = torch.ones(B, dtype=torch.long, device=device)
        pivots = self.pivot_proj(z_s)

        for step in range(num_steps):
            ratio = math.cos((step + 1) / num_steps * math.pi / 2)
            num_to_keep_masked = int(math.ceil(ratio * N_l))
            num_to_unmask = max(1, int(mask.sum(dim=1).max().item()) - num_to_keep_masked)

            # Embed tokens
            high_res_embedded = self.token_embed(tokens)
            high_res_embedded = high_res_embedded + self.high_res_pos_embed[:, :N_l, :]
            mask_tokens = self.mask_token.expand(B, N_l, -1)
            mask_expanded = mask.unsqueeze(-1).float()
            high_res_embedded = high_res_embedded * (1 - mask_expanded) + mask_tokens * mask_expanded

            x_with_pivots = torch.cat([pivots, high_res_embedded], dim=1)
            c = self._get_text_conditioning(text_emb, scale_id)

            for block in self.blocks:
                x_with_pivots = block(x_with_pivots, c)
            x_with_pivots = self.norm(x_with_pivots)
            z_l = x_with_pivots[:, N_s:, :]

            if cfg_scale > 1.0:
                null_text = self.null_text_embed.expand(B, -1, -1)
                c_uncond = self._get_text_conditioning(null_text, scale_id)
                x_uncond = torch.cat([pivots, high_res_embedded], dim=1)
                for block in self.blocks:
                    x_uncond = block(x_uncond, c_uncond)
                x_uncond = self.norm(x_uncond)
                z_l_uncond = x_uncond[:, N_s:, :]
                z_l = z_l_uncond + cfg_scale * (z_l - z_l_uncond)

            sampled = self.diff_head2.sample(z_l, temperature=temperature, num_steps=10)
            tokens = tokens * (1 - mask.unsqueeze(-1)) + sampled * mask.unsqueeze(-1)

            if step < num_steps - 1:
                confidence = sampled.norm(dim=-1) * mask
                _, indices = confidence.topk(num_to_unmask, dim=1)
                for b in range(B):
                    mask[b, indices[b]] = 0
            else:
                mask = torch.zeros_like(mask)

        return tokens
