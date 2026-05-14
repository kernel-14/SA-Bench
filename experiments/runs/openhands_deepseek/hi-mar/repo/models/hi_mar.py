import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import HiMARTransformer, MLPDiffusionHead, DiffusionTransformerHead
from utils.diffusion import NoiseScheduler


class HiMAR(nn.Module):
    """
    Hierarchical Masked Autoregressive Model (Hi-MAR).
    Two-phase autoregressive modeling with low-resolution token pivots.

    Phase 1: Masked low-res tokens + context → Transformer → Z^s conditional tokens
             Z^s → MLP Diffusion Head → loss on masked tokens
    Phase 2: [Z^s, masked high-res tokens] + context → Transformer → Z_full → Z_high
             Z_high → Diffusion Transformer Head (all tokens with self-attention) → loss
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        token_dim = config['token_dim']  # 16 (VAE KL-16 latent channels)
        trans_dim = config['transformer_dim']  # 768/1024/1280

        # Token input projection: token_dim → transformer_dim
        self.token_in_proj = nn.Linear(token_dim, trans_dim)
        # Token output projection: transformer_dim → token_dim
        self.token_out_proj = nn.Linear(trans_dim, token_dim)

        # Transformer backbone (shared, bidirectional, scale-aware)
        self.transformer = HiMARTransformer(
            dim=trans_dim,
            depth=config['transformer_depth'],
            num_heads=config['transformer_num_heads'],
            mlp_ratio=config['mlp_ratio'],
            dropout=config.get('dropout', 0.0),
        )

        # Phase 1 diffusion head (MLP-based, per-token)
        self.phase1_head = MLPDiffusionHead(
            in_dim=token_dim,
            hidden_dim=config['head1_hidden_dim'],
            depth=config['head1_depth'],
            cond_out_dim=trans_dim,
        )

        # Phase 2 diffusion head (Diffusion Transformer, all tokens)
        self.phase2_head = DiffusionTransformerHead(
            in_dim=token_dim,
            hidden_dim=config['head2_hidden_dim'],
            depth=config['head2_depth'],
            num_heads=config['head2_num_heads'],
            cond_out_dim=trans_dim,
        )

        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_train_timesteps=config.get('num_train_timesteps', 1000),
            beta_start=config.get('beta_start', 1e-4),
            beta_end=config.get('beta_end', 0.02),
        )

        # Class embedding for class-conditional generation
        if config.get('num_classes', 0) > 0:
            self.class_embed = nn.Embedding(config['num_classes'], trans_dim)
        else:
            self.class_embed = None

        # Unconditional class embedding (for CFG)
        if config.get('num_classes', 0) > 0:
            self.uncond_embed = nn.Embedding(1, trans_dim)
        else:
            self.uncond_embed = None

        # Learnable mask token (in token_dim space)
        self.mask_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        # Positional embeddings (in transformer_dim)
        self.low_res_pos = nn.Parameter(torch.randn(1, config['low_res_num_tokens'], trans_dim) * 0.02)
        self.high_res_pos = nn.Parameter(torch.randn(1, config['high_res_num_tokens'], trans_dim) * 0.02)

        self.token_dim = token_dim
        self.trans_dim = trans_dim
        self.low_res_tokens_n = config['low_res_num_tokens']
        self.high_res_tokens_n = config['high_res_num_tokens']

    def _make_context_tokens(self, class_ids: torch.Tensor | None) -> tuple[torch.Tensor, int]:
        """Create context token sequence (class tokens prepended)."""
        B = class_ids.shape[0] if class_ids is not None else 1
        if class_ids is not None and self.class_embed is not None:
            ctx = self.class_embed(class_ids).unsqueeze(1)  # [B, 1, dim]
            return ctx, 1
        return None, 0

    def _make_uncond_context(self, B: int, device: torch.device) -> torch.Tensor:
        """Create unconditional context for CFG."""
        if self.uncond_embed is not None:
            idx = torch.zeros(B, dtype=torch.long, device=device)
            return self.uncond_embed(idx).unsqueeze(1)  # [B, 1, trans_dim]
        return torch.zeros(B, 1, self.trans_dim, device=device)

    def _phase1_forward(
        self,
        tokens: torch.Tensor,          # [B, N_low, token_dim]
        mask: torch.Tensor,            # [B, N_low] bool
        class_ids: torch.Tensor | None,
        training: bool = True,
    ) -> dict:
        """
        Phase 1: Low-resolution masked autoregressive modeling.
        masked_low_res (token_dim) → token_in_proj → + pos → Transformer → Z^s (trans_dim) → MLP head → loss
        """
        B, N, _ = tokens.shape

        # Apply mask in token_dim space
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token.to(tokens.device)

        # Project to transformer dimension
        masked_proj = self.token_in_proj(masked_tokens)  # [B, N, trans_dim]

        # Add positional embeddings
        img_tokens = masked_proj + self.low_res_pos  # [B, N, trans_dim]

        # Context tokens (class embeddings in trans_dim, prepended)
        ctx_tokens, ctx_len = self._make_context_tokens(class_ids)

        # Prepend context tokens
        if ctx_tokens is not None:
            x = torch.cat([ctx_tokens, img_tokens], dim=1)  # [B, ctx_len + N, trans_dim]
        else:
            x = img_tokens

        # Forward through transformer (scale 0 = phase 1)
        z = self.transformer(x, scale_idx=0)  # [B, ctx_len + N, trans_dim]

        # Extract image token conditional embeddings (in trans_dim)
        z_img = z[:, ctx_len:, :]  # [B, N, trans_dim]

        result = {'cond_tokens': z_img}

        if training:
            t = torch.randint(0, self.noise_scheduler.num_train_timesteps, (B,), device=tokens.device)
            noise = torch.randn(B, N, self.token_dim, device=tokens.device)
            x_t = self.noise_scheduler.add_noise(tokens, noise, t)

            # Loss only on masked positions
            masked_x_t = x_t[mask]
            masked_z = z_img[mask]
            masked_t = t.unsqueeze(-1).expand(-1, N).reshape(-1)[mask.reshape(-1)]

            if masked_x_t.numel() == 0:
                result['loss'] = torch.tensor(0.0, device=tokens.device, requires_grad=True)
            else:
                noise_pred = self.phase1_head(masked_x_t, masked_t, masked_z)
                target_noise = noise[mask]
                result['loss'] = F.mse_loss(noise_pred, target_noise)
                result['pred'] = noise_pred

        return result

    def _phase2_forward(
        self,
        tokens: torch.Tensor,          # [B, N_high, token_dim]
        mask: torch.Tensor,            # [B, N_high] bool
        z_s: torch.Tensor,             # [B, N_low, trans_dim] conditional tokens from phase 1
        class_ids: torch.Tensor | None,
        training: bool = True,
    ) -> dict:
        """
        Phase 2: High-resolution masked autoregressive modeling.
        [z_s (trans_dim), masked_high_res_proj (trans_dim)] + context → Transformer → Z_high (trans_dim)
        Z_high → Diffusion Transformer head → loss (token_dim)
        """
        B, N_high, _ = tokens.shape

        # Apply mask in token_dim, then project
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token.to(tokens.device)
        masked_proj = self.token_in_proj(masked_tokens)  # [B, N_high, trans_dim]

        # Context tokens
        ctx_tokens, ctx_len = self._make_context_tokens(class_ids)

        # Add positional embeddings
        z_s_pos = z_s + self.low_res_pos                              # [B, N_low, trans_dim]
        high_res_pos_tokens = masked_proj + self.high_res_pos         # [B, N_high, trans_dim]

        # Build sequence: [ctx, z_s, high_res_masked]
        parts = []
        if ctx_tokens is not None:
            parts.append(ctx_tokens)
        parts.append(z_s_pos)
        parts.append(high_res_pos_tokens)
        x = torch.cat(parts, dim=1)  # [B, ctx_len + N_low + N_high, trans_dim]

        # Forward through transformer (scale 1 = phase 2)
        z_full = self.transformer(x, scale_idx=1)

        # Extract high-res conditional tokens
        z_high_start = ctx_len + self.low_res_tokens_n
        z_high = z_full[:, z_high_start:, :]  # [B, N_high, trans_dim]

        result = {'cond_tokens': z_high}

        if training:
            t = torch.randint(0, self.noise_scheduler.num_train_timesteps, (B,), device=tokens.device)
            noise = torch.randn(B, N_high, self.token_dim, device=tokens.device)
            x_t = self.noise_scheduler.add_noise(tokens, noise, t)

            # Diffusion Transformer head: input = ALL high-res tokens (in token_dim)
            all_tokens = torch.where(mask.unsqueeze(-1), x_t, tokens)  # [B, N_high, token_dim]

            noise_pred = self.phase2_head(all_tokens, t, z_high)
            result['loss'] = F.mse_loss(noise_pred, noise)
            result['pred'] = noise_pred

        return result

    def forward(
        self,
        low_res_tokens: torch.Tensor,
        high_res_tokens: torch.Tensor,
        low_res_mask: torch.Tensor,
        high_res_mask: torch.Tensor,
        class_ids: torch.Tensor | None = None,
    ) -> dict:
        """
        Full forward pass during training.

        Args:
            low_res_tokens:  [B, N_low, D] ground-truth low-res tokens.
            high_res_tokens: [B, N_high, D] ground-truth high-res tokens.
            low_res_mask:    [B, N_low] bool mask for phase 1.
            high_res_mask:   [B, N_high] bool mask for phase 2.
            class_ids:       [B] class indices.
        Returns:
            dict with 'phase1_loss', 'phase2_loss', 'loss', etc.
        """
        # Phase 1: Low-resolution prediction
        p1 = self._phase1_forward(low_res_tokens, low_res_mask, class_ids, training=True)
        z_s = p1['cond_tokens'].detach()  # Paper: use predicted conditional tokens, not GT tokens

        # Phase 2: High-resolution prediction conditioned on phase 1 output
        p2 = self._phase2_forward(high_res_tokens, high_res_mask, z_s, class_ids, training=True)

        return {
            'phase1_loss': p1['loss'],
            'phase2_loss': p2['loss'],
            'loss': p1['loss'] + p2['loss'],
            'phase1_cond': z_s,
            'phase2_cond': p2['cond_tokens'],
        }

    @torch.no_grad()
    def generate(
        self,
        class_ids: torch.Tensor | None = None,
        phase1_steps: int = 32,
        phase2_steps: int = 4,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Full two-phase generation with iterative masked decoding.

        Phase 1: 32 steps, MLP diffusion head.
        Phase 2: 4 steps, Diffusion Transformer head.
        CFG optionally applied in phase 2 only (as described in paper).
        """
        B = class_ids.shape[0] if class_ids is not None else 1
        device = next(self.parameters()).device

        if class_ids is None:
            class_ids = torch.zeros(B, dtype=torch.long, device=device)

        # Phase 1: Generate low-res tokens → get Z^s pivots
        z_s = self._phase1_decode(class_ids, steps=phase1_steps)  # [B, N_low, D]

        # Phase 2: Generate high-res tokens conditioned on Z^s
        if cfg_scale > 1.0:
            high_res = self._phase2_decode_cfg(class_ids, z_s, steps=phase2_steps, cfg_scale=cfg_scale)
        else:
            high_res = self._phase2_decode(class_ids, z_s, steps=phase2_steps)

        return high_res

    @torch.no_grad()
    def _phase1_decode(
        self,
        class_ids: torch.Tensor,
        steps: int = 32,
    ) -> torch.Tensor:
        """Iterative decoding for phase 1. Returns Z^s conditional tokens (trans_dim)."""
        B, N = class_ids.shape[0], self.low_res_tokens_n
        device = class_ids.device

        # Context tokens
        ctx_tokens, ctx_len = self._make_context_tokens(class_ids)

        # Start with all tokens masked (in token_dim)
        predicted_tokens = self.mask_token.expand(B, N, self.token_dim).clone()
        mask = torch.ones(B, N, dtype=torch.bool, device=device)

        for step in range(steps):
            # Project tokens to trans_dim and add pos
            img_proj = self.token_in_proj(predicted_tokens) + self.low_res_pos
            if ctx_tokens is not None:
                x = torch.cat([ctx_tokens, img_proj], dim=1)
            else:
                x = img_proj

            # Forward through transformer
            z = self.transformer(x, scale_idx=0)
            z_img = z[:, ctx_len:, :]  # [B, N, trans_dim]

            # Predict tokens at masked positions using MLP head
            sub_steps = 20
            t_seq = torch.linspace(self.noise_scheduler.num_train_timesteps - 1, 0, sub_steps + 1, device=device).long()
            current_noise = torch.randn(B, N, self.token_dim, device=device)
            for i in range(sub_steps):
                t = t_seq[i].unsqueeze(0).expand(B)
                t_next = t_seq[i + 1].unsqueeze(0).expand(B)
                noise_pred = self.phase1_head(
                    current_noise.reshape(B * N, self.token_dim),
                    t.unsqueeze(-1).expand(-1, N).reshape(-1),
                    z_img.reshape(B * N, self.trans_dim),
                ).reshape(B, N, self.token_dim)
                alpha_t = self.noise_scheduler.alphas_cumprod.to(device)[t]
                alpha_next = self.noise_scheduler.alphas_cumprod.to(device)[t_next]
                x0_pred = (current_noise - (1 - alpha_t).sqrt().view(-1, 1, 1) * noise_pred) / alpha_t.sqrt().view(-1, 1, 1)
                if i < sub_steps - 1:
                    current_noise = alpha_next.sqrt().view(-1, 1, 1) * x0_pred + (1 - alpha_next).sqrt().view(-1, 1, 1) * torch.randn(B, N, self.token_dim, device=device)
                else:
                    predicted_tokens[mask] = x0_pred[mask]

            # Unmask according to cosine schedule
            progress = (step + 1) / steps
            keep_masked = math.cos(math.pi / 2 * progress)
            num_mask = max(0, int(keep_masked * N + 0.5))
            if num_mask > 0 and step < steps - 1:
                mask[:] = False
                for b in range(B):
                    idx = torch.randperm(N, device=device)[:num_mask]
                    mask[b, idx] = True
            elif step == steps - 1:
                mask[:] = False

        # Re-forward to get final Z^s (trans_dim)
        img_proj = self.token_in_proj(predicted_tokens) + self.low_res_pos
        if ctx_tokens is not None:
            x = torch.cat([ctx_tokens, img_proj], dim=1)
        else:
            x = img_proj
        z_s = self.transformer(x, scale_idx=0)
        return z_s[:, ctx_len:, :]

    @torch.no_grad()
    def _phase2_decode(
        self,
        class_ids: torch.Tensor,
        z_s: torch.Tensor,
        steps: int = 4,
    ) -> torch.Tensor:
        """Iterative decoding for phase 2. Returns high-resolution tokens (token_dim)."""
        B, N = class_ids.shape[0], self.high_res_tokens_n
        device = class_ids.device

        ctx_tokens, ctx_len = self._make_context_tokens(class_ids)
        z_s_pos = z_s + self.low_res_pos                          # [B, N_low, trans_dim]

        predicted_tokens = self.mask_token.expand(B, N, self.token_dim).clone()
        mask = torch.ones(B, N, dtype=torch.bool, device=device)

        for step in range(steps):
            img_proj = self.token_in_proj(predicted_tokens) + self.high_res_pos  # [B, N, trans_dim]
            parts = []
            if ctx_tokens is not None:
                parts.append(ctx_tokens)
            parts.append(z_s_pos)
            parts.append(img_proj)
            x = torch.cat(parts, dim=1)

            z_full = self.transformer(x, scale_idx=1)
            z_high = z_full[:, ctx_len + self.low_res_tokens_n:, :]  # [B, N, trans_dim]

            # Diffusion Transformer head denoising (in token_dim space)
            sub_steps = 5
            t_seq = torch.linspace(self.noise_scheduler.num_train_timesteps - 1, 0, sub_steps + 1, device=device).long()
            current_noise = torch.randn(B, N, self.token_dim, device=device)
            for i in range(sub_steps):
                t = t_seq[i].unsqueeze(0).expand(B)
                t_next = t_seq[i + 1].unsqueeze(0).expand(B)
                noise_pred = self.phase2_head(current_noise, t, z_high)
                alpha_t = self.noise_scheduler.alphas_cumprod.to(device)[t]
                alpha_next = self.noise_scheduler.alphas_cumprod.to(device)[t_next]
                x0_pred = (current_noise - (1 - alpha_t).sqrt().view(-1, 1, 1) * noise_pred) / alpha_t.sqrt().view(-1, 1, 1)
                if i < sub_steps - 1:
                    current_noise = alpha_next.sqrt().view(-1, 1, 1) * x0_pred + (1 - alpha_next).sqrt().view(-1, 1, 1) * torch.randn(B, N, self.token_dim, device=device)
                else:
                    predicted_tokens[mask] = x0_pred[mask]

            # Cosine schedule
            progress = (step + 1) / steps
            keep_masked = math.cos(math.pi / 2 * progress)
            num_mask = max(0, int(keep_masked * N + 0.5))
            if num_mask > 0 and step < steps - 1:
                mask[:] = False
                for b in range(B):
                    idx = torch.randperm(N, device=device)[:num_mask]
                    mask[b, idx] = True
            elif step == steps - 1:
                mask[:] = False

        return predicted_tokens

    @torch.no_grad()
    def _phase2_decode_cfg(
        self,
        class_ids: torch.Tensor,
        z_s: torch.Tensor,
        steps: int = 4,
        cfg_scale: float = 3.0,
    ) -> torch.Tensor:
        """Phase 2 decoding with Classifier-Free Guidance."""
        B, N = class_ids.shape[0], self.high_res_tokens_n
        device = class_ids.device

        ctx_tokens, ctx_len = self._make_context_tokens(class_ids)
        uncond_ctx = self._make_uncond_context(B, device)
        z_s_pos = z_s + self.low_res_pos

        predicted_tokens = self.mask_token.expand(B, N, self.token_dim).clone()
        mask = torch.ones(B, N, dtype=torch.bool, device=device)

        for step in range(steps):
            img_proj = self.token_in_proj(predicted_tokens) + self.high_res_pos

            # Conditional path
            parts_cond = []
            if ctx_tokens is not None:
                parts_cond.append(ctx_tokens)
            parts_cond.append(z_s_pos)
            parts_cond.append(img_proj)
            x_cond = torch.cat(parts_cond, dim=1)
            z_full_cond = self.transformer(x_cond, scale_idx=1)
            z_high_cond = z_full_cond[:, ctx_len + self.low_res_tokens_n:, :]

            # Unconditional path
            parts_uncond = [uncond_ctx, z_s_pos, img_proj]
            x_uncond = torch.cat(parts_uncond, dim=1)
            z_full_uncond = self.transformer(x_uncond, scale_idx=1)
            z_high_uncond = z_full_uncond[:, 1 + self.low_res_tokens_n:, :]

            # CFG
            z_high = z_high_uncond + cfg_scale * (z_high_cond - z_high_uncond)

            # Denoising
            sub_steps = 5
            t_seq = torch.linspace(self.noise_scheduler.num_train_timesteps - 1, 0, sub_steps + 1, device=device).long()
            current_noise = torch.randn(B, N, self.token_dim, device=device)
            for i in range(sub_steps):
                t = t_seq[i].unsqueeze(0).expand(B)
                t_next = t_seq[i + 1].unsqueeze(0).expand(B)
                noise_pred = self.phase2_head(current_noise, t, z_high)
                alpha_t = self.noise_scheduler.alphas_cumprod.to(device)[t]
                alpha_next = self.noise_scheduler.alphas_cumprod.to(device)[t_next]
                x0_pred = (current_noise - (1 - alpha_t).sqrt().view(-1, 1, 1) * noise_pred) / alpha_t.sqrt().view(-1, 1, 1)
                if i < sub_steps - 1:
                    current_noise = alpha_next.sqrt().view(-1, 1, 1) * x0_pred + (1 - alpha_next).sqrt().view(-1, 1, 1) * torch.randn(B, N, self.token_dim, device=device)
                else:
                    predicted_tokens[mask] = x0_pred[mask]

            progress = (step + 1) / steps
            keep_masked = math.cos(math.pi / 2 * progress)
            num_mask = max(0, int(keep_masked * N + 0.5))
            if num_mask > 0 and step < steps - 1:
                mask[:] = False
                for b in range(B):
                    idx = torch.randperm(N, device=device)[:num_mask]
                    mask[b, idx] = True
            elif step == steps - 1:
                mask[:] = False

        return predicted_tokens
