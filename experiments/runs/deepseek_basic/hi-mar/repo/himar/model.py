"""
Hi-MAR: Hierarchical Masked Autoregressive Model.

Core model tying together:
1. Hi-MAR Transformer (scale-aware backbone)
2. MLP Diffusion Head (Phase 1)
3. Diffusion Transformer Head (Phase 2)
4. Two-phase training and inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .transformer import HiMARTransformer
from .diffusion_head import MLPDiffusionHead, DiffusionTransformerHead
from .masking import RandomMasking, CosineMasking, BetaMasking


def extract_condition_tokens(transformer_output, num_context, num_visual):
    """
    Extract conditional tokens corresponding to visual tokens from
    the Transformer output (which includes context + visual tokens).
    """
    return transformer_output[:, num_context:, :]


class HiMAR(nn.Module):
    """
    Hierarchical Masked Autoregressive Model.
    
    Two-phase architecture:
    Phase 1 (low-res): Hi-MAR Transformer → MLP Diffusion Head
    Phase 2 (high-res): Hi-MAR Transformer → Diffusion Transformer Head
    
    The Transformer is shared across both phases, with scale-aware
    conditioning differentiating the two scales.
    """
    
    def __init__(
        self,
        # Transformer config
        num_layers=24,
        hidden_size=768,
        num_heads=12,
        mlp_ratio=4.0,
        # Phase 1 diffusion head config
        head1_num_layers=6,
        head1_hidden_size=1024,
        # Phase 2 diffusion head config
        head2_num_layers=6,
        head2_hidden_size=512,
        head2_num_heads=8,
        # VAE config
        latent_dim=16,
        # Task config
        num_classes=None,  # for class-conditional on ImageNet
        # Token counts
        low_res_tokens=256,   # 16x16 for 128x128 image with VAE 8x downsample
        high_res_tokens=1024, # 32x32 for 256x256 image with VAE 8x downsample
        # Masking
        mask_range_phase1=(0.7, 1.0),
        mask_schedule_phase2='cosine',
        mask_beta_alpha=4.0,
        mask_beta_beta=1.0,
        # Diffusion
        num_diffusion_steps=1000,
        diffusion_beta_start=1e-4,
        diffusion_beta_end=0.02,
        # Classifier-free guidance
        cfg_drop_prob=0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.low_res_tokens = low_res_tokens
        self.high_res_tokens = high_res_tokens
        self.num_diffusion_steps = num_diffusion_steps
        self.cfg_drop_prob = cfg_drop_prob
        self.num_classes = num_classes
        
        # Learnable mask token in LATENT space (for masking before Transformer input proj)
        self.mask_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
        # Shared Hi-MAR Transformer backbone
        self.transformer = HiMARTransformer(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            max_seq_len=max(low_res_tokens, high_res_tokens) + 512,
            vocab_size=num_classes,
            input_dim=latent_dim,
        )
        
        # Phase 1: MLP-based diffusion head
        self.diffusion_head1 = MLPDiffusionHead(
            num_layers=head1_num_layers,
            hidden_size=head1_hidden_size,
            latent_dim=latent_dim,
            condition_dim=hidden_size,
        )
        
        # Phase 2: Diffusion Transformer head
        self.diffusion_head2 = DiffusionTransformerHead(
            num_layers=head2_num_layers,
            hidden_size=head2_hidden_size,
            num_heads=head2_num_heads,
            latent_dim=latent_dim,
            condition_dim=hidden_size,
        )
        
        # Masking strategies
        self.mask_phase1 = RandomMasking(mask_range=mask_range_phase1)
        if mask_schedule_phase2 == 'cosine':
            self.mask_phase2 = CosineMasking()
        else:
            self.mask_phase2 = BetaMasking(alpha=mask_beta_alpha, beta=mask_beta_beta)
        
        # Diffusion noise schedule
        self.register_buffer('betas', self._linear_beta_schedule(num_diffusion_steps, diffusion_beta_start, diffusion_beta_end))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
    
    def _linear_beta_schedule(self, timesteps, start, end):
        return torch.linspace(start, end, timesteps)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine beta schedule as in improved DDPM."""
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        alpha_bar = torch.cos((steps / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise, noise
    
    def diffusion_loss(self, head, z_cond, x_target, mask_pos=None):
        """
        Compute diffusion loss for a set of tokens.
        
        Args:
            head: MLPDiffusionHead or DiffusionTransformerHead
            z_cond: (B, N, C) conditional tokens from Transformer
            x_target: (B, N, latent_dim) ground truth tokens
            mask_pos: (B, N) boolean mask indicating which positions to compute loss on
        Returns:
            scalar loss
        """
        B, N, D = x_target.shape
        
        # Sample random timesteps
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=x_target.device)
        
        # Sample noise and corrupt
        noise = torch.randn_like(x_target)
        x_t, noise = self.q_sample(x_target, t, noise)
        
        # Predict noise using the diffusion head
        if isinstance(head, DiffusionTransformerHead):
            noise_pred = head(x_t, t, z_cond, mask_pos)
        else:
            noise_pred = head(x_t, t, z_cond)
        
        # Compute MSE loss
        loss = F.mse_loss(noise_pred, noise, reduction='none')
        loss = loss.mean(dim=-1)  # Average over latent dim
        
        if mask_pos is not None:
            # Only compute loss on masked positions
            loss = loss * mask_pos.float()
            loss = loss.sum() / (mask_pos.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    def forward_phase1(self, x_low, class_idx=None, context_embeds=None):
        """
        Phase 1: Low-resolution masked autoregressive modeling.
        
        Args:
            x_low: (B, N_low, latent_dim) low-resolution VAE tokens
            class_idx: (B,) optional class indices
            context_embeds: (B, N_ctx, C) optional text embeddings
        Returns:
            loss_phase1, conditional_tokens_low
        """
        B, N, D = x_low.shape
        
        # Generate masked low-resolution tokens (mask in latent space)
        mask_tok = self.mask_token.squeeze(0)  # (1, latent_dim) -> broadcast over N
        masked_x_low, mask_pos = self.mask_phase1(x_low, mask_tok)
        
        # Pass through Transformer (scale 0 = low-res)
        cond_tokens = self.transformer(
            masked_x_low,
            scale_idx=0,
            class_idx=class_idx,
            context_embeds=context_embeds,
        )
        
        # Extract conditional tokens for visual positions
        num_context = (1 if class_idx is not None else 0) + (context_embeds.shape[1] if context_embeds is not None else 0)
        cond_visual = extract_condition_tokens(cond_tokens, num_context, N)
        
        # Compute diffusion loss on masked positions
        loss = self.diffusion_loss(self.diffusion_head1, cond_visual, x_low, mask_pos)
        
        return loss, cond_visual, mask_pos
    
    def forward_phase2(self, x_high, cond_low, class_idx=None, context_embeds=None):
        """
        Phase 2: High-resolution masked autoregressive modeling.
        Uses conditional tokens from Phase 1 as pivots.
        
        Args:
            x_high: (B, N_high, latent_dim) high-resolution VAE tokens
            cond_low: (B, N_low, C) conditional tokens from Phase 1
            class_idx: (B,) optional class indices
            context_embeds: (B, N_ctx, C) optional text embeddings
        Returns:
            loss_phase2, conditional_tokens_high
        """
        B, N, D = x_high.shape
        
        # Generate masked high-resolution tokens (mask in latent space)
        mask_tok = self.mask_token.squeeze(0)  # (1, latent_dim)
        masked_x_high, mask_pos = self.mask_phase2(x_high, mask_tok)
        
        # For phase 2, the input to Transformer includes:
        # [masked_x_high] after input projection, prepended by cond_low
        # cond_low is already in hidden_size, masked_x_high is in latent_dim
        # The transformer handles the projection internally
        
        # First, project masked_x_high through transformer's input_proj
        masked_x_high_proj = self.transformer.input_proj(masked_x_high)
        # Add positional embeddings
        masked_x_high_proj = masked_x_high_proj + self.transformer.pos_embed[:, :N, :]
        
        # Prepend low-res conditional tokens (already in hidden_size) as pivots
        transformer_input = torch.cat([cond_low, masked_x_high_proj], dim=1)
        
        # Add class embedding
        if class_idx is not None and self.transformer.num_classes is not None:
            class_emb = self.transformer.class_embed(class_idx).unsqueeze(1)
            transformer_input = torch.cat([class_emb, transformer_input], dim=1)
        
        # Add context embeddings
        if context_embeds is not None:
            transformer_input = torch.cat([context_embeds, transformer_input], dim=1)
        
        # Get scale vector (scale 1 = high-res)
        scale_vec = self.transformer.get_scale_vector(1)
        scale_vec = scale_vec.expand(B, -1)
        
        # Pass through Transformer blocks
        x = transformer_input
        for block in self.transformer.blocks:
            x = block(x, condition=scale_vec)
        
        # Final layer norm with adaLN
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.transformer.final_adaLN(scale_vec)
        cond_tokens = gamma1.unsqueeze(1) * self.transformer.final_norm(
            alpha1.unsqueeze(1) * x + beta1.unsqueeze(1)
        )
        
        # Extract conditional tokens for high-res visual positions
        # Skip: context + class + cond_low
        num_context = (
            (1 if class_idx is not None and self.transformer.num_classes is not None else 0) + 
            (context_embeds.shape[1] if context_embeds is not None else 0) + 
            cond_low.shape[1]
        )
        cond_visual = extract_condition_tokens(cond_tokens, num_context, N)
        
        # Compute diffusion loss
        loss = self.diffusion_loss(self.diffusion_head2, cond_visual, x_high, mask_pos)
        
        return loss, cond_visual, mask_pos
    
    def forward(self, x_low, x_high, class_idx=None, context_embeds=None):
        """
        Full forward pass for training.
        
        Args:
            x_low: (B, N_low, D) low-resolution tokens
            x_high: (B, N_high, D) high-resolution tokens
            class_idx: (B,) class labels
            context_embeds: (B, N_ctx, C) text embeddings
        Returns:
            total_loss, loss_dict
        """
        # Phase 1
        loss1, cond_low, _ = self.forward_phase1(x_low, class_idx, context_embeds)
        
        # Phase 2 (using conditional tokens from Phase 1 as pivots)
        loss2, cond_high, _ = self.forward_phase2(x_high, cond_low, class_idx, context_embeds)
        
        total_loss = loss1 + loss2
        
        return total_loss, {'loss_phase1': loss1, 'loss_phase2': loss2}
    
    @torch.no_grad()
    def sample_phase1(
        self,
        batch_size,
        steps=32,
        class_idx=None,
        context_embeds=None,
        cfg_scale=1.0,
        device='cuda',
    ):
        """
        Phase 1 inference: Generate low-resolution tokens.
        
        Uses iterative denoising with the MLP diffusion head.
        """
        N = self.low_res_tokens
        D = self.latent_dim
        C = self.hidden_size
        
        # Start with fully masked tokens (all mask tokens)
        mask_tok = self.mask_token.expand(batch_size, N, -1)  # (B, N, D)
        x = torch.randn(batch_size, N, D, device=device)
        
        # Cosine schedule for progressive unmasking
        for step in range(steps):
            # Get mask ratio for this step
            r = self.mask_phase2.get_mask_ratio(torch.tensor([step], device=device))
            num_masked = max(1, int(r.item() * N))
            
            # Full mask token input to Transformer for conditional tokens
            masked_input = mask_tok  # All masked
            
            # Get conditional tokens from Transformer
            cond_tokens = self.transformer(
                masked_input,
                scale_idx=0,
                class_idx=class_idx,
                context_embeds=context_embeds,
            )
            
            num_context = (1 if class_idx is not None else 0) + (context_embeds.shape[1] if context_embeds is not None else 0)
            cond_visual = extract_condition_tokens(cond_tokens, num_context, N)
            
            # CFG
            if cfg_scale > 1.0 and class_idx is not None:
                cond_tokens_uncond = self.transformer(
                    masked_input,
                    scale_idx=0,
                    class_idx=None,
                    context_embeds=context_embeds,
                )
                cond_visual_uncond = extract_condition_tokens(cond_tokens_uncond, num_context, N)
                cond_visual = cond_visual_uncond + cfg_scale * (cond_visual - cond_visual_uncond)
            
            # Denoise using diffusion head (single step at t=0 for simplicity)
            t = torch.zeros(batch_size, dtype=torch.long, device=device)
            noise_pred = self.diffusion_head1(x, t, cond_visual)
            
            # Reconstruct x_0
            sqrt_alpha_0 = self.sqrt_alphas_cumprod[0]
            sqrt_one_minus_alpha_0 = self.sqrt_one_minus_alphas_cumprod[0]
            x_pred = (x - sqrt_one_minus_alpha_0 * noise_pred) / sqrt_alpha_0.clamp(min=1e-8)
            
            # Update: keep only the most confident predictions
            # For simplicity, use the prediction directly for unmasked positions
            x = x_pred
        
        return x, cond_visual
    
    @torch.no_grad()
    def sample_phase2(
        self,
        cond_low,
        steps=4,
        class_idx=None,
        context_embeds=None,
        cfg_scale=1.0,
        device='cuda',
    ):
        """
        Phase 2 inference: Generate high-resolution tokens.
        
        Uses the Diffusion Transformer head for denoising.
        """
        B = cond_low.shape[0]
        N_high = self.high_res_tokens
        D = self.latent_dim
        N_low = self.low_res_tokens
        
        # Start with random noise
        x_high = torch.randn(B, N_high, D, device=device)
        
        # Create mask tokens for Transformer input
        mask_tok = self.mask_token.expand(B, N_high, -1)
        
        for step in range(steps):
            # Project mask tokens through input_proj and add pos embed
            mask_proj = self.transformer.input_proj(mask_tok)
            mask_proj = mask_proj + self.transformer.pos_embed[:, :N_high, :]
            
            # Build Transformer input: cond_low + mask_proj
            transformer_input = torch.cat([cond_low, mask_proj], dim=1)
            
            # Add class and context
            if class_idx is not None and self.transformer.num_classes is not None:
                class_emb = self.transformer.class_embed(class_idx).unsqueeze(1)
                transformer_input = torch.cat([class_emb, transformer_input], dim=1)
            if context_embeds is not None:
                transformer_input = torch.cat([context_embeds, transformer_input], dim=1)
            
            # Get scale vector and run through blocks
            scale_vec = self.transformer.get_scale_vector(1).expand(B, -1)
            x = transformer_input
            for block in self.transformer.blocks:
                x = block(x, condition=scale_vec)
            
            alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.transformer.final_adaLN(scale_vec)
            cond_tokens = gamma1.unsqueeze(1) * self.transformer.final_norm(
                alpha1.unsqueeze(1) * x + beta1.unsqueeze(1)
            )
            
            num_context = (
                (1 if class_idx is not None and self.transformer.num_classes is not None else 0) +
                (context_embeds.shape[1] if context_embeds is not None else 0) +
                N_low
            )
            cond_visual = extract_condition_tokens(cond_tokens, num_context, N_high)
            
            # CFG
            if cfg_scale > 1.0 and class_idx is not None:
                # Unconditional branch
                transformer_input_u = torch.cat([cond_low, mask_proj], dim=1)
                if self.transformer.num_classes is not None:
                    # Use zero embedding or skip class for unconditional
                    pass
                if context_embeds is not None:
                    transformer_input_u = torch.cat([context_embeds, transformer_input_u], dim=1)
                
                x_u = transformer_input_u
                for block in self.transformer.blocks:
                    x_u = block(x_u, condition=scale_vec)
                cond_tokens_u = gamma1.unsqueeze(1) * self.transformer.final_norm(
                    alpha1.unsqueeze(1) * x_u + beta1.unsqueeze(1)
                )
                num_context_u = (
                    (context_embeds.shape[1] if context_embeds is not None else 0) + N_low
                )
                cond_visual_u = extract_condition_tokens(cond_tokens_u, num_context_u, N_high)
                cond_visual = cond_visual_u + cfg_scale * (cond_visual - cond_visual_u)
            
            # Denoise using Diffusion Transformer head
            t = torch.zeros(B, dtype=torch.long, device=device)
            noise_pred = self.diffusion_head2(x_high, t, cond_visual)
            
            sqrt_alpha_0 = self.sqrt_alphas_cumprod[0]
            sqrt_one_minus_alpha_0 = self.sqrt_one_minus_alphas_cumprod[0]
            x_high = (x_high - sqrt_one_minus_alpha_0 * noise_pred) / sqrt_alpha_0.clamp(min=1e-8)
        
        return x_high
    
    @torch.no_grad()
    def generate(
        self,
        batch_size=1,
        class_idx=None,
        context_embeds=None,
        phase1_steps=32,
        phase2_steps=4,
        cfg_scale=1.0,
        device='cuda',
    ):
        """
        Full generation pipeline: Phase 1 → Phase 2.
        """
        # Phase 1: Generate low-resolution tokens
        x_low, cond_low = self.sample_phase1(
            batch_size=batch_size,
            steps=phase1_steps,
            class_idx=class_idx,
            context_embeds=context_embeds,
            cfg_scale=cfg_scale,
            device=device,
        )
        
        # Phase 2: Generate high-resolution tokens
        x_high = self.sample_phase2(
            cond_low=cond_low,
            steps=phase2_steps,
            class_idx=class_idx,
            context_embeds=context_embeds,
            cfg_scale=cfg_scale,
            device=device,
        )
        
        return x_low, x_high
    
    def get_param_groups(self, weight_decay=0.02):
        """Get parameter groups for optimizer with weight decay."""
        decay = set()
        no_decay = set()
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (nn.LayerNorm,)):
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        return [
            {'params': [param_dict[pn] for pn in sorted(decay)], 'weight_decay': weight_decay},
            {'params': [param_dict[pn] for pn in sorted(no_decay)], 'weight_decay': 0.0},
        ]


def create_himar_model(config):
    """Create a Hi-MAR model from a configuration dictionary."""
    model_configs = {
        'Hi-MAR-B': {
            'num_layers': 24,
            'hidden_size': 768,
            'num_heads': 12,
            'head1_num_layers': 6,
            'head1_hidden_size': 1024,
            'head2_num_layers': 6,
            'head2_hidden_size': 512,
            'head2_num_heads': 8,
        },
        'Hi-MAR-L': {
            'num_layers': 32,
            'hidden_size': 1024,
            'num_heads': 16,
            'head1_num_layers': 8,
            'head1_hidden_size': 1280,
            'head2_num_layers': 8,
            'head2_hidden_size': 512,
            'head2_num_heads': 8,
        },
        'Hi-MAR-H': {
            'num_layers': 40,
            'hidden_size': 1280,
            'num_heads': 16,
            'head1_num_layers': 12,
            'head1_hidden_size': 1536,
            'head2_num_layers': 12,
            'head2_hidden_size': 768,
            'head2_num_heads': 12,
        },
    }
    
    model_name = config.get('model_name', 'Hi-MAR-B')
    cfg = model_configs[model_name].copy()
    cfg.update({
        'num_classes': config.get('num_classes', 1000),
        'latent_dim': config.get('latent_dim', 16),
        'mask_range_phase1': config.get('mask_range_phase1', (0.7, 1.0)),
        'mask_schedule_phase2': config.get('mask_schedule_phase2', 'cosine'),
        'mask_beta_alpha': config.get('mask_beta_alpha', 4.0),
        'mask_beta_beta': config.get('mask_beta_beta', 1.0),
        'num_diffusion_steps': config.get('num_diffusion_steps', 1000),
        'cfg_drop_prob': config.get('cfg_drop_prob', 0.1),
    })
    
    return HiMAR(**cfg)
