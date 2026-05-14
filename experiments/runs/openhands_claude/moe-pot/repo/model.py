from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import ModelConfig, get_model_config
from layers import PatchEmbedding, TemporalAggregation
from modules import MoEBlock, OutputProjection


class MoEPOT(nn.Module):
    """Mixture-of-Experts Pre-training Operator Transformer (MoE-POT).

    Architecture (Section 4):
    1. Patchification + positional encoding for each input timestep
    2. Temporal aggregation across T frames
    3. N MoE blocks (each: FourierLayer + MoELayer)
    4. Output projection back to original spatial resolution

    The model auto-regressively predicts the next frame given T previous frames.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Number of Fourier modes (use half the patch-grid size)
        patch_grid = cfg.spatial_size // cfg.patch_size
        num_modes = patch_grid // 2

        # 1. Patchification layer (shared across timesteps)
        self.patch_embed = PatchEmbedding(
            in_channels=cfg.in_channels,
            embed_dim=cfg.attention_dim,
            patch_size=cfg.patch_size,
            num_timesteps=cfg.num_timesteps,
            spatial_size=cfg.spatial_size,
        )

        # 2. Temporal aggregation
        self.temporal_agg = TemporalAggregation(
            embed_dim=cfg.attention_dim,
            num_timesteps=cfg.num_timesteps,
        )

        # 3. N MoE blocks
        self.blocks = nn.ModuleList(
            [
                MoEBlock(
                    channels=cfg.attention_dim,
                    num_heads=cfg.num_heads,
                    num_routed_experts=cfg.num_routed_experts,
                    num_shared_experts=cfg.num_shared_experts,
                    top_k=cfg.top_k,
                    load_balance_weight=cfg.load_balance_weight,
                    num_modes_h=num_modes,
                    num_modes_w=num_modes,
                    expert_hidden_channels=cfg.mlp_dim,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        # 4. Output projection
        self.output_proj = OutputProjection(
            embed_dim=cfg.attention_dim,
            out_channels=cfg.out_channels,
            patch_size=cfg.patch_size,
        )

    def forward(
        self, u_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u_frames: (B, T, C, H, W) — T input frames

        Returns:
            pred:         (B, C, H, W) — predicted next frame
            balance_loss: scalar — sum of load-balancing losses across all blocks
        """
        B, T, C, H, W = u_frames.shape

        # --- Step 1: Patchify each timestep ---
        patch_features: List[torch.Tensor] = []
        for t in range(T):
            z_t = self.patch_embed(u_frames[:, t], t)  # (B, D, Hp, Wp)
            patch_features.append(z_t)

        # --- Step 2: Temporal aggregation ---
        z = self.temporal_agg(patch_features)  # (B, D, Hp, Wp)

        # --- Step 3: MoE blocks ---
        balance_loss = torch.tensor(0.0, device=u_frames.device)
        for block in self.blocks:
            z = block(z)
            balance_loss = balance_loss + block.balance_loss

        # --- Step 4: Output projection ---
        pred = self.output_proj(z)  # (B, C_out, H, W)
        return pred, balance_loss

    def get_router_weights(self, u_frames: torch.Tensor) -> List[torch.Tensor]:
        """Return softmax router weights for each block (for interpretability).

        Args:
            u_frames: (B, T, C, H, W)

        Returns:
            List of (B, N_r) tensors, one per block.
        """
        B, T, C, H, W = u_frames.shape

        patch_features = [self.patch_embed(u_frames[:, t], t) for t in range(T)]
        z = self.temporal_agg(patch_features)

        router_weights = []
        for block in self.blocks:
            # Fourier layer output (input to MoE layer)
            z0 = block.fourier_layer(z)
            # Extract router weights before running the full MoE layer
            logits = block.moe_layer.router(z0)
            weights = torch.softmax(logits, dim=-1)  # (B, N_r)
            router_weights.append(weights.detach())
            # Continue forward pass through MoE layer
            z = block.moe_layer(z0)

        return router_weights

    def freeze_router(self) -> None:
        """Freeze all router-gating network parameters (used during fine-tuning)."""
        for block in self.blocks:
            for param in block.moe_layer.router.parameters():
                param.requires_grad_(False)

    def unfreeze_router(self) -> None:
        """Unfreeze all router-gating network parameters."""
        for block in self.blocks:
            for param in block.moe_layer.router.parameters():
                param.requires_grad_(True)

    def count_parameters(self) -> Dict[str, int]:
        """Count total and activated parameters."""
        total = sum(p.numel() for p in self.parameters())

        # Activated params = total - (N_r - K) * params_per_routed_expert * num_blocks
        params_per_expert = sum(
            p.numel() for p in self.blocks[0].moe_layer.routed_experts[0].parameters()
        )
        num_blocks = len(self.blocks)
        non_activated_per_block = (self.cfg.num_routed_experts - self.cfg.top_k) * params_per_expert
        activated = total - num_blocks * non_activated_per_block

        return {"total": total, "activated": activated}


def build_model(size: str = "tiny") -> MoEPOT:
    """Build a MoE-POT model of the given size."""
    cfg = get_model_config(size)
    return MoEPOT(cfg)
