# model/hiera.py
"""
Hiera – a hierarchical vision transformer backbone for SAM 2.

Implements the image encoder described in the SAM 2 paper (Section 4 and
Appendix D.1).  The backbone processes an RGB image of size (1024 × 1024)
through four stages, producing multi‑scale feature maps at strides 4, 8, 16 and
32.  These features are later fused by an FPN (stride 16 + 32) and used as
skip connections (stride 4, 8) by the mask decoder.

Key design choices (all align with the paper):
- absolute positional embeddings (no relative positional bias),
- window attention in most blocks, with a small set of global attention blocks
  whose indices are specified per variant,
- no 2D‑RoPE inside the image encoder (RoPE is used in memory attention),
- FlashAttention‑2 is enabled when available for global blocks.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, Mlp, PatchEmbed as TimmPatchEmbed, trunc_normal_

# Attempt to import FlashAttention for global attention blocks
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_func = None

# Try to load pretrained weights from Hugging Face Hub
try:
    from huggingface_hub import hf_hub_download
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False

# ---------------------------------------------------------------------------
# Variant configuration dictionaries
# ---------------------------------------------------------------------------
HIERA_VARIANTS: Dict[str, Dict[str, Any]] = {
    "hiera_tiny": {
        "embed_dims": [96, 192, 384, 768],
        "depths": [1, 2, 4, 2],
        "num_heads": [3, 6, 12, 24],
        "global_att_indices": [5, 7, 9],   # global indices across all blocks
    },
    "hiera_small": {
        "embed_dims": [96, 192, 384, 768],
        "depths": [2, 3, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "global_att_indices": [7, 10, 13],
    },
    "hiera_b_plus": {
        "embed_dims": [128, 256, 512, 1024],
        "depths": [2, 4, 12, 4],
        "num_heads": [4, 8, 16, 32],
        "global_att_indices": [12, 16, 20],
    },
    "hiera_large": {
        "embed_dims": [192, 384, 768, 1536],
        "depths": [3, 6, 18, 6],
        "num_heads": [6, 12, 24, 48],
        "global_att_indices": [23, 33, 43],
    },
}

# Default window size (matches 1024² input; tunable but not in config)
WINDOW_SIZE = 8

# ---------------------------------------------------------------------------
# Helper functions for window partitioning
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition a feature map into non‑overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: integer size of each window.

    Returns:
        windows: (B * num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, H: int, W: int, window_size: int) -> torch.Tensor:
    """
    Reverse the window partition.

    Args:
        windows: (B * num_windows, window_size, window_size, C)
        H, W: original spatial dimensions before partitioning.
        window_size: same as in partition.

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
# Absolute position embedding
# ---------------------------------------------------------------------------

class AbsolutePositionEmbedding(nn.Module):
    """
    A learnable 2D absolute position embedding that is added to tokens.

    The embedding has shape (1, H, W, C) and is interpolated to the actual
    feature map size at runtime if needed (though we build it for the exact
    expected size).

    In the Hiera design, this embedding is added once at the beginning of
    each stage, and the same embedding is used for all blocks within that stage.
    """

    def __init__(self, embed_dim: int, grid_size: Tuple[int, int]):
        super().__init__()
        H, W = grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, H, W, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)  # typical ViT init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add position embedding to input tensor.

        Args:
            x: (B, H, W, C) or (B, L, C) with L=H*W. If it is a flat sequence,
               we assume it can be reshaped to (B, H, W, C).

        Returns:
            x with position embedding added element‑wise. If input was flat,
            returned shape is flat.
        """
        is_flat = x.ndim == 3   # (B, L, C)
        if is_flat:
            B, L, C = x.shape
            H = int(math.sqrt(L))
            W = H
            x = x.view(B, H, W, C)

        # If the spatial size does not match, interpolate (should rarely happen
        # because we build the embedding for the expected size).
        if x.shape[1:3] != self.pos_embed.shape[1:3]:
            pos = F.interpolate(
                self.pos_embed.permute(0, 3, 1, 2),
                size=x.shape[1:3],
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        else:
            pos = self.pos_embed

        x = x + pos

        if is_flat:
            x = x.view(B, L, C)
        return x


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single transformer block used in the Hiera backbone.

    - Self‑attention can be either window‑based or global, depending on the
      `global_attention` flag.
    - Absolute position embedding is already added to the input tokens; no extra
      positional encoding is performed inside the block.
    - Attention uses standard scaled dot‑product with an option to leverage
      FlashAttention for global blocks (hardware permitting).
    - The block follows the Pre‑LN design: LayerNorm → Attention → DropPath → add,
      then LayerNorm → MLP → DropPath → add.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        global_attention: bool = False,
        window_size: int = WINDOW_SIZE,
        use_flash: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.global_attention = global_attention
        self.window_size = window_size
        self.use_flash = use_flash and HAS_FLASH_ATTN and global_attention

        # Layers
        self.norm1 = nn.LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, dim * 3)
        self.attn_proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Core attention operation. Handles both windowed and global cases.

        Args:
            x: (B, L, C) for global, or (B, H, W, C) for windowed.

        Returns:
            attended tensor of the same shape as input.
        """
        is_windows = not self.global_attention
        if is_windows:
            B, H, W, C = x.shape
            # Partition into windows: (B*Nw, window_size, window_size, C)
            x_windows = window_partition(x, self.window_size)
            # Flatten each window into a sequence: (B*Nw, window_size*window_size, C)
            B_win = x_windows.shape[0]
            x_flat = x_windows.view(B_win, -1, C)
        else:
            # Already flat: (B, L, C)
            x_flat = x

        B, L, C = x_flat.shape
        assert C % self.num_heads == 0, f"C ({C}) must be divisible by num_heads ({self.num_heads})"

        # QKV
        qkv = self.attn_qkv(x_flat)  # (B, L, 3*C)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nH, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_flash:
            # FlashAttention requires inputs to be contiguous and in shape (B, L, nH, head_dim)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            attn_out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=self.head_dim ** -0.5)
            # flash_attn returns (B, L, nH, head_dim); transpose back
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, C)
        else:
            # Standard scaled dot-product attention
            scale = self.head_dim ** -0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale
            attn_weights = attn_weights.softmax(dim=-1)
            attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, L, C)

        attn_out = self.attn_proj(attn_out)

        if is_windows:
            # Reverse window partition
            attn_out = attn_out.view(B_win, self.window_size, self.window_size, C)
            attn_out = window_reverse(attn_out, H, W, self.window_size)
        return attn_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: For global attention, shape (B, L, C) with L=H*W.
               For window attention, shape (B, H, W, C).

        Returns:
            output of the same shape as input.
        """
        shortcut = x

        # LayerNorm may be applied to flat or spatial; adapt accordingly
        if self.global_attention:
            # x: (B, L, C)
            x = self.norm1(x)
        else:
            # x: (B, H, W, C) => treat as (B, L, C) for LayerNorm, then revert
            B, H, W, C = x.shape
            x = self.norm1(x.view(B, -1, C)).view(B, H, W, C)

        x = self._attention(x)
        x = self.drop_path(x) + shortcut

        shortcut = x
        if self.global_attention:
            x = self.norm2(x)
        else:
            B, H, W, C = x.shape
            x = self.norm2(x.view(B, -1, C)).view(B, H, W, C)

        # MLP: for global, directly; for window, flatten, MLP, reshape
        if self.global_attention:
            x = self.mlp(x)
        else:
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.mlp(x)
            x = x.view(B, H, W, C)

        x = self.drop_path(x) + shortcut
        return x


# ---------------------------------------------------------------------------
# Patch Embedding (initial)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Converts the input image into a sequence of tokens with stride 4.

    Uses a single conv2d kernel of size 4×4 and stride 4, followed by LayerNorm.
    """

    def __init__(self, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            tokens of shape (B, L, embed_dim) where L = (H/4)*(W/4).
        """
        x = self.proj(x)   # (B, embed_dim, H/4, W/4)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, L, embed_dim)
        x = self.norm(x)
        return x, (H, W)   # return spatial dimensions for later reshaping


# ---------------------------------------------------------------------------
# Patch Merging (downsampling between stages)
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """
    Downsamples the spatial resolution by a factor of 2 and changes the channel
    dimension from `in_dim` to `out_dim`.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.reduction = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) or (B, L, C). If flat, we assume it can be reshaped to
               (B, H, W, C) where H==W.

        Returns:
            tokens of shape (B, new_L, out_dim).
        """
        is_flat = x.ndim == 3
        if is_flat:
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.reduction(x)       # (B, out_dim, H/2, W/2)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, L_new, out_dim)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Hiera stage
# ---------------------------------------------------------------------------

class HieraStage(nn.Module):
    """
    A single stage of the Hiera backbone, consisting of a sequence of
    TransformerBlocks.  Absolute position embedding is added once at the start.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        global_indices_stage: List[int],
        grid_size: Tuple[int, int],
        drop_path_rates: List[float],
        window_size: int = WINDOW_SIZE,
    ):
        super().__init__()
        self.dim = dim
        self.pos_embed = AbsolutePositionEmbedding(dim, grid_size)

        blocks = []
        for i in range(depth):
            is_global = i in global_indices_stage
            blocks.append(
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    drop_path=drop_path_rates[i],
                    global_attention=is_global,
                    window_size=window_size,
                    use_flash=is_global,   # flash only for global to keep window simple
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: flat tokens (B, L, C) corresponding to the input of this stage.
               L equals H*W where H,W are the spatial dimensions.

        Returns:
            flat tokens of the same shape after the stage.
        """
        # Add absolute position embedding (supports both flat and spatial, but here we keep spatial)
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C)
        x = self.pos_embed(x)   # (B, H, W, C)

        for blk in self.blocks:
            if blk.global_attention:
                # Need flat sequence for global attention
                x = x.view(B, -1, C)
                x = blk(x)
                x = x.view(B, H, W, C)
            else:
                x = blk(x)   # stays (B, H, W, C) for window attention

        # Return flat sequence for consistency
        x = x.view(B, L, C)
        return x


# ---------------------------------------------------------------------------
# Main Hiera module
# ---------------------------------------------------------------------------

class Hiera(nn.Module):
    """
    Hiera – Hierarchical Vision Transformer for SAM 2.

    Build one of four variants (tiny, small, B+, large).  The variant specific
    parameters are taken from :data:`HIERA_VARIANTS`.

    Args:
        variant: one of ``"hiera_tiny"``, ``"hiera_small"``, ``"hiera_b_plus"``, ``"hiera_large"``.
        pretrained: if ``True``, load official MAE‑pretrained weights via Hugging Face Hub
            (or a local path if provided in ``pretrained_path``).
        pretrained_path: optional local path to a checkpoint file.
    """

    def __init__(
        self,
        variant: str = "hiera_b_plus",
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if variant not in HIERA_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose one of {list(HIERA_VARIANTS.keys())}")
        cfg = HIERA_VARIANTS[variant]
        self.embed_dims = cfg["embed_dims"]
        self.depths = cfg["depths"]
        self.num_heads = cfg["num_heads"]
        self.global_att_indices = set(cfg["global_att_indices"])

        self.num_stages = 4
        window_size = WINDOW_SIZE

        # Patch embedding (stride 4)
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=self.embed_dims[0])

        # Build stages and mergings
        self.stages = nn.ModuleList()
        self.mergings = nn.ModuleList()

        # Drop path rates: linearly increase from 0 to pre-defined max per stage
        # The paper mentions drop_path values 0.1 (T/S), 0.2 (B+), 0.3 (L) but these are
        # applied during pretraining; here we set a mild schedule for reproducibility.
        drop_path_max = 0.1 if variant.startswith("hiera_tiny") else \
                        0.2 if variant == "hiera_b_plus" else \
                        0.3 if variant == "hiera_large" else 0.15  # safe default
        total_blocks = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_max, total_blocks)]

        # We'll keep an absolute block counter to identify global attention blocks.
        block_idx_abs = 0

        # Stage 1 does not have a merging before it.
        # The feature map size for stage 1 is determined by image size: 1024 -> stride4 -> 256.
        grid_size_1 = (1024 // 4, 1024 // 4)   # 256x256
        global_indices_stage1 = [idx for idx in range(block_idx_abs, block_idx_abs + self.depths[0])
                                 if idx in self.global_att_indices]
        dpr_stage1 = dpr[block_idx_abs:block_idx_abs + self.depths[0]]
        block_idx_abs += self.depths[0]
        self.stages.append(
            HieraStage(
                dim=self.embed_dims[0],
                num_heads=self.num_heads[0],
                depth=self.depths[0],
                global_indices_stage=global_indices_stage1,
                grid_size=grid_size_1,
                drop_path_rates=dpr_stage1,
                window_size=window_size,
            )
        )

        # For stages 2..4, insert a PatchMerging before each.
        for s in range(1, self.num_stages):
            merging = PatchMerging(in_dim=self.embed_dims[s - 1], out_dim=self.embed_dims[s])
            self.mergings.append(merging)

            # Determine grid size: each merging halves spatial dims.
            H = 1024 // (4 * (2 ** s))
            W = H
            grid_size = (H, W)

            global_indices_stage = [idx for idx in range(block_idx_abs, block_idx_abs + self.depths[s])
                                    if idx in self.global_att_indices]
            dpr_stage = dpr[block_idx_abs:block_idx_abs + self.depths[s]]
            block_idx_abs += self.depths[s]

            self.stages.append(
                HieraStage(
                    dim=self.embed_dims[s],
                    num_heads=self.num_heads[s],
                    depth=self.depths[s],
                    global_indices_stage=global_indices_stage,
                    grid_size=grid_size,
                    drop_path_rates=dpr_stage,
                    window_size=window_size,
                )
            )

        self.apply(self._init_weights)

        if pretrained:
            self._load_pretrained(variant, pretrained_path)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _load_pretrained(self, variant: str, path: Optional[str]) -> None:
        """
        Load MAE‑pretrained Hiera weights from Hugging Face Hub or a local path.

        The expected Hub repository pattern is:
            "facebook/sam2-hiera-<variant>"
        and the checkpoint file is "model.pth" or "pytorch_model.bin".
        """
        if path is None and HAS_HUGGINGFACE:
            repo_id = f"facebook/sam2-hiera-{variant}"
            # The SAM 2 repo stores the Hiera checkpoints inside the sam2/ directory,
            # but the public HF repo might be different.  The paper states the code is
            # released at https://github.com/facebookresearch/sam2.  The HF Hub repo
            # "facebook/sam2-hiera-<variant>" may not exist yet; we provide a fallback.
            # For now, we assume the community or Meta will provide such a repo.
            try:
                path = hf_hub_download(repo_id=repo_id, filename="model.pth")
            except Exception as e:
                warnings.warn(
                    f"Could not download pretrained weights from Hugging Face Hub ({e}). "
                    "Training will proceed with random initialization."
                )
                return

        if path is not None and os.path.isfile(path):
            checkpoint = torch.load(path, map_location="cpu")
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Map keys if necessary (official checkpoint keys may differ slightly)
            state_dict = self._remap_pretrained_keys(state_dict)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"[Hiera] Loaded pretrained weights from {path}")
            if missing:
                print(f"  Missing keys: {len(missing)} (e.g. {missing[:3]})")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        else:
            warnings.warn(f"No pretrained checkpoint found at {path}. Starting from scratch.")

    def _remap_pretrained_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Adapt the checkpoint keys to match our module naming.
        This function is generic; actual mapping will depend on the official release.
        If the keys already match, it returns the state_dict unchanged.
        The following is a placeholder – the exact mapping must be verified against the
        public checkpoint.
        """
        new_state = {}
        for k, v in state_dict.items():
            # Example adaptation: remove "backbone." prefix if present
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            # Our patch_embed.proj.weight is a Conv2d; some checkpoints store it as "patch_embed.proj.weight"
            # If the keys match exactly, no change needed.
            new_state[k] = v
        return new_state

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process an RGB image and return multi‑scale feature maps.

        Args:
            x: (B, 3, 1024, 1024) image tensor (expected to be normalized [0,1]).
               The resolution must be exactly 1024×1024 for the pre‑computed grid sizes.

        Returns:
            dict with keys:
                "stage1": (B, C1, 256, 256)
                "stage2": (B, C2, 128, 128)
                "stage3": (B, C3,  64,  64)
                "stage4": (B, C4,  32,  32)
        """
        B, C, H_img, W_img = x.shape
        if H_img != 1024 or W_img != 1024:
            raise ValueError(f"Hiera expects input of size 1024x1024, got {H_img}x{W_img}")

        # Stage 1
        x, (H1, W1) = self.patch_embed(x)  # (B, L, C1), L = 256*256 = 65536
        x = self.stages[0](x)               # (B, L, C1)

        # Reshape to spatial for feature map output
        feat1 = x.view(B, H1, W1, self.embed_dims[0])
        feat1 = feat1.permute(0, 3, 1, 2).contiguous()  # (B, C1, H1, W1)

        # Intermediate features are stored in a list, then reshaped at the end
        features = {"stage1": feat1}
        current_tokens = x  # keep the flat tokens for next merging

        # Stages 2–4
        for stage_idx in range(1, self.num_stages):
            # Apply PatchMerging (input flat tokens, output flat tokens)
            current_tokens = self.mergings[stage_idx - 1](current_tokens)  # (B, L_new, C_stage)
            current_tokens = self.stages[stage_idx](current_tokens)        # (B, L_new, C_stage)

            # Determine new spatial size: each merging halves resolution
            H_new = 1024 // (4 * (2 ** stage_idx))
            W_new = H_new
            feat = current_tokens.view(B, H_new, W_new, self.embed_dims[stage_idx])
            feat = feat.permute(0, 3, 1, 2).contiguous()  # (B, C_stage, H_new, W_new)
            features[f"stage{stage_idx+1}"] = feat

        return features


# ---------------------------------------------------------------------------
# Quick test (when executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple shape check
    variant = "hiera_b_plus"
    model = Hiera(variant=variant, pretrained=False)
    img = torch.randn(2, 3, 1024, 1024)
    outs = model(img)
    for k, v in outs.items():
        print(f"{k}: {v.shape}")
    # Expected:
    # stage1: (2, 128, 256, 256)
    # stage2: (2, 256, 128, 128)
    # stage3: (2, 512,  64,  64)
    # stage4: (2, 1024, 32,  32)
