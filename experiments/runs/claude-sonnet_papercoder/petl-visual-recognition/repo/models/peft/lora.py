## Code: models/peft/lora.py

```python
## models/peft/lora.py
"""LoRA (Low-Rank Adaptation) implementation for the PEFT Visual Recognition study.

This module implements LoRA as described in Section 2.2 and Appendix B.2.3 of:

    "Lessons and Insights from a Unifying Study of Parameter-Efficient
    Fine-Tuning (PEFT) in Visual Recognition"

LoRA applies low-rank decomposition to the Q and V projection weights within
the MSA block of each Transformer layer. The K projection is left unchanged.

Paper equations (Appendix B.2.3):
    W_Q/V + ΔW_Q/V = W_Q/V + W_down^{Q/V} · W_up^{Q/V}
    h₃ = LoRA(h₂) + h₃
    LoRA(h₂) = [W_down^Q · W_up^Q · h₂, 0, W_down^V · W_up^V · h₂]

Initialization:
    W_down: kaiming_uniform_ (standard He init)
    W_up:   zeros → ΔW = 0 at init → model starts from pretrained behavior

Config reference (config.yaml):
    peft_methods.lora.apply_to: [query, value]
    peft_methods.lora.search_grid.lora_rank: [1, 8, 16, 32]
    peft_methods.lora.params_range_M: [0.036, 1.179]
    backbones.imagenet21k_vit.embed_dim: 768
    backbones.imagenet21k_vit.num_layers: 12

Parameter count verification (12 layers × 4 matrices × rank × embed_dim):
    rank=1:  12 × 4 × 1  × 768 =    36,864 ≈ 0.037M ✓
    rank=8:  12 × 4 × 8  × 768 =   294,912 ≈ 0.295M ✓
    rank=16: 12 × 4 × 16 × 768 =   589,824 ≈ 0.590M ✓
    rank=32: 12 × 4 × 32 × 768 = 1,179,648 ≈ 1.180M ✓ (within 1.5% cap)

Typical usage (called by PEFTFactory):
    import copy
    backbone = copy.deepcopy(vit_wrapper.get_backbone())
    # freeze_backbone() called before this on the copy

    lora_module = LoRAModule(rank=8, embed_dim=768, num_layers=12)
    lora_module.apply_to_backbone(backbone)

    # LoRA parameters are now part of backbone's module tree via attn.qkv
    # replacement. get_params() returns only W_down/W_up parameters.
    trainable = lora_module.get_params()
"""

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture constants from config.yaml: backbones.imagenet21k_vit
# ---------------------------------------------------------------------------

# Token embedding dimension D for ViT-B/16.
# config.yaml: backbones.imagenet21k_vit.embed_dim: 768
_DEFAULT_EMBED_DIM: int = 768

# Number of Transformer layers M for ViT-B/16.
# config.yaml: backbones.imagenet21k_vit.num_layers: 12
_DEFAULT_NUM_LAYERS: int = 12

# LoRA is applied to Q and V projections only (K is unchanged).
# config.yaml: peft_methods.lora.apply_to: [query, value]
_APPLY_TO_Q: bool = True
_APPLY_TO_V: bool = True
_APPLY_TO_K: bool = False  # K projection is NOT modified per paper


# ===========================================================================
# LoRALinear: drop-in replacement for a single nn.Linear with LoRA residual
# ===========================================================================

class LoRALinear(nn.Module):
    """Low-rank adaptation of a single nn.Linear layer.

    Replaces a frozen pretrained Linear layer with a LoRA-augmented version:
        output = F.linear(x, original_weight, original_bias)   # frozen path
                 + F.linear(F.linear(x, W_down), W_up)         # low-rank residual

    The original weight and bias are stored as frozen parameters
    (requires_grad=False). Only W_down and W_up are trainable.

    Initialization ensures identity residual at the start of training:
        W_down: kaiming_uniform_ (He init, a=sqrt(5))
        W_up:   zeros → W_up @ W_down = 0 → output = F.linear(x, original_weight)

    This class is used as a building block within LoRAFusedQKV and can also
    be used standalone for individual Q or V projection replacement when the
    backbone uses separate Q/K/V projections.

    Attributes:
        in_features: Input dimension (D = 768 for ViT-B/16 Q/V projections).
        out_features: Output dimension (D = 768 for ViT-B/16 Q/V projections).
        rank: LoRA bottleneck rank r. Must satisfy r < min(in_features, out_features).
        original_weight: Frozen pretrained weight of shape (out_features, in_features).
            Stored as nn.Parameter with requires_grad=False.
        original_bias: Frozen pretrained bias of shape (out_features,), or None.
            Stored as nn.Parameter with requires_grad=False if present.
        W_down: Trainable down-projection of shape (rank, in_features).
            Initialized with kaiming_uniform_.
        W_up: Trainable up-projection of shape (out_features, rank).
            Initialized to zeros (ensures ΔW = 0 at init).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        original_weight: torch.Tensor,
        original_bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialises LoRALinear with frozen pretrained weights and trainable LoRA params.

        Args:
            in_features: Input feature dimension. For ViT-B/16 Q/V: D = 768.
            out_features: Output feature dimension. For ViT-B/16 Q/V: D = 768.
            rank: LoRA bottleneck rank r. Search grid: {1, 8, 16, 32}
                (config.yaml: peft_methods.lora.search_grid.lora_rank).
                Must satisfy 0 < rank < min(in_features, out_features).
            original_weight: Pretrained weight tensor of shape
                (out_features, in_features). Cloned and stored as a frozen
                parameter (requires_grad=False).
            original_bias: Pretrained bias tensor of shape (out_features,),
                or None if the original layer had no bias. Cloned and stored
                as a frozen parameter if not None.

        Raises:
            ValueError: If rank <= 0 or rank >= min(in_features, out_features).
            ValueError: If original_weight shape does not match
                (out_features, in_features).
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Input validation.
        # ------------------------------------------------------------------
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if rank >= min(in_features, out_features):
            raise ValueError(
                f"rank ({rank}) must be less than min(in_features={in_features}, "
                f"out_features={out_features}) = {min(in_features, out_features)} "
                "to achieve parameter efficiency."
            )
        expected_weight_shape: tuple = (out_features, in_features)
        if tuple(original_weight.shape) != expected_weight_shape:
            raise ValueError(
                f"original_weight shape {tuple(original_weight.shape)} does not match "
                f"expected shape {expected_weight_shape} "
                f"(out_features={out_features}, in_features={in_features})."
            )

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = rank

        # ------------------------------------------------------------------
        # Store frozen pretrained weight as nn.Parameter with requires_grad=False.
        # Using nn.Parameter (not register_buffer) so it appears in state_dict()
        # and can be loaded/saved correctly. requires_grad=False prevents
        # optimizer updates.
        # ------------------------------------------------------------------
        self.original_weight: nn.Parameter = nn.Parameter(
            original_weight.clone().float(),
            requires_grad=False,
        )

        # Store frozen bias if present.
        if original_bias is not None:
            if original_bias.shape != (out_features,):
                raise ValueError(
                    f"original_bias shape {tuple(original_bias.shape)} does not match "
                    f"expected shape ({out_features},)."
                )
            self.original_bias: Optional[nn.Parameter] = nn.Parameter(
                original_bias.clone().float(),
                requires_grad=False,
            )
        else:
            self.original_bias = None

        # ------------------------------------------------------------------
        # Trainable LoRA parameters.
        # W_down: (rank, in_features) — down-projection
        # W_up:   (out_features, rank) — up-projection
        # ------------------------------------------------------------------
        self.W_down: nn.Parameter = nn.Parameter(
            torch.empty(rank, in_features, dtype=torch.float32),
            requires_grad=True,
        )
        self.W_up: nn.Parameter = nn.Parameter(
            torch.empty(out_features, rank, dtype=torch.float32),
            requires_grad=True,
        )

        # ------------------------------------------------------------------
        # Identity-preserving initialization:
        # W_down: kaiming_uniform_ with a=sqrt(5) (matches PyTorch Linear default)
        # W_up:   zeros → W_up @ W_down = 0 → ΔW = 0 at init
        # ------------------------------------------------------------------
        nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))
        nn.init.zeros_(self.W_up)

        _logger.debug(
            "LoRALinear initialised: in=%d, out=%d, rank=%d, "
            "has_bias=%s",
            in_features,
            out_features,
            rank,
            original_bias is not None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the LoRA-augmented linear transformation.

        Computes:
            output = F.linear(x, original_weight, original_bias)  # frozen pretrained
                     + F.linear(F.linear(x, W_down), W_up)        # low-rank residual

        The low-rank residual path:
            F.linear(x, W_down) computes x @ W_down.T → shape (..., rank)
            F.linear(..., W_up) computes ... @ W_up.T  → shape (..., out_features)
        This implements W_up @ W_down @ x in the linear algebra sense.

        Args:
            x: Input tensor of shape (..., in_features). For ViT attention:
                shape (B, N+1, D) where B=batch, N+1=197 tokens, D=768.

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Frozen pretrained path: x @ original_weight.T + original_bias
        pretrained_out: torch.Tensor = F.linear(
            x, self.original_weight, self.original_bias
        )

        # Low-rank residual path: W_up @ W_down @ x
        # Step 1: x @ W_down.T → (..., rank)
        down: torch.Tensor = F.linear(x, self.W_down)
        # Step 2: down @ W_up.T → (..., out_features)
        lora_delta: torch.Tensor = F.linear(down, self.W_up)

        return pretrained_out + lora_delta


# ===========================================================================
# LoRAFusedQKV: handles timm's fused QKV projection with LoRA on Q and V
# ===========================================================================

class LoRAFusedQKV(nn.Module):
    """LoRA adaptation for timm's fused QKV projection layer.

    timm's ViT uses a single fused Linear(D, 3D) layer for Q, K, V projections
    rather than three separate layers. This class replaces that fused layer
    with a LoRA-augmented version that applies LoRA only to the Q and V slices
    of the weight matrix, leaving K unchanged.

    The fused weight matrix W_qkv ∈ ℝ^{3D×D} is partitioned as:
        W_qkv[:D, :]   = W_Q  (Q projection weight)
        W_qkv[D:2D, :] = W_K  (K projection weight — NOT modified)
        W_qkv[2D:, :]  = W_V  (V projection weight)

    Forward computation:
        qkv = F.linear(x, W_qkv, bias_qkv)          # frozen pretrained path
        q_delta = W_up_Q @ W_down_Q @ x              # LoRA delta for Q
        v_delta = W_up_V @ W_down_V @ x              # LoRA delta for V
        qkv[:, :, :D]  += q_delta                    # add to Q slice
        qkv[:, :, 2D:] += v_delta                    # add to V slice

    Paper: "LoRA update methodology is strategically applied to the
    Query/Value projection weights W_{Q/V} within the MSA block."
    (Appendix B.2.3)

    Config: config.yaml -> peft_methods.lora.apply_to: [query, value]

    Attributes:
        embed_dim: Per-head embedding dimension D (768 for ViT-B/16).
        rank: LoRA bottleneck rank r.
        original_weight: Frozen fused QKV weight of shape (3D, D).
        original_bias: Frozen fused QKV bias of shape (3D,), or None.
        W_down_Q: Trainable Q down-projection of shape (rank, D).
        W_up_Q: Trainable Q up-projection of shape (D, rank).
        W_down_V: Trainable V down-projection of shape (rank, D).
        W_up_V: Trainable V up-projection of shape (D, rank).
    """

    def __init__(
        self,
        embed_dim: int,
        rank: int,
        original_weight: torch.Tensor,
        original_bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialises LoRAFusedQKV with frozen QKV weights and trainable LoRA params.

        Args:
            embed_dim: Per-projection embedding dimension D. For ViT-B/16: 768.
                The fused weight has shape (3*embed_dim, embed_dim).
            rank: LoRA bottleneck rank r. Search grid: {1, 8, 16, 32}
                (config.yaml: peft_methods.lora.search_grid.lora_rank).
            original_weight: Pretrained fused QKV weight of shape (3D, D).
                Cloned and stored as a frozen parameter (requires_grad=False).
            original_bias: Pretrained fused QKV bias of shape (3D,), or None.
                Cloned and stored as a frozen parameter if not None.

        Raises:
            ValueError: If rank <= 0 or rank >= embed_dim.
            ValueError: If original_weight shape does not match (3*embed_dim, embed_dim).
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Input validation.
        # ------------------------------------------------------------------
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if rank >= embed_dim:
            raise ValueError(
                f"rank ({rank}) must be less than embed_dim ({embed_dim}) "
                "to achieve parameter efficiency."
            )

        fused_out_dim: int = 3 * embed_dim
        expected_weight_shape: tuple = (fused_out_dim, embed_dim)
        if tuple(original_weight.shape) != expected_weight_shape:
            raise ValueError(
                f"original_weight shape {tuple(original_weight.shape)} does not match "
                f"expected fused QKV shape {expected_weight_shape} "
                f"(3*embed_dim={fused_out_dim}, embed_dim={embed_dim})."
            )

        self.embed_dim: int = embed_dim
        self.rank: int = rank

        # ------------------------------------------------------------------
        # Store frozen pretrained fused QKV weight and bias.
        # ------------------------------------------------------------------
        self.original_weight: nn.Parameter = nn.Parameter(
            original_weight.clone().float(),
            requires_grad=False,
        )

        if original_bias is not None:
            if original_bias.shape != (fused_out_dim,):
                raise ValueError(
                    f"original_bias shape {tuple(original_bias.shape)} does not match "
                    f"expected shape ({fused_out_dim},)."
                )
            self.original_bias: Optional[nn.Parameter] = nn.Parameter(
                original_bias.clone().float(),
                requires_grad=False,
            )
        else:
            self.original_bias = None

        # ------------------------------------------------------------------
        # Trainable LoRA parameters for Q projection.
        # W_down_Q: (rank, D) — down-projection for Q
        # W_up_Q:   (D, rank) — up-projection for Q
        # ------------------------------------------------------------------
        self.W_down_Q: nn.Parameter = nn.Parameter(
            torch.empty(rank, embed_dim, dtype=torch.float32),
            requires_grad=True,
        )
        self.W_up_Q: nn.Parameter = nn.Parameter(
            torch.empty(embed_dim, rank, dtype=torch.float32),
            requires_grad=True,
        )

        # ------------------------------------------------------------------
        # Trainable LoRA parameters for V projection.
        # W_down_V: (rank, D) — down-projection for V
        # W_up_V:   (D, rank) — up-projection for V
        # ------------------------------------------------------------------
        self.W_down_V: nn.Parameter = nn.Parameter(
            torch.empty(rank, embed_dim, dtype=torch.float32),
            requires_grad=True,
        )
        self.W_up_V: nn.Parameter = nn.Parameter(
            torch.empty(embed_dim, rank, dtype=torch.float32),
            requires_grad=True,
        )

        # ------------------------------------------------------------------
        # Identity-preserving initialization:
        # W_down_Q/V: kaiming_uniform_ (He init, a=sqrt(5))
        # W_up_Q/V:   zeros → ΔW = 0 at init
        # ------------------------------------------------------------------
        nn.init.kaiming_uniform_(self.W_down_Q, a=math.sqrt(5))
        nn.init.zeros_(self.W_up_Q)
        nn.init.kaiming_uniform_(self.W_down_V, a=math.sqrt(5))
        nn.init.zeros_(self.W_up_V)

        _logger.debug(
            "LoRAFusedQKV initialised: embed_dim=%d, rank=%d, "
            "has_bias=%s, trainable_params=%d",
            embed_dim,
            rank,
            original_bias is not None,
            4 * rank * embed_dim,  # W_down_Q + W_up_Q + W_down_V + W_up_V
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the LoRA-augmented fused QKV projection.

        Computes the full QKV output with LoRA deltas added to Q and V slices:
            qkv = F.linear(x, original_weight, original_bias)  # (B, N+1, 3D)
            q_delta = W_up_Q @ W_down_Q @ x                    # (B, N+1, D)
            v_delta = W_up_V @ W_down_V @ x                    # (B, N+1, D)
            qkv[..., :D]  += q_delta   # add LoRA delta to Q slice
            qkv[..., 2D:] += v_delta   # add LoRA delta to V slice

        The K slice (qkv[..., D:2D]) is unchanged (no LoRA applied to K).

        Paper: "LoRA(h₂) = [W_down^Q · W_up^Q · h₂, 0, W_down^V · W_up^V · h₂]"
        (Appendix B.2.3, Equation 18)

        Args:
            x: Input tensor of shape (B, N+1, D) where:
                B = batch size, N+1 = 197 tokens (1 CLS + 196 patches), D = 768.

        Returns:
            Fused QKV output tensor of shape (B, N+1, 3D) = (B, 197, 2304).
            Q slice: qkv[..., :D] includes LoRA delta.
            K slice: qkv[..., D:2D] is unchanged (pretrained only).
            V slice: qkv[..., 2D:] includes LoRA delta.
        """
        # ------------------------------------------------------------------
        # Step 1: Frozen pretrained fused QKV projection.
        # x: (B, N+1, D) → qkv: (B, N+1, 3D)
        # ------------------------------------------------------------------
        qkv: torch.Tensor = F.linear(x, self.original_weight, self.original_bias)

        # ------------------------------------------------------------------
        # Step 2: Compute LoRA delta for Q projection.
        # W_up_Q @ W_down_Q @ x:
        #   F.linear(x, W_down_Q) → (B, N+1, rank)
        #   F.linear(..., W_up_Q) → (B, N+1, D)
        # ------------------------------------------------------------------
        q_down: torch.Tensor = F.linear(x, self.W_down_Q)   # (B, N+1, rank)
        q_delta: torch.Tensor = F.linear(q_down, self.W_up_Q)  # (B, N+1, D)

        # ------------------------------------------------------------------
        # Step 3: Compute LoRA delta for V projection.
        # W_up_V @ W_down_V @ x:
        #   F.linear(x, W_down_V) → (B, N+1, rank)
        #   F.linear(..., W_up_V) → (B, N+1, D)
        # ------------------------------------------------------------------
        v_down: torch.Tensor = F.linear(x, self.W_down_V)   # (B, N+1, rank)
        v_delta: torch.Tensor = F.linear(v_down, self.W_up_V)  # (B, N+1, D)

        # ------------------------------------------------------------------
        # Step 4: Add LoRA deltas to Q and V slices of the fused QKV output.
        # Use clone() to avoid in-place modification of the computation graph.
        # ------------------------------------------------------------------
        qkv = qkv.clone()
        qkv[..., :self.embed_dim] = qkv[..., :self.embed_dim] + q_delta
        qkv[..., 2 * self.embed_dim:] = qkv[..., 2 * self.embed_dim:] + v_delta

        return qkv

    def get_lora_params(self) -> List[nn.Parameter]:
        """Returns the trainable LoRA parameters for this layer.

        Returns:
            List of 4 nn.Parameter objects:
            [W_down_Q, W_up_Q, W_down_V, W_up_V]
        """
        return [self.W_down_Q, self.W_up_Q, self.W_down_V, self.W_up_V]

    def get_merged_weight(self) -> torch.Tensor:
        """Computes the merged QKV weight for inference-time re-parameterization.

        Merges LoRA deltas into the original weight:
            W_Q_merged = W_Q_original + W_up_Q @ W_down_Q
            W_V_merged = W_V_original + W_up_V @ W_down_V
            W_K unchanged

        Returns:
            Merged weight tensor of shape (3D, D) with LoRA deltas absorbed.
            Can be used to replace this module with a standard nn.Linear
            for zero-overhead inference.
        """
        with torch.no_grad():
            merged: torch.Tensor = self.original_weight.data.clone()

            # Merge Q: W_up_Q @ W_down_Q → shape (D, D)
            # W_up_Q: (D, rank), W_down_Q: (rank, D) → product: (D, D)
            q_delta_weight: torch.Tensor = self.W_up_Q @ self.W_down_Q
            merged[:self.embed_dim, :] = merged[:self.embed_dim, :] + q_delta_weight

            # Merge V: W_up_V @ W_down_V → shape (D, D)
            v_delta_weight: torch.Tensor = self.W_up_V @ self.W_down_V
            merged[2 * self.embed_dim:, :] = (
                merged[2 * self.embed_dim:, :] + v_delta_weight
            )

        return merged


# ===========================================================================
# LoRAModule: manages LoRA application across all Transformer blocks
# ===========================================================================

class LoRAModule(nn.Module):
    """Manages LoRA application across all Transformer blocks of a ViT backbone.

    Coordinates the replacement of fused QKV projection layers in all 12
    Transformer blocks with LoRAFusedQKV instances. After apply_to_backbone()
    is called, the LoRA parameters are part of the backbone's module tree
    (registered as submodules of each block's attention module).

    This class is an nn.Module so that any extra state can be tracked, but
    its primary role is as a coordinator — the actual trainable parameters
    live in the LoRAFusedQKV instances registered within the backbone.

    Attributes:
        rank: LoRA bottleneck rank r. Search grid: {1, 8, 16, 32}
            (config.yaml: peft_methods.lora.search_grid.lora_rank).
        embed_dim: Token embedding dimension D = 768
            (config.yaml: backbones.imagenet21k_vit.embed_dim).
        num_layers: Number of Transformer blocks M = 12
            (config.yaml: backbones.imagenet21k_vit.num_layers).
        lora_layers: List of LoRAFusedQKV instances, one per Transformer block.
            Populated by apply_to_backbone(). Length = num_layers after application.
    """

    def __init__(
        self,
        rank: int,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        num_layers: int = _DEFAULT_NUM_LAYERS,
    ) -> None:
        """Initialises the LoRAModule coordinator.

        Does NOT apply LoRA to any backbone — call apply_to_backbone() explicitly.

        Args:
            rank: LoRA bottleneck rank r. Must be positive and less than embed_dim.
                Search grid from config.yaml: peft_methods.lora.search_grid.lora_rank:
                [1, 8, 16, 32].
                Parameter counts:
                    rank=1:  ~0.037M (config: params_range_M min = 0.036)
                    rank=32: ~1.180M (config: params_range_M max = 1.179)
            embed_dim: Token embedding dimension D. Default: 768
                (config.yaml: backbones.imagenet21k_vit.embed_dim: 768).
            num_layers: Number of Transformer blocks M. Default: 12
                (config.yaml: backbones.imagenet21k_vit.num_layers: 12).

        Raises:
            ValueError: If rank <= 0 or rank >= embed_dim.
            ValueError: If embed_dim <= 0 or num_layers <= 0.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Input validation.
        # ------------------------------------------------------------------
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if rank >= embed_dim:
            raise ValueError(
                f"rank ({rank}) must be less than embed_dim ({embed_dim}) "
                "to achieve parameter efficiency."
            )

        self.rank: int = rank
        self.embed_dim: int = embed_dim
        self.num_layers: int = num_layers

        # Populated by apply_to_backbone(). Not an nn.ModuleList here because
        # the LoRAFusedQKV instances are already registered as submodules of
        # the backbone (via attn.qkv replacement). Storing as a plain list
        # avoids double registration while still providing easy access for
        # get_params() and merge_weights().
        self.lora_layers: List[LoRAFusedQKV] = []

        _logger.info(
            "LoRAModule initialised: rank=%d, embed_dim=%d, num_layers=%d. "
            "Expected trainable params: %d (%.4fM). "
            "Call apply_to_backbone() to inject LoRA into a backbone.",
            rank,
            embed_dim,
            num_layers,
            4 * rank * embed_dim * num_layers,
            4 * rank * embed_dim * num_layers / 1_000_000,
        )

    def apply_to_backbone(self, backbone: nn.Module) -> None:
        """Injects LoRA into all Transformer blocks of the ViT backbone.

        Iterates over all blocks in backbone.blocks and replaces each block's
        attention module's fused QKV layer (attn.qkv) with a LoRAFusedQKV
        instance. The original QKV weights are preserved as frozen parameters
        within each LoRAFusedQKV.

        This method must be called AFTER the backbone is frozen (all parameters
        have requires_grad=False). The LoRAFusedQKV instances introduce new
        trainable parameters (W_down_Q, W_up_Q, W_down_V, W_up_V) that are
        registered as part of the backbone's module tree.

        After this call:
        - backbone.blocks[i].attn.qkv is a LoRAFusedQKV instance for all i
        - self.lora_layers contains references to all 12 LoRAFusedQKV instances
        - Only W_down_Q, W_up_Q, W