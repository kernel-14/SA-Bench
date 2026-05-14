## Code: models/peft/fact.py

```python
## models/peft/fact.py
"""FacT (Factor Tuning) implementation for the PEFT Visual Recognition study.

This module implements FacT-TT (Tensor-Train) and FacT-TK (Tucker) as described
in Section 2.2 and Appendix B.2.3 of the paper:

    "Lessons and Insights from a Unifying Study of Parameter-Efficient
    Fine-Tuning (PEFT) in Visual Recognition"

FacT extends LoRA's low-rank update idea from individual matrices to a global
tensor decomposition across all Transformer layers. All weight matrices from all
layers are stacked into a 3D tensor W_FacT ∈ ℝ^{12L×D×D}, and the update
ΔW_FacT is decomposed using either Tensor-Train (TT) or Tucker (TK) format.

The 12 matrices per layer are:
    - W_Q, W_K, W_V, W_O from MSA block (4 matrices, each D×D)
    - W_1 ∈ ℝ^{D×4D} split into 4 D×D blocks along output dim (indices 4-7)
    - W_2 ∈ ℝ^{4D×D} split into 4 D×D blocks along input dim (indices 8-11)

Paper equations (Appendix B.2.3, Equation 19):
    FacT-TT: ΔW_FacT = s · Σ ×₂ Uᵀ ×₃ Vᵀ
    FacT-TK: ΔW_FacT = s · A ×₁ Bᵀ ×₂ Uᵀ ×₃ Vᵀ

In einsum notation:
    TT: delta[n,l,m] = s · Sigma[n,i,j] · U[l,i] · V[m,j]
    TK: delta[n,l,m] = s · A[i,j,k] · B[n,i] · U[l,j] · V[m,k]

Config reference (config.yaml):
    peft_methods.fact_tt.search_grid.fact_scale: [0.01, 0.1, 1.0, 10.0, 100.0]
    peft_methods.fact_tt.search_grid.fact_rank: [8, 16, 32]
    peft_methods.fact_tt.params_range_M: [0.021, 0.196]
    peft_methods.fact_tk.search_grid.fact_rank: [16, 32, 64]
    peft_methods.fact_tk.search_grid.fact_scale: [0.01, 0.1, 1.0, 10.0, 100.0]
    peft_methods.fact_tk.params_range_M: [0.030, 0.369]
    backbones.imagenet21k_vit.embed_dim: 768
    backbones.imagenet21k_vit.num_layers: 12

Parameter count verification (L=12, D=768):
    TT r=8:  2×768×8 + 144×8×8   = 12288 + 9216   = 21504  ≈ 0.021M ✓
    TT r=32: 2×768×32 + 144×32×32 = 49152 + 147456 = 196608 ≈ 0.196M ✓
    TK r=16: 2×768×16 + 144×16 + 16³ = 24576 + 2304 + 4096 = 30976 ≈ 0.031M ✓
    TK r=64: 2×768×64 + 144×64 + 64³ = 98304 + 9216 + 262144 = 369664 ≈ 0.370M ✓

Typical usage (called by PEFTFactory):
    import copy
    backbone = copy.deepcopy(vit_wrapper.get_backbone())
    # freeze_backbone() called before this on the copy

    fact_module = FacTModule(rank=8, embed_dim=768, num_layers=12, mode='tt', scale=1.0)
    fact_module.apply_to_backbone(backbone)

    # FacT parameters are in fact_module (U, V, Sigma/B/A)
    # Backbone parameters remain frozen
    trainable = fact_module.get_params()
"""

import logging
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn

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

# Number of weight matrices per Transformer layer.
# 4 from MSA (W_Q, W_K, W_V, W_O) + 4 from W_1 split + 4 from W_2 split = 12.
_MATRICES_PER_LAYER: int = 12

# MLP expansion factor: W_1 ∈ ℝ^{D×4D}, W_2 ∈ ℝ^{4D×D}.
_MLP_EXPANSION: int = 4

# Valid FacT modes.
_VALID_MODES: set = {"tt", "tk"}

# Initialization standard deviation for U and V (matches ViT patch embedding init).
_INIT_STD: float = 0.02

# Matrix index assignments within each layer's 12-matrix block.
# MSA matrices (indices 0-3):
_IDX_WQ: int = 0   # W_Q projection
_IDX_WK: int = 1   # W_K projection
_IDX_WV: int = 2   # W_V projection
_IDX_WO: int = 3   # W_O output projection
# W_1 blocks (indices 4-7): W_1[:, k*D:(k+1)*D] for k=0,1,2,3
_IDX_W1_START: int = 4
# W_2 blocks (indices 8-11): W_2[k*D:(k+1)*D, :] for k=0,1,2,3
_IDX_W2_START: int = 8


class FacTModule(nn.Module):
    """Factor Tuning (FacT) PEFT module for ViT backbones.

    Implements both FacT-TT (Tensor-Train) and FacT-TK (Tucker) decomposition
    formats. The module owns the shared decomposition factors (U, V, Sigma/B/A)
    and registers forward hooks on the backbone's linear layers to inject the
    low-rank weight updates during the forward pass.

    The update ΔW_FacT ∈ ℝ^{12L×D×D} is computed from the shared factors and
    sliced per-layer to produce per-matrix weight deltas. These deltas are added
    to the frozen pretrained weights via forward hooks, without modifying the
    backbone's parameter tensors.

    Initialization ensures ΔW_FacT = 0 at the start of training:
    - U, V: initialized with truncated normal (std=0.02)
    - Sigma (TT): initialized to zeros → ΔW = s·0·Uᵀ·Vᵀ = 0
    - A (TK): initialized to zeros → ΔW = s·0·Bᵀ·Uᵀ·Vᵀ = 0
    - B (TK): initialized with truncated normal (std=0.02)

    Attributes:
        rank: Bottleneck rank r. TT search grid: {8,16,32}; TK: {16,32,64}.
        embed_dim: Token embedding dimension D = 768 for ViT-B/16.
        num_layers: Number of Transformer blocks L = 12 for ViT-B/16.
        num_matrices: Total matrices = 12 * num_layers = 144 for ViT-B/16.
        mode: Decomposition format. Either 'tt' (Tensor-Train) or 'tk' (Tucker).
        scale: Scalar scale factor s applied to ΔW_FacT.
        U: Shared left factor, shape (embed_dim, rank) = (768, r).
        V: Shared right factor, shape (embed_dim, rank) = (768, r).
        Sigma: TT-mode per-matrix coefficients, shape (num_matrices, rank, rank).
            Only present when mode='tt'. None when mode='tk'.
        B: TK-mode per-matrix projection, shape (num_matrices, rank).
            Only present when mode='tk'. None when mode='tt'.
        A: TK-mode core Tucker tensor, shape (rank, rank, rank).
            Only present when mode='tk'. None when mode='tt'.
        _hooks: List of registered forward hook handles for cleanup.
        _delta_W_cache: Cached ΔW_FacT tensor computed once per forward pass.
            Set by the pre-forward hook, cleared by the post-forward hook.
        _pre_hook_handle: Handle for the backbone pre-forward hook (caching).
        _post_hook_handle: Handle for the backbone post-forward hook (cache clear).
    """

    def __init__(
        self,
        rank: int,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        num_layers: int = _DEFAULT_NUM_LAYERS,
        mode: str = "tt",
        scale: float = 1.0,
    ) -> None:
        """Initialises FacTModule with decomposition factors.

        Does NOT apply hooks to any backbone — call apply_to_backbone() explicitly.

        Args:
            rank: Bottleneck rank r.
                TT search grid: {8, 16, 32}
                    (config.yaml: peft_methods.fact_tt.search_grid.fact_rank)
                TK search grid: {16, 32, 64}
                    (config.yaml: peft_methods.fact_tk.search_grid.fact_rank)
            embed_dim: Token embedding dimension D. Default: 768
                (config.yaml: backbones.imagenet21k_vit.embed_dim: 768).
            num_layers: Number of Transformer blocks L. Default: 12
                (config.yaml: backbones.imagenet21k_vit.num_layers: 12).
            mode: Decomposition format. Either 'tt' (Tensor-Train) or 'tk' (Tucker).
                Maps from config.yaml:
                    peft_methods.fact_tt.mode: tensor_train → 'tt'
                    peft_methods.fact_tk.mode: tucker → 'tk'
            scale: Scalar scale factor s applied to ΔW_FacT. Default: 1.0.
                Search grids from config.yaml:
                    peft_methods.fact_tt.search_grid.fact_scale: [0.01, 0.1, 1.0, 10.0, 100.0]
                    peft_methods.fact_tk.search_grid.fact_scale: [0.01, 0.1, 1.0, 10.0, 100.0]

        Raises:
            ValueError: If mode is not 'tt' or 'tk'.
            ValueError: If rank, embed_dim, or num_layers are non-positive.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Input validation.
        # ------------------------------------------------------------------
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid FacT mode: '{mode}'. Must be one of {_VALID_MODES}. "
                "Use 'tt' for Tensor-Train (FacT-TT) or 'tk' for Tucker (FacT-TK)."
            )
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.rank: int = rank
        self.embed_dim: int = embed_dim
        self.num_layers: int = num_layers
        self.num_matrices: int = _MATRICES_PER_LAYER * num_layers  # 144 for ViT-B/16
        self.mode: str = mode
        self.scale: float = scale

        # ------------------------------------------------------------------
        # Shared factors U and V (present in both TT and TK modes).
        # U ∈ ℝ^{D×r}: shared left basis (row space)
        # V ∈ ℝ^{D×r}: shared right basis (column space)
        # Initialized with truncated normal (std=0.02) — standard ViT init scale.
        # ------------------------------------------------------------------
        self.U: nn.Parameter = nn.Parameter(
            torch.empty(embed_dim, rank, dtype=torch.float32),
            requires_grad=True,
        )
        self.V: nn.Parameter = nn.Parameter(
            torch.empty(embed_dim, rank, dtype=torch.float32),
            requires_grad=True,
        )
        nn.init.trunc_normal_(self.U, mean=0.0, std=_INIT_STD)
        nn.init.trunc_normal_(self.V, mean=0.0, std=_INIT_STD)

        # ------------------------------------------------------------------
        # Mode-specific parameters.
        # ------------------------------------------------------------------
        if mode == "tt":
            # Tensor-Train mode:
            # Sigma ∈ ℝ^{12L×r×r}: per-matrix coefficients.
            # Initialized to zeros → ΔW_FacT = s·0·Uᵀ·Vᵀ = 0 at init.
            self.Sigma: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(self.num_matrices, rank, rank, dtype=torch.float32),
                requires_grad=True,
            )
            # TK-specific parameters are None in TT mode.
            self.B: Optional[nn.Parameter] = None
            self.A: Optional[nn.Parameter] = None

        else:  # mode == "tk"
            # Tucker mode:
            # B ∈ ℝ^{12L×r}: per-matrix projection onto rank-r space.
            # Initialized with truncated normal.
            self.B = nn.Parameter(
                torch.empty(self.num_matrices, rank, dtype=torch.float32),
                requires_grad=True,
            )
            nn.init.trunc_normal_(self.B, mean=0.0, std=_INIT_STD)

            # A ∈ ℝ^{r×r×r}: core Tucker tensor.
            # Initialized to zeros → ΔW_FacT = s·0·Bᵀ·Uᵀ·Vᵀ = 0 at init.
            self.A = nn.Parameter(
                torch.zeros(rank, rank, rank, dtype=torch.float32),
                requires_grad=True,
            )
            # TT-specific parameter is None in TK mode.
            self.Sigma = None

        # ------------------------------------------------------------------
        # Hook management.
        # _hooks: list of RemovableHook handles for per-layer linear hooks.
        # _pre_hook_handle: handle for backbone pre-forward hook (cache compute).
        # _post_hook_handle: handle for backbone post-forward hook (cache clear).
        # ------------------------------------------------------------------
        self._hooks: List[Any] = []
        self._pre_hook_handle: Optional[Any] = None
        self._post_hook_handle: Optional[Any] = None

        # Cache for ΔW_FacT computed once per forward pass.
        # Set by pre-forward hook, cleared by post-forward hook.
        self._delta_W_cache: Optional[torch.Tensor] = None

        # ------------------------------------------------------------------
        # Log parameter count for verification against paper Table 3.
        # ------------------------------------------------------------------
        total_params: int = self._count_fact_params()
        _logger.info(
            "FacTModule initialised: mode='%s', rank=%d, embed_dim=%d, "
            "num_layers=%d, num_matrices=%d, scale=%.4f, "
            "total_fact_params=%d (%.4fM). "
            "Call apply_to_backbone() to inject FacT into a backbone.",
            mode,
            rank,
            embed_dim,
            num_layers,
            self.num_matrices,
            scale,
            total_params,
            total_params / 1_000_000,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_delta_W(self) -> torch.Tensor:
        """Computes the full ΔW_FacT tensor from the decomposition factors.

        This method is called during the forward pass (inside hooks or via
        the pre-forward caching hook) and participates in autograd. Gradients
        flow through U, V, and Sigma/B/A to enable learning.

        TT mode (Equation 19, paper):
            ΔW_FacT[n, l, m] = s · Sigma[n, i, j] · U[l, i] · V[m, j]
            Einsum: 'nij,li,mj->nlm'
            Shape: (num_matrices, embed_dim, embed_dim) = (144, 768, 768)

        TK mode (Equation 19, paper):
            ΔW_FacT[n, l, m] = s · A[i, j, k] · B[n, i] · U[l, j] · V[m, k]
            Einsum: 'ijk,ni,lj,mk->nlm'
            Shape: (num_matrices, embed_dim, embed_dim) = (144, 768, 768)

        Returns:
            ΔW_FacT tensor of shape (num_matrices, embed_dim, embed_dim).
            For ViT-B/16: shape (144, 768, 768).
            The tensor is on the same device as self.U.
            Gradients are maintained for backpropagation through U, V, Sigma/B/A.

        Raises:
            RuntimeError: If mode-specific parameters (Sigma/B/A) are None
                (should not happen if __init__ completed successfully).
        """
        if self.mode == "tt":
            if self.Sigma is None:
                raise RuntimeError(
                    "FacT-TT mode requires Sigma parameter, but it is None. "
                    "This indicates a bug in FacTModule.__init__."
                )
            # TT decomposition:
            # delta[n, l, m] = scale * Sigma[n, i, j] * U[l, i] * V[m, j]
            # Einsum contracts: n (matrix index), i (rank dim 1), j (rank dim 2),
            #                   l (embed_dim row), m (embed_dim col)
            delta: torch.Tensor = self.scale * torch.einsum(
                "nij,li,mj->nlm",
                self.Sigma,  # (num_matrices, rank, rank)
                self.U,      # (embed_dim, rank)
                self.V,      # (embed_dim, rank)
            )
            # delta shape: (num_matrices, embed_dim, embed_dim) = (144, 768, 768)

        else:  # mode == "tk"
            if self.B is None or self.A is None:
                raise RuntimeError(
                    "FacT-TK mode requires B and A parameters, but one or both are None. "
                    "This indicates a bug in FacTModule.__init__."
                )
            # Tucker decomposition:
            # delta[n, l, m] = scale * A[i, j, k] * B[n, i] * U[l, j] * V[m, k]
            # Einsum contracts: n (matrix index), i (rank dim 1), j (rank dim 2),
            #                   k (rank dim 3), l (embed_dim row), m (embed_dim col)
            delta = self.scale * torch.einsum(
                "ijk,ni,lj,mk->nlm",
                self.A,  # (rank, rank, rank)
                self.B,  # (num_matrices, rank)
                self.U,  # (embed_dim, rank)
                self.V,  # (embed_dim, rank)
            )
            # delta shape: (num_matrices, embed_dim, embed_dim) = (144, 768, 768)

        return delta

    def apply_to_backbone(self, backbone: nn.Module) -> None:
        """Injects FacT weight updates into all Transformer blocks via forward hooks.

        Registers forward hooks on the following nn.Linear layers in each block:
        - block.attn.qkv: fused QKV projection (D → 3D)
        - block.attn.proj: attention output projection (D → D)
        - block.mlp.fc1: MLP first layer (D → 4D)
        - block.mlp.fc2: MLP second layer (4D → D)

        Each hook adds the corresponding slice of ΔW_FacT to the layer's output:
            output_new = output_original + input @ delta.T

        A pre-forward hook on the backbone computes and caches ΔW_FacT once per
        forward pass. A post-forward hook clears the cache. This avoids redundant
        einsum computations (would otherwise be 4 per layer × 12 layers = 48).

        This method must be called AFTER the backbone is frozen (all parameters
        have requires_grad=False). The FacT parameters (U, V, Sigma/B/A) remain
        in self (not in the backbone), so they are not affected by backbone freezing.

        Args:
            backbone: The timm VisionTransformer backbone with num_classes=0.
                Must have backbone.blocks (nn.Sequential of timm Block instances).
                Modified in-place by registering forward hooks.

        Raises:
            AttributeError: If backbone does not have 'blocks' attribute.
            RuntimeError: If backbone.blocks is empty.
            AttributeError: If any block does not have the expected sub-modules
                (attn.qkv, attn.proj, mlp.fc1, mlp.fc2).
        """
        # ------------------------------------------------------------------
        # Input validation.
        # ------------------------------------------------------------------
        if not hasattr(backbone, "blocks"):
            raise AttributeError(
                "Backbone does not have a 'blocks' attribute. "
                "Expected a timm VisionTransformer with backbone.blocks "
                "(nn.Sequential or nn.ModuleList of Block instances)."
            )

        actual_num_layers: int = len(backbone.blocks)
        if actual_num_layers == 0:
            raise RuntimeError(
                "backbone.blocks is empty. Cannot apply FacT to a backbone "
                "with no Transformer blocks."
            )

        if actual_num_layers != self.num_layers:
            _logger.warning(
                "FacTModule was initialised with num_layers=%d, but backbone has "
                "%d blocks. Using backbone's actual layer count for hook registration. "
                "Note: ΔW_FacT was computed for %d matrices; only the first %d "
                "layers' matrices will be used.",
                self.num_layers,
                actual_num_layers,
                self.num_matrices,
                actual_num_layers,
            )

        _logger.info(
            "Applying FacT to backbone: mode='%s', rank=%d, num_layers=%d, "
            "scale=%.4f",
            self.mode,
            self.rank,
            actual_num_layers,
            self.scale,
        )

        # ------------------------------------------------------------------
        # Register pre-forward hook on backbone to compute and cache ΔW_FacT
        # once per forward pass. This avoids 48 redundant einsum computations.
        # ------------------------------------------------------------------
        def _pre_forward_hook(
            module: nn.Module,
            input: Any,
        ) -> None:
            """Computes and caches ΔW_FacT before the backbone forward pass."""
            self._delta_W_cache = self.compute_delta_W()

        def _post_forward_hook(
            module: nn.Module,
            input: Any,
            output: Any,
        ) -> None:
            """Clears the ΔW_FacT cache after the backbone forward pass."""
            self._delta_W_cache = None

        self._pre_hook_handle = backbone.register_forward_pre_hook(_pre_forward_hook)
        self._post_hook_handle = backbone.register_forward_hook(_post_forward_hook)

        # ------------------------------------------------------------------
        # Register per-layer hooks on attn.qkv, attn.proj, mlp.fc1, mlp.fc2.
        # ------------------------------------------------------------------
        for layer_idx in range(actual_num_layers):
            block: nn.Module = backbone.blocks[layer_idx]

            # Validate that the block has the expected sub-modules.
            self._validate_block_structure(block, layer_idx)

            # Base matrix index for this layer: layer_idx * 12.
            base_idx: int = layer_idx * _MATRICES_PER_LAYER

            # ------------------------------------------------------------------
            # Hook 1: attn.qkv — fused QKV projection (D → 3D).
            # Adds deltas for W_Q (idx 0), W_K (idx 1), W_V (idx 2).
            # Delta assembly: cat([ΔW_Q, ΔW_K, ΔW_V], dim=0) → (3D, D)
            # ------------------------------------------------------------------
            qkv_hook: Callable = self._make_qkv_hook(base_idx)
            handle_qkv = block.attn.qkv.register_forward_hook(qkv_hook)
            self._hooks.append(handle_qkv)

            # ------------------------------------------------------------------
            # Hook 2: attn.proj — attention output projection (D → D).
            # Adds delta for W_O (idx 3).
            # Delta: ΔW_O ∈ ℝ^{D×D}
            # ------------------------------------------------------------------
            proj_hook: Callable = self._make_proj_hook(base_idx + _IDX_WO)
            handle_proj = block.attn.proj.register_forward_hook(proj_hook)
            self._hooks.append(handle_proj)

            # ------------------------------------------------------------------
            # Hook 3: mlp.fc1 — MLP first layer (D → 4D).
            # Adds deltas for W_1 blocks (indices 4, 5, 6, 7).
            # Delta assembly: cat([ΔW_1_0, ΔW_1_1, ΔW_1_2, ΔW_1_3], dim=1) → (D, 4D)
            # ------------------------------------------------------------------
            fc1_hook: Callable = self._make_fc1_hook(base_idx + _IDX_W1_START)
            handle_fc1 = block.mlp.fc1.register_forward_hook(fc1_hook)
            self._hooks.append(handle_fc1)

            # ------------------------------------------------------------------
            # Hook 4: mlp.fc2 — MLP second layer (4D → D).
            # Adds deltas for W_2 blocks (indices 8, 9, 10, 11).
            # Delta assembly: cat([ΔW_2_0, ΔW_2_1, ΔW_2_2, ΔW_2_3], dim=0) → (4D, D)
            # ------------------------------------------------------------------
            fc2_hook: Callable = self._make_fc2_hook(base_idx + _IDX_W2_START)
            handle_fc2 = block.mlp.fc2.register_forward_hook(fc2_hook)
            self._hooks.append(handle_fc2)

            _logger.debug(
                "FacT hooks registered for block %d (base_idx=%d): "
                "attn.qkv, attn.proj, mlp.fc1, mlp.fc2.",
                layer_idx,
                base_idx,
            )

        _logger.info(
            "FacT applied successfully: %d forward hooks registered across "
            "%d blocks (4 hooks per block) + 2 backbone-level hooks.",
            len(self._hooks),
            actual_num_layers,
        )

    def get_params(self) -> List[nn.Parameter]:
        """Returns the trainable FacT decomposition parameters.

        These are the only parameters that should be passed to the optimizer.
        The backbone parameters remain frozen (requires_grad=False).

        Returns:
            List of nn.Parameter objects:
            - TT mode: [U, V, Sigma] (3 parameters)
            - TK mode: [U, V, B, A] (4 parameters)

        Note:
            The classification head parameters are managed separately by
            PEFTModel and are not included here.
        """
        if self.mode == "tt":
            assert self.Sigma is not None, "Sigma must not be None in TT mode."
            return [self.U, self.V, self.Sigma]
        else:  # mode == "tk"
            assert self.B is not None, "B must not be None in TK mode."
            assert self.A is not None, "A must not be None in TK mode."
            return [self.U, self.V, self.B, self.A]

    def remove_hooks(self) -> None:
        """Removes all registered forward hooks from the backbone.

        Should be called when the model is no longer needed or when
        re-applying FacT with different parameters. After calling this
        method, the backbone's forward pass will no longer include FacT
        weight updates.

        Also clears the delta cache and removes backbone-level hooks.
        """
        # Remove per-layer hooks.
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        # Remove backbone-level pre/post hooks.
        if self._pre_hook_handle is not None:
            self._pre_hook_handle.remove()
            self._pre_hook_handle = None

        if self._post_hook_handle is not None:
            self._post_hook_handle.remove()
            self._post_hook_handle = None

        # Clear cache.
        self._delta_W_cache = None

        _logger.info("FacT hooks removed.")

    # ------------------------------------------------------------------
    # Private hook factory methods
    # ------------------------------------------------------------------

    def _make_qkv_hook(
        self,
        base_idx: int,
    ) -> Callable:
        """Creates a forward hook for the fused QKV projection layer.

        The hook adds the concatenated [ΔW_Q, ΔW_K, ΔW_V] delta to the
        fused QKV output. The delta is assembled from three D×D blocks
        (indices base_idx+0, base_idx+1, base_idx+2) concatenated along
        dimension 0 to form a (3D, D) matrix.

        Paper: "LoRA(h₂) = [W_down^Q · W_up^Q · h₂, 0, W_down^V · W_up^V · h₂]"