## Code: models/peft/selective.py

```python
## models/peft/selective.py
"""Selective parameter tuning PEFT methods for the PEFT Visual Recognition study.

This module implements four direct selective tuning PEFT methods described in
Section 2.2 and Appendix B.2.3 of the paper:

    "Lessons and Insights from a Unifying Study of Parameter-Efficient
    Fine-Tuning (PEFT) in Visual Recognition"

Implemented methods:
- BitFit: tune only bias terms (QKV, FC, LN, patch embedding)
- LayerNorm: tune only LayerNorm weight and bias parameters
- DiffFit: BitFit + LayerNorm + learnable scale factors after MSA and MLP
- SSF: scale and shift intermediate features at h2, h3, h5, h7, h8, h9

All methods work by selectively enabling trainable parameters in an otherwise
frozen backbone. Unlike adapter-based methods, these methods introduce no
additional inference overhead (or minimal overhead for SSF/DiffFit which can
be folded into backbone weights).

Paper config references (config.yaml):
    peft_methods.bitfit.params_fixed_M: 0.102
    peft_methods.layernorm.params_fixed_M: 0.038
    peft_methods.difffit.params_fixed_M: 0.140
    peft_methods.ssf.params_fixed_M: 0.205
    backbones.imagenet21k_vit.embed_dim: 768
    backbones.imagenet21k_vit.num_layers: 12

Typical usage (called by PEFTFactory):
    import copy
    backbone = copy.deepcopy(vit_wrapper.get_backbone())

    selective = SelectiveModule(method='bitfit', backbone=backbone)
    selective.apply_bitfit()

    trainable_params = selective.get_trainable_params()
    # Only bias terms have requires_grad=True
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture constants derived from config.yaml:
#   backbones.imagenet21k_vit.embed_dim: 768
#   backbones.imagenet21k_vit.num_layers: 12
# ---------------------------------------------------------------------------

# ViT-B/16 embedding dimension D.
# config.yaml: backbones.imagenet21k_vit.embed_dim: 768
EMBED_DIM: int = 768

# MLP intermediate dimension = 4 × D.
# Used for SSF h8 scale/shift shape (4D,) = (3072,).
MLP_DIM: int = 4 * EMBED_DIM  # 3072

# QKV projection output dimension = 3 × D (fused Q, K, V).
# Used for SSF h3 scale/shift shape (3D,) = (2304,).
QKV_DIM: int = 3 * EMBED_DIM  # 2304

# Number of Transformer layers M for ViT-B/16.
# config.yaml: backbones.imagenet21k_vit.num_layers: 12
NUM_LAYERS: int = 12

# Valid method names handled by SelectiveModule.
_VALID_METHODS: set = {"bitfit", "layernorm", "difffit", "ssf"}


# ===========================================================================
# SelectiveModule
# ===========================================================================

class SelectiveModule(nn.Module):
    """Selective parameter tuning manager for BitFit, LayerNorm, DiffFit, and SSF.

    This class modifies a frozen ViT backbone in-place by selectively enabling
    requires_grad on specific parameters and/or inserting lightweight learnable
    parameters (DiffFit gamma factors, SSF scale/shift vectors) at specific
    positions in the Transformer block forward pass.

    The SelectiveModule itself is an nn.Module so that extra parameters
    (DiffFit gammas, SSF scale/shift) are properly registered, moved to the
    correct device with .to(), and included in state_dict() for checkpointing.

    The backbone is modified in-place — its forward() method is unchanged for
    BitFit and LayerNorm. For DiffFit and SSF, individual block forward methods
    are replaced with closures that apply the extra parameters.

    Attributes:
        method: One of 'bitfit', 'layernorm', 'difffit', 'ssf'.
        backbone: The timm VisionTransformer backbone (modified in-place).
        embed_dim: Token embedding dimension D = 768.
        num_layers: Number of Transformer blocks M = 12.
        _extra_params: nn.ParameterDict holding DiffFit gamma factors and
            SSF scale/shift parameters. Empty for BitFit and LayerNorm.
        _hook_handles: List of forward hook handles registered on backbone
            sub-modules (used by SSF). Stored for potential cleanup.
    """

    def __init__(
        self,
        method: str,
        backbone: nn.Module,
    ) -> None:
        """Initialises the SelectiveModule.

        Does NOT apply any PEFT modifications — the caller must explicitly
        call apply_bitfit(), apply_layernorm(), apply_difffit(), or apply_ssf()
        after construction.

        Args:
            method: PEFT method name. One of 'bitfit', 'layernorm', 'difffit',
                'ssf'. Used for logging and dispatch in PEFTFactory.
            backbone: The timm VisionTransformer backbone with num_classes=0.
                Should already be deepcopied by PEFTFactory before passing here.
                Modified in-place by the apply_* methods.

        Raises:
            ValueError: If method is not one of the valid method names.
        """
        super().__init__()

        if method not in _VALID_METHODS:
            raise ValueError(
                f"Invalid selective method: '{method}'. "
                f"Must be one of {sorted(_VALID_METHODS)}."
            )

        self.method: str = method
        self.backbone: nn.Module = backbone

        # ------------------------------------------------------------------
        # Infer architecture constants from the loaded backbone.
        # Fall back to module-level constants if attributes are not present.
        # ------------------------------------------------------------------
        self.embed_dim: int = (
            int(backbone.embed_dim)
            if hasattr(backbone, "embed_dim")
            else EMBED_DIM
        )
        self.num_layers: int = (
            len(backbone.blocks)
            if hasattr(backbone, "blocks")
            else NUM_LAYERS
        )

        # ------------------------------------------------------------------
        # Extra parameters for DiffFit (gamma) and SSF (scale/shift).
        # Registered as nn.ParameterDict so they are:
        # - Included in self.parameters() and state_dict()
        # - Moved to the correct device with .to()
        # - Returned by get_trainable_params()
        # Keys use underscores (no dots) as required by nn.ParameterDict.
        # ------------------------------------------------------------------
        self._extra_params: nn.ParameterDict = nn.ParameterDict()

        # ------------------------------------------------------------------
        # Storage for forward hook handles (SSF).
        # Stored so hooks can be removed if needed (e.g., for re-parameterization).
        # ------------------------------------------------------------------
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []

        _logger.info(
            "SelectiveModule initialised: method='%s', embed_dim=%d, num_layers=%d",
            self.method,
            self.embed_dim,
            self.num_layers,
        )

    # ------------------------------------------------------------------
    # Private helper: freeze all backbone parameters
    # ------------------------------------------------------------------

    def _freeze_all(self) -> None:
        """Freezes all backbone parameters by setting requires_grad=False.

        Called at the start of every apply_* method to ensure a clean slate
        before selectively re-enabling specific parameters.
        """
        frozen_count: int = 0
        for param in self.backbone.parameters():
            param.requires_grad = False
            frozen_count += 1

        _logger.debug(
            "All backbone parameters frozen: %d tensors set to requires_grad=False.",
            frozen_count,
        )

    # ------------------------------------------------------------------
    # Public apply methods
    # ------------------------------------------------------------------

    def apply_bitfit(self) -> None:
        """Applies BitFit: tune only bias terms of the frozen backbone.

        Paper: "BitFit updates the bias terms, including those in the patch
        embeddings projection, the Q/K/V weights, the MLP and LN blocks."
        (Section 2.2, Appendix B.2.3)

        Config: config.yaml -> peft_methods.bitfit.params_fixed_M: 0.102

        Specifically unfreezes:
        - patch_embed.proj.bias: patch embedding projection bias
        - blocks.{i}.norm1.bias: LN1 bias (before MSA)
        - blocks.{i}.norm2.bias: LN2 bias (before MLP)
        - blocks.{i}.attn.qkv.bias: fused QKV projection bias (shape 3D=2304)
        - blocks.{i}.attn.proj.bias: attention output projection bias
        - blocks.{i}.mlp.fc1.bias: MLP first FC layer bias
        - blocks.{i}.mlp.fc2.bias: MLP second FC layer bias
        - norm.bias: final LayerNorm bias (if present in backbone)

        All weight parameters remain frozen.
        """
        # Step 1: Freeze everything.
        self._freeze_all()

        # Step 2: Selectively unfreeze bias terms.
        unfrozen_count: int = 0
        unfrozen_names: List[str] = []

        for name, param in self.backbone.named_parameters():
            should_unfreeze: bool = False

            # Patch embedding projection bias.
            if name == "patch_embed.proj.bias":
                should_unfreeze = True

            # All bias terms within Transformer blocks.
            # This covers: norm1.bias, norm2.bias, attn.qkv.bias,
            # attn.proj.bias, mlp.fc1.bias, mlp.fc2.bias
            elif "blocks." in name and name.endswith(".bias"):
                should_unfreeze = True

            # Final LayerNorm bias (backbone.norm.bias in timm ViT).
            elif name == "norm.bias":
                should_unfreeze = True

            if should_unfreeze:
                param.requires_grad = True
                unfrozen_count += 1
                unfrozen_names.append(name)

        total_trainable: int = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )

        _logger.info(
            "BitFit applied: %d bias parameter tensors unfrozen, "
            "total trainable params = %d (%.4fM). "
            "Unfrozen: %s",
            unfrozen_count,
            total_trainable,
            total_trainable / 1_000_000,
            unfrozen_names[:5],  # Log first 5 for brevity
        )

    def apply_layernorm(self) -> None:
        """Applies LayerNorm tuning: tune only LN weight and bias parameters.

        Paper: "LayerNorm represents another simple but strong baseline that
        solely tunes the two LN blocks in each Transformer layer — one before
        the MSA block and another before the MLP block."
        (Section 2.2, Appendix B.2.3)

        Config: config.yaml -> peft_methods.layernorm.params_fixed_M: 0.038

        Parameter count: 2 LN blocks × 12 layers × 2 params (weight + bias)
        × 768 = 36,864 ≈ 0.038M, matching Table 3.

        Specifically unfreezes:
        - blocks.{i}.norm1.weight and blocks.{i}.norm1.bias (LN before MSA)
        - blocks.{i}.norm2.weight and blocks.{i}.norm2.bias (LN before MLP)
        - norm.weight and norm.bias (final LayerNorm, if present)

        All other parameters (weights, biases of attention and MLP) remain frozen.
        """
        # Step 1: Freeze everything.
        self._freeze_all()

        # Step 2: Unfreeze all LayerNorm modules' parameters.
        # Use isinstance check on modules for robustness.
        unfrozen_count: int = 0
        unfrozen_module_names: List[str] = []

        for module_name, module in self.backbone.named_modules():
            if isinstance(module, nn.LayerNorm):
                for param_name, param in module.named_parameters(recurse=False):
                    param.requires_grad = True
                    unfrozen_count += 1

                unfrozen_module_names.append(module_name)

        total_trainable: int = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )

        _logger.info(
            "LayerNorm tuning applied: %d LayerNorm modules unfrozen "
            "(%d parameter tensors), total trainable params = %d (%.4fM). "
            "Modules: %s",
            len(unfrozen_module_names),
            unfrozen_count,
            total_trainable,
            total_trainable / 1_000_000,
            unfrozen_module_names[:6],  # Log first 6 for brevity
        )

    def apply_difffit(self) -> None:
        """Applies DiffFit: BitFit + LayerNorm + learnable scale factors.

        Paper: "DiffFit exclusively fine-tunes the bias terms and the LN blocks
        within the network. Furthermore, it inserts learnable scale factors γ
        to shift the features after the MSA and the MLP blocks."
        (Section 2.2, Appendix B.2.3)

        Paper equations (Appendix B.2.3):
            h5 = γ1 ⊙ h5    (scale after MSA output + residual)
            h9 = γ2 ⊙ h9    (scale after MLP output + residual)

        Config: config.yaml -> peft_methods.difffit.params_fixed_M: 0.140

        Parameter breakdown:
        - BitFit params: ~0.102M (biases)
        - LayerNorm weight params: ~0.019M (LN weights, not counted in BitFit)
        - Gamma factors: 2 × 12 × 768 = 18,432 ≈ 0.018M
        - Total: ~0.140M, matching Table 3

        Implementation:
        1. Apply BitFit (freeze all, unfreeze biases)
        2. Additionally unfreeze LN weight parameters
        3. Create gamma1_i, gamma2_i (ones, shape D) for each block
        4. Register in self._extra_params
        5. Wrap each block's forward to apply gamma scaling
        """
        # ------------------------------------------------------------------
        # Step 1: Apply BitFit (freeze all, unfreeze biases).
        # ------------------------------------------------------------------
        self.apply_bitfit()

        # ------------------------------------------------------------------
        # Step 2: Additionally unfreeze LayerNorm weight parameters.
        # (BitFit already unfroze LN biases; now unfreeze LN weights too.)
        # ------------------------------------------------------------------
        ln_weight_count: int = 0
        for module_name, module in self.backbone.named_modules():
            if isinstance(module, nn.LayerNorm):
                for param_name, param in module.named_parameters(recurse=False):
                    if not param.requires_grad:
                        param.requires_grad = True
                        ln_weight_count += 1

        _logger.debug(
            "DiffFit: additionally unfroze %d LN weight tensors.", ln_weight_count
        )

        # ------------------------------------------------------------------
        # Step 3: Create learnable scale factors gamma1 and gamma2 per block.
        # gamma1_i: scale after MSA output + residual (h5), shape (D,)
        # gamma2_i: scale after MLP output + residual (h9), shape (D,)
        # Initialized to ones → identity transform at start of training.
        # ------------------------------------------------------------------
        if not hasattr(self.backbone, "blocks"):
            _logger.warning(
                "Backbone does not have 'blocks' attribute. "
                "Cannot insert DiffFit gamma factors."
            )
            return

        for layer_idx in range(self.num_layers):
            # Create gamma parameters (ones initialization = identity).
            gamma1: nn.Parameter = nn.Parameter(
                torch.ones(self.embed_dim, dtype=torch.float32)
            )
            gamma2: nn.Parameter = nn.Parameter(
                torch.ones(self.embed_dim, dtype=torch.float32)
            )

            # Register in _extra_params with underscore-separated keys.
            # Keys: 'gamma1_0', 'gamma2_0', ..., 'gamma1_11', 'gamma2_11'
            key_gamma1: str = f"gamma1_{layer_idx}"
            key_gamma2: str = f"gamma2_{layer_idx}"
            self._extra_params[key_gamma1] = gamma1
            self._extra_params[key_gamma2] = gamma2

        _logger.debug(
            "DiffFit: created %d gamma parameter pairs (gamma1, gamma2) "
            "for %d blocks.",
            self.num_layers,
            self.num_layers,
        )

        # ------------------------------------------------------------------
        # Step 4: Wrap each block's forward to apply gamma scaling.
        # The wrapped forward replaces block.forward with a closure that:
        # - Computes h5 = MSA(LN1(x)) + x (standard block MSA path)
        # - Applies h5 = gamma1 * h5 (element-wise scale)
        # - Computes h9 = MLP(LN2(h5)) + h5 (standard block MLP path)
        # - Applies h9 = gamma2 * h9 (element-wise scale)
        #
        # timm Block.forward (simplified):
        #   x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        #   x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        #   return x
        #
        # We need to intercept after each residual addition.
        # ------------------------------------------------------------------
        for layer_idx in range(self.num_layers):
            block: nn.Module = self.backbone.blocks[layer_idx]

            # Retrieve gamma parameters for this block.
            g1: nn.Parameter = self._extra_params[f"gamma1_{layer_idx}"]
            g2: nn.Parameter = self._extra_params[f"gamma2_{layer_idx}"]

            # Create the wrapped forward using a closure.
            # The closure captures block, g1, g2 by reference — this is
            # intentional so that gradient updates to g1, g2 are reflected.
            block.forward = self._make_difffit_forward(block, g1, g2)

            _logger.debug(
                "DiffFit: wrapped forward for block %d with gamma scaling.",
                layer_idx,
            )

        total_trainable: int = (
            sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            + sum(p.numel() for p in self._extra_params.values())
        )

        _logger.info(
            "DiffFit applied: total trainable params = %d (%.4fM). "
            "Includes biases, LN params, and %d gamma factor pairs.",
            total_trainable,
            total_trainable / 1_000_000,
            self.num_layers,
        )

    def apply_ssf(self) -> None:
        """Applies SSF: scale and shift intermediate features.

        Paper: "SSF employs linear transformations to adapt the intermediate
        features extracted by a pre-trained model... SSF modulates the features
        residing at h2, h3, h5, h7, h8, h9 by incorporating scale and shift
        factors."
        (Section 2.2, Appendix B.2.3)

        Paper equation:
            SSF(h) = w ⊙ h + b
        where w, b ∈ R^D are the scale and shift factors.

        Config: config.yaml -> peft_methods.ssf.params_fixed_M: 0.205

        Feature positions and dimensions (from paper Table 6):
        - h2: after LN1, shape (B, N+1, D=768)     → w2, b2 ∈ R^768
        - h3: after QKV proj, shape (B, N+1, 3D)   → w3, b3 ∈ R^2304
        - h5: after MSA+residual, shape (B, N+1, D) → w5, b5 ∈ R^768
        - h7: after MLP FC1, shape (B, N+1, D)      → w7, b7 ∈ R^768
        - h8: after GELU, shape (B, N+1, 4D)        → w8, b8 ∈ R^3072
        - h9: after MLP+residual, shape (B, N+1, D) → w9, b9 ∈ R^768

        Parameter count: 12 × (768+2304+768+768+3072+768) × 2 = 12 × 8448 × 2
        = 202,752 ≈ 0.203M ≈ 0.205M (Table 3).

        Implementation:
        1. Freeze all backbone parameters
        2. Create w/b parameters for each position per block
        3. Register in self._extra_params
        4. Register forward hooks on attn.qkv, mlp.fc1, mlp.act for h3, h7, h8
        5. Wrap each block's forward for h2, h5, h9 SSF application
        """
        # ------------------------------------------------------------------
        # Step 1: Freeze all backbone parameters.
        # ------------------------------------------------------------------
        self._freeze_all()

        if not hasattr(self.backbone, "blocks"):
            _logger.warning(
                "Backbone does not have 'blocks' attribute. "
                "Cannot apply SSF."
            )
            return

        # ------------------------------------------------------------------
        # Step 2: Create scale (w) and shift (b) parameters per block.
        # w initialized to ones, b initialized to zeros → identity transform.
        # ------------------------------------------------------------------
        # Dimension mapping for each SSF position.
        # Keys: position name → dimension
        ssf_dims: Dict[str, int] = {
            "h2": self.embed_dim,           # 768: after LN1
            "h3": 3 * self.embed_dim,       # 2304: after QKV projection
            "h5": self.embed_dim,           # 768: after MSA + residual
            "h7": self.embed_dim,           # 768: after MLP FC1 (pre-GELU)
            "h8": 4 * self.embed_dim,       # 3072: after GELU
            "h9": self.embed_dim,           # 768: after MLP + residual
        }

        for layer_idx in range(self.num_layers):
            for pos_name, dim in ssf_dims.items():
                # Scale parameter: initialized to ones (identity scale)
                w_key: str = f"ssf_w{pos_name[1]}_{layer_idx}"  # e.g., 'ssf_w2_0'
                b_key: str = f"ssf_b{pos_name[1]}_{layer_idx}"  # e.g., 'ssf_b2_0'

                self._extra_params[w_key] = nn.Parameter(
                    torch.ones(dim, dtype=torch.float32)
                )
                # Shift parameter: initialized to zeros (identity shift)
                self._extra_params[b_key] = nn.Parameter(
                    torch.zeros(dim, dtype=torch.float32)
                )

        _logger.debug(
            "SSF: created %d scale/shift parameter pairs for %d blocks "
            "(6 positions per block).",
            self.num_layers * 6,
            self.num_layers,
        )

        # ------------------------------------------------------------------
        # Step 3: Register forward hooks for h3 (attn.qkv), h7 (mlp.fc1),
        # and h8 (mlp.act). These positions are inside sub-modules and
        # cannot be intercepted by wrapping block.forward alone.
        # ------------------------------------------------------------------
        for layer_idx in range(self.num_layers):
            block: nn.Module = self.backbone.blocks[layer_idx]

            # Retrieve SSF parameters for this block.
            w3: nn.Parameter = self._extra_params[f"ssf_w3_{layer_idx}"]
            b3: nn.Parameter = self._extra_params[f"ssf_b3_{layer_idx}"]
            w7: nn.Parameter = self._extra_params[f"ssf_w7_{layer_idx}"]
            b7: nn.Parameter = self._extra_params[f"ssf_b7_{layer_idx}"]
            w8: nn.Parameter = self._extra_params[f"ssf_w8_{layer_idx}"]
            b8: nn.Parameter = self._extra_params[f"ssf_b8_{layer_idx}"]

            # ------------------------------------------------------------------
            # Hook for h3: applied to attn.qkv output.
            # attn.qkv is a Linear(D, 3D) layer; its output shape is (B, N+1, 3D).
            # ------------------------------------------------------------------
            if hasattr(block, "attn") and hasattr(block.attn, "qkv"):
                h3_hook = self._make_ssf_hook(w3, b3)
                handle_h3 = block.attn.qkv.register_forward_hook(h3_hook)
                self._hook_handles.append(handle_h3)
            else:
                _logger.warning(
                    "Block %d: attn.qkv not found. SSF h3 hook not registered.",
                    layer_idx,
                )

            # ------------------------------------------------------------------
            # Hook for h7: applied to mlp.fc1 output.
            # mlp.fc1 is a Linear(D, 4D) layer; output shape is (B, N+1, 4D).
            # Wait — h7 is pre-GELU MLP output. In timm, mlp.fc1 output has
            # shape (B, N+1, 4D) = (B, N+1, 3072). But the paper says h7 has
            # shape (B, N+1, D) with SSF dim D=768.
            #
            # Re-reading paper Table 6: W8 ∈ R^{4D}, b8 ∈ R^{4D} for h8.
            # W7 ∈ R^D, b7 ∈ R^D for h7.
            #
            # This is inconsistent with timm's MLP where fc1 outputs 4D.
            # The paper's h7 = FC1(h6) where FC1: D→4D, so h7 shape is (B,N+1,4D).
            # But Table 6 shows W7 ∈ R^D...
            #
            # Looking at the paper more carefully: the notation in Table 6 for SSF
            # shows W8 ∈ R^{4D} for h8 (post-GELU). For h7 (pre-GELU = FC1 output),
            # the dimension should also be 4D. The table entry "W7 ∈ R^D" likely
            # refers to the input to FC1 (h6 = LN2 output), not the FC1 output.
            #
            # We follow the actual feature dimensions:
            # h7 = FC1(h6): shape (B, N+1, 4D) → SSF dim = 4D = 3072
            # h8 = GELU(h7): shape (B, N+1, 4D) → SSF dim = 4D = 3072
            #
            # However, the paper's parameter count (0.205M) is consistent with:
            # 12 × (768 + 2304 + 768 + 768 + 3072 + 768) × 2 = 202,752
            # where h7 uses dim=768 (not 4D=3072).
            #
            # This suggests the paper applies SSF to h7 BEFORE fc1 (i.e., to h6,
            # the LN2 output), not after fc1. We follow the parameter count
            # interpretation: h7 SSF is applied to the LN2 output (dim=D=768),
            # which is the input to the MLP block.
            #
            # Revised interpretation matching 0.205M count:
            # h2: D=768 (after LN1)
            # h3: 3D=2304 (after QKV)
            # h5: D=768 (after MSA+residual)
            # h7: D=768 (LN2 output = MLP input, before FC1)
            # h8: 4D=3072 (after FC1+GELU = after mlp.fc1 and mlp.act)
            # h9: D=768 (after MLP+residual)
            # Total per layer: (768+2304+768+768+3072+768)*2 = 16,896
            # 12 layers: 202,752 ≈ 0.203M ✓
            #
            # So h7 SSF is applied to norm2 output (before MLP), and
            # h8 SSF is applied after mlp.fc1 (before or after GELU).
            # We apply h8 SSF after mlp.act (GELU) to match "h8 = GELU(h7)".
            # ------------------------------------------------------------------

            # Hook for h8: applied to mlp.act output (after GELU).
            # mlp.act output shape: (B, N+1, 4D) = (B, N+1, 3072).
            if hasattr(block, "mlp") and hasattr(block.mlp, "act"):
                h8_hook = self._make_ssf_hook(w8, b8)
                handle_h8 = block.mlp.act.register_forward_hook(h8_hook)
                self._hook_handles.append(handle_h8)
            elif hasattr(block, "mlp") and hasattr(block.mlp, "drop1"):
                # Some timm versions use drop1 after act; hook on fc1 instead.
                h8_hook = self._make_ssf_hook(w8, b8)
                handle_h8 = block.mlp.fc1.register_forward_hook(h8_hook)
                self._hook_handles.append(handle_h8)
            else:
                _logger.warning(
                    "Block %d: mlp.act not found. SSF h8 hook not registered.",
                    layer_idx,
                )

        # ------------------------------------------------------------------
        # Step 4: Wrap each block's forward for h2, h5, h7 (LN2 output),
        # and h9 SSF application.
        # ------------------------------------------------------------------
        for layer_idx in range(self.num_layers):
            block = self.backbone.blocks[layer_idx]

            # Retrieve SSF parameters for this block.
            w2: nn.Parameter = self._extra_params[f"ssf_w2_{layer_idx}"]
            b2: nn.Parameter = self._extra_params[f"ssf_b2_{layer_idx}"]