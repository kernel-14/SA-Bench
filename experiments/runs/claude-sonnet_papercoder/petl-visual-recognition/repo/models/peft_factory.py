## Code: models/peft_factory.py
```python
## models/peft_factory.py
"""PEFT model factory for the PEFT Visual Recognition reproduction study.

This module provides PEFTFactory and PEFTModel — the central assembly point
for all 16 PEFT methods (14 PEFT + linear probing + full fine-tuning) evaluated
in the paper:

    "Lessons and Insights from a Unifying Study of Parameter-Efficient
    Fine-Tuning (PEFT) in Visual Recognition"

PEFTFactory.build() is the single entry point for constructing a training-ready
model for any method. It deep-copies the backbone (ensuring trial isolation),
applies the appropriate PEFT modifications, creates a randomly initialized
classification head, and returns a PEFTModel wrapped around the result.

PEFTModel is a thin nn.Module that unifies the forward pass and provides
parameter introspection utilities (get_trainable_params, count_trainable_params)
used by Trainer and HyperparamSearch.

Config references (config.yaml):
    peft_param_cap.ratio: 0.015
    peft_param_cap.absolute_max_params: 1_290_000
    backbones.imagenet21k_vit.total_params: 86_000_000
    peft_methods.*: method-specific hyperparameter search grids

Typical usage (called by HyperparamSearch and main.py):
    factory = PEFTFactory()
    model = factory.build(
        method='lora',
        backbone_wrapper=vit_wrapper,
        num_classes=47,
        peft_params={'lora_rank': 8},
        device='cuda',
    )
    if factory.check_param_cap(model):
        trainer = Trainer(model, train_loader, val_loader, config)
        best_val_acc = trainer.train()
"""

import copy
import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# PEFT method imports
# ---------------------------------------------------------------------------
from models.peft.vpt import VPTModule, apply_to_vit
from models.peft.adapters import (
    HoulsbyAdapterBlock,
    PfeifferAdapterBlock,
    AdaptFormerBlock,
    ConvPassBlock,
    RepAdapterModule,
)
from models.peft.selective import SelectiveModule
from models.peft.lora import LoRAModule
from models.peft.fact import FacTModule

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported PEFT method names (16 total: 14 PEFT + 2 baselines).
# Order matches paper Table 1 and config.yaml peft_methods keys.
# Imported by main.py for CLI validation and by HyperparamSearch.
# ---------------------------------------------------------------------------
SUPPORTED_METHODS: List[str] = [
    "linear",
    "full",
    "vpt_shallow",
    "vpt_deep",
    "bitfit",
    "layernorm",
    "difffit",
    "ssf",
    "pfeiffer_adapter",
    "houlsby_adapter",
    "adaptformer",
    "convpass",
    "repadapter",
    "lora",
    "fact_tt",
    "fact_tk",
]

# ---------------------------------------------------------------------------
# Total ViT-B/16 parameter count for param cap enforcement.
# config.yaml: backbones.imagenet21k_vit.total_params: 86_000_000
# ---------------------------------------------------------------------------
TOTAL_VIT_B16_PARAMS: int = 86_000_000

# ---------------------------------------------------------------------------
# PEFT parameter cap ratio.
# config.yaml: peft_param_cap.ratio: 0.015
# Absolute cap: 0.015 * 86_000_000 = 1_290_000
# config.yaml: peft_param_cap.absolute_max_params: 1_290_000
# ---------------------------------------------------------------------------
PEFT_PARAM_CAP_RATIO: float = 0.015
PEFT_PARAM_CAP_ABSOLUTE: int = 1_290_000

# ---------------------------------------------------------------------------
# Default RepAdapter group count.
# config.yaml: peft_methods.repadapter.groups: 8
# ---------------------------------------------------------------------------
_REPADAPTER_DEFAULT_GROUPS: int = 8

# ---------------------------------------------------------------------------
# Head initialization standard deviation (matches ViT patch embedding init).
# ---------------------------------------------------------------------------
_HEAD_INIT_STD: float = 0.02

# ---------------------------------------------------------------------------
# Methods exempt from the 1.5% parameter cap.
# 'linear' has ~0 backbone params; 'full' intentionally exceeds the cap.
# ---------------------------------------------------------------------------
_CAP_EXEMPT_METHODS: set = {"linear", "full"}


# ===========================================================================
# PEFTModel
# ===========================================================================

class PEFTModel(nn.Module):
    """Unified nn.Module wrapper for backbone + classification head.

    Holds the (potentially PEFT-modified) backbone and the classification head
    together. Provides a unified forward pass and parameter introspection
    utilities used by Trainer, HyperparamSearch, and WiSE.

    The backbone is always a timm VisionTransformer with num_classes=0, so
    its forward() returns the CLS token feature of shape (B, embed_dim).
    VPT modifications are applied in-place to the backbone's blocks, so the
    external interface remains identical.

    Attributes:
        backbone: The (potentially PEFT-modified) timm VisionTransformer.
            Returns CLS token features of shape (B, embed_dim) from forward().
        head: Randomly initialized nn.Linear(embed_dim, num_classes).
            Always has requires_grad=True.
        method: PEFT method name string. Used by WiSE for interpolation dispatch
            and by Logger for experiment identification.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Linear,
        method: str,
    ) -> None:
        """Initialises PEFTModel.

        No PEFT logic happens here. All backbone modifications (freezing,
        module insertion, parameter selection) are performed by PEFTFactory.build()
        before this constructor is called.

        Args:
            backbone: The timm VisionTransformer backbone with num_classes=0,
                already modified in-place by the PEFT factory. Its forward()
                must return CLS token features of shape (B, embed_dim).
            head: Randomly initialized nn.Linear(embed_dim, num_classes).
                Created by PEFTFactory.build() with requires_grad=True.
            method: PEFT method name string. One of SUPPORTED_METHODS.
                Stored for downstream dispatch (WiSE, logging).
        """
        super().__init__()

        self.backbone: nn.Module = backbone
        self.head: nn.Linear = head
        self.method: str = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes classification logits for input images.

        Passes images through the backbone to obtain CLS token features,
        then through the classification head to produce logits.

        The backbone handles all PEFT-specific logic internally:
        - VPT: prompt injection is handled by VPTWrappedBlock.forward()
        - Adapters: injected via wrapped block forwards
        - LoRA: LoRAFusedQKV replaces attn.qkv transparently
        - FacT: forward hooks add weight deltas transparently
        - Selective (BitFit/LN/DiffFit/SSF): modified backbone params/hooks

        Args:
            x: Input image tensor of shape (B, C, H, W) where:
                B = batch size, C = 3 (RGB), H = W = 224 (ViT-B/16 input size).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        # ------------------------------------------------------------------
        # Step 1: Extract CLS token features from the backbone.
        # timm VisionTransformer with num_classes=0 returns the CLS token
        # feature vector of shape (B, embed_dim) = (B, 768) for ViT-B/16.
        # ------------------------------------------------------------------
        features: torch.Tensor = self.backbone(x)

        # ------------------------------------------------------------------
        # Safety check: handle the case where timm returns the full sequence
        # instead of just the CLS token. This can happen with some timm
        # versions or configurations.
        # ------------------------------------------------------------------
        if features.dim() == 3:
            # Shape (B, seq_len, embed_dim) — extract CLS token at position 0.
            features = features[:, 0, :]
        elif features.dim() != 2:
            raise RuntimeError(
                f"Unexpected backbone output shape: {features.shape}. "
                "Expected (B, embed_dim) or (B, seq_len, embed_dim)."
            )

        # ------------------------------------------------------------------
        # Step 2: Classification head.
        # features: (B, embed_dim) → logits: (B, num_classes)
        # ------------------------------------------------------------------
        logits: torch.Tensor = self.head(features)

        return logits

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Returns all parameters with requires_grad=True.

        This list is passed to AdamW in Trainer._build_optimizer(). The
        correctness of this list depends entirely on which parameters were
        frozen/unfrozen by PEFTFactory.build() before this model was created.

        Returns:
            List of nn.Parameter objects with requires_grad=True from both
            the backbone and the head. The head always contributes its
            weight and bias. The backbone contributes only PEFT-specific
            parameters (or all parameters for full fine-tuning).
        """
        return [p for p in self.parameters() if p.requires_grad]

    def count_trainable_params(self) -> int:
        """Returns the total number of trainable parameters.

        Sums numel() for all parameters with requires_grad=True. Used by
        PEFTFactory.check_param_cap() to enforce the ≤1.5% constraint and
        by Logger for experiment metadata.

        Returns:
            Integer count of trainable parameters across backbone and head.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# PEFTFactory
# ===========================================================================

class PEFTFactory:
    """Stateless factory for constructing PEFT-modified models.

    Each call to build() is fully independent — the backbone_wrapper is never
    modified, and each trial gets a fresh deepcopy of the backbone. This
    statelessness is critical for HyperparamSearch, which calls build() many
    times in a loop with different hyperparameter combinations.

    All 16 methods are supported:
    - 2 baselines: linear probing, full fine-tuning
    - 2 prompt-based: VPT-Shallow, VPT-Deep
    - 5 adapter-based: Pfeiffer, Houlsby, AdaptFormer, ConvPass, RepAdapter
    - 4 direct selective: BitFit, LayerNorm, DiffFit, SSF
    - 3 efficient selective: LoRA, FacT-TT, FacT-TK
    """

    def __init__(self) -> None:
        """Initialises the factory. No state is maintained."""
        pass

    def build(
        self,
        method: str,
        backbone_wrapper: Any,
        num_classes: int,
        peft_params: Dict[str, Any],
        device: str = "cuda",
    ) -> PEFTModel:
        """Constructs a training-ready PEFTModel for the given method.

        Performs five steps:
        1. Deep-copy the backbone from backbone_wrapper (trial isolation)
        2. Read architecture constants (embed_dim, num_layers)
        3. Apply PEFT modifications based on method
        4. Create randomly initialized classification head
        5. Wrap in PEFTModel and move to device

        Args:
            method: PEFT method name. Must be one of SUPPORTED_METHODS.
                Example: 'lora', 'bitfit', 'vpt_deep', 'linear', 'full'.
            backbone_wrapper: A ViTWrapper or CLIPWrapper instance providing
                get_backbone() and architecture constants (embed_dim, num_layers).
                Never modified — only get_backbone() is called.
            num_classes: Number of output classes for the classification head.
                Varies per VTAB task (e.g., 47 for DTD, 102 for Caltech101).
            peft_params: Dict of method-specific hyperparameters. Keys must
                match config.yaml's peft_methods search grids. Examples:
                    {'lora_rank': 8}
                    {'adapter_bottleneck': 16, 'adapter_scale': 0.1}
                    {'num_prompts': 10}
                    {'fact_rank': 16, 'fact_scale': 1.0}
                Empty dict {} is valid for methods with no hyperparameters
                (bitfit, layernorm, difffit, ssf, linear, full).
            device: Target device string. Default: 'cuda'.
                The returned model is moved to this device.

        Returns:
            A PEFTModel instance with:
            - backbone: PEFT-modified timm VisionTransformer
            - head: Randomly initialized nn.Linear(embed_dim, num_classes)
            - method: The method string for downstream dispatch
            The model is on the specified device and ready for training.

        Raises:
            ValueError: If method is not in SUPPORTED_METHODS.
            KeyError: If a required peft_params key is missing for the method.
            RuntimeError: If backbone modification fails.
        """
        # ------------------------------------------------------------------
        # Validate method name.
        # ------------------------------------------------------------------
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown PEFT method: '{method}'. "
                f"Supported methods: {SUPPORTED_METHODS}"
            )

        # ------------------------------------------------------------------
        # Step 1: Deep-copy the backbone for trial isolation.
        # The backbone_wrapper holds clean pretrained weights and is never
        # modified. Each trial gets a fresh copy.
        # ------------------------------------------------------------------
        backbone: nn.Module = copy.deepcopy(backbone_wrapper.get_backbone())

        # ------------------------------------------------------------------
        # Step 2: Read architecture constants from the backbone_wrapper.
        # ------------------------------------------------------------------
        embed_dim: int = getattr(backbone_wrapper, "embed_dim", 768)
        num_layers: int = getattr(backbone_wrapper, "num_layers", 12)

        # Verify embed_dim from the actual backbone if possible.
        if hasattr(backbone, "embed_dim"):
            embed_dim = int(backbone.embed_dim)
        if hasattr(backbone, "blocks"):
            num_layers = len(backbone.blocks)

        _logger.info(
            "Building PEFTModel: method='%s', num_classes=%d, "
            "embed_dim=%d, num_layers=%d, device='%s'",
            method,
            num_classes,
            embed_dim,
            num_layers,
            device,
        )

        # ------------------------------------------------------------------
        # Step 3: Apply PEFT modifications based on method.
        # ------------------------------------------------------------------
        self._apply_peft(
            method=method,
            backbone=backbone,
            embed_dim=embed_dim,
            num_layers=num_layers,
            peft_params=peft_params,
        )

        # ------------------------------------------------------------------
        # Step 4: Create randomly initialized classification head.
        # Paper: "The prediction head is randomly initialized for each dataset."
        # ------------------------------------------------------------------
        head: nn.Linear = nn.Linear(embed_dim, num_classes)
        nn.init.normal_(head.weight, std=_HEAD_INIT_STD)
        nn.init.zeros_(head.bias)
        # Head always has requires_grad=True (default for new nn.Linear).

        # ------------------------------------------------------------------
        # Step 5: Wrap in PEFTModel and move to device.
        # ------------------------------------------------------------------
        model: PEFTModel = PEFTModel(
            backbone=backbone,
            head=head,
            method=method,
        )
        model = model.to(device)

        # ------------------------------------------------------------------
        # Log trainable parameter count for verification.
        # ------------------------------------------------------------------
        trainable_count: int = model.count_trainable_params()
        _logger.info(
            "PEFTModel built: method='%s', trainable_params=%d (%.4fM), "
            "cap_check=%s",
            method,
            trainable_count,
            trainable_count / 1_000_000,
            "EXEMPT" if method in _CAP_EXEMPT_METHODS
            else ("PASS" if self.check_param_cap(model) else "FAIL"),
        )

        return model

    def check_param_cap(
        self,
        model: PEFTModel,
        cap_ratio: float = PEFT_PARAM_CAP_RATIO,
        total_params: int = TOTAL_VIT_B16_PARAMS,
    ) -> bool:
        """Checks whether the model's trainable parameters are within the cap.

        Paper: "We set a cap for PEFT size ≤ 1.5% of ViT-B/16."
        Config: peft_param_cap.ratio: 0.015, total_params: 86_000_000
        Absolute cap: 0.015 * 86_000_000 = 1_290_000 trainable parameters.

        This is called in HyperparamSearch.run_search() after each build() call.
        If it returns False, the trial is skipped (not trained). Methods in
        _CAP_EXEMPT_METHODS ('linear', 'full') should not be checked by the
        caller (or always return True for them).

        Args:
            model: The PEFTModel to check.
            cap_ratio: Maximum fraction of total_params allowed as trainable.
                Default: 0.015 (config.yaml: peft_param_cap.ratio).
            total_params: Total backbone parameter count used as denominator.
                Default: 86_000_000 (config.yaml: backbones.imagenet21k_vit.total_params).

        Returns:
            True if model.count_trainable_params() <= cap_ratio * total_params.
            False otherwise.
        """
        max_trainable: int = int(cap_ratio * total_params)
        actual_trainable: int = model.count_trainable_params()
        return actual_trainable <= max_trainable

    def list_methods(self) -> List[str]:
        """Returns the list of all supported PEFT method names.

        Used by main.py for CLI argument validation:
            if args.method not in factory.list_methods():
                raise ValueError(...)

        Returns:
            Copy of SUPPORTED_METHODS list (16 method name strings).
        """
        return list(SUPPORTED_METHODS)

    # ------------------------------------------------------------------
    # Private dispatch method
    # ------------------------------------------------------------------

    def _apply_peft(
        self,
        method: str,
        backbone: nn.Module,
        embed_dim: int,
        num_layers: int,
        peft_params: Dict[str, Any],
    ) -> None:
        """Dispatches to the appropriate PEFT modification logic.

        Modifies the backbone in-place. After this method returns, the backbone
        has the correct requires_grad settings and any PEFT modules are
        registered as submodules (so their parameters appear in
        PEFTModel.parameters()).

        Args:
            method: PEFT method name string.
            backbone: Deep-copied timm VisionTransformer to modify in-place.
            embed_dim: Token embedding dimension D (768 for ViT-B/16).
            num_layers: Number of Transformer blocks M (12 for ViT-B/16).
            peft_params: Method-specific hyperparameter dict.
        """
        if method == "linear":
            self._apply_linear(backbone)
        elif method == "full":
            self._apply_full(backbone)
        elif method == "vpt_shallow":
            self._apply_vpt(backbone, embed_dim, num_layers, peft_params, mode="shallow")
        elif method == "vpt_deep":
            self._apply_vpt(backbone, embed_dim, num_layers, peft_params, mode="deep")
        elif method == "bitfit":
            self._apply_selective(backbone, "bitfit")
        elif method == "layernorm":
            self._apply_selective(backbone, "layernorm")
        elif method == "difffit":
            self._apply_selective(backbone, "difffit")
        elif method == "ssf":
            self._apply_selective(backbone, "ssf")
        elif method == "pfeiffer_adapter":
            self._apply_pfeiffer_adapter(backbone, embed_dim, num_layers, peft_params)
        elif method == "houlsby_adapter":
            self._apply_houlsby_adapter(backbone, embed_dim, num_layers, peft_params)
        elif method == "adaptformer":
            self._apply_adaptformer(backbone, embed_dim, num_layers, peft_params)
        elif method == "convpass":
            self._apply_convpass(backbone, embed_dim, num_layers, peft_params)
        elif method == "repadapter":
            self._apply_repadapter(backbone, embed_dim, num_layers, peft_params)
        elif method == "lora":
            self._apply_lora(backbone, embed_dim, num_layers, peft_params)
        elif method == "fact_tt":
            self._apply_fact(backbone, embed_dim, num_layers, peft_params, mode="tt")
        elif method == "fact_tk":
            self._apply_fact(backbone, embed_dim, num_layers, peft_params, mode="tk")
        else:
            raise ValueError(
                f"Unhandled method '{method}' in _apply_peft dispatch. "
                "This should not happen if SUPPORTED_METHODS is correct."
            )

    # ------------------------------------------------------------------
    # Baseline methods
    # ------------------------------------------------------------------

    def _apply_linear(self, backbone: nn.Module) -> None:
        """Linear probing: freeze entire backbone, train only the head.

        Paper: "linear probing ... only updates the prediction heads while
        keeping the backbone frozen."

        After this call, backbone.parameters() all have requires_grad=False.
        The head (created separately in build()) has requires_grad=True.
        """
        for param in backbone.parameters():
            param.requires_grad = False

        _logger.debug("Linear probing: all backbone parameters frozen.")

    def _apply_full(self, backbone: nn.Module) -> None:
        """Full fine-tuning: all backbone parameters are trainable.

        Paper: "full FT ... updates all the model parameters end-to-end."

        After this call, backbone.parameters() all have requires_grad=True.
        This is the default state after deepcopy from a pretrained model,
        but we explicitly set it for clarity.
        """
        for param in backbone.parameters():
            param.requires_grad = True

        total: int = sum(p.numel() for p in backbone.parameters())
        _logger.debug(
            "Full fine-tuning: all %d backbone parameters unfrozen (%.2fM).",
            sum(1 for _ in backbone.parameters()),
            total / 1_000_000,
        )

    # ------------------------------------------------------------------
    # VPT methods
    # ------------------------------------------------------------------

    def _apply_vpt(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_layers: int,
        peft_params: Dict[str, Any],
        mode: str,
    ) -> None:
        """Applies Visual Prompt Tuning (VPT-Shallow or VPT-Deep).

        Paper: "VPT-Shallow adds l prompts P₀ to the input of the first
        Transformer layer only. VPT-Deep inserts l prompts to the input of
        every Transformer layer but their outputs are discarded."
        (Section 2.2, Appendix B.2.1)

        Config search grid:
            vpt_shallow.num_prompts: [5, 10, 50, 100, 200]
            vpt_deep.num_prompts: [5, 10, 50, 100]

        Steps:
        1. Extract num_prompts from peft_params
        2. Freeze backbone
        3. Create VPTModule (owns learnable prompt parameters)
        4. Apply VPT to backbone (wraps blocks with VPTWrappedBlock)
        5. Attach VPTModule to backbone so prompts appear in parameters()

        Args:
            backbone: Backbone to modify in-place.
            embed_dim: Token embedding dimension D.
            num_layers: Number of Transformer blocks M.
            peft_params: Must contain 'num_prompts' key.
            mode: Either 'shallow' or 'deep'.

        Raises:
            KeyError: If 'num_prompts' is not in peft_params.
        """
        if "num_prompts" not in peft_params:
            raise KeyError(
                f"VPT-{mode} requires 'num_prompts' in peft_params. "
                f"Got keys: {list(peft_params.keys())}. "
                f"Config search grid: vpt_{mode}.search_grid.num_prompts"
            )

        num_prompts: int = int(peft_params["num_prompts"])

        # ------------------------------------------------------------------
        # Step 1: Freeze backbone parameters.
        # VPT only trains the prompt tokens and the head.
        # ------------------------------------------------------------------
        for param in backbone.parameters():
            param.requires_grad = False

        # ------------------------------------------------------------------
        # Step 2: Create VPTModule with learnable prompt parameters.
        # ------------------------------------------------------------------
        vpt_module: VPTModule = VPTModule(
            num_prompts=num_prompts,
            embed_dim=embed_dim,
            num_layers=num_layers,
            mode=mode,
        )

        # ------------------------------------------------------------------
        # Step 3: Apply VPT to backbone — wraps each block with VPTWrappedBlock.
        # ------------------------------------------------------------------
        apply_to_vit(backbone=backbone, vpt_module=vpt_module, mode=mode)

        # ------------------------------------------------------------------
        # Step 4: Attach VPTModule to backbone so its parameters appear in
        # backbone.parameters() and thus in PEFTModel.parameters().
        # This is critical: without this, prompt parameters are invisible
        # to the optimizer.
        # ------------------------------------------------------------------
        backbone.vpt_module = vpt_module  # type: ignore[assignment]

        trainable: int = sum(
            p.numel() for p in backbone.parameters() if p.requires_grad
        )
        _logger.debug(
            "VPT-%s applied: num_prompts=%d, trainable_backbone_params=%d (%.4fM).",
            mode,
            num_prompts,
            trainable,
            trainable / 1_000_000,
        )

    # ------------------------------------------------------------------
    # Selective tuning methods
    # ------------------------------------------------------------------

    def _apply_selective(
        self,
        backbone: nn.Module,
        method: str,
    ) -> None:
        """Applies a selective parameter tuning method (BitFit/LayerNorm/DiffFit/SSF).

        Creates a SelectiveModule and calls the appropriate apply_* method.
        The SelectiveModule modifies the backbone in-place and registers any
        extra parameters (DiffFit gammas, SSF scale/shift) as attributes on
        the backbone blocks.

        For DiffFit and SSF, the extra parameters are stored in
        selective._extra_params (an nn.ParameterDict). We attach the
        SelectiveModule to the backbone so these parameters appear in
        backbone.parameters().

        Args:
            backbone: Backbone to modify in-place.
            method: One of 'bitfit', 'layernorm', 'difffit', 'ssf'.
        """
        selective: SelectiveModule = SelectiveModule(
            method=method,
            backbone=backbone,
        )

        if method == "bitfit":
            selective.apply_bitfit()
        elif method == "layernorm":
            selective.apply_layernorm()
        elif method == "difffit":
            selective.apply_difffit()
        elif method == "ssf":
            selective.apply_ssf()
        else:
            raise ValueError(
                f"Unknown selective method: '{method}'. "
                "Expected one of: bitfit, layernorm, difffit, ssf."
            )

        # ------------------------------------------------------------------
        # Attach SelectiveModule to backbone so extra parameters (DiffFit
        # gammas, SSF scale/shift) appear in backbone.parameters().
        # For BitFit and LayerNorm, selective._extra_params is empty, so
        # this attachment is harmless but consistent.
        # ------------------------------------------------------------------
        backbone.selective_module = selective  # type: ignore[assignment]

        trainable: int = sum(
            p.numel() for p in backbone.parameters() if p.requires_grad
        )
        _logger.debug(
            "%s applied: trainable_backbone_params=%d (%.4fM).",
            method,
            trainable,
            trainable / 1_000_000,
        )

    # ------------------------------------------------------------------
    # Adapter-based methods
    # ------------------------------------------------------------------

    def _apply_pfeiffer_adapter(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_layers: int,
        peft_params: Dict[str, Any],
    ) -> None:
        """Applies Pfeiffer Adapter: one sequential adapter after MLP per layer.

        Paper: "Pfeif. Adapter inserts the Adapter solely after the MLP block."
        (Section 2.2, Appendix B.2.2)

        Config search grid:
            pfeiffer_adapter.adapter_scale: [0.01, 0.1, 1.0, 10.0]
            pfeiffer_adapter.adapter_bottleneck: [4, 8, 16, 32]

        Args:
            backbone: Backbone to modify in-place.
            embed_dim: Token embedding dimension D.
            num_layers: Number of Transformer blocks M.
            peft_params: Must contain 'adapter_bottleneck' and 'adapter_scale'.

        Raises:
            KeyError: If required keys are missing from peft_params.
        """
        bottleneck: int = self._get_required_param(
            peft_params, "adapter_bottleneck", "pfeiffer_adapter"
        )
        scale: float = float(self._get_required_param(
            peft_params, "adapter_scale", "pfeiffer_adapter"
        ))

        # Freeze backbone.
        for param in backbone.parameters():
            param.requires_grad = False

        if not hasattr(backbone, "blocks"):
            raise RuntimeError(
                "Backbone does not have 'blocks' attribute. "
                "Cannot apply Pfeiffer Adapter."
            )

        # Insert one PfeifferAdapterBlock per Transformer block.
        for layer_idx in range(min(num_layers, len(backbone.blocks))):
            block: nn.Module = backbone.blocks[layer_idx]

            adapter_block: PfeifferAdapterBlock = PfeifferAdapterBlock(
                embed_dim=embed_dim,
                bottleneck=bottleneck,
                scale=scale,
            )

            # Register adapter as a submodule of the block so its parameters
            # appear in backbone.parameters().
            block.pfeiffer_adapter = adapter_block  # type: ignore[assignment]

            # Wrap the block's forward to apply the adapter after MLP.
            original_forward = block.forward

            def _make_pfeiffer_forward(
                orig_fwd: Any,
                adp: PfeifferAdapterBlock,
            ) -> Any:
                def _forward(x: torch.Tensor) -> torch.Tensor:
                    # Standard block forward: LN1 -> MSA -> residual -> LN2 -> MLP -> residual
                    out: torch.Tensor = orig_fwd(x)
                    # Apply Pfeiffer adapter after M