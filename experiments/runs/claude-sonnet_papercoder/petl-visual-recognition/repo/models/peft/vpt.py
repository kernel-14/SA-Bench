## models/peft/vpt.py
"""Visual Prompt Tuning (VPT) implementation for the PEFT Visual Recognition study.

This module implements VPT-Shallow and VPT-Deep as described in Section 2.2
and Appendix B.2.1 of the paper:

    "Lessons and Insights from a Unifying Study of Parameter-Efficient
    Fine-Tuning (PEFT) in Visual Recognition"

VPT adapts a frozen ViT backbone by prepending learnable "soft prompt" tokens
to the input sequence of Transformer layers. Only the prompt parameters and
the classification head are trained; the backbone remains fully frozen.

Two variants:
- VPT-Shallow: prompts prepended only to the first layer; they persist through
  all subsequent layers as part of the token sequence.
- VPT-Deep: prompts prepended to every layer; their outputs are discarded at
  each layer boundary so fresh prompts are injected at the next layer.

Paper equations (Appendix B.2.1):
    VPT-Shallow: [P̃₁, Z₁] = L₁([P₀, Z₀])
                 [P̃ₘ, Zₘ] = Lₘ([P̃ₘ₋₁, Zₘ₋₁])  m = 2, 3, ..., M
    VPT-Deep:    [_, Zₘ] = Lₘ([Pₘ₋₁, Zₘ₋₁])     m = 1, 2, 3, ..., M

Config reference (config.yaml):
    peft_methods.vpt_shallow.search_grid.num_prompts: [5, 10, 50, 100, 200]
    peft_methods.vpt_shallow.params_range_M: [0.0003, 0.153]
    peft_methods.vpt_deep.search_grid.num_prompts: [5, 10, 50, 100]
    peft_methods.vpt_deep.params_range_M: [0.046, 0.921]
    backbones.imagenet21k_vit.embed_dim: 768
    backbones.imagenet21k_vit.num_layers: 12

Typical usage (called by PEFTFactory):
    import copy
    backbone = copy.deepcopy(vit_wrapper.get_backbone())
    vit_wrapper.freeze_backbone()  # freeze original backbone params

    vpt_module = VPTModule(
        num_prompts=10,
        embed_dim=768,
        num_layers=12,
        mode='deep',
    )
    apply_to_vit(backbone, vpt_module, mode='deep')

    # vpt_module.get_params() returns only the prompt parameters (trainable)
    # backbone parameters remain frozen
"""

import logging
from typing import List

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture constants (config.yaml: backbones.imagenet21k_vit)
# ---------------------------------------------------------------------------
_DEFAULT_EMBED_DIM: int = 768    # config.yaml: backbones.imagenet21k_vit.embed_dim
_DEFAULT_NUM_LAYERS: int = 12    # config.yaml: backbones.imagenet21k_vit.num_layers

# Prompt initialization standard deviation — matches ViT patch embedding init.
_PROMPT_INIT_STD: float = 0.02

# Valid VPT modes.
_VALID_MODES: set = {"shallow", "deep"}


# ---------------------------------------------------------------------------
# VPTModule: owns and manages learnable prompt parameters
# ---------------------------------------------------------------------------

class VPTModule(nn.Module):
    """Learnable prompt token parameters for Visual Prompt Tuning.

    Owns the prompt parameters as an nn.ParameterList so they are properly
    registered with PyTorch and returned by model.parameters(). The actual
    injection logic lives in VPTWrappedBlock.

    For VPT-Deep: one prompt set per Transformer layer (num_layers entries).
    For VPT-Shallow: one prompt set for the first layer only (1 entry).

    Attributes:
        num_prompts: Number of prompt tokens prepended per layer.
        embed_dim: Token embedding dimension D (768 for ViT-B/16).
        num_layers: Number of Transformer layers M (12 for ViT-B/16).
        mode: Either 'shallow' or 'deep'.
        prompt_tokens: nn.ParameterList of learnable prompt tensors.
            Deep mode: num_layers entries, each shape (num_prompts, embed_dim).
            Shallow mode: 1 entry, shape (num_prompts, embed_dim).
    """

    def __init__(
        self,
        num_prompts: int,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        num_layers: int = _DEFAULT_NUM_LAYERS,
        mode: str = "deep",
    ) -> None:
        """Initialises prompt parameters with truncated normal distribution.

        Args:
            num_prompts: Number of learnable prompt tokens per layer.
                VPT-Shallow search grid: {5, 10, 50, 100, 200}
                    (config.yaml: peft_methods.vpt_shallow.search_grid.num_prompts)
                VPT-Deep search grid: {5, 10, 50, 100}
                    (config.yaml: peft_methods.vpt_deep.search_grid.num_prompts)
            embed_dim: Token embedding dimension. Default: 768
                (config.yaml: backbones.imagenet21k_vit.embed_dim).
            num_layers: Number of Transformer layers. Default: 12
                (config.yaml: backbones.imagenet21k_vit.num_layers).
                Only used for mode='deep' to determine the number of prompt sets.
            mode: VPT variant. Either 'shallow' or 'deep'. Default: 'deep'.
                - 'shallow': one prompt set for the first layer only.
                - 'deep': one prompt set per Transformer layer.

        Raises:
            ValueError: If mode is not 'shallow' or 'deep'.
            ValueError: If num_prompts <= 0 or embed_dim <= 0 or num_layers <= 0.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Input validation.
        # ------------------------------------------------------------------
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid VPT mode: '{mode}'. Must be one of {_VALID_MODES}."
            )
        if num_prompts <= 0:
            raise ValueError(
                f"num_prompts must be positive, got {num_prompts}."
            )
        if embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be positive, got {embed_dim}."
            )
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {num_layers}."
            )

        self.num_prompts: int = num_prompts
        self.embed_dim: int = embed_dim
        self.num_layers: int = num_layers
        self.mode: str = mode

        # ------------------------------------------------------------------
        # Build prompt_tokens as nn.ParameterList.
        # Deep mode: num_layers prompt sets (one per Transformer layer).
        # Shallow mode: 1 prompt set (only for the first layer).
        # ------------------------------------------------------------------
        num_prompt_sets: int = num_layers if mode == "deep" else 1

        self.prompt_tokens: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(torch.empty(num_prompts, embed_dim))
                for _ in range(num_prompt_sets)
            ]
        )

        # ------------------------------------------------------------------
        # Initialise with truncated normal (std=0.02), matching ViT's
        # patch embedding initialization convention.
        # ------------------------------------------------------------------
        for prompt_param in self.prompt_tokens:
            nn.init.trunc_normal_(prompt_param, mean=0.0, std=_PROMPT_INIT_STD)

        # ------------------------------------------------------------------
        # Log parameter count for verification against paper Table 3.
        # ------------------------------------------------------------------
        total_params: int = sum(p.numel() for p in self.prompt_tokens)
        _logger.info(
            "VPTModule initialised: mode='%s', num_prompts=%d, embed_dim=%d, "
            "num_layers=%d, num_prompt_sets=%d, total_prompt_params=%d (%.4fM)",
            mode,
            num_prompts,
            embed_dim,
            num_layers,
            num_prompt_sets,
            total_params,
            total_params / 1_000_000,
        )

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Returns the prompt tensor for the given Transformer layer.

        This is a simple accessor; the actual injection (concatenation with
        the token sequence) is performed in VPTWrappedBlock.forward().

        Args:
            x: Input token tensor (unused; present for interface consistency).
                Shape: (B, seq_len, embed_dim).
            layer_idx: Index of the Transformer layer requesting prompts.
                For deep mode: must be in [0, num_layers - 1].
                For shallow mode: any value is accepted; always returns
                prompt_tokens[0].

        Returns:
            Prompt tensor of shape (num_prompts, embed_dim).
            The caller is responsible for expanding the batch dimension.

        Raises:
            IndexError: If mode='deep' and layer_idx is out of range.
        """
        if self.mode == "deep":
            if not (0 <= layer_idx < len(self.prompt_tokens)):
                raise IndexError(
                    f"layer_idx={layer_idx} is out of range for VPT-Deep with "
                    f"{len(self.prompt_tokens)} prompt sets "
                    f"(valid range: 0 to {len(self.prompt_tokens) - 1})."
                )
            return self.prompt_tokens[layer_idx]
        else:
            # Shallow mode: always return the single prompt set.
            return self.prompt_tokens[0]

    def get_params(self) -> List[nn.Parameter]:
        """Returns all learnable prompt parameters.

        Used by PEFTFactory and Trainer to confirm only prompt parameters
        are trainable (backbone parameters remain frozen).

        Returns:
            List of nn.Parameter objects from self.prompt_tokens.
            Deep mode: num_layers parameters, each shape (num_prompts, embed_dim).
            Shallow mode: 1 parameter, shape (num_prompts, embed_dim).
        """
        return list(self.prompt_tokens)


# ---------------------------------------------------------------------------
# VPTWrappedBlock: wraps a single timm Block to inject prompts
# ---------------------------------------------------------------------------

class VPTWrappedBlock(nn.Module):
    """Wrapper around a timm Transformer Block that injects VPT prompt tokens.

    Replaces each Block in backbone.blocks with this wrapper. The original
    block's parameters remain frozen; only the prompt parameters (owned by
    VPTModule) are trainable.

    Token ordering convention (CLS first, then prompts, then patches):
        Input to block:  [CLS | prompts | patch_1 | ... | patch_N]
        Output of block: [CLS | prompts | patch_1 | ... | patch_N]

    For VPT-Deep, prompt outputs are stripped after each block:
        Stripped output: [CLS | patch_1 | ... | patch_N]
    Fresh prompts are then prepended at the next block.

    For VPT-Shallow, prompts persist through all layers (no stripping).
    The first block prepends prompts; subsequent blocks receive the full
    sequence including prompt tokens from the previous layer.

    Attributes:
        original_block: The frozen timm Block being wrapped.
        prompt_module: The VPTModule owning the learnable prompt parameters.
        layer_idx: Index of this block in the backbone (0-indexed).
        mode: Either 'shallow' or 'deep'.
        num_prompts: Number of prompt tokens (cached from prompt_module).
    """

    def __init__(
        self,
        original_block: nn.Module,
        prompt_module: VPTModule,
        layer_idx: int,
        mode: str,
    ) -> None:
        """Initialises the wrapped block.

        Args:
            original_block: The timm Block instance to wrap. Its parameters
                must already be frozen (requires_grad=False) by the time
                forward() is called during training.
            prompt_module: The VPTModule instance owning the prompt parameters.
                Shared across all VPTWrappedBlock instances for the same model.
            layer_idx: Index of this block in backbone.blocks (0-indexed).
                Used to retrieve the correct prompt set in deep mode.
            mode: VPT variant. Either 'shallow' or 'deep'.

        Raises:
            ValueError: If mode is not 'shallow' or 'deep'.
        """
        super().__init__()

        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid VPT mode: '{mode}'. Must be one of {_VALID_MODES}."
            )

        self.original_block: nn.Module = original_block
        self.prompt_module: VPTModule = prompt_module
        self.layer_idx: int = layer_idx
        self.mode: str = mode
        self.num_prompts: int = prompt_module.num_prompts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Injects prompt tokens and runs the original Transformer block.

        Handles both VPT-Shallow and VPT-Deep injection logic.

        VPT-Shallow logic:
            - Layer 0: prepend prompts → run block → return full output
              (CLS + prompts + patches persist to next layer).
            - Layer > 0: x already contains prompt tokens from previous layer.
              Pass directly through original_block unchanged.

        VPT-Deep logic:
            - Every layer: prepend fresh prompts → run block → strip prompt
              outputs → return (CLS + patches) for next layer.

        Args:
            x: Input token tensor of shape (B, seq_len, D) where:
                - For VPT-Deep: seq_len = 1 + N (CLS + N patches) at every layer.
                - For VPT-Shallow at layer 0: seq_len = 1 + N.
                - For VPT-Shallow at layer > 0: seq_len = 1 + num_prompts + N.
                B = batch size, N = number of patch tokens (196 for 224×224/16²),
                D = embed_dim = 768.

        Returns:
            Output token tensor:
            - VPT-Deep: shape (B, 1 + N, D) — prompt outputs stripped.
            - VPT-Shallow layer 0: shape (B, 1 + num_prompts + N, D).
            - VPT-Shallow layer > 0: shape (B, 1 + num_prompts + N, D).
        """
        batch_size: int = x.shape[0]

        if self.mode == "shallow":
            return self._forward_shallow(x, batch_size)
        else:
            return self._forward_deep(x, batch_size)

    def _forward_shallow(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """VPT-Shallow forward pass.

        Paper equation (Appendix B.2.1):
            [P̃₁, Z₁] = L₁([P₀, Z₀])
            [P̃ₘ, Zₘ] = Lₘ([P̃ₘ₋₁, Zₘ₋₁])  m = 2, 3, ..., M

        At layer 0: prepend P₀ to Z₀ and run through L₁.
        At layer > 0: P̃ₘ₋₁ is already part of x (persisted from previous layer).

        Args:
            x: Input token tensor of shape (B, seq_len, D).
            batch_size: Batch size B.

        Returns:
            Output tensor:
            - Layer 0: shape (B, 1 + num_prompts + N, D).
            - Layer > 0: shape (B, 1 + num_prompts + N, D) — unchanged seq_len.
        """
        if self.layer_idx == 0:
            # ------------------------------------------------------------------
            # First layer: prepend prompt tokens to the input sequence.
            # x shape: (B, 1 + N, D) — CLS token + N patch tokens.
            # ------------------------------------------------------------------

            # Retrieve prompt tensor: shape (num_prompts, D).
            prompts: torch.Tensor = self.prompt_module.forward(x, layer_idx=0)

            # Expand to batch dimension: (B, num_prompts, D).
            # expand() avoids memory copy — prompts are shared across batch.
            prompts_expanded: torch.Tensor = prompts.unsqueeze(0).expand(
                batch_size, -1, -1
            )

            # Concatenate: [CLS | prompts | patches]
            # x[:, :1, :] = CLS token, shape (B, 1, D)
            # prompts_expanded = shape (B, num_prompts, D)
            # x[:, 1:, :] = patch tokens, shape (B, N, D)
            x_with_prompts: torch.Tensor = torch.cat(
                [x[:, :1, :], prompts_expanded, x[:, 1:, :]],
                dim=1,
            )
            # x_with_prompts shape: (B, 1 + num_prompts + N, D)

            # Run through the original frozen Transformer block.
            output: torch.Tensor = self.original_block(x_with_prompts)
            # output shape: (B, 1 + num_prompts + N, D)

            return output

        else:
            # ------------------------------------------------------------------
            # Subsequent layers: prompt tokens from the previous layer are
            # already part of x (they persist in VPT-Shallow).
            # x shape: (B, 1 + num_prompts + N, D)
            # Pass directly through the original block unchanged.
            # ------------------------------------------------------------------
            return self.original_block(x)

    def _forward_deep(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """VPT-Deep forward pass.

        Paper equation (Appendix B.2.1):
            [_, Zₘ] = Lₘ([Pₘ₋₁, Zₘ₋₁])  m = 1, 2, 3, ..., M

        At every layer: prepend fresh prompts, run block, discard prompt outputs.

        Args:
            x: Input token tensor of shape (B, 1 + N, D).
                CLS token at position 0, N patch tokens at positions 1..N.
                Prompt outputs from the previous layer have already been stripped.
            batch_size: Batch size B.

        Returns:
            Output tensor of shape (B, 1 + N, D) — prompt outputs stripped.
            The CLS token is at position 0; patch tokens follow.
        """
        # ------------------------------------------------------------------
        # Step 1: Retrieve fresh prompt tokens for this layer.
        # Shape: (num_prompts, D).
        # ------------------------------------------------------------------
        prompts: torch.Tensor = self.prompt_module.forward(x, layer_idx=self.layer_idx)

        # Expand to batch dimension: (B, num_prompts, D).
        # expand() avoids memory copy — same prompts for all samples in batch.
        prompts_expanded: torch.Tensor = prompts.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # ------------------------------------------------------------------
        # Step 2: Concatenate [CLS | prompts | patches].
        # x[:, :1, :] = CLS token, shape (B, 1, D)
        # prompts_expanded = shape (B, num_prompts, D)
        # x[:, 1:, :] = patch tokens, shape (B, N, D)
        # ------------------------------------------------------------------
        x_with_prompts: torch.Tensor = torch.cat(
            [x[:, :1, :], prompts_expanded, x[:, 1:, :]],
            dim=1,
        )
        # x_with_prompts shape: (B, 1 + num_prompts + N, D)

        # ------------------------------------------------------------------
        # Step 3: Run through the original frozen Transformer block.
        # ------------------------------------------------------------------
        output: torch.Tensor = self.original_block(x_with_prompts)
        # output shape: (B, 1 + num_prompts + N, D)

        # ------------------------------------------------------------------
        # Step 4: Strip prompt outputs.
        # Keep CLS (position 0) and patch tokens (positions 1+num_prompts onward).
        # Discard prompt outputs (positions 1 to num_prompts inclusive).
        # ------------------------------------------------------------------
        # output[:, :1, :] = CLS token output, shape (B, 1, D)
        # output[:, 1:1+num_prompts, :] = prompt outputs (DISCARDED)
        # output[:, 1+num_prompts:, :] = patch token outputs, shape (B, N, D)
        stripped_output: torch.Tensor = torch.cat(
            [output[:, :1, :], output[:, 1 + self.num_prompts:, :]],
            dim=1,
        )
        # stripped_output shape: (B, 1 + N, D)

        return stripped_output


# ---------------------------------------------------------------------------
# apply_to_vit: modifies backbone in-place by replacing all blocks
# ---------------------------------------------------------------------------

def apply_to_vit(
    backbone: nn.Module,
    vpt_module: VPTModule,
    mode: str,
) -> None:
    """Modifies the ViT backbone in-place by wrapping all Transformer blocks.

    Replaces each Block in backbone.blocks with a VPTWrappedBlock that
    injects prompt tokens at the correct position during the forward pass.

    This function must be called AFTER the backbone is deepcopied (to avoid
    modifying the original ViTWrapper's backbone) and BEFORE or AFTER
    freeze_backbone() — either order is correct since VPTWrappedBlock holds
    a reference to the original block whose parameters are frozen by the
    backbone freeze call.

    Recommended call order in PEFTFactory.build():
        1. backbone = copy.deepcopy(vit_wrapper.get_backbone())
        2. vit_wrapper.freeze_backbone()  # or freeze the copy directly
        3. vpt_module = VPTModule(num_prompts, embed_dim, num_layers, mode)
        4. apply_to_vit(backbone, vpt_module, mode)
        5. model = PEFTModel(backbone, head, peft_module=vpt_module, method=...)

    After this function returns:
    - backbone.blocks[i] is a VPTWrappedBlock for all i in [0, num_layers-1].
    - The original Block instances are stored inside each VPTWrappedBlock.
    - timm's VisionTransformer.forward_features() calls self.blocks sequentially,
      so the wrapped blocks are invoked automatically.
    - The final norm layer (backbone.norm) and CLS token extraction
      (x[:, 0]) in timm's forward_features() are unaffected.

    Args:
        backbone: The timm VisionTransformer backbone (with num_classes=0).
            Must have a backbone.blocks attribute (nn.Sequential or nn.ModuleList
            of timm Block instances). Modified in-place.
        vpt_module: The VPTModule instance owning the learnable prompt parameters.
            Shared across all VPTWrappedBlock instances.
        mode: VPT variant. Either 'shallow' or 'deep'.

    Raises:
        AttributeError: If backbone does not have a 'blocks' attribute.
        ValueError: If mode is not 'shallow' or 'deep'.
        RuntimeError: If backbone.blocks is empty.
    """
    # ------------------------------------------------------------------
    # Input validation.
    # ------------------------------------------------------------------
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid VPT mode: '{mode}'. Must be one of {_VALID_MODES}."
        )

    if not hasattr(backbone, "blocks"):
        raise AttributeError(
            "Backbone does not have a 'blocks' attribute. "
            "Expected a timm VisionTransformer with backbone.blocks "
            "(nn.Sequential or nn.ModuleList of Block instances)."
        )

    num_blocks: int = len(backbone.blocks)

    if num_blocks == 0:
        raise RuntimeError(
            "backbone.blocks is empty. Cannot apply VPT to a backbone "
            "with no Transformer blocks."
        )

    _logger.info(
        "Applying VPT to backbone: mode='%s', num_blocks=%d, "
        "num_prompts=%d, embed_dim=%d",
        mode,
        num_blocks,
        vpt_module.num_prompts,
        vpt_module.embed_dim,
    )

    # ------------------------------------------------------------------
    # Validate that vpt_module has the correct number of prompt sets.
    # Deep mode requires num_layers prompt sets; shallow requires 1.
    # ------------------------------------------------------------------
    expected_prompt_sets: int = num_blocks if mode == "deep" else 1
    actual_prompt_sets: int = len(vpt_module.prompt_tokens)

    if actual_prompt_sets != expected_prompt_sets:
        raise ValueError(
            f"VPTModule has {actual_prompt_sets} prompt set(s), but mode='{mode}' "
            f"with {num_blocks} blocks requires {expected_prompt_sets} prompt set(s). "
            "Ensure VPTModule was constructed with the correct num_layers and mode."
        )

    # ------------------------------------------------------------------
    # Replace each Block with a VPTWrappedBlock.
    # backbone.blocks is a timm nn.Sequential; we replace entries by index.
    # ------------------------------------------------------------------
    for layer_idx in range(num_blocks):
        original_block: nn.Module = backbone.blocks[layer_idx]

        wrapped_block: VPTWrappedBlock = VPTWrappedBlock(
            original_block=original_block,
            prompt_module=vpt_module,
            layer_idx=layer_idx,
            mode=mode,
        )

        # Replace the block in the backbone's blocks container.
        # timm uses nn.Sequential for backbone.blocks, which supports
        # item assignment via __setitem__.
        backbone.blocks[layer_idx] = wrapped_block

        _logger.debug(
            "Replaced backbone.blocks[%d] with VPTWrappedBlock "
            "(mode='%s', num_prompts=%d).",
            layer_idx,
            mode,
            vpt_module.num_prompts,
        )

    _logger.info(
        "VPT applied successfully: %d blocks wrapped with VPTWrappedBlock "
        "(mode='%s').",
        num_blocks,
        mode,
    )
