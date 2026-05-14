```python
## model/navil_model.py
"""Top-level NaViL multimodal model assembly.

This module implements ``NaViLModel``, the central class that wires together:
- ``VisualEncoder``: bidirectional transformer that converts image patches to
  semantic visual token embeddings.
- ``Connector``: pixel-shuffle + MLP projector that bridges visual encoder
  output to the LLM's embedding space.
- ``MoELLM``: MoE-extended causal LLM with modality-specific experts.
- ``SpecialTokens``: structural tokens for image layout encoding.
- ``AutoTokenizer``: tokenizer extended with the four special tokens.

Training stage parameter freezing follows the paper's three-stage recipe:
- S1.1: only visual encoder, connector, and MoE visual experts are trainable.
- S1.2: additionally unfreezes linguistic expert attention projections.
- S2:   all parameters are trainable.

Modality mask convention (shared across the codebase):
    modality_mask: LongTensor of shape (B, L)
    0 = visual token  → routes to visual_expert in MoELayer
    1 = text token    → routes to linguistic_expert in MoELayer
    All four special tokens are treated as visual tokens (mask=0).

Config alignment (configs/navil_2b.yaml):
    model.visual_encoder.{depth, width, mlp_width, num_heads, patch_size}
    model.connector.pixel_shuffle_factor
    model.llm.{name_or_path, width}
    training.{s1_1, s1_2, s2}.{trainable_modules, frozen_modules}
    inference.{tau, min_area_threshold, patch_size}
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.connector import Connector
from model.moe_llm import MoELLM
from model.special_tokens import SpecialTokens
from model.visual_encoder import VisualEncoder

logger: logging.Logger = logging.getLogger(__name__)


class NaViLModel(nn.Module):
    """Native Multimodal Large Language Model (NaViL).

    Assembles the visual encoder, connector, MoE-extended LLM, tokenizer,
    and special tokens into a single end-to-end trainable module. Supports
    three training stages with different parameter freezing configurations,
    multi-scale visual token packing, and autoregressive generation.

    Args:
        config: OmegaConf DictConfig loaded from ``configs/navil_2b.yaml``
                or ``configs/navil_9b.yaml``. All sub-component constructors
                read their hyperparameters from this config.

    Attributes:
        config:          The OmegaConf config used to construct this model.
        special_tokens:  ``SpecialTokens`` instance managing the four
                         structural image tokens.
        tokenizer:       HuggingFace tokenizer extended with special tokens.
        visual_encoder:  Bidirectional visual transformer encoder.
        connector:       Pixel-shuffle + MLP projector.
        llm:             MoE-extended causal LLM.
        patch_size:      Patch size in pixels (16 for both NaViL-2B and 9B).
        pixel_shuffle_factor: Spatial compression factor for the connector.

    Example::

        from omegaconf import OmegaConf
        config = OmegaConf.load("configs/navil_2b.yaml")
        model = NaViLModel(config)
        model.freeze_params_for_stage("s1_1")
    """

    def __init__(self, config: DictConfig) -> None:
        """Construct all sub-components from config and wire them together.

        Steps:
        1. Instantiate SpecialTokens.
        2. Load tokenizer and register special tokens.
        3. Instantiate VisualEncoder from config.model.visual_encoder.
        4. Instantiate Connector from config.model.connector + llm.width.
        5. Load MoELLM from pretrained weights (replaces dense layers with MoE).
        6. Resize token embeddings to cover the 4 newly added special tokens.

        Args:
            config: OmegaConf DictConfig with model, training, and inference
                    sections as defined in configs/navil_2b.yaml.
        """
        super().__init__()

        self.config: DictConfig = config

        # ------------------------------------------------------------------ #
        # 1. Special tokens                                                    #
        # ------------------------------------------------------------------ #
        self.special_tokens: SpecialTokens = SpecialTokens()

        # ------------------------------------------------------------------ #
        # 2. Tokenizer — load then register special tokens                    #
        # ------------------------------------------------------------------ #
        llm_path: str = config.model.llm.name_or_path
        logger.info("Loading tokenizer from %s", llm_path)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            trust_remote_code=True,
            use_fast=False,
        )

        # Ensure pad token is set (required for batched generation)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # Register the four NaViL structural tokens
        self.tokenizer = self.special_tokens.register_with_tokenizer(self.tokenizer)
        logger.info(
            "Tokenizer vocabulary size after special token registration: %d",
            len(self.tokenizer),
        )

        # ------------------------------------------------------------------ #
        # 3. Visual encoder                                                    #
        # ------------------------------------------------------------------ #
        ve_cfg = config.model.visual_encoder
        logger.info(
            "Building VisualEncoder: depth=%d, width=%d, mlp_width=%d, "
            "num_heads=%d, patch_size=%d",
            ve_cfg.depth,
            ve_cfg.width,
            ve_cfg.mlp_width,
            ve_cfg.num_heads,
            ve_cfg.patch_size,
        )
        self.visual_encoder: VisualEncoder = VisualEncoder(
            depth=int(ve_cfg.depth),
            width=int(ve_cfg.width),
            mlp_width=int(ve_cfg.mlp_width),
            num_heads=int(ve_cfg.num_heads),
            patch_size=int(ve_cfg.patch_size),
        )

        # ------------------------------------------------------------------ #
        # 4. Connector                                                         #
        # ------------------------------------------------------------------ #
        conn_cfg = config.model.connector
        llm_cfg = config.model.llm
        logger.info(
            "Building Connector: visual_dim=%d, llm_dim=%d, "
            "pixel_shuffle_factor=%d",
            ve_cfg.width,
            llm_cfg.width,
            conn_cfg.pixel_shuffle_factor,
        )
        self.connector: Connector = Connector(
            visual_dim=int(ve_cfg.width),
            llm_dim=int(llm_cfg.width),
            pixel_shuffle_factor=int(conn_cfg.pixel_shuffle_factor),
        )

        # ------------------------------------------------------------------ #
        # 5. MoE-extended LLM                                                 #
        # ------------------------------------------------------------------ #
        logger.info("Loading MoELLM from pretrained: %s", llm_path)
        self.llm: MoELLM = MoELLM.from_pretrained(llm_path)

        # ------------------------------------------------------------------ #
        # 6. Resize token embeddings for the 4 new special tokens             #
        # ------------------------------------------------------------------ #
        new_vocab_size: int = len(self.tokenizer)
        logger.info(
            "Resizing token embeddings to %d (added %d special tokens)",
            new_vocab_size,
            4,
        )
        self.llm.resize_token_embeddings(new_vocab_size)

        # ------------------------------------------------------------------ #
        # Convenience attributes used in build_multimodal_embeds              #
        # ------------------------------------------------------------------ #
        self.patch_size: int = int(ve_cfg.patch_size)
        self.pixel_shuffle_factor: int = int(conn_cfg.pixel_shuffle_factor)

    # ---------------------------------------------------------------------- #
    # Class methods for serialisation                                          #
    # ---------------------------------------------------------------------- #

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: DictConfig,
    ) -> "NaViLModel":
        """Load a saved NaViL checkpoint.

        Constructs the model architecture from ``config``, then loads the
        full state dict from ``path``. The base LLM weights are NOT
        re-downloaded — the saved state dict already contains the
        MoE-extended weights.

        Args:
            path:   Path to the checkpoint directory created by
                    ``save_pretrained``. Must contain either
                    ``navil_model.safetensors`` or ``pytorch_model.bin``.
            config: OmegaConf DictConfig for model construction.

        Returns:
            A ``NaViLModel`` instance with weights loaded from ``path``.

        Raises:
            FileNotFoundError: If no weight file is found in ``path``.
        """
        ckpt_path: Path = Path(path)

        # Construct the model architecture
        model: "NaViLModel" = cls(config)

        # Determine weight file path
        safetensors_path: Path = ckpt_path / "navil_model.safetensors"
        pytorch_bin_path: Path = ckpt_path / "pytorch_model.bin"

        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file as safetensors_load
                state_dict: Dict[str, torch.Tensor] = safetensors_load(
                    str(safetensors_path), device="cpu"
                )
                logger.info("Loading weights from %s", safetensors_path)
            except ImportError:
                logger.warning(
                    "safetensors not available; falling back to torch.load"
                )
                state_dict = torch.load(str(safetensors_path), map_location="cpu")
        elif pytorch_bin_path.exists():
            state_dict = torch.load(str(pytorch_bin_path), map_location="cpu")
            logger.info("Loading weights from %s", pytorch_bin_path)
        else:
            raise FileNotFoundError(
                f"No weight file found in {ckpt_path}. "
                "Expected 'navil_model.safetensors' or 'pytorch_model.bin'."
            )

        missing_keys: List[str]
        unexpected_keys: List[str]
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=False
        )

        if missing_keys:
            logger.warning(
                "Missing keys when loading NaViLModel: %s", missing_keys[:10]
            )
        if unexpected_keys:
            logger.warning(
                "Unexpected keys when loading NaViLModel: %s",
                unexpected_keys[:10],
            )

        # Optionally load tokenizer from checkpoint if present
        tokenizer_dir: Path = ckpt_path / "tokenizer"
        if tokenizer_dir.exists():
            model.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_dir),
                trust_remote_code=True,
                use_fast=False,
            )
            logger.info("Tokenizer loaded from checkpoint: %s", tokenizer_dir)

        logger.info("NaViLModel loaded from %s", path)
        return model

    def save_pretrained(self, path: str) -> None:
        """Persist the full model state to disk.

        Saves:
        - Model state dict as ``navil_model.safetensors`` (or
          ``pytorch_model.bin`` if safetensors is unavailable).
        - Tokenizer to ``tokenizer/`` subdirectory.
        - Config as ``config.json``.

        Args:
            path: Directory path to save into. Created if it does not exist.
        """
        save_path: Path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        state_dict: Dict[str, torch.Tensor] = self.state_dict()

        try:
            from safetensors.torch import save_file as safetensors_save
            weight_path: Path = save_path / "navil_model.safetensors"
            # safetensors requires contiguous tensors
            contiguous_state_dict: Dict[str, torch.Tensor] = {
                k: v.contiguous() for k, v in state_dict.items()
            }
            safetensors_save(contiguous_state_dict, str(weight_path))
            logger.info("Model weights saved to %s", weight_path)
        except ImportError:
            weight_path = save_path / "pytorch_model.bin"
            torch.save(state_dict, str(weight_path))
            logger.info(
                "safetensors not available; weights saved to %s", weight_path
            )

        # Save tokenizer
        tokenizer_dir: Path = save_path / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(tokenizer_dir))
        logger.info("Tokenizer saved to %s", tokenizer_dir)

        # Save config as JSON
        config_path: Path = save_path / "config.json"
        try:
            config_dict: Any = OmegaConf.to_container(
                self.config, resolve=True, throw_on_missing=False
            )
            with open(str(config_path), "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2)
            logger.info("Config saved to %s", config_path)
        except Exception as exc:
            logger.warning("Failed to save config.json: %s", exc)

    # ---------------------------------------------------------------------- #
    # Parameter management                                                     #
    # ---------------------------------------------------------------------- #

    def get_trainable_params(self, stage: str) -> List[nn.Parameter]:
        """Return parameters that should have requires_grad=True for a stage.

        Uses the ``trainable_modules`` list from the config YAML to determine
        which named parameters are trainable. The matching is done via
        string prefix/pattern matching against the full parameter name
        (e.g., ``"visual_encoder.layers.0.norm1.weight"``).

        Special case: if ``trainable_modules`` contains ``"all"``, all
        parameters are returned (used for stage s2).

        Args:
            stage: Training stage identifier: ``"s1_1"``, ``"s1_2"``,
                   or ``"s2"``.

        Returns:
            List of ``nn.Parameter`` objects that should be trainable.

        Raises:
            ValueError: If ``stage`` is not one of the three valid stages.
        """
        valid_stages: Tuple[str, ...] = ("s1_1", "s1_2", "s2")
        if stage not in valid_stages:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of {valid_stages}."
            )

        # Retrieve trainable module patterns from config
        stage_cfg: DictConfig = getattr(self.config.training, stage)
        trainable_modules: List[str] = list(
            OmegaConf.to_container(stage_cfg.trainable_modules, resolve=True)
        )

        # Special case: "all" means every parameter is trainable
        if "all" in trainable_modules:
            return list(self.parameters())

        trainable_params: List[nn.Parameter] = []
        seen_ids: set = set()

        for param_name, param in self.named_parameters():
            if self._matches_trainable_pattern(param_name, trainable_modules):
                param_id: int = id(param)
                if param_id not in seen_ids:
                    trainable_params.append(param)
                    seen_ids.add(param_id)

        logger.debug(
            "Stage '%s': %d trainable parameter tensors found.",
            stage,
            len(trainable_params),
        )
        return trainable_params

    def _matches_trainable_pattern(
        self,
        param_name: str,
        patterns: List[str],
    ) -> bool:
        """Check if a parameter name matches any of the trainable patterns.

        Patterns from the config YAML use dot-separated module paths with
        optional wildcard ``*`` segments. Examples:
            - ``"visual_encoder"``         → matches all visual_encoder params
            - ``"connector"``              → matches all connector params
            - ``"llm.moe_layers.*.visual_expert"`` → matches visual expert params
              in any MoE layer (``*`` matches any single path segment)

        The matching logic:
        1. If the pattern is a prefix of the param_name (with a ``.`` separator
           or exact match), it matches.
        2. If the pattern contains ``*``, split both pattern and name by ``.``
           and match segment by segment (``*`` matches any single segment).

        Args:
            param_name: Full dotted parameter name from ``named_parameters()``.
            patterns:   List of pattern strings from config trainable_modules.

        Returns:
            True if param_name matches any pattern, False otherwise.
        """
        for pattern in patterns:
            if "*" in pattern:
                # Wildcard matching: split into segments and match
                if self._wildcard_match(param_name, pattern):
                    return True
            else:
                # Prefix matching: param_name starts with pattern
                # Allow exact match or match followed by "."
                if param_name == pattern or param_name.startswith(pattern + "."):
                    return True
        return False

    def _wildcard_match(self, name: str, pattern: str) -> bool:
        """Match a parameter name against a pattern with ``*`` wildcards.

        ``*`` matches exactly one dot-separated path segment.

        Args:
            name:    Full dotted parameter name.
            pattern: Pattern string with optional ``*`` segments.

        Returns:
            True if name matches the pattern, False otherwise.

        Example::

            _wildcard_match(
                "llm.layers.3.visual_expert.q_proj.weight",
                "llm.moe_layers.*.visual_expert"
            )
            # Returns True because "3" matches "*" and the prefix matches.
        """
        # Normalize: the config uses "llm.moe_layers.*" but the actual
        # parameter names use "llm.layers.*" (from MoELLM's module structure).
        # Handle both naming conventions.
        normalized_pattern: str = pattern.replace("llm.moe_layers.", "llm.layers.")

        name_parts: List[str] = name.split(".")
        pattern_parts: List[str] = normalized_pattern.split(".")

        # The pattern must be a prefix of the name (name can have more segments)
        if len(pattern_parts) > len(name_parts):
            return False

        for name_seg, pat_seg in zip(name_parts, pattern_parts):
            if pat_seg == "*":
                continue  # wildcard matches any single segment
            if name_seg != pat_seg:
                return False

        return True

    def freeze_params_for_stage(self, stage: str) -> None:
        """Set requires_grad appropriately for the given training stage.

        Two-pass approach:
        1. Freeze all parameters (requires_grad=False).
        2. Unfreeze only the trainable parameters for this stage.

        This ensures no parameter is accidentally left trainable from a
        previous stage when resuming or switching stages.

        Args:
            stage: Training stage: ``"s1_1"``, ``"s1_2"``, or ``"s2"``.
        """
        # Pass 1: freeze everything
        param: nn.Parameter
        for param in self.parameters():
            param.requires_grad_(False)

        # Pass 2: unfreeze stage-specific params
        trainable_params: List[nn.Parameter] = self.get_trainable_params(stage)
        for param in trainable_params:
            param.requires_grad_(True)

        # Log summary
        total_params: int = sum(p.numel() for p in self.parameters())
        trainable_count: int = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        frozen_count: int = total_params - trainable_count

        logger.info(
            "Stage '%s' parameter freeze: total=%d, trainable=%d (%.1f%%), "
            "frozen=%d (%.1f%%)",
            stage,
            total_params,
            trainable_count,
            100.0 * trainable_count / max(total_params, 1),
            frozen_count,
            100.0 * frozen_count / max(total_params, 1),
        )

    # ---------------------------------------------------------------------- #
    # Core multimodal embedding construction                                   #
    # ---------------------------------------------------------------------- #

    def build_multimodal_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values_list: Optional[List[List[torch.Tensor]]],
        grid_sizes_list: Optional[List[List[Tuple[int, int]]]],
        modality_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct the full multimodal embedding sequence for the LLM.

        Replaces image placeholder token embeddings in the text embedding
        sequence with actual visual token embeddings produced by the visual
        encoder and connector. Handles multi-scale visual packing with
        special token insertion.

        The dataset pre-structures ``input_ids`` so that image positions
        contain placeholder token IDs (``begin_of_image`` ID repeated for
        each visual token slot). This method replaces those placeholder
        embeddings with the actual visual embeddings.

        Special token insertion order per image:
            [begin_of_image_embed]
              for each scale i:
                for each row r in compressed grid (H//r, W//r):
                  [row_r_token_embeds]  (W//r tokens)
                  [end_of_line_embed]
                [end_of_scale_embed]
            [end_of_image_embed]

        All visual tokens and special tokens within the image subsequence
        have modality_mask=0 (visual). Text tokens have modality_mask=1.

        Args:
            input_ids:        LongTensor of shape (B, L) containing tokenized
                              text with image placeholder tokens at visual
                              positions. Placeholder token ID is
                              ``begin_of_image`` token ID.
            pixel_values_list: Outer list (batch), inner list (scales). Each
                               element is a float tensor of shape (C, H_i, W_i)
                               for scale i of a batch item. Pass ``None`` for
                               text-only batches.
            grid_sizes_list:  Corresponding grid sizes (H_grid, W_grid) per
                              scale per batch item. Pass ``None`` for text-only.
            modality_mask:    LongTensor of shape (B, L) with 0=visual
                              placeholder, 1=text. Used to identify which
                              positions in input_ids are image placeholders.

        Returns:
            Tuple (inputs_embeds, updated_modality_mask, attention_mask):
                inputs_embeds:        Float tensor (B, L, llm_dim) with visual
                                      embeddings substituted at image positions.
                updated_modality_mask: LongTensor (B, L) — same as input
                                       modality_mask (positions already correct).
                attention_mask:        LongTensor (B, L) with 1 for real tokens
                                       and 0 for padding positions.

        Note:
            When ``pixel_values_list`` is None (text-only batch), returns
            ``embed_tokens(input_ids)`` directly with all-ones attention mask
            and all-ones modality mask (all text).
        """
        B: int
        L: int
        B, L = input_ids.shape

        # ------------------------------------------------------------------ #
        # Text-only batch: skip visual processing entirely                    #
        # ------------------------------------------------------------------ #
        if pixel_values_list is None or len(pixel_values_list) == 0:
            text_embeds: torch.Tensor = self._embed_tokens(input_ids)
            attention_mask: torch.Tensor = (
                input_ids != self.tokenizer.pad_token_id
            ).long()
            # All tokens are text tokens in a text-only batch
            text_only_mask: torch.Tensor = torch.ones(
                B, L, dtype=torch.long, device=input_ids.device
            )
            return text_embeds, text_only_mask, attention_mask

        # ------------------------------------------------------------------ #
        # Step 1: Get base text embeddings for all positions                  #
        # ------------------------------------------------------------------ #
        # This gives embeddings for all positions including placeholders.
        # Visual positions will be overwritten with actual visual embeddings.
        inputs_embeds: torch.Tensor = self._embed_tokens(input_ids)
        # inputs_embeds: (B, L, llm_dim)

        # ------------------------------------------------------------------ #
        # Step 2: Get special token embeddings (shared across batch)          #
        # ------------------------------------------------------------------ #
        begin_img_id: int = self.special_tokens.get_token_id("BEGIN_IMAGE")
        end_img_id: int = self.special_tokens.get_token_id("END_IMAGE")
        eol_id: int = self.special_tokens.get_token_id("END_OF_LINE")
        eos_id: int = self.special_tokens.get_token_id("END_OF_SCALE")

        device: torch.device = input_ids.device
        dtype: torch.dtype = inputs_embeds.dtype

        def _get_special_embed(token_id: int) -> torch.Tensor:
            """Get embedding for a single special token. Shape: (1, llm_dim)."""
            tid_tensor: torch.Tensor = torch.tensor(
                [[token_id]], dtype=torch.long, device=device
            )
            return self._embed_tokens(tid_tensor).squeeze(0)  # (1, llm_dim)

        begin_img_embed: torch.Tensor = _get_special_embed(begin_img_id)
        end_img_embed: torch.Tensor = _get_special_embed(end_img_id)
        eol_embed: torch.Tensor = _get_special_embed(eol_id)
        eos_embed: torch.Tensor = _get_special_embed(eos_id)

        # ------------------------------------------------------------------ #
        # Step 3: Process each batch item                                      #
        # ------------------------------------------------------------------ #
        for b in range(B):
            if b >= len(pixel_values_list) or pixel_values_list[b] is None:
                # No image for this batch item — skip
                continue

            scales: List[torch.Tensor] = pixel_values_list[b]
            grid_sizes: List[Tuple[int, int]] = grid_sizes_list[b]

            if len(scales) == 0:
                continue

            # -------------------------------------------------------------- #
            # 3a. Encode each scale through visual encoder + connector        #
            # -------------------------------------------------------------- #
            projected_scales: List[torch.Tensor] = []
            compressed_grids: List[Tuple[int, int]] = []

            for scale_idx, (scale_img, orig_grid) in enumerate(
                zip(scales, grid_sizes)
            ):
                # scale_img: (C, H_i, W_i) — add batch dim
                scale_batch: torch.Tensor = scale_img.unsqueeze(0).to(
                    device=device, dtype=dtype
                )
                # (1, C, H_i, W_i)

                # Visual encoder: (1, N_patches, visual_width)
                visual_tokens: torch.Tensor
                grid_hw: Tuple[int, int]
                visual_tokens, grid_hw = self.visual_encoder(scale_batch)

                # Connector: (1, N_compressed, llm_dim)
                projected_tokens: torch.Tensor
                new_grid: Tuple[int, int]
                projected_tokens, new_grid = self.connector(
                    visual_tokens, grid_hw
                )

                # Remove batch dim: (N_compressed, llm_dim)
                projected_scales.append(projected_tokens.squeeze(0))
                compressed_grids.append(new_grid)

            # -------------------------------------------------------------- #
            # 3b. Build the full visual token sequence with special tokens    #
            # -------------------------------------------------------------- #
            # Structure:
            #   [begin_of_image]
            #     for each scale:
            #       for each row:
            #         [row_tokens] [end_of_line]
            #       [end_of_scale]
            #   [end_of_image]
            visual_seq_parts: List[torch.Tensor] = [begin_img_embed]

            for scale_idx, (proj_tokens, comp_grid) in enumerate(
                zip(projected_scales, compressed_grids)
            ):
                H_comp: int
                W_comp: int
                H_comp, W_comp = comp_grid
                # proj_tokens: (H_comp * W_comp, llm_dim)

                # Insert end_of_line after each row
                for row in range(H_comp):
                    row_start: int = row * W_comp
                    row_end: int = row_start + W_comp
                    row_tokens: torch.Tensor = proj_tokens[row_start:row_end]
                    # row_tokens: (W_comp, llm_dim)
                    visual_seq_parts.append(row_tokens)
                    visual_seq_parts.append(eol_embed)

                # Insert end_of_scale after each scale
                visual_seq_parts.append(eos_embed)

            visual_seq_parts.append(end_img_embed)

            # Concatenate all parts: (N_visual_total, llm_dim)
            visual_seq: torch.Tensor = torch.cat(visual_seq_parts, dim=0)
            N_visual: int = visual_seq.shape[0]

            # -------------------------------------------------------------- #
            # 3c. Find image placeholder positions in input_ids[b]            #
            # -------------------------------------------------------------- #
            # The dataset marks visual positions with modality_mask[b] == 0.
            # We replace those positions with the visual sequence embeddings.
            vis_positions: torch.Tensor = (modality_mask[b] == 0).nonzero(
                as_tuple=False
            ).squeeze(-1)
            # vis_positions: (N_placeholder,) — indices in [0, L)

            N_placeholder: int = vis_positions.shape[0]

            if N_placeholder == 0:
                logger.warning(
                    "Batch item %d has pixel_values but no visual placeholder "
                    "positions in modality_mask. Skipping visual embedding "
                    "substitution.",
                    b,
                )
                continue

            if N_placeholder != N_visual:
                # Mismatch: truncate or pad the visual sequence to fit
                logger.warning(
                    "Batch item %d: placeholder count (%d) != visual token "
                    "count (%d). Adjusting visual sequence to fit.",
                    b,
                    N