# models/navil_model.py

"""
Top‑level NaViL model that assembles the visual encoder, pixel‑shuffle connector,
modality‑aware MoE‑extended LLM, and a text tokenizer.

Provides a unified interface for training (``forward``) and for evaluation
(``generate``).  All architectural hyper‑parameters are drawn from a
``ModelConfig`` dataclass (populated from ``config.yaml``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig
from models.connector import Connector
from models.moe_llm import MoELLM
from models.visual_encoder import VisualEncoder


# ---------------------------------------------------------------------------
# Helper: compute visual features once and return per‑scale shapes
# ---------------------------------------------------------------------------

def _compute_visual_features(
    pixel_values: List[torch.Tensor],
    visual_encoder: VisualEncoder,
    connector: Connector,
    shuffle_ratio: int,
    patch_size: int,
) -> Tuple[Optional[torch.Tensor], Optional[List[List[Tuple[int, int]]]]]:
    """
    Processes a batch of multi‑scale image pyramids and returns:

    1. ``visual_embeds`` : a tensor of shape ``(B, total_patches, llm_hidden)``
       containing all visual embeddings concatenated across scales.
    2. ``per_scale_shapes`` : a list of length ``B``, each entry a list of
       ``(h, w)`` tuples (one per scale) giving the spatial dimensions of
       the visual tokens *after* the connector.

    If ``pixel_values`` is ``None`` or empty, both return values are ``None``.
    """
    if not pixel_values:
        return None, None

    B = pixel_values[0].shape[0]
    device = pixel_values[0].device

    visual_embeds_list: List[torch.Tensor] = []
    per_scale_shapes: List[List[Tuple[int, int]]] = []

    # Process each sample individually – allows variable image resolutions within
    # the batch (though in practice all samples of a stage share the same resolution).
    for b in range(B):
        sample_embs: List[torch.Tensor] = []
        sample_shapes: List[Tuple[int, int]] = []

        for scale_tensor in pixel_values:
            # scale_tensor: (B, 3, H, W)  → extract sample b
            img = scale_tensor[b : b + 1]  # keep batch dim → (1, 3, H, W)
            _, _, H, W = img.shape
            h_patches = H // patch_size
            w_patches = W // patch_size

            # ---- 1. Visual encoder (bidirectional ViT) ----
            vis_feat = visual_encoder(img)  # (1, L, encoder_width)

            # ---- 2. Connector: pixel‑unshuffle + MLP ----
            proj_feat = connector(
                vis_feat,
                patch_grid_sizes=[(h_patches, w_patches)],
            )[
                0
            ]  # (L_down, llm_hidden)
            sample_embs.append(proj_feat)

            # after connector the grid size shrinks by `shuffle_ratio` in each dim
            new_h = h_patches // shuffle_ratio
            new_w = w_patches // shuffle_ratio
            sample_shapes.append((new_h, new_w))

        # Concatenate all scales → (total_patches, llm_hidden)
        sample_embs_cat = torch.cat(sample_embs, dim=0)
        visual_embeds_list.append(sample_embs_cat)
        per_scale_shapes.append(sample_shapes)

    # Stack across batch (assumes all samples have the same total_patches)
    visual_embeds = torch.stack(visual_embeds_list, dim=0)  # (B, total_patches, D)
    return visual_embeds, per_scale_shapes


# ---------------------------------------------------------------------------
# NaViL Model
# ---------------------------------------------------------------------------


class NaViLModel(nn.Module):
    """
    Native Multimodal MLLM as described in the NaViL paper.

    Args:
        config: Model configuration loaded from ``config.yaml``.  The
            ``ModelConfig`` dataclass contains all architectural parameters
            for the visual encoder, connector, LLM MoE extensions, and
            special tokens.
        tokenizer: Optional pre‑loaded HuggingFace tokenizer.  If ``None``,
            the tokenizer is loaded from the base LLM path with additional
            NaViL‑specific special tokens added automatically.  The tokenizer
            is stored as ``self.tokenizer``.
    """

    def __init__(
        self,
        config: ModelConfig,
        tokenizer: Optional["PreTrainedTokenizer"] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # 1. Text tokenizer
        # ------------------------------------------------------------------
        if tokenizer is None:
            tokenizer_name = config.llm.base_model
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
            # Add the four NaViL‑specific control tokens + a placeholder token for
            # individual visual patches.
            special_tokens_dict = {
                "additional_special_tokens": [
                    config.special_tokens.begin_of_image,
                    config.special_tokens.end_of_image,
                    config.special_tokens.end_of_line,
                    config.special_tokens.end_of_scale,
                    "<image_patch>",  # position placeholder for visual features
                ]
            }
            tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer = tokenizer

        # Cache special token IDs for later use.
        self.beg_image_id = tokenizer.convert_tokens_to_ids(
            config.special_tokens.begin_of_image
        )
        self.end_image_id = tokenizer.convert_tokens_to_ids(
            config.special_tokens.end_of_image
        )
        self.end_line_id = tokenizer.convert_tokens_to_ids(
            config.special_tokens.end_of_line
        )
        self.end_scale_id = tokenizer.convert_tokens_to_ids(
            config.special_tokens.end_of_scale
        )
        self.patch_token_id = tokenizer.convert_tokens_to_ids("<image_patch>")

        # ------------------------------------------------------------------
        # 2. Visual encoder (bidirectional ViT with 2D‑RoPE)
        # ------------------------------------------------------------------
        ve_cfg = config.visual_encoder
        self.visual_encoder = VisualEncoder(
            depth=ve_cfg.depth,
            width=ve_cfg.width,
            patch_size=ve_cfg.patch_size,
            mlp_width=ve_cfg.mlp_width,
            num_heads=ve_cfg.num_attention_heads,
        )

        # ------------------------------------------------------------------
        # 3. Base LLM (loaded, then extended with MoE)
        # ------------------------------------------------------------------
        base_model = AutoModelForCausalLM.from_pretrained(
            config.llm.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # always train in bfloat16
        )
        llm_hidden_size = base_model.config.hidden_size

        self.moe_llm = MoELLM(
            base_model=base_model,
            num_experts=config.llm.num_experts,
        )

        # Enable MoE extensions as requested.
        if config.llm.attention_experts:
            self.moe_llm.replace_attention_with_moe()
        if config.llm.ffn_experts:
            self.moe_llm.replace_ffn_with_moe()

        # Resize token embeddings to accommodate the newly added special tokens.
        self.moe_llm.model.resize_token_embeddings(len(tokenizer))

        # ------------------------------------------------------------------
        # 4. Connector (pixel‑unshuffle + MLP)
        # ------------------------------------------------------------------
        cn_cfg = config.connector
        self.connector = Connector(
            in_dim=ve_cfg.width,
            out_dim=llm_hidden_size,
            shuffle_ratio=cn_cfg.pixel_shuffle_ratio,
            mlp_hidden_dim=cn_cfg.mlp_hidden_dim,
        )

    # ------------------------------------------------------------------
    # Forward pass (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: Optional[List[torch.Tensor]] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        image_token_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            pixel_values: List of per‑scale image tensors, each of shape
                ``(B, 3, H_scale, W_scale)``.  May be ``None`` for
                text‑only inputs.
            input_ids: Tokenised text sequence of shape ``(B, seq_len)``.
                Must already contain the special image‑delimiter tokens
                and placeholder ``<image_patch>`` tokens at positions
                indicated by ``image_token_mask``.
            attention_mask: Standard causal‑LM attention mask; 1 for real
                tokens, 0 for padding.  If ``None``, all ones are assumed.
            image_token_mask: Boolean mask of shape ``(B, seq_len)``.
                ``True`` at positions occupied by ``<image_patch>``
                placeholder tokens.  ``False`` for delimiters and text.
            labels: Token IDs for next‑token prediction.  Positions
                corresponding to visual patches should be set to ``-100``.
                If ``None``, only logits are returned.

        Returns:
            dict containing ``"loss"`` (scalar if labels provided) and
            ``"logits"`` (shape ``(B, seq_len, vocab_size)``).
        """
        # 1. Compute visual features once.
        visual_embeds, _ = _compute_visual_features(
            pixel_values,
            self.visual_encoder,
            self.connector,
            self.config.connector.pixel_shuffle_ratio,
            self.config.visual_encoder.patch_size,
        )

        # 2. Pass through the MoE‑extended LLM.
        return self.moe_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=visual_embeds,
            image_token_mask=image_token_mask,
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Convenience: forward with pre‑computed visual embeddings
    # (used by `generate` to avoid recomputing the visual features at
    # every step)
    # ------------------------------------------------------------------
    def _forward_with_visual_embeds(
        self,
        visual_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        image_token_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """See ``self.forward`` – identical except visual features are given directly."""
        return self.moe_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=visual_embeds,
            image_token_mask=image_token_mask,
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Generation (inference)
    # ------------------------------------------------------------------

    def generate(
        self,
        pixel_values: List[torch.Tensor],
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a text response given an image and a prompt.

        Args:
            pixel_values: Multi‑scale image pyramid as a list of tensors
                (each ``(1, 3, H, W)``).  Batch size must be 1.
            input_ids: Tokenised text prompt, shape ``(1, prompt_len)``.
            attention_mask: Optional padding mask for the prompt; if not
                given, all tokens are considered valid.
            max_new_tokens: Maximum number of tokens to generate.
            do_sample: If ``True``, sample from the output distribution
                (after applying temperature, top‑p, top‑k).  Otherwise
                greedy decoding is used.
            temperature: Softmax temperature (only used when ``do_sample``
                is ``True``).
            top_p: Nucleus sampling threshold.
            top_k: Top‑k filtering threshold.  If ``None``, no top‑k filter
                is applied.
            **kwargs: Additional parameters (unused).

        Returns:
            Decoded text string (special tokens are retained in the output,
            but the caller may strip them).
        """
        if input_ids.shape[0] != 1:
            raise ValueError("generate() currently only supports batch_size=1")

        device = input_ids.device

        # ------------------------------------------------------------------
        # 1. Pre‑compute visual features.
        # ------------------------------------------------------------------
        visual_embeds, shapes_per_sample = _compute_visual_features(
            pixel_values,
            self.visual_encoder,
            self.connector,
            self.config.connector.pixel_shuffle_ratio,
            self.config.visual_encoder.patch_size,
        )
        if visual_embeds is None or shapes_per_sample is None:
            raise RuntimeError("No image input provided for generation")

        # We only support a single example – take the first sample's shapes.
        per_scale_shapes = shapes_per_sample[0]  # list of (h, w)

        # ------------------------------------------------------------------
        # 2. Build the token ID sequence for the image prefix.
        # ------------------------------------------------------------------
        prefix_ids: List[int] = [self.beg_image_id]
        image_mask_ids: List[int] = []  # will be 1 for placeholder positions

        for h, w in per_scale_shapes:
            code_ids, code_mask = self._build_scale_codes(h, w)
            prefix_ids.extend(code_ids)
            image_mask_ids.extend(code_mask)
            prefix_ids.append(self.end_scale_id)
            image_mask_ids.append(0)

        prefix_ids.append(self.end_image_id)
        image_mask_ids.append(0)

        # Convert to tensors.
        prefix_tensor = torch.tensor(
            [prefix_ids], device=device, dtype=torch.long
        )  # (1, L_pfx)
        prefix_image_mask = torch.tensor(
            [image_mask_ids], device=device, dtype=torch.bool
        )

        # ------------------------------------------------------------------
        # 3. Concatenate with the text prompt.
        # ------------------------------------------------------------------
        full_input_ids = torch.cat([prefix_tensor, input_ids], dim=1)
        full_image_mask = torch.cat(
            [prefix_image_mask, torch.zeros_like(input_ids, dtype=torch.bool)],
            dim=1,
        )
        # Attention mask: all ones (no padding in the prefix)
        if attention_mask is None:
            full_attn_mask = torch.ones_like(full_input_ids)
        else:
            pfx_attn = torch.ones(1, prefix_tensor.shape[1], device=device)
            full_attn_mask = torch.cat([pfx_attn, attention_mask], dim=1)

        # ------------------------------------------------------------------
        # 4. Autoregressive generation loop (no caching for simplicity)
        # ------------------------------------------------------------------
        generated_ids: List[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = -1  # fallback – will never match

        current_ids = full_input_ids
        current_attn = full_attn_mask
        current_image_mask = full_image_mask

        for _ in range(max_new_tokens):
            outputs = self._forward_with_visual_embeds(
                visual_embeds=visual_embeds,
                input_ids=current_ids,
                attention_mask=current_attn,
                image_token_mask=current_image_mask,
            )
            logits = outputs["logits"]  # (1, cur_len, vocab_size)

            # Use logits of the last token.
            next_logits = logits[0, -1, :]  # (vocab_size,)

            # Apply sampling or greedy decoding.
            if do_sample:
                # Temperature scaling
                if temperature > 0.0:
                    next_logits = next_logits / temperature
                # Top‑k filtering
                if top_k is not None and top_k > 0:
                    topk_values, _ = torch.topk(next_logits, top_k)
                    min_val = topk_values[-1]
                    next_logits[next_logits < min_val] = float("-inf")
                # Top‑p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True
                    )
                    cum_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cum_probs > top_p
                    # Shift the indices to keep at least one token
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[
                        :-1
                    ].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices[
                        sorted_indices_to_remove
                    ]
                    next_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze()
            else:
                next_token = torch.argmax(next_logits)

            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            # Append to running sequence.
            next_token_tensor = next_token.unsqueeze(0).unsqueeze(0)  # (1,1)
            current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
            current_attn = torch.cat(
                [current_attn, torch.ones(1, 1, device=device, dtype=current_attn.dtype)], dim=1
            )
            current_image_mask = torch.cat(
                [current_image_mask, torch.tensor([[False]], device=device)], dim=1
            )

            # Stop if end‑of‑sentence token is generated.
            if next_token_id == eos_token_id:
                break

        # ------------------------------------------------------------------
        # 5. Decode only the newly generated tokens.
        # ------------------------------------------------------------------
        if generated_ids:
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        else:
            generated_text = ""
        return generated_text

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model weights, tokenizer, and configuration.

        Args:
            save_directory: Path to the output directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        # Save PyTorch state dict
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        # Save configuration (OmegaConf dataclass serialised as JSON)
        config_dict = self.config.__dict__  # assuming dataclass fields are simple
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def load_pretrained(cls, load_directory: str) -> "NaViLModel":
        """
        Load a model from a directory saved by ``save_pretrained``.

        Args:
            load_directory: Path to the directory containing ``pytorch_model.bin``,
                ``config.json`` and the tokenizer files.

        Returns:
            A fully initialised ``NaViLModel``.
        """
        # Load configuration
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config_dict = json.load(f)
        # Reconstruct ModelConfig (using the default constructor and then setting fields)
        config = ModelConfig()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(load_directory, trust_remote_code=True)

        # Create model and load weights
        model = cls(config, tokenizer=tokenizer)
        state_dict = torch.load(
            os.path.join(load_directory, "pytorch_model.bin"),
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)
        return model

    # ------------------------------------------------------------------
    # Internal helper: build token IDs for a single scale’s patch grid
    # ------------------------------------------------------------------
    def _build_scale_codes(
        self, h: int, w: int
    ) -> Tuple[List[int], List[int]]:
        """
        Create a sequence of ``<image_patch>`` placeholders interleaved with
        ``<end_of_line>`` tokens for one scale.

        Args:
            h: Number of patch rows (after connector).
            w: Number of patch columns.

        Returns:
            Tuple ``(ids, mask)`` where ``ids`` are token IDs and ``mask`` is
            a binary list with 1 for each ``<image_patch>`` position.
        """
        ids: List[int] = []
        mask: List[int] = []
        for _ in range(h):
            ids.extend([self.patch_token_id] * w)
            mask.extend([1] * w)
            ids.append(self.end_line_id)
            mask.append(0)
        # The last end_of_line token is part of the row; no extra
        return ids, mask
