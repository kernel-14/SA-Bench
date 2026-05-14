# models/moe_llm.py

"""
Modality‑specific Mixture of Experts (MoE) extension for the LLM component of NaViL.

This module implements:

- ``ModalitySpecificAttention`` – replaces a standard transformer attention block
  with two sets of Q/K/V/O projections: one for linguistic tokens (copied from
  the pre‑trained LLM) and one for visual tokens (freshly initialised).
  The active expert is selected deterministically based on a boolean
  ``image_token_mask``.

- ``ModalitySpecificFFN`` – analogous split for the SwiGLU feed‑forward network
  (gate, up, down projections).

- ``MoELLM`` – a wrapper that loads a HuggingFace ``PreTrainedModel``, replaces
  every transformer layer’s ``self_attn`` and ``mlp`` with the above modules, and
  provides a custom ``forward`` method that:
  * concatenates visual tokens (from the encoder + connector) with text embeddings,
  * constructs a hybrid attention mask (bidirectional for image tokens,
    causal for text tokens),
  * iterates over the transformer layers manually, applying layer norms and the
    new attention / FFN modules,
  * and finally applies the final LM head to produce logits (and optionally loss).

The implementation follows the paper’s meta‑architecture (Sec. 3.1) and the
equations for MHA‑MMoE and FFN‑MMoE (Sec. 3.2.2).  No learned routing is used:
each token is hard‑assigned to a modality based on its position.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


# ---------------------------------------------------------------------------
# ModalitySpecificAttention
# ---------------------------------------------------------------------------


class ModalitySpecificAttention(nn.Module):
    """
    Self‑attention with modality‑specific linear projections, as defined by
    the MHA‑MMoE equations in the NaViL paper.

    The module holds **two** sets of projection weights: ``ling_*`` (copied
    from the pre‑trained LLM) and ``vis_*`` (freshly initialised).  During
    the forward pass the input sequence is split into image and text parts
    according to ``image_token_mask``, each part is projected using its own
    expert, and the results are concatenated before the shared attention
    operation.  The output projection is similarly handled.

    Args:
        ling_q_proj: Pre‑trained linear layer for linguistic query (copied).
        ling_k_proj: Pre‑trained linear layer for linguistic key.
        ling_v_proj: Pre‑trained linear layer for linguistic value.
        ling_o_proj: Pre‑trained linear layer for linguistic output.
        vis_q_proj: Freshly initialised linear layer for visual query.
        vis_k_proj: Freshly initialised linear layer for visual key.
        vis_v_proj: Freshly initialised linear layer for visual value.
        vis_o_proj: Freshly initialised linear layer for visual output.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
    """

    def __init__(
        self,
        ling_q_proj: nn.Linear,
        ling_k_proj: nn.Linear,
        ling_v_proj: nn.Linear,
        ling_o_proj: nn.Linear,
        vis_q_proj: nn.Linear,
        vis_k_proj: nn.Linear,
        vis_v_proj: nn.Linear,
        vis_o_proj: nn.Linear,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        # Linguistic projections (from pre‑trained weights)
        self.q_proj_ling = ling_q_proj
        self.k_proj_ling = ling_k_proj
        self.v_proj_ling = ling_v_proj
        self.o_proj_ling = ling_o_proj

        # Visual projections (from scratch)
        self.q_proj_vis = vis_q_proj
        self.k_proj_vis = vis_k_proj
        self.v_proj_vis = vis_v_proj
        self.o_proj_vis = vis_o_proj

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states : ``(B, L, C)`` where ``C = hidden_size``.
            attention_mask : optional additive mask of shape
                ``(B, 1, L, L)`` (or broadcastable).
            image_token_mask : ``(B, L)`` boolean tensor, ``True`` for visual
                positions, ``False`` for linguistic.  Visual tokens are assumed
                to be **contiguous at the beginning** of the sequence.

        Returns:
            Output tensor of shape ``(B, L, C)``.
        """
        B, L, C = hidden_states.shape

        # ------------------------------------------------------------------
        # Fallback: no visual tokens → use linguistic expert for everything
        # ------------------------------------------------------------------
        if image_token_mask is None or image_token_mask.sum() == 0:
            Q = self.q_proj_ling(hidden_states)
            K = self.k_proj_ling(hidden_states)
            V = self.v_proj_ling(hidden_states)
            attn_out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=attention_mask
            )
            return self.o_proj_ling(attn_out)

        # ------------------------------------------------------------------
        # Split the sequence into image and text parts (image first)
        # ------------------------------------------------------------------
        img_len = int(image_token_mask.sum(dim=1).max().item())
        txt_len = L - img_len

        img_hidden = hidden_states[:, :img_len, :]
        txt_hidden = hidden_states[:, img_len:, :]

        # ----------- Q/K/V projections per modality -----------
        Q_img = self.q_proj_vis(img_hidden)
        K_img = self.k_proj_vis(img_hidden)
        V_img = self.v_proj_vis(img_hidden)

        Q_txt = self.q_proj_ling(txt_hidden)
        K_txt = self.k_proj_ling(txt_hidden)
        V_txt = self.v_proj_ling(txt_hidden)

        # Concatenate along sequence dimension
        Q = torch.cat([Q_img, Q_txt], dim=1)
        K = torch.cat([K_img, K_txt], dim=1)
        V = torch.cat([V_img, V_txt], dim=1)

        # ----------- Shared attention -----------
        attn_out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attention_mask
        )

        # ----------- Output projection per modality -----------
        attn_out_img = attn_out[:, :img_len, :]
        attn_out_txt = attn_out[:, img_len:, :]

        o_img = self.o_proj_vis(attn_out_img)
        o_txt = self.o_proj_ling(attn_out_txt)

        return torch.cat([o_img, o_txt], dim=1)


# ---------------------------------------------------------------------------
# ModalitySpecificFFN
# ---------------------------------------------------------------------------


class ModalitySpecificFFN(nn.Module):
    """
    SwiGLU feed‑forward network with modality‑specific gate, up, and down
    projections (FFN‑MMoE from the NaViL paper).

    Args:
        ling_gate_proj: Pre‑trained linguistic gate projection (copied).
        ling_up_proj:   Pre‑trained linguistic up projection.
        ling_down_proj: Pre‑trained linguistic down projection.
        vis_gate_proj:  Freshly initialised visual gate projection.
        vis_up_proj:    Freshly initialised visual up projection.
        vis_down_proj:  Freshly initialised visual down projection.
    """

    def __init__(
        self,
        ling_gate_proj: nn.Linear,
        ling_up_proj: nn.Linear,
        ling_down_proj: nn.Linear,
        vis_gate_proj: nn.Linear,
        vis_up_proj: nn.Linear,
        vis_down_proj: nn.Linear,
    ) -> None:
        super().__init__()
        self.gate_proj_ling = ling_gate_proj
        self.up_proj_ling = ling_up_proj
        self.down_proj_ling = ling_down_proj

        self.gate_proj_vis = vis_gate_proj
        self.up_proj_vis = vis_up_proj
        self.down_proj_vis = vis_down_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states : ``(B, L, C)``.
            image_token_mask : ``(B, L)`` boolean mask; visual tokens first.

        Returns:
            Output of same shape ``(B, L, C)``.
        """
        B, L, C = hidden_states.shape

        if image_token_mask is None or image_token_mask.sum() == 0:
            gate = self.gate_proj_ling(hidden_states)
            up = self.up_proj_ling(hidden_states)
            activated = F.silu(gate) * up
            return self.down_proj_ling(activated)

        img_len = int(image_token_mask.sum(dim=1).max().item())
        img_hidden = hidden_states[:, :img_len, :]
        txt_hidden = hidden_states[:, img_len:, :]

        # Visual FFN
        gate_img = self.gate_proj_vis(img_hidden)
        up_img = self.up_proj_vis(img_hidden)
        activated_img = F.silu(gate_img) * up_img
        down_img = self.down_proj_vis(activated_img)

        # Linguistic FFN
        gate_txt = self.gate_proj_ling(txt_hidden)
        up_txt = self.up_proj_ling(txt_hidden)
        activated_txt = F.silu(gate_txt) * up_txt
        down_txt = self.down_proj_ling(activated_txt)

        return torch.cat([down_img, down_txt], dim=1)


# ---------------------------------------------------------------------------
# MoELLM – main wrapper
# ---------------------------------------------------------------------------


class MoELLM(nn.Module):
    """
    Wraps a HuggingFace ``PreTrainedModel`` and extends it with
    modality‑specific MoE for attention and FFN, following the NaViL paper.

    After construction, call ``replace_attention_with_moe()`` and
    ``replace_ffn_with_moe()`` to inject the expert modules.

    Args:
        base_model: Already loaded HF causal LM (e.g., InternLM2‑1.8B).
        num_experts: Number of experts per modality (fixed to 2).
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        num_experts: int = 2,
    ) -> None:
        super().__init__()
        self.model = base_model
        # Common sub‑modules (confirm existence)
        self.embed_tokens = self.model.model.embed_tokens
        # Final norm: try common attributes
        self.final_norm = getattr(self.model.model, "norm", None) or getattr(
            self.model.model, "final_layernorm", None
        )
        if self.final_norm is None:
            raise AttributeError(
                "Could not locate final layer normalisation in the base model."
            )
        self.lm_head = self.model.lm_head

        self.layers = self.model.model.layers
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size

        # Stored after `replace_*` for parameter freezing patterns
        self._num_experts = num_experts

    def replace_attention_with_moe(self) -> None:
        """
        Replace every transformer layer’s ``self_attn`` with
        ``ModalitySpecificAttention``.  Linguistic weights are deep‑copied
        from the original pre‑trained attention, visual weights are freshly
        initialised.
        """
        init_std = 0.02  # common for models like InternLM2 / Qwen3
        for layer in self.layers:
            orig_attn = layer.self_attn

            # Gather original projection layers
            ling_q = copy.deepcopy(orig_attn.q_proj)
            ling_k = copy.deepcopy(orig_attn.k_proj)
            ling_v = copy.deepcopy(orig_attn.v_proj)
            ling_o = copy.deepcopy(orig_attn.o_proj)

            # Dimensions
            in_features = ling_q.in_features
            out_features = ling_q.out_features

            # New visual projections
            vis_q = nn.Linear(in_features, out_features, bias=False)
            vis_k = nn.Linear(in_features, out_features, bias=False)
            vis_v = nn.Linear(in_features, out_features, bias=False)
            vis_o = nn.Linear(in_features, out_features, bias=False)

            for m in (vis_q, vis_k, vis_v, vis_o):
                nn.init.normal_(m.weight, std=init_std)

            # Attention head configuration
            num_heads = orig_attn.num_heads
            head_dim = out_features // num_heads

            new_attn = ModalitySpecificAttention(
                ling_q_proj=ling_q,
                ling_k_proj=ling_k,
                ling_v_proj=ling_v,
                ling_o_proj=ling_o,
                vis_q_proj=vis_q,
                vis_k_proj=vis_k,
                vis_v_proj=vis_v,
                vis_o_proj=vis_o,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            layer.self_attn = new_attn

    def replace_ffn_with_moe(self) -> None:
        """
        Replace every transformer layer’s ``mlp`` with
        ``ModalitySpecificFFN``.  Linguistic weights are deep‑copied from
        the original pre‑trained FFN, visual weights are freshly initialised.
        """
        init_std = 0.02
        for layer in self.layers:
            orig_mlp = layer.mlp

            ling_gate = copy.deepcopy(orig_mlp.gate_proj)
            ling_up = copy.deepcopy(orig_mlp.up_proj)
            ling_down = copy.deepcopy(orig_mlp.down_proj)

            # Dimensions
            hid_size = ling_gate.in_features
            intermediate_size = ling_gate.out_features

            vis_gate = nn.Linear(hid_size, intermediate_size, bias=False)
            vis_up = nn.Linear(hid_size, intermediate_size, bias=False)
            vis_down = nn.Linear(intermediate_size, hid_size, bias=False)

            for m in (vis_gate, vis_up, vis_down):
                nn.init.normal_(m.weight, std=init_std)

            new_ffn = ModalitySpecificFFN(
                ling_gate_proj=ling_gate,
                ling_up_proj=ling_up,
                ling_down_proj=ling_down,
                vis_gate_proj=vis_gate,
                vis_up_proj=vis_up,
                vis_down_proj=vis_down,
            )
            layer.mlp = new_ffn

    def _build_hybrid_attn_mask(
        self,
        img_len: int,
        txt_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Construct a 4D additive attention mask for the hybrid (bidirectional image,
        causal text) attention pattern.

        Args:
            img_len: number of image tokens (constant across batch).
            txt_len: number of text tokens.
            batch_size: batch size.
            device: target device.
            dtype: model dtype (float32 or bfloat16).
            text_padding_mask: ``(B, txt_len)`` with 1 for real tokens, 0 for padding.

        Returns:
            Tensor of shape ``(B, 1, total_len, total_len)`` with 0 where
            attention is allowed and ``-inf`` otherwise.
        """
        total_len = img_len + txt_len
        row_idx = torch.arange(total_len, device=device).unsqueeze(1)  # (L, 1)
        col_idx = torch.arange(total_len, device=device).unsqueeze(0)  # (1, L)

        # Allowed attention: query is image OR key is image OR (query is text AND key <= query)
        allowed = (
            (row_idx < img_len)  # query is image
            | (col_idx < img_len)  # key is image
            | ((row_idx >= img_len) & (col_idx <= row_idx))  # text causal
        )

        # Base mask: 0.0 where allowed, -inf elsewhere
        mask = torch.full(
            (1, 1, total_len, total_len), fill_value=float("-inf"), dtype=dtype, device=device
        )
        mask[0, 0, allowed] = 0.0
        mask = mask.expand(batch_size, -1, -1, -1)

        # Merge text padding mask (image tokens are never padded)
        if text_padding_mask is not None:
            # Build full padding mask: image positions always valid
            full_pad = torch.ones(batch_size, total_len, device=device)
            full_pad[:, img_len:] = text_padding_mask.to(device)
            # Keys that are padding should be masked out for all queries
            key_pad = (full_pad == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, total_len)
            mask = mask.masked_fill(key_pad, float("-inf"))
            # Also, queries that are padding should not attend to anything? Usually that's handled
            # by not selecting them as queries; we can also mask them to avoid numerical issues:
            query_pad = (full_pad == 0).unsqueeze(1).unsqueeze(3)  # (B, 1, total_len, 1)
            mask = mask.masked_fill(query_pad, float("-inf"))

        return mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,  # noqa: ARG002
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: ``(B, text_len)`` – tokenised text.
            attention_mask: ``(B, text_len)`` – standard padding mask (1 = real).
            image_embeds: ``(B, img_len, hidden_size)`` – visual tokens after
                encoder + connector.
            labels: ``(B, total_len)`` – ground‑truth token IDs for next‑token
                prediction (``-100`` for ignored positions, including all image
                positions).
            **kwargs: Additional parameters forwarded to the transformer layers
                (currently unused).

        Returns:
            dict with keys ``"loss"`` (scalar if labels provided, else None)
            and ``"logits"`` (shape ``(B, total_len, vocab_size)``).
        """
        B, text_len = input_ids.shape
        if image_embeds is None:
            img_len = 0
            combined_embeds = self.embed_tokens(input_ids)
        else:
            img_len = image_embeds.shape[1]
            text_embeds = self.embed_tokens(input_ids)
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        total_len = combined_embeds.shape[1]
        device = combined_embeds.device
        dtype = combined_embeds.dtype

        # ----- Image token mask (all image tokens come first) -----
        image_token_mask = torch.zeros(B, total_len, dtype=torch.bool, device=device)
        image_token_mask[:, :img_len] = True

        # ----- Hybrid attention mask -----
        attn_mask = self._build_hybrid_attn_mask(
            img_len=img_len,
            txt_len=text_len,
            batch_size=B,
            device=device,
            dtype=dtype,
            text_padding_mask=attention_mask,
        )

        # ----- Iterate over transformer layers -----
        hidden_states = combined_embeds
        for layer in self.layers:  # type: ignore[attr-defined]
            # Self‑attention block
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)  # type: ignore[union-attr]
            attn_out = layer.self_attn(  # type: ignore[operator]
                normed,
                attention_mask=attn_mask,
                image_token_mask=image_token_mask,
            )
            hidden_states = residual + attn_out

            # FFN block
            residual = hidden_states
            normed = layer.post_attention_layernorm(hidden_states)  # type: ignore[union-attr]
            ffn_out = layer.mlp(  # type: ignore[operator]
                normed,
                image_token_mask=image_token_mask,
            )
            hidden_states = residual + ffn_out

        # ----- Final norm and LM head -----
        hidden_states = self.final_norm(hidden_states)  # type: ignore[union-attr]
        logits = self.lm_head(hidden_states)  # (B, total_len, vocab)

        # ----- Loss -----
        loss = None
        if labels is not None:
            # Shift for next‑token prediction (ignore image positions if labelled -100)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits}

