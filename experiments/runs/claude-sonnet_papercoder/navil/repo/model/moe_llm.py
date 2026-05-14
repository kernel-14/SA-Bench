```python
## model/moe_llm.py
"""MoE-extended LLM with modality-specific experts for NaViL.

This module implements the core linguistic component of the NaViL native
multimodal architecture (Sections 3.2.2 and 4.1 of the paper). Every
standard transformer layer is replaced with a MoELayer that routes tokens
to modality-specific experts based on a hard binary mask — no learned
router is needed since token modality is known at input time.

Key design choices (from the paper):
- MHA-MMoE: modality-specific Q, K, V, O projection matrices per layer.
  Unified global (causal) attention is computed across all tokens together;
  only the projections are modality-specific.
- FFN-MMoE: modality-specific gate, up, down projection matrices per layer.
  SwiGLU activation: FFN(x) = down(SiLU(gate(x)) * up(x)).
- Hard routing: modality_mask (0=visual, 1=text) determines which expert
  processes each token. No routing overhead at inference time.
- Warm initialization: both visual and linguistic experts are initialized
  from the same pretrained dense LLM weights. During training they diverge
  as each only receives gradients from its respective token positions.
- GQA compatibility: Qwen3-8B uses Grouped Query Attention (fewer KV heads
  than Q heads). ModalityExpert handles this via num_key_value_heads.

Architecture parameters (NaViL-2B, Table 6):
    Base LLM: InternLM2-1.8B
    hidden_size=2048, num_heads=16, mlp_width=8192, num_experts=2

Architecture parameters (NaViL-9B, Table 6):
    Base LLM: Qwen3-8B
    hidden_size=4096, num_heads=32, mlp_width=12288, num_experts=2

Modality mask convention (shared across the codebase):
    modality_mask: LongTensor of shape (B, L)
    0 = visual token (routes to visual_expert)
    1 = text/linguistic token (routes to linguistic_expert)
    Special tokens (<begin_of_image>, <end_of_line>, etc.) are treated
    as visual tokens (mask=0) since they are structural image markers.

Config alignment (configs/navil_2b.yaml):
    model.llm.name_or_path:    "internlm/internlm2-1_8b"
    model.llm.depth:           24
    model.llm.width:           2048
    model.llm.mlp_width:       8192
    model.llm.num_heads:       16
    model.llm.num_experts:     2
    training.precision:        "bfloat16"
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local RMSNorm implementation
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Implements: x_norm = x / sqrt(mean(x^2) + eps) * weight

    Runs the variance computation in float32 for numerical stability, then
    casts back to the input dtype. This matches the behavior of most LLM
    implementations (InternLM2, Qwen3, LLaMA).

    Args:
        hidden_size: Dimension of the input features.
        eps:         Small constant for numerical stability. Default: 1e-6.

    Example::

        norm = RMSNorm(2048, eps=1e-6)
        x = torch.randn(2, 128, 2048, dtype=torch.bfloat16)
        out = norm(x)  # (2, 128, 2048), bfloat16
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(hidden_size))
        self.eps: float = eps
        self.hidden_size: int = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Normalized tensor of the same shape and dtype as x.
        """
        input_dtype: torch.dtype = x.dtype
        # Compute variance in float32 for stability
        x_float: torch.Tensor = x.float()
        variance: torch.Tensor = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normalized: torch.Tensor = x_float * torch.rsqrt(variance + self.eps)
        # Cast back to original dtype and apply learned scale
        return (self.weight * x_normalized).to(input_dtype)


# ---------------------------------------------------------------------------
# Helper: extract layer weights with model-type normalization
# ---------------------------------------------------------------------------

def _get_layer_weights(
    layer: nn.Module,
) -> Dict[str, Optional[torch.Tensor]]:
    """Extract attention and FFN weight tensors from a transformer layer.

    Handles the attribute name differences between InternLM2 and Qwen3:

    InternLM2 attention:
        - layer.attention.wqkv  (fused QKV, shape: (3*hidden, hidden) or GQA variant)
        - layer.attention.wo    (output projection)
        - layer.feed_forward.w1 (gate), .w2 (down), .w3 (up)
        - layer.attention_norm  (pre-attention norm)
        - layer.ffn_norm        (pre-FFN norm)

    Qwen3 attention:
        - layer.self_attn.q_proj, .k_proj, .v_proj, .o_proj
        - layer.mlp.gate_proj, .up_proj, .down_proj
        - layer.input_layernorm, .post_attention_layernorm

    Returns a dict with normalized keys:
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        gate_proj_weight, up_proj_weight, down_proj_weight,
        norm1_weight, norm2_weight

    Args:
        layer: A single transformer decoder layer from the base LLM.

    Returns:
        Dictionary mapping normalized weight names to weight tensors.
        Values may be None if a weight is not found (logged as warning).
    """
    weights: Dict[str, Optional[torch.Tensor]] = {}

    # ------------------------------------------------------------------ #
    # Attention projections                                                #
    # ------------------------------------------------------------------ #
    if hasattr(layer, "self_attn"):
        # Qwen3 / LLaMA style
        attn = layer.self_attn
        weights["q_proj_weight"] = attn.q_proj.weight.data if hasattr(attn, "q_proj") else None
        weights["k_proj_weight"] = attn.k_proj.weight.data if hasattr(attn, "k_proj") else None
        weights["v_proj_weight"] = attn.v_proj.weight.data if hasattr(attn, "v_proj") else None
        weights["o_proj_weight"] = attn.o_proj.weight.data if hasattr(attn, "o_proj") else None
    elif hasattr(layer, "attention"):
        # InternLM2 style
        attn = layer.attention
        if hasattr(attn, "wqkv"):
            # Fused QKV weight: shape (3*hidden_size, hidden_size) for MHA
            # or ((num_heads + 2*num_kv_heads) * head_dim, hidden_size) for GQA
            wqkv: torch.Tensor = attn.wqkv.weight.data
            # Determine split sizes from the weight shape
            total_out: int = wqkv.shape[0]
            hidden_size: int = wqkv.shape[1]
            # For standard MHA: total_out = 3 * hidden_size
            # For GQA: total_out = (num_heads + 2*num_kv_heads) * head_dim
            # We split evenly into 3 parts as a best-effort for MHA
            # GQA InternLM2 models will be handled by the caller
            q_size: int = total_out // 3
            k_size: int = total_out // 3
            v_size: int = total_out - q_size - k_size
            weights["q_proj_weight"] = wqkv[:q_size]
            weights["k_proj_weight"] = wqkv[q_size: q_size + k_size]
            weights["v_proj_weight"] = wqkv[q_size + k_size:]
        elif hasattr(attn, "q_proj"):
            weights["q_proj_weight"] = attn.q_proj.weight.data
            weights["k_proj_weight"] = attn.k_proj.weight.data if hasattr(attn, "k_proj") else None
            weights["v_proj_weight"] = attn.v_proj.weight.data if hasattr(attn, "v_proj") else None
        else:
            weights["q_proj_weight"] = None
            weights["k_proj_weight"] = None
            weights["v_proj_weight"] = None

        # Output projection
        if hasattr(attn, "wo"):
            weights["o_proj_weight"] = attn.wo.weight.data
        elif hasattr(attn, "out_proj"):
            weights["o_proj_weight"] = attn.out_proj.weight.data
        else:
            weights["o_proj_weight"] = None
    else:
        logger.warning(
            "Could not find attention module in layer %s. "
            "Expected 'self_attn' (Qwen3/LLaMA) or 'attention' (InternLM2).",
            type(layer).__name__,
        )
        weights["q_proj_weight"] = None
        weights["k_proj_weight"] = None
        weights["v_proj_weight"] = None
        weights["o_proj_weight"] = None

    # ------------------------------------------------------------------ #
    # FFN projections                                                      #
    # ------------------------------------------------------------------ #
    if hasattr(layer, "mlp"):
        # Qwen3 / LLaMA style
        mlp = layer.mlp
        weights["gate_proj_weight"] = mlp.gate_proj.weight.data if hasattr(mlp, "gate_proj") else None
        weights["up_proj_weight"] = mlp.up_proj.weight.data if hasattr(mlp, "up_proj") else None
        weights["down_proj_weight"] = mlp.down_proj.weight.data if hasattr(mlp, "down_proj") else None
    elif hasattr(layer, "feed_forward"):
        # InternLM2 style: w1=gate, w2=down, w3=up
        ff = layer.feed_forward
        weights["gate_proj_weight"] = ff.w1.weight.data if hasattr(ff, "w1") else None
        weights["up_proj_weight"] = ff.w3.weight.data if hasattr(ff, "w3") else None
        weights["down_proj_weight"] = ff.w2.weight.data if hasattr(ff, "w2") else None
    else:
        logger.warning(
            "Could not find FFN module in layer %s. "
            "Expected 'mlp' (Qwen3/LLaMA) or 'feed_forward' (InternLM2).",
            type(layer).__name__,
        )
        weights["gate_proj_weight"] = None
        weights["up_proj_weight"] = None
        weights["down_proj_weight"] = None

    # ------------------------------------------------------------------ #
    # Layer norms                                                          #
    # ------------------------------------------------------------------ #
    if hasattr(layer, "input_layernorm"):
        # Qwen3 / LLaMA style
        weights["norm1_weight"] = layer.input_layernorm.weight.data
    elif hasattr(layer, "attention_norm"):
        # InternLM2 style
        weights["norm1_weight"] = layer.attention_norm.weight.data
    else:
        weights["norm1_weight"] = None

    if hasattr(layer, "post_attention_layernorm"):
        # Qwen3 / LLaMA style
        weights["norm2_weight"] = layer.post_attention_layernorm.weight.data
    elif hasattr(layer, "ffn_norm"):
        # InternLM2 style
        weights["norm2_weight"] = layer.ffn_norm.weight.data
    else:
        weights["norm2_weight"] = None

    return weights


# ---------------------------------------------------------------------------
# ModalityExpert
# ---------------------------------------------------------------------------

class ModalityExpert(nn.Module):
    """One complete set of projection matrices for a single modality.

    Each MoELayer contains exactly two ModalityExpert instances — one for
    visual tokens and one for linguistic tokens. The expert holds:
    - Attention projections: q_proj, k_proj, v_proj, o_proj
    - FFN projections: gate_proj, up_proj, down_proj

    All projection matrices use bias=False, consistent with the LLM
    architecture convention.

    GQA support: When num_key_value_heads < num_attention_heads (Grouped
    Query Attention), k_proj and v_proj output a smaller dimension:
    head_dim * num_key_value_heads instead of hidden_size. The q_proj
    always outputs hidden_size (= head_dim * num_attention_heads).

    Args:
        hidden_size:          Token embedding dimension (e.g., 2048 for
                              NaViL-2B, 4096 for NaViL-9B).
        mlp_width:            FFN intermediate dimension (e.g., 8192 for
                              NaViL-2B, 12288 for NaViL-9B).
        num_heads:            Number of query attention heads (e.g., 16
                              for NaViL-2B, 32 for NaViL-9B).
        modality:             String label: ``'visual'`` or ``'linguistic'``.
                              Stored for identification/debugging only.
        num_key_value_heads:  Number of key/value heads. Equals num_heads
                              for standard MHA (InternLM2-1.8B). May be
                              less for GQA (Qwen3-8B). Defaults to
                              num_heads (full MHA).

    Raises:
        ValueError: If hidden_size is not divisible by num_heads.
        ValueError: If num_key_value_heads > num_heads.

    Example::

        expert = ModalityExpert(
            hidden_size=2048, mlp_width=8192,
            num_heads=16, modality='visual'
        )
        x_vis = torch.randn(50, 2048)  # 50 visual tokens
        Q, K, V = expert.attn_proj(x_vis)
        ffn_out = expert.ffn_forward(x_vis)
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_width: int,
        num_heads: int,
        modality: str,
        num_key_value_heads: int = -1,  # -1 means use num_heads (full MHA)
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}."
            )

        # Resolve default: -1 means full MHA (num_kv_heads == num_heads)
        if num_key_value_heads < 0:
            num_key_value_heads = num_heads

        if num_key_value_heads > num_heads:
            raise ValueError(
                f"num_key_value_heads={num_key_value_heads} cannot exceed "
                f"num_heads={num_heads}."
            )

        self.hidden_size: int = hidden_size
        self.mlp_width: int = mlp_width
        self.num_heads: int = num_heads
        self.num_key_value_heads: int = num_key_value_heads
        self.head_dim: int = hidden_size // num_heads
        self.modality: str = modality

        # KV projection output dimension (smaller than hidden_size for GQA)
        kv_dim: int = self.head_dim * num_key_value_heads

        # ------------------------------------------------------------------ #
        # Attention projection matrices (all bias=False)                      #
        # ------------------------------------------------------------------ #
        # Q always projects to full hidden_size (num_heads * head_dim)
        self.q_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=False)
        # K and V project to kv_dim (= hidden_size for MHA, smaller for GQA)
        self.k_proj: nn.Linear = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj: nn.Linear = nn.Linear(hidden_size, kv_dim, bias=False)
        # Output projection: hidden_size → hidden_size
        self.o_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=False)

        # ------------------------------------------------------------------ #
        # FFN projection matrices (SwiGLU, all bias=False)                   #
        # ------------------------------------------------------------------ #
        self.gate_proj: nn.Linear = nn.Linear(hidden_size, mlp_width, bias=False)
        self.up_proj: nn.Linear = nn.Linear(hidden_size, mlp_width, bias=False)
        self.down_proj: nn.Linear = nn.Linear(mlp_width, hidden_size, bias=False)

    def attn_proj(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply modality-specific attention input projections.

        Args:
            x: Token tensor of shape (N_m, hidden_size) where N_m is the
               number of tokens belonging to this modality.

        Returns:
            Tuple (Q, K, V) where:
                Q: shape (N_m, hidden_size)  — query projections
                K: shape (N_m, kv_dim)       — key projections
                V: shape (N_m, kv_dim)       — value projections
                kv_dim = head_dim * num_key_value_heads
        """
        Q: torch.Tensor = self.q_proj(x)   # (N_m, hidden_size)
        K: torch.Tensor = self.k_proj(x)   # (N_m, kv_dim)
        V: torch.Tensor = self.v_proj(x)   # (N_m, kv_dim)
        return Q, K, V

    def ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply modality-specific SwiGLU FFN.

        Computes: down_proj(SiLU(gate_proj(x)) * up_proj(x))

        Args:
            x: Token tensor of shape (N_m, hidden_size).

        Returns:
            Output tensor of shape (N_m, hidden_size).
        """
        gate: torch.Tensor = F.silu(self.gate_proj(x))  # (N_m, mlp_width)
        up: torch.Tensor = self.up_proj(x)               # (N_m, mlp_width)
        out: torch.Tensor = self.down_proj(gate * up)    # (N_m, hidden_size)
        return out


# ---------------------------------------------------------------------------
# MoELayer
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """Single transformer decoder layer with modality-specific MoE experts.

    Replaces a standard transformer layer with two sets of modality-specific
    projection matrices (one for visual tokens, one for linguistic tokens).
    The unified global causal attention is computed across all tokens together;
    only the projection matrices are modality-specific.

    Paper equations (Section 3.2.2):
        x' = x + MHA-MMoE(RMSNorm(x))
        x_out = x' + FFN-MMoE(RMSNorm(x'))

    MHA-MMoE:
        Q_{i,m} = x_{i,m} W_Q^m
        K_{i,m} = x_{i,m} W_K^m
        V_{i,m} = x_{i,m} W_V^m
        attn = softmax(QK^T / sqrt(d)) V
        out_{i,m} = attn_{i,m} W_O^m

    FFN-MMoE:
        FFN(x_{i,m}) = (SiLU(x_{i,m} W_gate^m) ⊙ x_{i,m} W_up^m) W_down^m

    Args:
        hidden_size:          Token embedding dimension.
        num_heads:            Number of query attention heads.
        mlp_width:            FFN intermediate dimension.
        num_key_value_heads:  Number of KV heads (< num_heads for GQA).
                              Defaults to num_heads (full MHA).

    Attributes:
        visual_expert:    ModalityExpert for visual tokens (modality_mask==0).
        linguistic_expert: ModalityExpert for text tokens (modality_mask==1).
        norm1:            RMSNorm applied before attention.
        norm2:            RMSNorm applied before FFN.
        rotary_emb:       1D rotary embedding module (set by MoELLM after
                          construction, not in __init__). Used in mha_mmoe
                          to apply position-dependent rotation to Q and K.

    Example::

        layer = MoELayer(hidden_size=2048, num_heads=16, mlp_width=8192)
        x = torch.randn(2, 128, 2048)
        modality_mask = torch.zeros(2, 128, dtype=torch.long)
        modality_mask[:, 64:] = 1  # second half is text
        out = layer(x, modality_mask, attention_mask=None, position_ids=None)
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        mlp_width: int = 8192,
        num_key_value_heads: int = -1,
    ) -> None:
        super().__init__()

        # Resolve default KV heads
        if num_key_value_heads < 0:
            num_key_value_heads = num_heads

        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.mlp_width: int = mlp_width
        self.num_key_value_heads: int = num_key_value_heads
        self.head_dim: int = hidden_size // num_heads
        self.num_groups: int = num_heads // num_key_value_heads  # 1 for MHA

        # ------------------------------------------------------------------ #
        # Modality-specific experts                                            #
        # ------------------------------------------------------------------ #
        self.visual_expert: ModalityExpert = ModalityExpert(
            hidden_size=hidden_size,
            mlp_width=mlp_width,
            num_heads=num_heads,
            modality="visual",
            num_key_value_heads=num_key_value_heads,
        )
        self.linguistic_expert: ModalityExpert = ModalityExpert(
            hidden_size=hidden_size,
            mlp_width=mlp_width,
            num_heads=num_heads,
            modality="linguistic",
            num_key_value_heads=num_key_value_heads,
        )

        # ------------------------------------------------------------------ #
        # Pre-norm layers                                                      #
        # ------------------------------------------------------------------ #
        self.norm1: RMSNorm = RMSNorm(hidden_size, eps=1e-6)
        self.norm2: RMSNorm = RMSNorm(hidden_size, eps=1e-6)

        # ------------------------------------------------------------------ #
        # Rotary embedding — set externally by MoELLM._replace_layers_with_moe
        # after construction. Stored as a plain attribute (not a sub-module)
        # to avoid duplicate state_dict entries when shared across layers.
        # ------------------------------------------------------------------ #
        self.rotary_emb: Optional[nn.Module] = None

    def _split_by_modality(
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split token sequence into visual and text subsets.

        Uses nonzero indexing to extract tokens belonging to each modality.
        The returned indices are needed by _merge_by_modality to scatter
        outputs back to their original positions.

        Args:
            x:             Token tensor of shape (B, L, hidden_size).
            modality_mask: LongTensor of shape (B, L) where 0=visual, 1=text.

        Returns:
            Tuple (x_vis, x_txt, vis_indices, txt_indices) where:
                x_vis:       Float tensor of shape (N_vis, hidden_size)
                             containing all visual tokens across the batch.
                x_txt:       Float tensor of shape (N_txt, hidden_size)
                             containing all text tokens across the batch.
                vis_indices: LongTensor of shape (N_vis, 2) with columns
                             [batch_idx, seq_idx] for each visual token.
                txt_indices: LongTensor of shape (N_txt, 2) with columns
                             [batch_idx, seq_idx] for each text token.
        """
        # nonzero returns (N, 2) tensor of [batch_idx, seq_idx] pairs
        vis_indices: torch.Tensor = (modality_mask == 0).nonzero(as_tuple=False)
        txt_indices: torch.Tensor = (modality_mask == 1).nonzero(as_tuple=False)

        # Extract tokens at the identified positions
        x_vis: torch.Tensor = x[vis_indices[:, 0], vis_indices[:, 1]]
        # shape: (N_vis, hidden_size)

        x_txt: torch.Tensor = x[txt_indices[:, 0], txt_indices[:, 1]]
        # shape: (N_txt, hidden_size)

        return x_vis, x_txt, vis_indices, txt_indices

    def _merge_by_modality(
        self,
        vis_out: torch.Tensor,
        txt_out: torch.Tensor,
        vis_indices: torch.Tensor,
        txt_indices: torch.Tensor,
        B: int,
        L: int,
    ) -> torch.Tensor:
        """Reconstruct the full token sequence from modality-split outputs.

        Scatters visual and text expert outputs back to their original
        positions in the (B, L, hidden_size) sequence.

        Args:
            vis_out:     Float tensor of shape (N_vis, hidden_size).
            txt_out:     Float tensor of shape (N_txt, hidden_size).
            vis_indices: LongTensor of shape (N_vis, 2) with [batch, seq] indices.
            txt_indices: LongTensor of shape (N_txt, 2) with [batch, seq] indices.
            B:           Batch size.
            L:           Sequence length.

        Returns:
            Float tensor of shape (B, L, hidden_size) with all tokens
            in their original positions.
        """
        out: torch.Tensor = torch.zeros(
            B, L, self.hidden_size,
            device=vis_out.device,
            dtype=vis_out.dtype,
        )

        # Scatter visual expert outputs
        if vis_indices.shape[0] > 0:
            out[vis_indices[:, 0], vis_indices[:, 1]] = vis_out

        # Scatter linguistic expert outputs
        if txt_indices.shape[0] > 0:
            out[txt_indices[:, 0], txt_indices[:, 1]] = txt_out

        return out

    def _apply_rotary_emb(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 1D rotary position embeddings to Q and K tensors.

        Handles multiple rotary embedding API styles:
        1. HuggingFace style: rotary_emb(x, position_ids) returns (cos, sin)
           then apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
        2. Simple style: rotary_emb(Q, position_ids) returns rotated Q directly
        3. Fallback: return Q and K unchanged if rotary_emb is None

        Args:
            Q:            Query tensor of shape (B, num_heads, L, head_dim).
            K:            Key tensor of shape (B, num_kv_heads, L, head_dim).
            position_ids: LongTensor of shape (B, L).

        Returns:
            Tuple (Q_rotated, K_rotated) with the same shapes as inputs.
        """
        if self.rotary_emb is None:
            return Q, K

        try:
            # Try HuggingFace transformers style (most common)
            # rotary_emb returns (cos, sin) tensors
            # We pass a dummy x tensor with the right sequence length
            # Some implementations need the actual hidden states for dtype/device
            dummy_x: torch.Tensor = Q.transpose(1, 2)  # (B, L, num_heads, head_dim)

            cos: torch.Tensor
            sin: torch.Tensor

            # Try calling with (x, position_ids) — newer HF style
            try:
                cos, sin = self.rotary_emb(dummy_x, position_ids)
            except TypeError:
                # Older style: rotary_emb(seq_len, device)
                seq_len: int = Q.shape[2]
                cos, sin = self.rotary_emb(seq_len, Q.device)
                # cos/sin shape: (seq_len, head_dim) or (1, seq_len, head_dim)
                if cos.dim() == 2:
                    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, head_dim)
                    sin = sin.unsqueeze(0).unsqueeze(0)

            # Apply rotation: x_rot = x * cos + rotate_half(x) * sin
            # cos/sin may be (B, 1, L, head_dim) or (1, 1, L, head_dim)
            # Ensure they broadcast correctly with (B, num_heads, L, head_dim)
            if cos.dim() == 3:
                # (B, L, head_dim) → (B, 1, L, head_dim)
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            elif cos.dim() == 2:
                # (L, head_dim) → (1, 1, L, head_dim)
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)

            #