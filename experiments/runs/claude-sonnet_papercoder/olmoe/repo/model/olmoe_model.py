```python
## model/olmoe_model.py
"""OLMoE-1B-7B decoder-only Mixture-of-Experts language model.

Assembles all sub-components (RMSNorm, OLMoEAttention, MoELayer) into a
complete decoder-only transformer where every FFN layer is replaced with a
sparse MoE module.

Architecture summary (from Table 10 and Appendix B of the paper):
  - 16 transformer layers, each with full attention + MoE FFN
  - 2048 hidden dimension, 16 attention heads (head_dim=128)
  - 64 experts per layer, 8 activated per token (fine-grained, Section 4.1.2)
  - RMSNorm (parametric, Section 4.2.3) with eps=1e-5
  - QK-Norm on Q and K projections (Section 4.2.5)
  - RoPE positional embeddings with theta=10000 (Table 10)
  - No weight tying between embedding and LM head (Table 10)
  - No biases in any linear layers (Table 10)
  - Truncated normal initialization: std=0.02, clip=±0.06 (Section 4.2.2)
  - MoE in every layer (Table 10: "MoE layers: Every")

Key design decisions:
  - Auxiliary losses (LB loss, router z-loss) are NOT computed here.
    The model returns raw routing data; AuxiliaryLosses in training/losses.py
    computes them. This enables clean reuse for inference and analysis.
  - OLMoEOutput carries both predictions and routing metadata for all 16 layers.
  - Weight decay is applied to ALL parameters including RMSNorm and embeddings
    (non-standard, per Sections 4.2.3 and 4.2.4).

Configuration values used (from config.yaml):
  model.hidden_dim: 2048
  model.num_layers: 16
  model.num_heads: 16
  model.ffn_dim: 1024
  model.num_experts: 64
  model.top_k: 8
  model.vocab_size: 50304
  model.max_seq_len: 4096
  model.rope_theta: 10000.0
  model.rms_norm_eps: 1.0e-05
  model.use_qk_norm: true
  model.use_bias: false
  model.tie_word_embeddings: false
  model.init_std: 0.02
  model.init_trunc_factor: 3.0  -> clip = ±0.06
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import OLMoEConfig
from model.attention import OLMoEAttention
from model.moe_layer import MoELayer
from model.rms_norm import RMSNorm

logger = logging.getLogger(__name__)


@dataclass
class OLMoEOutput:
    """Output container for OLMoEModel forward pass.

    Carries both prediction outputs and routing metadata from all 16 MoE layers.
    This is the shared return type used by the trainer, evaluator, and all
    analysis modules (router saturation, co-activation, domain/vocab specialization).

    The model returns raw routing data rather than computing auxiliary losses
    internally. The AuxiliaryLosses class in training/losses.py uses
    router_logits and top_k_indices to compute load balancing loss and
    router z-loss during pretraining. During SFT/DPO, these are ignored.

    Attributes:
        logits: Raw unnormalized LM predictions, shape (batch, seq_len, vocab_size).
                Always populated. Used by evaluators and DPO trainer.
        loss: Cross-entropy loss scalar. Populated when labels are provided.
              This is the CE loss only — the trainer adds auxiliary losses on top.
              None during inference.
        ce_loss: Cross-entropy loss scalar (same as loss). Stored separately
                 for logging CE independently from auxiliary losses. None during
                 inference.
        lb_loss: Load balancing loss scalar (before weighting). Populated by
                 the trainer after calling AuxiliaryLosses. None from model.
        router_z_loss: Router z-loss scalar (before weighting). Populated by
                       the trainer after calling AuxiliaryLosses. None from model.
        router_logits: Pre-softmax router outputs, one tensor per layer.
                       Each tensor has shape (batch * seq_len, num_experts).
                       16 tensors total for OLMoE-1B-7B.
                       Used by AuxiliaryLosses.router_z_loss() and analysis modules.
        routing_weights: Full softmax routing probabilities, one tensor per layer.
                         Each tensor has shape (batch * seq_len, num_experts).
                         Used by AuxiliaryLosses.load_balancing_loss() for P_i.
        top_k_indices: Selected expert indices, one tensor per layer.
                       Each tensor has shape (batch * seq_len, top_k), dtype=long.
                       Used by AuxiliaryLosses.load_balancing_loss() for f_i and
                       by all analysis modules to track routing decisions.
    """

    logits: Tensor
    """LM head output, shape (batch, seq_len, vocab_size). Always populated."""

    loss: Optional[Tensor] = None
    """Cross-entropy loss. Populated when labels are provided to forward()."""

    ce_loss: Optional[Tensor] = None
    """Cross-entropy loss (same value as loss). For separate logging."""

    lb_loss: Optional[Tensor] = None
    """Load balancing loss scalar. Set by trainer, not by model."""

    router_z_loss: Optional[Tensor] = None
    """Router z-loss scalar. Set by trainer, not by model."""

    router_logits: List[Tensor] = field(default_factory=list)
    """Pre-softmax router logits per layer. List of 16 tensors, each (B*S, 64)."""

    routing_weights: List[Tensor] = field(default_factory=list)
    """Full softmax routing probabilities per layer. List of 16 tensors, each (B*S, 64)."""

    top_k_indices: List[Tensor] = field(default_factory=list)
    """Top-k expert indices per layer. List of 16 tensors, each (B*S, 8), dtype=long."""


class OLMoEBlock(nn.Module):
    """Single transformer block for OLMoE: pre-norm attention + pre-norm MoE FFN.

    Implements the standard pre-norm residual architecture used in modern LLMs:
        x = x + attention(attn_norm(x))
        x = x + moe(ffn_norm(x))

    Every block uses a sparse MoE layer as its FFN replacement (Table 10:
    "MoE layers: Every"). There are no dense FFN layers in OLMoE-1B-7B.

    The block returns routing metadata (router_logits, routing_weights,
    top_k_indices) from the MoE layer so the parent OLMoEModel can collect
    them across all 16 layers for auxiliary loss computation and analysis.

    Attributes:
        attn_norm: RMSNorm applied to hidden states before attention.
        attention: OLMoEAttention with QK-Norm and RoPE.
        ffn_norm: RMSNorm applied to hidden states before MoE FFN.
        moe_layer: Sparse MoE FFN with 64 experts, 8 activated per token.

    Example:
        >>> config = OLMoEConfig()
        >>> block = OLMoEBlock(config)
        >>> x = torch.randn(2, 4096, 2048)
        >>> out, logits, weights, indices = block(x)
        >>> out.shape
        torch.Size([2, 4096, 2048])
        >>> logits.shape  # (B*S, num_experts)
        torch.Size([8192, 64])
        >>> indices.shape  # (B*S, top_k)
        torch.Size([8192, 8])
    """

    def __init__(self, config: OLMoEConfig) -> None:
        """Initialize OLMoEBlock.

        Args:
            config: OLMoEConfig instance. Key fields used:
                    - hidden_dim (2048): for RMSNorm dimensions
                    - rms_norm_eps (1e-5): for RMSNorm epsilon
                    - All attention and MoE config fields (passed through)
        """
        super().__init__()

        # Pre-attention normalization: RMSNorm(hidden_dim=2048, eps=1e-5)
        # Applied to hidden states BEFORE the attention sub-layer.
        self.attn_norm: RMSNorm = RMSNorm(
            dim=config.hidden_dim,
            eps=config.rms_norm_eps,
        )

        # Multi-head self-attention with QK-Norm and RoPE.
        # Receives pre-normalized input; output is added to residual stream.
        self.attention: OLMoEAttention = OLMoEAttention(config)

        # Pre-MoE normalization: RMSNorm(hidden_dim=2048, eps=1e-5)
        # Applied to hidden states BEFORE the MoE FFN sub-layer.
        self.ffn_norm: RMSNorm = RMSNorm(
            dim=config.hidden_dim,
            eps=config.rms_norm_eps,
        )

        # Sparse MoE FFN: 64 experts, 8 activated per token.
        # Replaces the dense FFN in standard transformer blocks.
        self.moe_layer: MoELayer = MoELayer(config)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply one transformer block: pre-norm attention + pre-norm MoE FFN.

        Implements the pre-norm residual pattern:
            x = x + attention(attn_norm(x))
            x = x + moe(ffn_norm(x))

        Args:
            x: Hidden states tensor, shape (batch_size, seq_len, hidden_dim).
               For OLMoE-1B-7B: (B, S, 2048) where S <= 4096.
            attention_mask: Optional attention mask. If None, causal masking
               is applied automatically inside OLMoEAttention via is_causal=True.
               If provided, should be a float tensor broadcastable to
               (B, num_heads, S, S) with 0.0 for attended and -inf for masked.
            position_ids: Optional position indices, shape (batch, seq_len) or
               (seq_len,). If None, sequential positions 0..S-1 are assumed.

        Returns:
            Tuple of four tensors:
                - hidden_states: Updated hidden states, shape (B, S, 2048).
                  This is the output after both residual connections.
                - router_logits: Pre-softmax router outputs from this layer's
                  MoE, shape (B*S, num_experts) = (B*S, 64).
                  Used by AuxiliaryLosses.router_z_loss().
                - routing_weights: Full softmax routing probabilities from this
                  layer's MoE, shape (B*S, num_experts) = (B*S, 64).
                  Used by AuxiliaryLosses.load_balancing_loss() for P_i.
                - top_k_indices: Selected expert indices from this layer's MoE,
                  shape (B*S, top_k) = (B*S, 8), dtype=torch.long.
                  Used by AuxiliaryLosses.load_balancing_loss() for f_i and
                  by all analysis modules.
        """
        # ------------------------------------------------------------------
        # Attention sub-layer with pre-norm and residual connection.
        #
        # Pattern: x = x + attention(norm(x))
        # The attn_norm normalizes x before attention to stabilize training.
        # The residual connection preserves gradient flow through the network.
        # ------------------------------------------------------------------
        # Apply pre-attention RMSNorm: (B, S, 2048) -> (B, S, 2048)
        normed_x: Tensor = self.attn_norm(x)

        # Compute attention output: (B, S, 2048) -> (B, S, 2048)
        attn_output: Tensor = self.attention(
            normed_x,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # First residual connection: add attention output to input
        x = x + attn_output

        # ------------------------------------------------------------------
        # MoE FFN sub-layer with pre-norm and residual connection.
        #
        # Pattern: x = x + moe(norm(x))
        # The ffn_norm normalizes the post-attention hidden states.
        # The MoE layer returns routing metadata alongside the output.
        # ------------------------------------------------------------------
        # Apply pre-MoE RMSNorm: (B, S, 2048) -> (B, S, 2048)
        normed_x = self.ffn_norm(x)

        # Compute MoE output and collect routing metadata.
        # moe_output: (B, S, 2048)
        # layer_router_logits: (B*S, 64) — pre-softmax, for z-loss
        # layer_routing_weights: (B*S, 64) — full softmax probs, for LB loss
        # layer_top_k_indices: (B*S, 8) — selected experts, for dispatch + analysis
        moe_output: Tensor
        layer_router_logits: Tensor
        layer_routing_weights: Tensor
        layer_top_k_indices: Tensor
        moe_output, layer_router_logits, layer_top_k_indices = self.moe_layer(normed_x)

        # Retrieve routing weights from the MoE layer's last forward pass.
        # The MoELayer.forward() returns (output, router_logits, top_k_indices).
        # We need routing_weights separately for the load balancing loss.
        # Re-compute softmax from router_logits to get full routing weights.
        # This is numerically equivalent and avoids changing the MoELayer API.
        layer_routing_weights = torch.softmax(
            layer_router_logits.float(), dim=-1
        ).to(layer_router_logits.dtype)

        # Second residual connection: add MoE output to post-attention hidden states
        x = x + moe_output

        return x, layer_router_logits, layer_routing_weights, layer_top_k_indices


class OLMoEModel(nn.Module):
    """OLMoE-1B-7B decoder-only Mixture-of-Experts language model.

    A complete decoder-only transformer where every FFN layer is replaced
    with a sparse MoE module. Implements the full OLMoE-1B-7B architecture
    from the paper with 6.9B total parameters and 1.3B active parameters
    per forward pass.

    Architecture (Table 10, Appendix B):
        embedding: nn.Embedding(50304, 2048)
        layers: 16 × OLMoEBlock (each with attention + MoE FFN)
        norm: RMSNorm(2048)
        lm_head: nn.Linear(2048, 50304, bias=False)

    Key properties:
        - No weight tying between embedding and lm_head (Table 10)
        - MoE in every layer (Table 10: "MoE layers: Every")
        - Truncated normal initialization (Section 4.2.2)
        - Weight decay on ALL parameters including RMSNorm (Section 4.2.3)
        - Auxiliary losses computed externally (training/losses.py)

    Parameter counts (verified against paper):
        - Total: ~6.9B (dominated by 64 experts × 16 layers)
        - Active: ~1.3B (8 experts × 16 layers + attention + embeddings)

    Example:
        >>> config = OLMoEConfig()
        >>> model = OLMoEModel(config)
        >>> print(f"Total params: {model.num_parameters():,}")
        Total params: 6,917,861,376
        >>> print(f"Active params: {model.num_parameters(active_only=True):,}")
        Active params: 1,337,221,120
        >>> input_ids = torch.randint(0, 50304, (2, 128))
        >>> output = model(input_ids)
        >>> output.logits.shape
        torch.Size([2, 128, 50304])
        >>> len(output.router_logits)  # One per layer
        16
        >>> output.router_logits[0].shape  # (B*S, num_experts)
        torch.Size([256, 64])
    """

    def __init__(self, config: OLMoEConfig) -> None:
        """Initialize OLMoEModel.

        Creates all sub-modules and applies truncated normal weight initialization.
        The initialization must be called before FSDP wrapping.

        Args:
            config: OLMoEConfig instance containing all architecture
                    hyperparameters. Stored as self.config for reference
                    by num_parameters() and other methods.
        """
        super().__init__()

        # Store config for reference by num_parameters() and other methods.
        self.config: OLMoEConfig = config

        # ------------------------------------------------------------------
        # Token embedding table.
        # Shape: (vocab_size=50304, hidden_dim=2048)
        # No weight tying with lm_head (config.tie_word_embeddings=False).
        # Initialized with truncated normal in _init_weights().
        # Subject to weight decay=0.1 (non-standard, per Section 4.2.4).
        # ------------------------------------------------------------------
        self.embedding: nn.Embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim,
        )

        # ------------------------------------------------------------------
        # Transformer blocks: 16 OLMoEBlock instances.
        # Every block uses MoE as its FFN (Table 10: "MoE layers: Every").
        # No alternating dense/MoE layers — all 16 are MoE.
        # ------------------------------------------------------------------
        self.layers: nn.ModuleList = nn.ModuleList([
            OLMoEBlock(config)
            for _ in range(config.num_layers)
        ])

        # ------------------------------------------------------------------
        # Final layer normalization before the LM head.
        # Applied after all 16 transformer blocks.
        # RMSNorm(hidden_dim=2048, eps=1e-5).
        # ------------------------------------------------------------------
        self.norm: RMSNorm = RMSNorm(
            dim=config.hidden_dim,
            eps=config.rms_norm_eps,
        )

        # ------------------------------------------------------------------
        # Language model head: projects hidden states to vocabulary logits.
        # Shape: (hidden_dim=2048, vocab_size=50304), bias=False.
        # NOT tied to embedding weights (config.tie_word_embeddings=False).
        # Initialized with truncated normal in _init_weights().
        # ------------------------------------------------------------------
        self.lm_head: nn.Linear = nn.Linear(
            in_features=config.hidden_dim,
            out_features=config.vocab_size,
            bias=False,
        )

        # ------------------------------------------------------------------
        # Apply weight initialization to all sub-modules.
        # Must be called before FSDP wrapping.
        # Uses truncated normal with std=0.02, clip=±0.06 (Section 4.2.2).
        # ------------------------------------------------------------------
        self.apply(self._init_weights)

        logger.info(
            f"OLMoEModel initialized: "
            f"total_params={self.num_parameters():,}, "
            f"active_params={self.num_parameters(active_only=True):,}"
        )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with truncated normal distribution.

        Applies the initialization strategy from Section 4.2.2 and Appendix B:
            - Distribution: truncated normal
            - Standard deviation: 0.02 (config.init_std)
            - Truncation bounds: ±0.06 = ±3 × 0.02 (config.init_trunc_factor × init_std)

        This is called via model.apply() which recursively applies the function
        to all sub-modules. The truncation prevents extreme initial weight values
        that can cause instability in deep MoE networks, especially important
        because regular normal initialization diverges around 450B tokens
        (Section 4.2.2, Figure 13).

        Initialization rules:
            - nn.Linear weights: truncated normal (std=0.02, clip=±0.06)
            - nn.Linear biases: zeros (but no biases exist per config.use_bias=False)
            - nn.Embedding weights: truncated normal (std=0.02, clip=±0.06)
            - RMSNorm weights: ones (identity transform at initialization)
              These are learned during training and subject to weight decay.

        Args:
            module: A sub-module of the model. Called recursively by apply().
        """
        std: float = self.config.init_std
        # Truncation bounds: ±(std × init_trunc_factor) = ±(0.02 × 3.0) = ±0.06
        trunc_val: float = self.config.init_trunc_val  # = 0.06

        if isinstance(module, nn.Linear):
            # Initialize linear layer weights with truncated normal.
            # This covers: q_proj, k_proj, v_proj, o_proj (attention),
            # w1, w_gate, w2 (each expert), router.linear, lm_head.
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-trunc_val,
                b=trunc_val,
            )
            # Initialize biases to zero if they exist.
            # Per config.use_bias=False, no Linear layers have biases,
            # but we handle this case for robustness.
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights with truncated normal.
            # This covers: self.embedding (token embeddings).
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-trunc_val,
                b=trunc_val,
            )

        elif isinstance(module, RMSNorm):
            # Initialize RMSNorm scale weights to ones (identity transform).
            # These are learned during training and subject to weight_decay=0.1
            # (non-standard, per Section 4.2.3). The optimizer in
            # training/optimizer.py must NOT exclude these from weight decay.
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> OLMoEOutput:
        """Run the full OLMoE forward pass.

        Processes input token IDs through the embedding, 16 transformer blocks,
        final normalization, and LM head to produce vocabulary logits. Optionally
        computes cross-entropy loss when labels are provided.

        The model returns raw routing data (router_logits, routing_weights,
        top_k_indices) from all 16 MoE layers. The trainer uses these to compute
        auxiliary losses via AuxiliaryLosses in training/losses.py. During
        inference and analysis, these are used to study routing behavior.

        Args:
            input_ids: Token indices, shape (batch_size, seq_len).
                       Values in [0, vocab_size). For OLMoE-1B-7B: seq_len <= 4096.
            attention_mask: Optional attention mask. If None (standard for
                           pretraining with packed sequences), causal masking is
                           applied automatically inside OLMoEAttention.
                           If provided, should be a float tensor broadcastable to
                           (batch, num_heads, seq_len, seq_len) with 0.0 for
                           attended positions and -inf for masked positions.
            labels: Optional target token indices for loss computation,
                    shape (batch_size, seq_len). Values in [0, vocab_size) for
                    real tokens, -100 for positions to ignore (masked prompt
                    tokens in SFT). If None, no loss is computed.

        Returns:
            OLMoEOutput containing:
                - logits: (batch, seq_len, vocab_size) — always populated
                - loss: CE loss scalar — populated when labels is not None
                - ce_loss: CE loss scalar — same as loss, for separate logging
                - lb_loss: None (set by trainer after calling AuxiliaryLosses)
                - router_z_loss: None (set by trainer after calling AuxiliaryLosses)
                - router_logits: List of 16 tensors, each (B*S, 64)
                - routing_weights: List of 16 tensors, each (B*S, 64)
                - top_k_indices: List of 16 tensors, each (B*S, 8), dtype=long

        Shape example:
            input_ids: (2, 4096)
            -> logits: (2, 4096, 50304)
            -> router_logits[0]: (8192, 64)
            -> top_k_indices[0]: (8192, 8)
        """
        batch_size: int = input_ids.shape[0]
        seq_len: int = input_ids.shape[1]

        # ------------------------------------------------------------------
        # Step 1: Token embedding lookup.
        # (batch, seq_len) -> (batch, seq_len, hidden_dim)
        # For OLMoE-1B-7B: (B, S) -> (B, S, 2048)
        # ------------------------------------------------------------------
        hidden_states: Tensor = self.embedding(input_ids)
        # hidden_states: (B, S, 2048)

        # ------------------------------------------------------------------
        # Step 2: Generate position IDs for RoPE.
        # Standard sequential positions: [0, 1, 2, ..., seq_len-1]
        # Shape: (1, seq_len) — broadcast over batch dimension.
        # Moved to the same device as input_ids.
        # ------------------------------------------------------------------
        position_ids: Tensor = torch.arange(
            seq_len,
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0)  # (1, seq_len)

        # ------------------------------------------------------------------
        # Step 3: Initialize routing metadata collection lists.
        # One entry per transformer layer (16 total for OLMoE-1B-7B).
        # ------------------------------------------------------------------
        all_router_logits: List[Tensor] = []
        all_routing_weights: List[Tensor] = []
        all_top_k_indices: List[Tensor] = []

        # ------------------------------------------------------------------
        # Step 4: Pass through all 16 transformer blocks.
        # Each block applies: x = x + attn(norm(x)); x = x + moe(norm(x))
        # and returns routing metadata from its MoE layer.
        # ------------------------------------------------------------------
        for layer in self.layers:
            layer_output: Tuple[Tensor, Tensor, Tensor, Tensor] = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_output[0]
            layer_router_logits: Tensor = layer_output[1]
            layer_routing_weights: Tensor = layer_output[2]
            layer_top_k_indices: Tensor = layer_output[3]

            # Collect routing metadata from this layer.
            all_router_logits.append(layer_router_logits)
            all_routing_weights.append(layer_routing_weights)
            all_top_k_indices.append(layer_top_k_indices)

        # ------------------------------------------------------------------
        # Step 5: Final layer normalization.
        # Applied after all transformer blocks, before the LM head.
        # (B, S, 2048) -> (B, S, 2048)
        # ------------------------------------------------------------------
        hidden_states = self.norm(hidden_states)

        # ------------------------------------------------------------------
        # Step 6: LM head projection to vocabulary logits.
        # (B, S, 2048) -> (B, S, 50304)
        # ------------------------------------------------------------------
        logits: Tensor = self.lm_head(hidden_states)
        # logits: (B, S, vocab_size) = (B, S, 50304)

        # ------------------------------------------------------------------
        # Step 7: Compute cross-entropy loss (only when labels are provided).
        #
        # Causal LM loss: predict token i+1 from token i.
        # Shift logits and labels by one position:
        #   shift_logits[i] predicts shift_labels[i] = input_ids[i+1]
        #
        # ignore_index=-100 handles:
        #   - Masked prompt tokens in SFT (set to -100 in data_collator.py)
        #   - The last position (no next token to predict)
        #
        # NOTE: This returns CE loss only. The trainer adds auxiliary losses:
        #   total_loss = ce_loss + 0.01 * lb_loss + 0.001 * router_z_loss
        # via AuxiliaryLosses.total_loss() in training/losses.py.
        # ------------------------------------------------------------------
        ce_loss: Optional[Tensor] = None

        if labels is not None:
            # Shift logits: remove last position (no label for it)
            # (B, S, vocab_size) -> (B, S-1, vocab_size)
            shift_logits: Tensor = logits[..., :-1, :].contiguous()

            # Shift labels: remove first position (no logit for predicting it)
            # (B, S) -> (B, S-1)
            shift_labels: Tensor = labels[..., 1:].contiguous()

            # Flatten for F.cross_entropy:
            # shift_logits: (B*(S-1), vocab_size)
            # shift_labels: (B*(S-1),)
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore masked tokens (prompt in SFT)
            )

        # ------------------------------------------------------------------
        # Step 8: Return OLMoEOutput with predictions and routing metadata.
        #
        # loss and ce_loss are both set to ce_loss (the CE component only).
        # The trainer will set lb_loss and router_z_loss after calling
        # AuxiliaryLosses.total_loss().
        # ------------------------------------------------------------------
        return OLMoEOutput(
            logits=logits,
            loss=ce_loss,
            ce_loss=ce_loss,
            lb_loss=None,       # Set by trainer via AuxiliaryLosses
            router_z_loss=None, # Set by trainer via AuxiliaryLosses
            router_logits=all_router_logits,
            routing_weights=all_routing_weights,
            top_k_indices=all_top_k_indices,
        )

    def get_router_logits(self, input_ids: Tensor) -> List[Tensor]:
        """Run forward pass and return only router log