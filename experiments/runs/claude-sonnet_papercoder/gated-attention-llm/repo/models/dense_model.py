## models/dense_model.py
"""Dense transformer model for 1.7B parameter experiments.

This module implements DenseModel, the 1.7B parameter transformer used for:
    - Table 2: Dense model experiments across various LR/batch/depth configurations
    - Figure 1 (right): Training loss curves comparing baseline vs. gated models
    - Table 5: Long-context extension experiments (after context extension phase)
    - Sections 4.2, 4.3: Analysis of gating scores and attention sink behavior

The model is a standard pre-norm transformer with:
    - Token embedding with weight tying to lm_head
    - N layers of TransformerBlock (GatedMultiHeadAttention + SwiGLU FFN)
    - Final RMSNorm + linear language model head

When gating is enabled and ffn_reduce_for_gate=True, the FFN intermediate
dimension is automatically reduced to maintain parameter parity with the
ungated baseline. This matches the paper's Sec 3.2.2: "we reduce the width
of FFN to maintain the parameter size."

Analysis methods (get_gate_scores_all_layers, get_attention_weights_all_layers)
rely on two contracts with lower-level modules:
    - GateModule.forward() sets self._last_gate_scores (detached) after each call
    - TransformerBlock.forward() sets self.last_attn_weights (detached) after each call

Config values used (from config.yaml):
    model.type: Must be 'dense'
    model.num_layers: Number of transformer layers (28 or 48)
    model.d_model: Hidden dimension (2048 for 28L, 1536 for 48L)
    model.vocab_size: Vocabulary size (32000)
    model.ffn_dim: Base FFN intermediate dimension (8192 for 28L, 6144 for 48L)
    model.ffn_reduce_for_gate: Whether to reduce FFN width for parameter parity
    model.num_heads: Query heads q (32 for 28L, 24 for 48L)
    model.num_kv_heads: KV heads k (8 for both)
    model.d_k: Per-head dimension (64 for both)
    model.max_seq_len: Maximum sequence length (4096)
    model.sandwich_norm: Whether to apply sandwich normalization
    gate.position: Gate position ('G1'–'G5' or 'none')
    gate.granularity: 'elementwise' or 'headwise'
    gate.head_specific: bool
    gate.gate_type: 'multiplicative' or 'additive'
    gate.activation: activation function name
    rope.base: RoPE base frequency (10000.0)
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.gate_module import GateModule
from models.transformer_block import TransformerBlock


class DenseModel(nn.Module):
    """Dense transformer language model with configurable gated attention.

    Implements a standard pre-norm transformer stack for autoregressive
    language modeling. Supports all gating variants from the paper through
    the configurable GatedMultiHeadAttention inside each TransformerBlock.

    When ffn_reduce_for_gate=True and a gate is configured, the FFN
    intermediate dimension is automatically reduced so that the total
    parameter count matches the ungated baseline. This ensures fair
    comparison in Table 2 experiments.

    Attributes:
        config: The original configuration object (not mutated).
        num_layers: Number of transformer layers.
        d_model: Hidden dimension of the model.
        vocab_size: Vocabulary size.
        ffn_dim: Effective FFN intermediate dimension (possibly reduced).
        embedding: Token embedding table, shape [vocab_size, d_model].
        layers: ModuleList of TransformerBlock instances.
        norm: Final RMSNorm before lm_head.
        lm_head: Linear projection to vocabulary logits, shape [d_model, vocab_size].
            Weight-tied to embedding.weight.
    """

    def __init__(self, config: object) -> None:
        """Initialize DenseModel from configuration.

        Constructs the full model including optional FFN dimension reduction
        for parameter parity when gating is enabled.

        Args:
            config: Configuration object (OmegaConf DictConfig or compatible).
                Must expose model.*, gate.*, and rope.* fields as described
                in the module docstring. The config is never mutated in place;
                a deep copy is used for block construction when ffn_dim is
                overridden.

        Raises:
            ValueError: If model.type is not 'dense'.
            ValueError: If the computed reduced_ffn_dim falls below d_model // 2,
                indicating the gate is too large relative to the FFN.
        """
        super().__init__()

        # Validate model type
        model_type: str = str(getattr(config.model, "type", "dense"))
        if model_type != "dense":
            raise ValueError(
                f"DenseModel requires config.model.type='dense', got '{model_type}'"
            )

        # Store original config (never mutated)
        self.config = config

        # Extract scalar model dimensions from config
        self.num_layers: int = int(config.model.num_layers)
        self.d_model: int = int(config.model.d_model)
        self.vocab_size: int = int(config.model.vocab_size)
        base_ffn_dim: int = int(config.model.ffn_dim)
        ffn_reduce_for_gate: bool = bool(
            getattr(config.model, "ffn_reduce_for_gate", True)
        )

        # Gate position — needed to determine if reduction applies
        gate_position: str = str(getattr(config.gate, "position", "none"))

        # -----------------------------------------------------------------------
        # FFN dimension reduction for parameter parity
        # Paper Sec 3.2.2: "we reduce the width of FFN to maintain the parameter size"
        # Only applies when both conditions hold:
        #   1. ffn_reduce_for_gate is True
        #   2. A gate is actually configured (gate_position != 'none')
        # -----------------------------------------------------------------------
        effective_ffn_dim: int = self._compute_effective_ffn_dim(
            base_ffn_dim=base_ffn_dim,
            ffn_reduce_for_gate=ffn_reduce_for_gate,
            gate_position=gate_position,
            config=config,
        )
        self.ffn_dim: int = effective_ffn_dim

        # -----------------------------------------------------------------------
        # Build block_config with effective_ffn_dim
        # Use a deep copy to avoid mutating the original config object.
        # OmegaConf configs support structured access; we use copy.deepcopy
        # for compatibility with both OmegaConf and plain Python objects.
        # -----------------------------------------------------------------------
        block_config = self._make_block_config(config, effective_ffn_dim)

        # -----------------------------------------------------------------------
        # Token embedding
        # Shape: [vocab_size, d_model]
        # Weight-tied to lm_head below (standard for LLMs, reduces param count
        # by vocab_size * d_model ≈ 32000 * 2048 ≈ 65M for the 28L model).
        # -----------------------------------------------------------------------
        self.embedding: nn.Embedding = nn.Embedding(self.vocab_size, self.d_model)

        # -----------------------------------------------------------------------
        # Transformer layers
        # Each TransformerBlock contains GatedMultiHeadAttention + SwiGLU FFN.
        # The gate configuration is read from block_config.gate.* fields.
        # -----------------------------------------------------------------------
        self.layers: nn.ModuleList = nn.ModuleList(
            [
                TransformerBlock(block_config, layer_idx=i)
                for i in range(self.num_layers)
            ]
        )

        # -----------------------------------------------------------------------
        # Final normalization and language model head
        # -----------------------------------------------------------------------
        self.norm: nn.RMSNorm = nn.RMSNorm(self.d_model)
        self.lm_head: nn.Linear = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # -----------------------------------------------------------------------
        # Weight tying: lm_head.weight = embedding.weight
        # Standard for modern LLMs. PyTorch's parameters() deduplicates shared
        # tensors, so get_num_params() correctly counts them once.
        # -----------------------------------------------------------------------
        self.lm_head.weight = self.embedding.weight

        # -----------------------------------------------------------------------
        # Weight initialization
        # -----------------------------------------------------------------------
        self._init_weights()

    def _compute_effective_ffn_dim(
        self,
        base_ffn_dim: int,
        ffn_reduce_for_gate: bool,
        gate_position: str,
        config: object,
    ) -> int:
        """Compute the effective FFN intermediate dimension with optional reduction.

        When gating is added to the model, the FFN width is reduced to maintain
        total parameter parity with the ungated baseline. This implements the
        paper's Sec 3.2.2 requirement.

        The reduction formula from the design spec:
            gate_params_per_layer = GateModule(...).count_params()
            total_gate_params = gate_params_per_layer * num_layers
            ffn_param_reduction = total_gate_params
            reduced_ffn_dim = base_ffn_dim - ffn_param_reduction // (2 * d_model * num_layers)

        The divisor 2 * d_model * num_layers accounts for the two FFN matrices
        (gate_proj and up_proj) that each contribute d_model * d_ffn parameters.

        Args:
            base_ffn_dim: The configured FFN intermediate dimension from config.
            ffn_reduce_for_gate: Whether to apply the reduction.
            gate_position: Gate position string; reduction skipped if 'none'.
            config: Full configuration object for GateModule instantiation.

        Returns:
            Effective FFN intermediate dimension (possibly reduced and rounded
            down to nearest multiple of 64 for hardware efficiency).

        Raises:
            ValueError: If the reduced dimension falls below d_model // 2.
        """
        # No reduction needed if gate is disabled or reduction is not requested
        if not ffn_reduce_for_gate or gate_position == "none":
            return base_ffn_dim

        # Instantiate a temporary GateModule to get the exact parameter count.
        # This ensures consistency with whatever _build_projection() computes.
        try:
            temp_gate = GateModule(
                position=str(getattr(config.gate, "position", "G1")),
                granularity=str(getattr(config.gate, "granularity", "elementwise")),
                head_specific=bool(getattr(config.gate, "head_specific", True)),
                gate_type=str(getattr(config.gate, "gate_type", "multiplicative")),
                activation=str(getattr(config.gate, "activation", "sigmoid")),
                d_model=int(config.model.d_model),
                num_heads=int(config.model.num_heads),
                num_kv_heads=int(config.model.num_kv_heads),
                d_k=int(config.model.d_k),
                input_independent=bool(
                    getattr(config.gate, "input_independent", False)
                ),
            )
            gate_params_per_layer: int = temp_gate.count_params()
        except Exception:
            # If gate instantiation fails for any reason, skip reduction
            return base_ffn_dim

        # No reduction if gate adds no parameters (e.g., silu_only variant)
        if gate_params_per_layer == 0:
            return base_ffn_dim

        # Compute total gate parameters across all layers
        total_gate_params: int = gate_params_per_layer * self.num_layers

        # Reduction formula from design spec:
        # The SwiGLU FFN has 3 matrices: gate_proj [d_model, d_ffn],
        # up_proj [d_model, d_ffn], down_proj [d_ffn, d_model].
        # Total per layer = 3 * d_model * d_ffn.
        # We reduce d_ffn by: total_gate_params // (2 * d_model * num_layers)
        # (using 2 as a conservative approximation for gate_proj + up_proj)
        reduction_per_dim: int = total_gate_params // (
            2 * self.d_model * self.num_layers
        )
        reduced_ffn_dim: int = base_ffn_dim - reduction_per_dim

        # Round down to nearest multiple of 64 for hardware efficiency
        # (tensor cores prefer dimensions divisible by 64 or 128)
        alignment: int = 64
        reduced_ffn_dim = (reduced_ffn_dim // alignment) * alignment

        # Guard: ensure reduced dimension is reasonable
        min_ffn_dim: int = self.d_model // 2
        if reduced_ffn_dim < min_ffn_dim:
            raise ValueError(
                f"Computed reduced_ffn_dim={reduced_ffn_dim} is below minimum "
                f"{min_ffn_dim} (d_model // 2 = {self.d_model} // 2). "
                f"The gate adds too many parameters relative to the FFN. "
                f"Consider disabling ffn_reduce_for_gate or using a smaller gate."
            )

        return reduced_ffn_dim

    def _make_block_config(self, config: object, effective_ffn_dim: int) -> object:
        """Create a block config with the effective FFN dimension.

        Creates a deep copy of the config and overrides model.ffn_dim with
        the computed effective_ffn_dim. This avoids mutating the original
        config object while ensuring TransformerBlock uses the correct FFN width.

        Args:
            config: Original configuration object.
            effective_ffn_dim: The (possibly reduced) FFN intermediate dimension.

        Returns:
            A copy of config with model.ffn_dim set to effective_ffn_dim.
        """
        # Try OmegaConf-style copy first, fall back to deepcopy
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(config, DictConfig):
                block_config = OmegaConf.to_container(config, resolve=True)
                block_config = OmegaConf.create(block_config)
                # Override ffn_dim using OmegaConf's structured update
                with OmegaConf.open_dict(block_config):
                    block_config.model.ffn_dim = effective_ffn_dim
                return block_config
        except (ImportError, Exception):
            pass

        # Fallback: deep copy and set attribute directly
        block_config = copy.deepcopy(config)
        try:
            block_config.model.ffn_dim = effective_ffn_dim
        except Exception:
            # If config is immutable, wrap it in a simple namespace
            block_config = _ConfigWrapper(config, ffn_dim_override=effective_ffn_dim)

        return block_config

    def _init_weights(self) -> None:
        """Initialize model weights following standard LLM conventions.

        Applies:
            - Embedding: normal(0, 0.02)
            - Linear layers: normal(0, 0.02)
            - Output projections (W_O, down_proj): scaled by 1/sqrt(2 * num_layers)
              to prevent residual stream growth (GPT-2 style initialization)
            - Gate projections: normal(0, 0.02)
            - RMSNorm weights: 1.0 (default, not overridden)

        The output projection scaling is important for deep models (48 layers)
        to prevent the residual stream from growing too large during initialization.
        """
        std: float = 0.02
        output_proj_std: float = std / math.sqrt(2.0 * self.num_layers)

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Scale output projections (W_O and down_proj) to prevent residual growth
        # These are the "output" projections that feed directly into the residual stream.
        for layer in self.layers:
            # Attention output projection W_O
            nn.init.normal_(
                layer.attention.W_O.weight,
                mean=0.0,
                std=output_proj_std,
            )
            # FFN down projection (only for FeedForward, not MoELayer)
            if hasattr(layer.ffn, "down_proj"):
                nn.init.normal_(
                    layer.ffn.down_proj.weight,
                    mean=0.0,
                    std=output_proj_std,
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through the dense transformer model.

        Computes autoregressive language model logits for the input token
        sequence. The causal masking is handled internally by each
        TransformerBlock's GatedMultiHeadAttention when mask=None.

        Args:
            input_ids: Integer token indices, shape [batch, seq_len].
            mask: Optional additive causal attention mask.
                Shape [1, 1, seq_len, seq_len] or [batch, 1, seq_len, seq_len].
                Values: 0.0 for allowed positions, -inf for masked positions.
                If None, each attention layer builds the causal mask internally
                using F.scaled_dot_product_attention(is_causal=True).
            position_ids: Token position indices for RoPE.
                Shape [batch, seq_len] or [seq_len].
                If None, defaults to sequential [0, 1, ..., seq_len-1].

        Returns:
            Tuple of:
                - logits: Vocabulary logits, shape [batch, seq_len, vocab_size].
                  Used for cross-entropy loss computation in Trainer._compute_loss.
                - aux_losses: Empty dict {} for dense model (no MoE auxiliary losses).
                  MoEModel returns {'lbl_loss': ..., 'z_loss': ...} in the same slot.
        """
        batch_size: int = input_ids.shape[0]
        seq_len: int = input_ids.shape[1]

        # Generate default position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)

        # Step 1: Token embedding
        # [batch, seq_len] → [batch, seq_len, d_model]
        X: torch.Tensor = self.embedding(input_ids)

        # Step 2: Pass through all transformer layers
        # Each TransformerBlock returns [batch, seq_len, d_model]
        # and stores last_attn_weights for analysis.
        layer: TransformerBlock
        for layer in self.layers:
            X = layer(X, mask=mask, position_ids=position_ids)

        # Step 3: Final RMSNorm
        X = self.norm(X)

        # Step 4: Language model head (weight-tied to embedding)
        # [batch, seq_len, d_model] → [batch, seq_len, vocab_size]
        logits: torch.Tensor = self.lm_head(X)

        # Return empty aux_losses dict (dense model has no MoE losses)
        return logits, {}

    def get_num_params(self) -> int:
        """Count the total number of unique learnable parameters.

        Because lm_head.weight is tied to embedding.weight, PyTorch's
        parameters() iterator deduplicates them, counting the shared tensor
        only once. This gives the correct total unique parameter count.

        Returns:
            Total number of unique learnable parameters (int).

        Example:
            For the 28-layer dense model with d_model=2048, vocab_size=32000:
            - Embedding: 32000 * 2048 = 65.5M (shared with lm_head)
            - 28 × TransformerBlock: ~1.6B
            - Final norm: 2048
            Total ≈ 1.7B parameters
        """
        return sum(p.numel() for p in self.parameters())

    def get_gate_scores_all_layers(
        self,
        input_ids: torch.Tensor,
    ) -> List[Optional[torch.Tensor]]:
        """Capture gate scores from all layers for analysis.

        Runs a forward pass and collects the gate score tensors stored by
        GateModule._last_gate_scores during the forward pass. Used by
        GatingScoreAnalyzer to reproduce Figure 3, Figure 7, and Table 4's
        "Gate Score" column.

        This method relies on the contract that GateModule.forward() sets
        self._last_gate_scores = gate_scores.detach() after computing gate
        scores. If GateModule does not set this attribute (e.g., for
        silu_only or rmsnorm variants), None is returned for that layer.

        Args:
            input_ids: Integer token indices, shape [batch, seq_len].
                Typically a single batch from the evaluation dataset.

        Returns:
            List of length num_layers. Each element is either:
                - Tensor of gate scores for that layer (detached, on original device).
                  Shape depends on gate variant:
                    G1 elementwise head-specific: [batch, seq, num_heads, d_k]
                    G1 headwise head-specific:    [batch, seq, num_heads, 1]
                    G1 elementwise head-shared:   [batch, seq, 1, d_k]
                - None if the layer has no gate or gate scores are unavailable.

        Note:
            This method temporarily switches to eval() mode and restores the
            original training mode afterward. It runs with torch.no_grad() to
            avoid unnecessary gradient computation.
        """
        # Save current training mode and switch to eval
        was_training: bool = self.training
        self.eval()

        try:
            with torch.no_grad():
                # Run forward pass — this populates _last_gate_scores in each GateModule
                self.forward(input_ids, mask=None, position_ids=None)

            # Collect gate scores from each layer
            gate_scores: List[Optional[torch.Tensor]] = []
            for layer in self.layers:
                gate_module: Optional[GateModule] = layer.attention.gate
                if gate_module is None:
                    gate_scores.append(None)
                    continue

                # Retrieve cached gate scores set during forward pass
                last_scores: Optional[torch.Tensor] = getattr(
                    gate_module, "_last_gate_scores", None
                )
                gate_scores.append(last_scores)

        finally:
            # Restore original training mode
            if was_training:
                self.train()

        return gate_scores

    def get_attention_weights_all_layers(
        self,
        input_ids: torch.Tensor,
    ) -> List[Optional[torch.Tensor]]:
        """Capture attention weight matrices from all layers for analysis.

        Runs a forward pass with return_attn_weights=True (forcing manual SDPA
        computation) and collects the attention weight tensors stored by each
        TransformerBlock in self.last_attn_weights. Used by AttentionSinkAnalyzer
        to reproduce Figure 2 and the "F-Attn" column in Table 4.

        Paper finding (Sec 4.3): Baseline model allocates 46.7% of attention
        to the first token on average; gated model reduces this to 4.8%.

        This method relies on the contract that TransformerBlock.forward() sets
        self.last_attn_weights after each forward pass. When flash attention is
        used (return_attn_weights=False), last_attn_weights may be None.

        To ensure attention weights are captured, this method temporarily patches
        each TransformerBlock to call attention with return_attn_weights=True.

        Args:
            input_ids: Integer token indices, shape [batch, seq_len].
                Typically a single batch from the evaluation dataset.

        Returns:
            List of length num_layers. Each element is either:
                - Tensor of attention weights, shape [batch, num_heads, seq, seq].
                  Entry [b, h, i, j] is the attention weight from query position i
                  to key position j in head h of batch element b.
                - None if attention weights were not captured (flash attention path).

        Note:
            This method temporarily switches to eval() mode and restores the
            original training mode afterward. It runs with torch.no_grad() to
            avoid unnecessary gradient computation.

            For long sequences, storing full attention matrices for all layers
            is memory-intensive. This method is intended for analysis on short
            sequences (e.g., seq_len=512 or 1024).
        """
        # Save current training mode and switch to eval
        was_training: bool = self.training
        self.eval()

        try:
            with torch.no_grad():
                # Run forward pass with attention weight capture enabled.
                # We directly call each layer with return_attn_weights=True
                # by running the forward pass manually here.
                batch_size: int = input_ids.shape[0]
                seq_len: int = input_ids.shape[1]

                position_ids: torch.Tensor = torch.arange(
                    seq_len,
                    device=input_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(batch_size, -1)

                X: torch.Tensor = self.embedding(input_ids)

                # Run each layer, capturing attention weights explicitly
                for layer in self.layers:
                    # Pre-norm
                    attn_input: torch.Tensor = layer.norm1(X)

                    # Run attention with return_attn_weights=True to force
                    # manual SDPA computation (bypasses flash attention)
                    attn_out: torch.Tensor
                    attn_weights: Optional[torch.Tensor]
                    attn_out, attn_weights = layer.attention(
                        attn_input,
                        mask=None,
                        position_ids=position_ids,
                        return_attn_weights=True,
                    )

                    # Store attention weights in the block for retrieval below
                    layer.last_attn_weights = (
                        attn_weights.detach() if attn_weights is not None else None
                    )

                    # Apply sandwich norm if configured
                    if layer.sandwich_norm and layer.norm_attn_out is not None:
                        attn_out = layer.norm_attn_out(attn_out)

                    X = X + attn_out

                    # FFN sublayer
                    ffn_input: torch.Tensor = layer.norm2(X)
                    ffn_out: torch.Tensor = layer.ffn(ffn_input)

                    if layer.sandwich_norm and layer.norm_ffn_out is not None:
                        ffn_out = layer.norm_ffn_out(ffn_out)

                    X = X + ffn_out

            # Collect stored attention weights from each layer
            attention_weights: List[Optional[torch.Tensor]] = [
                layer.last_attn_weights for layer in self.layers
            ]

        finally:
            # Restore original training mode
            if was_training:
                self.train()

        return attention_weights

    def extra_repr(self) -> str:
        """Return human-readable model description for print(model).

        Returns:
            String summarizing the model configuration.
        """
        gate_pos: str = str(getattr(self.config.gate, "position", "none"))
        return (
            f"num_layers={self.num_layers}, "
            f"d_model={self.d_model}, "
            f"vocab_size={self.vocab_size}, "
            f"ffn_dim={self.ffn_dim}, "
            f"gate_position={gate_pos}, "
            f"total_params={self.get_num_params():,}"
        )


class _ConfigWrapper:
    """Lightweight wrapper to override ffn_dim in an immutable config.

    Used as a fallback when the config object cannot be deep-copied or
    mutated directly (e.g., frozen OmegaConf configs). Proxies all
    attribute access to the original config except for model.ffn_dim.

    This is an internal implementation detail of DenseModel and should
    not be used outside this module.
    """

    def __init__(self, original_config: object, ffn_dim_override: int) -> None:
        """Initialize the config wrapper.

        Args:
            original_config: The original configuration object to proxy.
            ffn_dim_override: The value to return for model.ffn_dim.
        """
        # Use object.__setattr__ to avoid triggering __setattr__ recursion
        object.__setattr__(self, "_original", original_config)
        object.__setattr__(self, "_ffn_dim_override", ffn_dim_override)
        # Create a wrapped model sub-config
        object.__setattr__(
            self,
            "_model_wrapper",
            _ModelConfigWrapper(original_config.model, ffn_dim_override),
        )

    def __getattr__(self, name: str) -> object:
        """Proxy attribute access to original config, except for 'model'."""
        if name == "model":
            return object.__getattribute__(self, "_model_wrapper")
        original = object.__getattribute__(self, "_original")
        return getattr(original, name)


class _ModelConfigWrapper:
    """Lightweight wrapper to override ffn_dim in the model sub-config.

    Proxies all attribute access to the original model config except for
    ffn_dim, which returns the overridden value.
    """

    def __init__(self, original_model_config: object, ffn_dim_override: int) -> None:
        """Initialize the model config wrapper.

        Args:
            original_model_config: The original model configuration sub-object.
            ffn_dim_override: The value to return for ffn_dim.
        """
        object.__setattr__(self, "_original", original_model_config)
        object.__setattr__(self, "_ffn_dim_override", ffn_dim_override)

    def __getattr__(self, name: str) -> object:
        """Proxy attribute access, overriding ffn_dim."""
        if name == "ffn_dim":
            return object.__getattribute__(self, "_ffn_dim_override")
        original = object.__getattribute__(self, "_original")
        return getattr(original, name)
