import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple, Union

# Import Config and GatingModule, and apply_rope from utils
try:
    from config import Config
    from model.gating_module import GatingModule
    from utils import apply_rope
except ImportError:
    # Fallback for testing or if imports are structured differently
    print("Warning: Could not import Config, GatingModule, or apply_rope. Using dummy classes/functions.")

    class Config:  # Dummy Config for isolated testing
        def __init__(self):
            self.model = self  # Self-reference for model config
            self.d_model = 2048
            self.q_heads = 32
            self.kv_heads = 4
            self.head_dim = 128
            self.attn_dropout = 0.1
            self.rope_base = 10000.0

            self.gating_enabled = True
            self.gating_position = "G1"  # Default to G1 for testing
            self.gating_granularity = "elementwise"
            self.gating_head_specific = True
            self.gating_type = "multiplicative"
            self.gating_activation_fn = "sigmoid"

            self.evaluation = type('EvaluationConfig', (object,), {
                'attention_sink_analysis_enabled': True,
                'gating_score_analysis_enabled': True,
                'massive_activation_analysis_enabled': True,
            })()

    class GatingModule(nn.Module):  # Dummy GatingModule
        def __init__(self, input_dim: int, score_input_dim: int, granularity: str, head_specific: bool,
                     num_heads: int, activation_fn_name: str, gating_type: str):
            super().__init__()
            self.head_specific = head_specific
            self.num_heads = num_heads
            self.input_dim = input_dim
            self.score_input_dim = score_input_dim
            self.w_theta = nn.Linear(self.score_input_dim, self.input_dim if granularity == "elementwise" else num_heads)
            self.activation_fn = torch.sigmoid # simplified
            self.gating_type = gating_type

        def forward(self, modulated_input: torch.Tensor, score_computation_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # Simplified dummy logic for forward pass
            score_input_for_w_theta = score_computation_input
            if not self.head_specific and score_computation_input.dim() == 4:
                score_input_for_w_theta = score_computation_input.mean(dim=-2) # Dummy average

            raw_gate_scores = self.w_theta(score_input_for_w_theta)
            if raw_gate_scores.dim() == 3 and modulated_input.dim() == 4: # (B, S, N) -> (B, S, N, 1) for headwise
                if self.num_heads == modulated_input.shape[-2] and self.input_dim != modulated_input.shape[-1]: # headwise, not elementwise
                    raw_gate_scores = raw_gate_scores.unsqueeze(-1)
            
            gate_scores = self.activation_fn(raw_gate_scores)

            if self.gating_type == "multiplicative":
                return modulated_input * gate_scores, gate_scores
            else:
                return modulated_input + gate_scores, gate_scores

    def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:  # Dummy apply_rope
        return x  # No-op for dummy


class GatedAttention(nn.Module):
    """
    Multi-Head Attention (or Grouped Query Attention) layer with optional gating mechanisms.
    This module integrates QKV projections, RoPE, SDPA, and GatingModule instances at
    various positions (G1, G2, G3, G4, G5) as described in the paper.
    """

    def __init__(self, config: Config):
        """
        Initializes the GatedAttention module.

        Args:
            config: Configuration object containing model and gating hyperparameters.
        """
        super().__init__()
        self.config = config

        self.d_model: int = config.d_model
        self.q_heads: int = config.q_heads
        self.kv_heads: int = config.kv_heads
        self.head_dim: int = config.head_dim
        self.rope_base: float = config.rope_base

        if self.d_model % self.q_heads != 0:
            raise ValueError(
                f"Model dimension {self.d_model} must be divisible by "
                f"number of query heads {self.q_heads}."
            )
        if self.head_dim != self.d_model // self.q_heads:
            raise ValueError(
                f"Calculated head_dim ({self.d_model // self.q_heads}) does not match "
                f"config.head_dim ({self.head_dim}). Please ensure consistency."
            )

        # QKV projection layers
        self.q_proj: nn.Linear = nn.Linear(self.d_model, self.q_heads * self.head_dim, bias=False)
        self.k_proj: nn.Linear = nn.Linear(self.d_model, self.kv_heads * self.head_dim, bias=False)
        self.v_proj: nn.Linear = nn.Linear(self.d_model, self.kv_heads * self.head_dim, bias=False)
        self.o_proj: nn.Linear = nn.Linear(self.q_heads * self.head_dim, self.d_model, bias=False)

        self.attn_dropout_layer: nn.Dropout = nn.Dropout(config.attn_dropout)

        # Gating Module Instantiation
        self.gating_g1: Optional[GatingModule] = None
        self.gating_g2: Optional[GatingModule] = None
        self.gating_g3: Optional[GatingModule] = None
        self.gating_g4: Optional[GatingModule] = None
        self.gating_g5: Optional[GatingModule] = None

        if self.config.gating_enabled:
            def get_gating_module_num_heads(is_q_head_position: bool) -> int:
                """Helper to determine num_heads for GatingModule based on head_specific flag."""
                if self.config.gating_head_specific:
                    return self.q_heads if is_q_head_position else self.kv_heads
                else:
                    # For head-shared, the GatingModule might operate on an aggregated input (e.g., averaged across heads)
                    # or an input that doesn't have a head dimension. The GatingModule itself will be instantiated once.
                    return 1

            gating_kwargs: Dict[str, Any] = {
                "granularity": self.config.gating_granularity,
                "head_specific": self.config.gating_head_specific,
                "activation_fn_name": self.config.gating_activation_fn,
                "gating_type": self.config.gating_type,
            }

            if self.config.gating_position == "G4":  # Query Gating
                self.gating_g4 = GatingModule(
                    input_dim=self.head_dim,
                    score_input_dim=self.head_dim,
                    num_heads=get_gating_module_num_heads(True),  # Q heads
                    **gating_kwargs
                )
            elif self.config.gating_position == "G3":  # Key Gating
                self.gating_g3 = GatingModule(
                    input_dim=self.head_dim,
                    score_input_dim=self.head_dim,
                    num_heads=get_gating_module_num_heads(False),  # KV heads
                    **gating_kwargs
                )
            elif self.config.gating_position == "G2":  # Value Gating
                self.gating_g2 = GatingModule(
                    input_dim=self.head_dim,
                    score_input_dim=self.head_dim,
                    num_heads=get_gating_module_num_heads(False),  # KV heads
                    **gating_kwargs
                )
            elif self.config.gating_position == "G1":  # SDPA Output Gating
                self.gating_g1 = GatingModule(
                    input_dim=self.head_dim,
                    score_input_dim=self.head_dim,
                    num_heads=get_gating_module_num_heads(True),  # Q heads (output comes from q heads)
                    **gating_kwargs
                )
            elif self.config.gating_position == "G5":  # Dense Output Gating
                # G5 is always elementwise and implicitly head-shared as it acts on d_model output
                self.gating_g5 = GatingModule(
                    input_dim=self.d_model,
                    score_input_dim=self.d_model,
                    granularity="elementwise",  # G5 is typically elementwise
                    head_specific=False,        # G5 acts on the full d_model output
                    num_heads=1,                # Only one "group" for the entire output
                    activation_fn_name=self.config.gating_activation_fn,
                    gating_type=self.config.gating_type,
                )
            else:
                raise ValueError(f"Unknown gating_position: {self.config.gating_position}")

        # Attributes to store intermediate values for analysis
        self._last_gating_scores: Optional[torch.Tensor] = None
        self._last_attention_weights: Optional[torch.Tensor] = None

    def _apply_gating_logic(
        self,
        gating_module: GatingModule,
        modulated_input: torch.Tensor,
        score_computation_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper method to apply a GatingModule.
        Handles specific reshaping for head-shared gating if needed (e.g., averaging across heads).

        Args:
            gating_module: The GatingModule instance.
            modulated_input: The tensor (Y) to be modulated. Expected shape can be (B, S, D) or (B, S, H, D_head).
            score_computation_input: The tensor (X) used to compute the gating scores.
                                     Expected shape can be (B, S, D) or (B, S, H, D_head).

        Returns:
            A tuple containing the gated output tensor and the raw gating scores (before activation).
        """
        final_score_computation_input = score_computation_input

        # Handle head-shared gating input for G1-G4 variants specifically.
        # "we average over the query head dimension q to obtain an n x d_k score" (Sec 3.2.1, Point ii)
        # This implies: if head_specific is False and the input has a head dim, average it out.
        # This applies to inputs to GatingModules at positions G1, G2, G3, G4.
        # For G5, score_computation_input is already (B, S, d_model) and has no head dimension,
        # so this conditional block will not apply, which is correct.
        
        is_qkv_or_sdpa_output_gating = (gating_module == self.gating_g1 or
                                        gating_module == self.gating_g2 or
                                        gating_module == self.gating_g3 or
                                        gating_module == self.gating_g4)

        if not gating_module.head_specific and is_qkv_or_sdpa_output_gating:
            # If `score_computation_input` is (B, S, NumHeads, HeadDim) for Q, K, V, or SDPA output
            # We need to average over NumHeads to get (B, S, HeadDim) to be passed to a single Linear layer.
            if final_score_computation_input.dim() == 4: # (B, S, H, D_head)
                final_score_computation_input = final_score_computation_input.mean(dim=-2) # -> (B, S, D_head)
            else:
                # If it's already (B, S, D_head) (e.g., if it was already averaged or had no head dim), use as is.
                pass 
        
        # Apply the gating module
        # GatingModule returns (gated_output, raw_gating_scores)
        gated_output, raw_gating_scores = gating_module(modulated_input, final_score_computation_input)
        return gated_output, raw_gating_scores


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,  # Added for explicit RoPE positions
    ) -> torch.Tensor:
        """
        Performs the forward pass for the GatedAttention layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask: Optional mask for attention scores (batch_size, 1, seq_len, seq_len).
                            Typically a causal mask, with large negative values for masked positions.
            positions: Optional tensor of absolute token positions for RoPE. If None, it will be generated
                       as `torch.arange(seq_len)`.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, S, D = hidden_states.shape

        # Reset analysis metrics for this forward pass
        self._last_gating_scores = None
        self._last_attention_weights = None

        # 1. QKV Projections
        queries = self.q_proj(hidden_states)  # (B, S, q_heads * head_dim)
        keys = self.k_proj(hidden_states)     # (B, S, kv_heads * head_dim)
        values = self.v_proj(hidden_states)   # (B, S, kv_heads * head_dim)

        # Reshape for multi-head attention: (B, S, NumHeads, HeadDim)
        queries = queries.view(B, S, self.q_heads, self.head_dim)
        keys = keys.view(B, S, self.kv_heads, self.head_dim)
        values = values.view(B, S, self.kv_heads, self.head_dim)

        # 2. Apply Gating G4 (Query Gating)
        if self.gating_g4:
            queries, gating_scores_g4 = self._apply_gating_logic(self.gating_g4, queries, queries)
            if self.config.evaluation.gating_score_analysis_enabled:
                self._last_gating_scores = gating_scores_g4.detach().cpu()

        # 3. Apply RoPE
        if positions is None:
            # Generate positions if not provided (e.g., during training with fixed length)
            # Positions should be (S,) or (B, S) for batch-wise
            positions = torch.arange(S, dtype=torch.long, device=hidden_states.device)

        queries = apply_rope(queries, positions, self.rope_base)
        keys = apply_rope(keys, positions, self.rope_base)
        
        # 4. Apply Gating G3 (Key Gating)
        if self.gating_g3:
            keys, gating_scores_g3 = self._apply_gating_logic(self.gating_g3, keys, keys)
            if self.config.evaluation.gating_score_analysis_enabled:
                self._last_gating_scores = gating_scores_g3.detach().cpu()

        # 5. Apply Gating G2 (Value Gating)
        if self.gating_g2:
            values, gating_scores_g2 = self._apply_gating_logic(self.gating_g2, values, values)
            if self.config.evaluation.gating_score_analysis_enabled:
                self._last_gating_scores = gating_scores_g2.detach().cpu()

        # GQA: Repeat keys and values if kv_heads < q_heads
        if self.kv_heads != self.q_heads:
            # kv_heads must divide q_heads evenly
            if self.q_heads % self.kv_heads != 0:
                raise ValueError(
                    f"Number of query heads ({self.q_heads}) must be a multiple of "
                    f"number of key/value heads ({self.kv_heads}) for GQA."
                )
            num_kv_groups = self.q_heads // self.kv_heads
            # Expand keys/values: (B, S, kv_heads, head_dim) -> (B, S, kv_heads, 1, head_dim)
            # -> (B, S, kv_heads, num_kv_groups, head_dim) -> (B, S, q_heads, head_dim)
            keys = keys.unsqueeze(-2).repeat(1, 1, 1, num_kv_groups, 1).view(B, S, self.q_heads, self.head_dim)
            values = values.unsqueeze(-2).repeat(1, 1, 1, num_kv_groups, 1).view(B, S, self.q_heads, self.head_dim)

        # Transpose for SDPA: (B, num_heads, S, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 6. Scaled Dot-Product Attention
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply attention mask
        if attention_mask is not None:
            # attention_mask is typically (B, 1, S, S) or (1, 1, S, S) for causal mask.
            # Add mask (large negative values for padded/future positions)
            attn_scores = attn_scores + attention_mask

        # Apply softmax to get attention weights. Ensure type consistency.
        # Use float for softmax for numerical stability, then cast back to original type.
        attn_weights = F.softmax(attn_scores.float(), dim=-1).type_as(attn_scores)

        # Store attention weights for analysis before dropout
        if self.config.evaluation.attention_sink_analysis_enabled:
            self._last_attention_weights = attn_weights.detach().cpu()

        attn_weights = self.attn_dropout_layer(attn_weights)

        attn_output = torch.matmul(attn_weights, values)  # (B, q_heads, S, head_dim)

        # Reshape attention output back to (B, S, q_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.q_heads * self.head_dim)

        # 7. Apply Gating G1 (SDPA Output Gating)
        if self.gating_g1:
            # Reshape attn_output to (B, S, q_heads, head_dim) for G1 GatingModule
            reshaped_attn_output_for_gating = attn_output.view(B, S, self.q_heads, self.head_dim)
            
            gated_attn_output_reshaped, gating_scores_g1 = self._apply_gating_logic(
                self.gating_g1, reshaped_attn_output_for_gating, reshaped_attn_output_for_gating
            )
            # Flatten back to (B, S, q_heads * head_dim) for o_proj
            attn_output = gated_attn_output_reshaped.view(B, S, self.q_heads * self.head_dim)

            if self.config.evaluation.gating_score_analysis_enabled:
                self._last_gating_scores = gating_scores_g1.detach().cpu()

        # 8. Final Output Projection
        output = self.o_proj(attn_output)  # (B, S, d_model)

        # 9. Apply Gating G5 (Dense Output Gating)
        if self.gating_g5:
            # G5 operates on (B, S, d_model). modulated_input and score_computation_input are both `output`.
            gated_output_g5, gating_scores_g5 = self._apply_gating_logic(self.gating_g5, output, output)
            output = gated_output_g5

            if self.config.evaluation.gating_score_analysis_enabled:
                self._last_gating_scores = gating_scores_g5.detach().cpu()
        
        return output

