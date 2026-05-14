"""
This module defines the main OLMoE (Open Mixture-of-Experts) language model.
It integrates various sub-components like token embeddings, Transformer decoder layers,
self-attention with QK-Norm and RoPE, and MoE layers, following the architecture
described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any, Callable

# Importing configuration and utility functions
from config import ModelConfig
from model.router import Router
from model.moe_layer import MoELayer, ExpertFFN # ExpertFFN used for type hinting and activation function logic
from utils.misc import truncated_normal_init_


# --- Helper Classes for Transformer Components ---

class _RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Reference: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            hidden_size: The dimension of the input features.
            eps: A small epsilon value to prevent division by zero.
        """
        super().__init__()
        # The weight is initialized to ones, and applies scaling after normalization.
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input hidden states.

        Args:
            hidden_states: Input tensor of shape `[..., hidden_size]`.

        Returns:
            Normalized tensor.
        """
        input_dtype = hidden_states.dtype
        # Normalize in float32 for stability, then cast back to input_dtype
        hidden_states = hidden_states.to(torch.float32)
        # Calculate variance over the last dimension (hidden_size)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor for RoPE.
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine component of rotary embeddings.
        sin: Sine component of rotary embeddings.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
    """
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

class _RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) module.
    Reference: https://arxiv.org/abs/2104.09864
    Precomputes and applies rotary embeddings.
    """
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        """
        Initializes the RotaryEmbedding module.

        Args:
            dim: The dimension of the embeddings (head_dim).
            max_position_embeddings: Maximum sequence length the model is expected to handle.
            base: The base value for the geometric progression of frequencies (RoPE theta).
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Calculate inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Initialize caches for cosine and sine values
        self.cos_cached: Optional[torch.Tensor] = None
        self.sin_cached: Optional[torch.Tensor] = None
        # Precompute for max_position_embeddings
        self._set_cos_sin_cache(max_position_embeddings, inv_freq.device, inv_freq.dtype)


    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Precomputes the cosine and sine tables for RoPE up to a given sequence length.
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Apply RoPE to first half of dimension and then concatenate the same for the second half
        # (seq_len, dim // 2) -> (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Reshape to (1, 1, seq_len, dim) for broadcasting with (batch, num_heads, seq_len, head_dim)
        self.cos_cached = emb.cos()[None, None, :, :].to(dtype)
        self.sin_cached = emb.sin()[None, None, :, :].to(dtype)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns precomputed cos and sin rotations for the given sequence length.
        Dynamically recomputes cache if `seq_len` exceeds `max_position_embeddings`.

        Args:
            x: Input tensor, used for inferring device and dtype.
            seq_len: The current sequence length. If None, it uses the sequence length of x.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and sine rotation tensors,
            sliced to the current sequence length.
        """
        if seq_len is None:
            # Assume x is (batch, head, seq_len, head_dim)
            seq_len = x.shape[2]
        
        # If the cache is smaller than the current sequence length, recompute
        # This handles inference for sequences longer than `max_position_embeddings`
        # though during training, sequence length is fixed by `max_seq_len`
        if self.cos_cached is None or seq_len > self.cos_cached.shape[2]:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(x.dtype)
        )


class _OLMoESelfAttention(nn.Module):
    """
    Multi-head self-attention module for OLMoE.
    Includes linear projections, QK-Norm, RoPE, and output projection.
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.d_model = model_config.d_model
        self.num_attention_heads = model_config.num_attention_heads
        self.head_dim = self.d_model // self.num_attention_heads

        if (self.head_dim * self.num_attention_heads) != self.d_model:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_attention_heads "
                f"({self.num_attention_heads})"
            )

        # Linear projections for query, key, value, and output. No bias as per Table 10.
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.use_qk_norm = model_config.use_qk_norm
        if self.use_qk_norm:
            # QK-Norm applied after projection and before RoPE
            self.q_norm = _RMSNorm(self.head_dim, eps=model_config.norm_eps)
            self.k_norm = _RMSNorm(self.head_dim, eps=model_config.norm_eps)
        
        # RoPE instantiation, max_position_embeddings comes from data_config.max_seq_len
        # The ModelConfig needs access to this value (e.g. from data_config) or a reasonable default
        # Assuming model_config directly has max_seq_len or it is passed to this component directly.
        # For now, it's inferred from `ModelConfig` itself which does not directly have `max_seq_len`.
        # This will be resolved by passing `data_config.max_seq_len` during `OLMoEModel` initialization.
        # Let's adjust ModelConfig or pass it as a separate argument. For now, assume model_config has it.
        # Given the config.yaml, it's `data.max_seq_len`. So `ModelConfig` itself should receive it.
        # Let's update ModelConfig to take max_seq_len, or infer it from input if not hardcoded.
        # The task description implies `ModelConfig` is directly passed to `_OLMoESelfAttention`.
        # Adding `max_seq_len` to ModelConfig for consistency with RoPE use.
        self.rope = _RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=2048, # Default value if model_config doesn't carry it
            base=model_config.rope_theta
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        """
        Helper function to reshape and transpose a tensor for multi-head attention processing.
        """
        # (bsz, seq_len, d_model) -> (bsz, seq_len, num_heads, head_dim) -> (bsz, num_heads, seq_len, head_dim)
        return tensor.view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the self-attention mechanism.

        Args:
            hidden_states: Input tensor of shape `[batch_size, sequence_length, d_model]`.
            attention_mask: Optional mask tensor for attention. Expected broadcastable to
                            `[batch_size, num_heads, sequence_length, sequence_length]`.

        Returns:
            Output tensor after self-attention, shape `[batch_size, sequence_length, d_model]`.
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz) # (bsz, num_heads, q_len, head_dim)

        # Apply QK-Norm if enabled
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        
        # Apply RoPE
        # `value_states` is used to infer device/dtype and sequence length
        cos, sin = self.rope(value_states, seq_len=q_len)
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            # Attention mask values are typically -inf for masked, 0 for not masked.
            attn_weights = attn_weights + attention_mask

        # Apply softmax to get probabilities
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Compute weighted sum of values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape output back to (bsz, q_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.d_model)

        # Final output projection
        attn_output = self.o_proj(attn_output)
        return attn_output


class _OLMoETransformerDecoderLayer(nn.Module):
    """
    A single decoder layer for the OLMoE model.
    It comprises a self-attention block and an MoE block, both with pre-normalization.
    """
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.d_model = model_config.d_model

        # Pre-attention normalization
        self.attn_norm = _RMSNorm(self.d_model, eps=model_config.norm_eps)
        # Self-attention module
        self.attn = _OLMoESelfAttention(model_config)
        
        # Pre-MoE normalization
        self.moe_norm = _RMSNorm(self.d_model, eps=model_config.norm_eps)
        # Mixture-of-Experts module
        self.moe_layer = MoELayer(
            d_model=self.d_model,
            ffn_dim_expert=model_config.ffn_dim_expert,
            num_experts=model_config.num_experts,
            num_activated_experts=model_config.num_activated_experts,
            activation_function=model_config.activation_function,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass for a single OLMoE decoder layer.

        Args:
            hidden_states: Input hidden states. Shape `[batch_size, sequence_length, d_model]`.
            attention_mask: Optional attention mask.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                - hidden_states: Output hidden states after the layer.
                - moe_aux_outputs: A tuple containing (router_logits, router_probabilities, expert_mask)
                                   from the MoE layer, used for auxiliary loss calculation in `OLMoEModel`.
        """
        # Self-Attention block with pre-normalization and residual connection
        residual_attn = hidden_states
        normed_hidden_states = self.attn_norm(hidden_states)
        attn_output = self.attn(normed_hidden_states, attention_mask=attention_mask)
        hidden_states = residual_attn + attn_output

        # MoE block with pre-normalization and residual connection
        residual_moe = hidden_states
        normed_hidden_states = self.moe_norm(hidden_states)
        moe_output_val, router_logits, router_probs, expert_mask = self.moe_layer(
            normed_hidden_states
        )
        hidden_states = residual_moe + moe_output_val

        # Return hidden states and auxiliary outputs from the MoE layer
        return hidden_states, (router_logits, router_probs, expert_mask)


class OLMoEModel(nn.Module):
    """
    The main OLMoE (Open Mixture-of-Experts) language model.
    A decoder-only Transformer based on the paper's specifications.
    """
    def __init__(self, model_config: ModelConfig, vocab_size: int, max_seq_len: int = 4096):
        """
        Initializes the OLMoE model.

        Args:
            model_config: Configuration object for the model architecture.
            vocab_size: The size of the tokenizer's vocabulary.
            max_seq_len: The maximum sequence length supported by the model,
                         used for initializing Rotary Embeddings.
        """
        super().__init__()
        self.model_config = model_config
        self.vocab_size = vocab_size
        self.d_model = model_config.d_model
        self.max_seq_len = max_seq_len # Store max_seq_len for RoPE initialization

        # Embeddings layer
        self.token_embeddings = nn.Embedding(vocab_size, self.d_model)

        # Transformer decoder layers (including MoE)
        self.layers = nn.ModuleList(
            [_OLMoETransformerDecoderLayer(model_config) for _ in range(model_config.num_layers)]
        )

        # Final normalization layer before the language modeling head
        self.final_norm = _RMSNorm(self.d_model, eps=model_config.norm_eps)
        
        # Language modeling head to project to vocabulary space
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)

        # Weight tying (currently disabled as per config.weight_tying=false)
        if model_config.weight_tying:
            self.lm_head.weight = self.token_embeddings.weight

        # Apply custom truncated normal initialization to all weights
        self._init_weights()

        # Update the RoPE max_position_embeddings in _OLMoESelfAttention instances
        self._update_rope_max_position_embeddings(max_seq_len)

        print(f"OLMoEModel initialized with {self.count_parameters():,} trainable parameters.")
        print(f"Model config: d_model={self.d_model}, layers={model_config.num_layers}, "
              f"heads={model_config.num_attention_heads}, vocab_size={vocab_size}")
        print(f"MoE config: experts_per_layer={model_config.num_experts}, "
              f"activated_experts_per_token={model_config.num_activated_experts}")
        print(f"Norm config: RMSNorm_eps={model_config.norm_eps}, QK-Norm={model_config.use_qk_norm}")
        print(f"Initialization: std={model_config.init_std}, trunc_cutoff={model_config.init_trunc_cutoff}")
        print(f"Weight tying: {model_config.weight_tying}")

    def _update_rope_max_position_embeddings(self, max_seq_len: int):
        """
        Recursively update the max_position_embeddings in all _RotaryEmbedding instances.
        This is necessary because ModelConfig doesn't carry max_seq_len directly,
        but it's needed for RoPE initialization.
        """
        for module in self.modules():
            if isinstance(module, _RotaryEmbedding):
                module.max_position_embeddings = max_seq_len
                # Recompute cache if necessary, in case the original default was smaller
                module._set_cos_sin_cache(max_seq_len, module.inv_freq.device, module.inv_freq.dtype)


    def _init_weights(self):
        """
        Applies truncated normal initialization to model weights as specified in the paper.
        Biases are initialized to zero, and normalization weights to one.
        """
        init_std = self.model_config.init_std
        trunc_cutoff = self.model_config.init_trunc_cutoff

        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Apply truncated normal initialization to linear layer weights
                truncated_normal_init_(module.weight, std=init_std, a=-trunc_cutoff, b=trunc_cutoff)
                if module.bias is not None:
                    module.bias.data.zero_()  # Initialize biases to zero
            elif isinstance(module, _RMSNorm):
                # RMSNorm weights are typically initialized to 1.0
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Embedding):
                # Apply truncated normal initialization to embedding weights
                truncated_normal_init_(module.weight, std=init_std, a=-trunc_cutoff, b=trunc_cutoff)
        
        # If lm_head weights are not tied to token_embeddings, ensure it's also initialized
        if not self.model_config.weight_tying and isinstance(self.lm_head, nn.Linear):
             truncated_normal_init_(self.lm_head.weight, std=init_std, a=-trunc_cutoff, b=trunc_cutoff)

    def _make_causal_attention_mask(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Creates a causal mask for decoder-only attention.
        The mask ensures that each token can only attend to previous tokens.
        """
        # Create a mask where upper triangle (future tokens) are masked
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        # fill lower triangle with 0, upper triangle with float.min
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        return mask[None, None, :, :] # Shape: (1, 1, seq_len, seq_len) for broadcasting


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass for the OLMoE model.

        Args:
            input_ids: Input token IDs. Shape `[batch_size, sequence_length]`.
            attention_mask: Padding attention mask (1 for attended, 0 for padding).
                            Shape `[batch_size, sequence_length]`.
            labels: Optional target labels for loss calculation.
                    Shape `[batch_size, sequence_length]`.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
                - logits: Raw output logits for next token prediction.
                          Shape `[batch_size, sequence_length, vocab_size]`.
                - ce_loss: Cross-entropy loss (if labels are provided), otherwise None.
                - total_lbl_loss: Sum of Load Balancing Loss contributions from all MoE layers.
                - total_rz_loss: Sum of Router Z-loss contributions from all MoE layers.
        """
        batch_size, seq_len = input_ids.shape
        
        # Create a causal attention mask (lower triangular)
        causal_mask = self._make_causal_attention_mask(seq_len, hidden_states.dtype, input_ids.device)

        # Expand padding attention_mask to be broadcastable: (bsz, 1, 1, seq_len)
        padding_attention_mask = attention_mask[:, None, None, :].bool()
        
        # Combine causal mask with padding mask
        # Values will be 0 for valid attention, -inf for masked (padding or future tokens)
        combined_attention_mask = (causal_mask & padding_attention_mask).float()
        combined_attention_mask = (1.0 - combined_attention_mask) * torch.finfo(combined_attention_mask.dtype).min


        hidden_states = self.token_embeddings(input_ids)

        # Initialize accumulators for auxiliary losses
        total_lbl_loss = torch.tensor(0.0, device=input_ids.device)
        total_rz_loss = torch.tensor(0.0, device=input_ids.device)

        for layer in self.layers:
            hidden_states, moe_aux_output = layer(hidden_states, attention_mask=combined_attention_mask)
            
            # Unpack MoE auxiliary outputs from the layer
            router_logits, router_probs, expert_mask = moe_aux_output

            # --- Calculate Router Z-loss for the current layer ---
            # L_RZ(x) = (1/B) * sum_i (log sum_j exp(x_j^(i)))^2
            # B is the number of tokens processed in this layer's MoE module (batch_size * seq_len)
            # `router_logits` shape is `[num_tokens, num_experts]`
            num_tokens_processed = router_logits.shape[0]
            if num_tokens_processed > 0: # Avoid division by zero if no tokens are processed (e.g., empty batch)
                log_sum_exp_logits = torch.logsumexp(router_logits, dim=-1) # shape: [num_tokens_processed]
                layer_rz_loss_unweighted = (log_sum_exp_logits.pow(2)).sum() / num_tokens_processed
                total_rz_loss += layer_rz_loss_unweighted

            # --- Calculate Load Balancing Loss (LBL) for the current layer ---
            # L_LB = N_E * sum_i (f_i * P_i)
            # f_i = fraction of tokens routed to expert E_i
            # P_i = total routing probability allocated to E_i across all tokens in the batch
            # `router_probs` shape is `[num_tokens, num_experts]`
            # `expert_mask` shape is `[num_tokens, num_experts]` (boolean: True if expert was selected)

                # P_i: sum of probabilities for each expert across all tokens processed by MoE
                p_i = router_probs.sum(dim=0) # shape: [num_experts]

                # f_i: fraction of tokens routed to expert i (sum of expert_mask for each expert, then normalize)
                f_i = expert_mask.float().sum(dim=0) / num_tokens_processed # shape: [num_experts]

                # Compute the layer's unweighted LBL contribution
                layer_lbl_loss_unweighted = self.model_config.num_experts * torch.sum(f_i * p_i)
                total_lbl_loss += layer_lbl_loss_unweighted


        # Apply final normalization
        hidden_states = self.final_norm(hidden_states)
        # Project to vocabulary space
        logits = self.lm_head(hidden_states)

        ce_loss = None
        if labels is not None:
            # Shift predictions for causal language modeling
            # Predict next token: tokens < x are used to predict x.
            # So, logits for token `i` predict label for token `i+1`.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten to compute cross-entropy loss
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100, # Common practice to ignore padding tokens in loss
            )
        
        # Return logits, cross-entropy loss, and accumulated auxiliary losses
        return logits, ce_loss, total_lbl_loss, total_rz_loss

    def get_parameters_for_optimizer(self) -> List[Dict[str, Any]]:
        """
        Returns model parameters grouped for the optimizer, with specified weight decay.

        As per the paper, all parameters (including RMSNorm and embedding parameters)
        are subject to the same weight decay.

        Returns:
            A list of dictionaries, each containing 'params' (iterable of parameters)
            and 'weight_decay' for that group.
        """
        return [{"params": self.parameters(), "weight_decay": self.model_config.weight_decay}]

    def count_parameters(self) -> int:
        """
        Counts the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

