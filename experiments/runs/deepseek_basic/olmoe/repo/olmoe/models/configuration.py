"""Configuration for OLMoE-1B-7B model.

Contains the exact hyperparameters from Table 10 in the paper.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OLMoEConfig:
    """Configuration for OLMoE-1B-7B model.

    These values match Table 10 of the paper.
    """

    # Architecture
    d_model: int = 2048  # Hidden dimension
    n_layers: int = 16  # Number of transformer layers
    n_heads: int = 16  # Number of attention heads
    vocab_size: int = 50304  # Vocabulary size (GPT-NeoX tokenizer)
    max_seq_len: int = 4096  # Maximum sequence length

    # MoE settings
    moe_num_experts: int = 64  # Total experts per MoE layer
    moe_num_activated: int = 8  # Activated experts per token (k=8)
    moe_ffn_dim: int = 1024  # FFN dimension per expert
    moe_layers: str = "every"  # MoE in every layer (all 16 layers)
    moe_type: str = "dMoE"  # Dropless MoE

    # Shared expert (not used in OLMoE-1B-7B)
    use_shared_expert: bool = False

    # Normalization
    layer_norm_type: str = "rmsnorm"  # RMSNorm (not non-parametric)
    layer_norm_eps: float = 1e-5
    use_qk_norm: bool = True  # QK-Norm for stability

    # Position embedding
    pos_emb: str = "rope"  # Rotary Position Embedding
    rope_theta: float = 10000.0

    # Attention
    attention_variant: str = "full"  # Full attention (not MoA, MQA, GQA)
    use_bias: bool = False  # No biases

    # Weight tying
    weight_tying: bool = False

    # Initialization
    init_dist: str = "truncated_normal"  # Truncated normal init
    init_std: float = 0.02
    init_trunc: float = 3.0  # Truncate at ±3×std (i.e., ±0.06)

    # Auxiliary loss weights
    load_balancing_loss_weight: float = 0.01  # α
    router_z_loss_weight: float = 0.001  # β

    # Training hyperparameters (Table 10)
    batch_size_samples: int = 1024  # Per GPU batch = 1024 samples
    batch_size_tokens: int = 4194304  # ~4M tokens per batch
    warmup_steps: int = 2500
    peak_lr: float = 4.0e-4
    min_lr: float = 4.0e-5
    optimizer: str = "adamw"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adamw_eps: float = 1.0e-8
    lr_schedule: str = "cosine"
    gradient_clipping: float = 1.0  # Global gradient clipping
    gradient_reduce_dtype: str = "fp32"
    optimizer_state_dtype: str = "fp32"

    # Pretraining
    pretraining_tokens: int = 5_133_000_000_000  # 5.133T tokens
    annealing_tokens: int = 100_000_000_000  # 100B annealing tokens
    annealing_schedule: str = "linear"
    annealing_min_lr: float = 0.0

    # Adaptation
    sft_learning_rate: float = 2.0e-5
    sft_epochs: int = 2
    sft_batch_size: int = 128
    sft_max_seq_len: int = 4096
    dpo_learning_rate: float = 5.0e-7
    dpo_epochs: int = 3
    dpo_batch_size: int = 32
    dpo_beta: float = 0.1

    # Dropout
    dropout: float = 0.0

    # Activation
    activation: str = "swiglu"

    @property
    def num_active_params(self) -> int:
        """Compute approximate number of active parameters."""
        # Vocabulary params
        vocab_params = self.vocab_size * self.d_model

        # Per layer: attention + MoE (with k activated experts)
        attn_params = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        if self.use_qk_norm:
            attn_params += 4 * self.d_model  # QK norms (2 per Q and K)

        # RMSNorm params per layer
        norm_params = 2 * self.d_model  # Pre-attn norm + post-attn norm + pre-MoE norm

        # Router params
        router_params = self.d_model * self.moe_num_experts

        # Expert params (k activated experts, each with 3 weight matrices for SwiGLU)
        expert_params_per = 3 * self.d_model * self.moe_ffn_dim
        active_expert_params = self.moe_num_activated * expert_params_per

        per_layer = attn_params + norm_params + router_params + active_expert_params
        total_active = vocab_params + self.n_layers * per_layer

        return total_active

    @property
    def num_total_params(self) -> int:
        """Compute approximate number of total parameters."""
        vocab_params = self.vocab_size * self.d_model

        attn_params = 4 * self.d_model * self.d_model
        if self.use_qk_norm:
            attn_params += 4 * self.d_model

        norm_params = 2 * self.d_model
        router_params = self.d_model * self.moe_num_experts

        # All experts (not just activated ones)
        expert_params_per = 3 * self.d_model * self.moe_ffn_dim
        all_expert_params = self.moe_num_experts * expert_params_per

        per_layer = attn_params + norm_params + router_params + all_expert_params
        total = vocab_params + self.n_layers * per_layer

        return total
