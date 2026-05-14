from dataclasses import dataclass, field
from typing import Optional, Literal


# ---------------------------------------------------------------------------
# Gating configuration
# ---------------------------------------------------------------------------

@dataclass
class GatingConfig:
    """Configuration for the gating mechanism (Eq. 5 in the paper)."""

    # Where to apply the gate: G1=sdpa_output, G2=value, G3=key, G4=query, G5=dense_output
    position: Literal["none", "G1", "G2", "G3", "G4", "G5"] = "none"

    # Granularity of the gating score vector
    granularity: Literal["elementwise", "headwise"] = "elementwise"

    # Whether each head gets its own gate parameters (head-specific) or they are shared
    head_specific: bool = True

    # Multiplicative (Y * sigma(X*W)) or additive (Y + sigma(X*W))
    mode: Literal["multiplicative", "additive"] = "multiplicative"

    # Activation function applied to the gate logits
    activation: Literal["sigmoid", "silu", "identity", "ns_sigmoid", "rmsnorm"] = "sigmoid"


# ---------------------------------------------------------------------------
# Attention configuration
# ---------------------------------------------------------------------------

@dataclass
class AttentionConfig:
    d_model: int = 2048
    num_heads: int = 32          # query heads (q)
    num_kv_heads: int = 4        # key/value heads (k) — GQA
    head_dim: int = 128          # d_k
    max_seq_len: int = 4096
    rope_base: float = 10000.0
    dropout: float = 0.0
    gating: GatingConfig = field(default_factory=GatingConfig)


# ---------------------------------------------------------------------------
# FFN configuration
# ---------------------------------------------------------------------------

@dataclass
class FFNConfig:
    d_model: int = 2048
    d_ffn: int = 8192            # intermediate dimension (SwiGLU uses 2/3 * 4 * d_model)
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# MoE configuration  (DeepSeekMoE-style fine-grained experts, top-k softmax routing)
# ---------------------------------------------------------------------------

@dataclass
class MoEConfig:
    d_model: int = 2048
    num_experts: int = 128       # total experts
    num_experts_per_tok: int = 8 # top-k activated experts
    expert_d_ffn: int = 1024     # per-expert intermediate dim (fine-grained)
    dropout: float = 0.0
    # Z-loss coefficient (Zoph et al., 2022)
    z_loss_coeff: float = 1e-3
    # Load-balancing loss coefficient (global-batch LBL, Qiu et al., 2025)
    lb_loss_coeff: float = 1e-2
    # Shared expert (optional, set to 0 to disable)
    num_shared_experts: int = 0


# ---------------------------------------------------------------------------
# Transformer block configuration
# ---------------------------------------------------------------------------

@dataclass
class TransformerBlockConfig:
    d_model: int = 2048
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    use_moe: bool = False
    ffn: FFNConfig = field(default_factory=FFNConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    # Sandwich norm: apply RMSNorm to attn/ffn outputs before residual add
    sandwich_norm: bool = False
    norm_eps: float = 1e-5


# ---------------------------------------------------------------------------
# Full model configurations
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    num_layers: int = 28
    d_model: int = 2048
    vocab_size: int = 152064     # Qwen2 tokenizer vocab size
    max_seq_len: int = 4096
    norm_eps: float = 1e-5
    tie_embeddings: bool = False
    block: TransformerBlockConfig = field(default_factory=TransformerBlockConfig)


def get_dense_1_7b_config(
    num_layers: int = 28,
    gating: Optional[GatingConfig] = None,
    sandwich_norm: bool = False,
) -> ModelConfig:
    """1.7B dense model as described in Sec. 3.2.2."""
    if gating is None:
        gating = GatingConfig(position="none")

    d_model = 2048
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    # SwiGLU FFN: d_ffn ≈ 8/3 * d_model, rounded to multiple of 256
    d_ffn = 5504  # ~2.69 * d_model, standard for 1.7B

    attn_cfg = AttentionConfig(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=4096,
        rope_base=10000.0,
        gating=gating,
    )
    ffn_cfg = FFNConfig(d_model=d_model, d_ffn=d_ffn)
    block_cfg = TransformerBlockConfig(
        d_model=d_model,
        attention=attn_cfg,
        use_moe=False,
        ffn=ffn_cfg,
        sandwich_norm=sandwich_norm,
    )
    return ModelConfig(
        num_layers=num_layers,
        d_model=d_model,
        vocab_size=152064,
        max_seq_len=4096,
        block=block_cfg,
    )


def get_dense_1_7b_deep_config(
    gating: Optional[GatingConfig] = None,
    sandwich_norm: bool = False,
) -> ModelConfig:
    """48-layer 1.7B dense model (deep variant, Sec. 3.2.2 rows 5-10)."""
    cfg = get_dense_1_7b_config(num_layers=48, gating=gating, sandwich_norm=sandwich_norm)
    # Reduce d_ffn to keep parameter count at ~1.7B with 48 layers
    cfg.block.ffn.d_ffn = 3072
    return cfg


def get_moe_15a2b_config(
    gating: Optional[GatingConfig] = None,
) -> ModelConfig:
    """15A2B MoE model: 15B total params, 2.54B activated (Sec. 3.2.1).

    Architecture:
      - 128 total experts, top-8 softmax routing
      - Fine-grained experts (DeepSeekMoE)
      - GQA: q=32, k=4, dk=128
      - Global-batch LBL + Z-loss

    d_model=2048 is confirmed by Table 1 added-parameter counts:
      - SDPA Elementwise G1: gate proj d_model → q*dk = 2048 → 4096,
        24 layers × 8.4M = 201M ✓
      - Dense Output G5: gate proj d_model → d_model = 2048 → 2048,
        24 layers × 4.2M = 100M ✓
    """
    if gating is None:
        gating = GatingConfig(position="none")

    d_model = 2048
    num_heads = 32
    num_kv_heads = 4
    head_dim = 128

    attn_cfg = AttentionConfig(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=4096,
        rope_base=10000.0,
        gating=gating,
    )
    # Fine-grained expert FFN: 128 experts, each with small d_ffn.
    # expert_d_ffn=750 gives ~15B total / ~2.54B activated:
    #   expert params/layer = 128 * 3 * 2048 * 750 ≈ 590M
    #   24 layers × 590M = 14.2B expert params
    #   non-expert ≈ 0.8B  →  total ≈ 15B
    #   activated/layer = 8 * 3 * 2048 * 750 ≈ 36.9M
    #   24 layers × 36.9M + 0.8B ≈ 1.7B activated
    # (exact match to 2.54B depends on shared-expert and embedding details)
    moe_cfg = MoEConfig(
        d_model=d_model,
        num_experts=128,
        num_experts_per_tok=8,
        expert_d_ffn=750,
        z_loss_coeff=1e-3,
        lb_loss_coeff=1e-2,
        num_shared_experts=0,
    )
    block_cfg = TransformerBlockConfig(
        d_model=d_model,
        attention=attn_cfg,
        use_moe=True,
        moe=moe_cfg,
    )
    return ModelConfig(
        num_layers=24,
        d_model=d_model,
        vocab_size=152064,
        max_seq_len=4096,
        block=block_cfg,
    )


# ---------------------------------------------------------------------------
# Training configurations
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Optimizer
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eps: float = 1e-8

    # LR schedule
    max_lr: float = 2e-3
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    total_steps: int = 100_000
    lr_schedule: str = "cosine"

    # Batch
    batch_size: int = 1024       # global batch size (sequences)
    seq_len: int = 4096

    # Precision
    dtype: str = "bfloat16"

    # Logging / checkpointing
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "checkpoints"

    # Distributed
    gradient_accumulation_steps: int = 1


def get_moe_training_config() -> TrainingConfig:
    """Training config for 15A2B MoE models (Sec. 3.2.1)."""
    return TrainingConfig(
        max_lr=2e-3,
        min_lr=3e-5,
        warmup_steps=1000,
        total_steps=100_000,
        batch_size=1024,
        seq_len=4096,
    )


def get_dense_400b_training_config() -> TrainingConfig:
    """Training config for 1.7B dense models on 400B tokens (Sec. 3.2.2)."""
    return TrainingConfig(
        max_lr=4e-3,
        min_lr=4e-5,
        warmup_steps=1000,
        total_steps=97_656,   # 400B tokens / (1024 seqs * 4096 tokens)
        batch_size=1024,
        seq_len=4096,
    )


def get_dense_3_5t_training_config() -> TrainingConfig:
    """Training config for 1.7B dense models on 3.5T tokens (Sec. 3.2.2)."""
    return TrainingConfig(
        max_lr=4.5e-3,
        min_lr=4.5e-5,
        warmup_steps=2000,
        total_steps=854_492,  # 3.5T tokens / (2048 seqs * 4096 tokens)
        batch_size=2048,
        seq_len=4096,
    )


def get_dense_1t_training_config() -> TrainingConfig:
    """Training config for 1.7B 48-layer dense models on 1T tokens (Sec. 3.2.2)."""
    return TrainingConfig(
        max_lr=5.3e-3,
        min_lr=5.3e-5,
        warmup_steps=2000,
        total_steps=61_035,   # 1T tokens / (4096 seqs * 4096 tokens)
        batch_size=4096,
        seq_len=4096,
    )


# ---------------------------------------------------------------------------
# Gating variant presets (Table 1 in the paper)
# ---------------------------------------------------------------------------

GATING_VARIANTS = {
    # Baseline
    "baseline": GatingConfig(position="none"),

    # --- Gating position variants ---
    # Row 5: SDPA Elementwise G1 (best)
    "G1_elementwise": GatingConfig(
        position="G1", granularity="elementwise", head_specific=True,
        mode="multiplicative", activation="sigmoid",
    ),
    # Row 6: Value Elementwise G2
    "G2_elementwise": GatingConfig(
        position="G2", granularity="elementwise", head_specific=True,
        mode="multiplicative", activation="sigmoid",
    ),
    # Row 7: Key Elementwise G3
    "G3_elementwise": GatingConfig(
        position="G3", granularity="elementwise", head_specific=True,
        mode="multiplicative", activation="sigmoid",
    ),
    # Row 8: Query Elementwise G4
    "G4_elementwise": GatingConfig(
        position="G4", granularity="elementwise", head_specific=True,
        mode="multiplicative", activation="sigmoid",
    ),
    # Row 9: Dense Output G5
    "G5_elementwise": GatingConfig(
        position="G5", granularity="elementwise", head_specific=False,
        mode="multiplicative", activation="sigmoid",
    ),

    # --- Granularity variants ---
    # Row 10: SDPA Headwise G1
    "G1_headwise": GatingConfig(
        position="G1", granularity="headwise", head_specific=True,
        mode="multiplicative", activation="sigmoid",
    ),
    # Row 11: Value Headwise G2
    "G2_headwise": GatingConfig(
        position="G2", granularity="headwise", head_specific=True,
        mode="multiplicative", activation="sigmoid",
    ),

    # --- Head-shared variants ---
    # Row 12: SDPA Head-Shared G1
    "G1_head_shared": GatingConfig(
        position="G1", granularity="elementwise", head_specific=False,
        mode="multiplicative", activation="sigmoid",
    ),
    # Row 13: Value Head-Shared G2
    "G2_head_shared": GatingConfig(
        position="G2", granularity="elementwise", head_specific=False,
        mode="multiplicative", activation="sigmoid",
    ),

    # --- Additive variant ---
    # Row 14: SDPA Additive G1 (SiLU)
    "G1_additive_silu": GatingConfig(
        position="G1", granularity="elementwise", head_specific=True,
        mode="additive", activation="silu",
    ),

    # --- Activation variants ---
    # Row 15: SDPA Elementwise G1 with SiLU
    "G1_elementwise_silu": GatingConfig(
        position="G1", granularity="elementwise", head_specific=True,
        mode="multiplicative", activation="silu",
    ),

    # --- Non-linearity analysis variants (Table 3) ---
    # Row 5 (Table 3): SDPA GroupNorm (RMSNorm at G1)
    "G1_rmsnorm": GatingConfig(
        position="G1", granularity="elementwise", head_specific=True,
        mode="multiplicative", activation="rmsnorm",
    ),
    # Row 6 (Table 3): SDPA SiLU only (no gate params)
    "G1_silu_only": GatingConfig(
        position="G1", granularity="elementwise", head_specific=True,
        mode="multiplicative", activation="silu",
    ),
    # Row 7 (Table 3): SDPA Additive Gate with Identity
    "G1_additive_identity": GatingConfig(
        position="G1", granularity="elementwise", head_specific=True,
        mode="additive", activation="identity",
    ),

    # --- Sparsity analysis variants (Table 4) ---
    # NS-sigmoid: constrains scores to [0.5, 1.0]
    "G1_ns_sigmoid": GatingConfig(
        position="G1", granularity="elementwise", head_specific=True,
        mode="multiplicative", activation="ns_sigmoid",
    ),
}
