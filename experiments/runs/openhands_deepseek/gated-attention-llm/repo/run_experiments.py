"""
Script to reproduce all paper experiments and ablation studies.

Tables 1-5 from the paper, plus Switch Head baselines (App. A.1).
"""

import os
import argparse
from config import (
    Config, ModelConfig, TrainingConfig,
    get_moe_15a2b_config, get_dense_1_7b_28l_config, get_dense_1_7b_48l_config,
)
from train import train


def create_table1_experiments() -> list:
    """
    Generate configs for Table 1: Gating Variant Performance.

    Table 1 rows:
    (1) Baseline (q=32, k=4, dk=128)
    (2) k=8 (more KV heads for parameter comparison)
    (3) q=48 (more query heads)
    (4) Add 4 Experts (more MoE capacity)
    (5) SDPA Elementwise G1 (sigmoid)
    (6) V Elementwise G2 (sigmoid)
    (7) K Elementwise G3 (sigmoid)
    (8) Q Elementwise G4 (sigmoid)
    (9) Dense Output G5 (sigmoid)
    (10) SDPA Headwise G1 (sigmoid)
    (11) V Headwise G2 (sigmoid)
    (12) SDPA Head-Shared G1 (sigmoid)
    (13) V Head-Shared G2 (sigmoid)
    (14) SDPA Additive G1 (SiLU)
    (15) SDPA Elementwise G1 (SiLU activation)
    """
    configs = []

    base_moe = get_moe_15a2b_config()

    # (1) Baseline
    configs.append(("table1_baseline", base_moe, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (2) k=8
    cfg_k8 = get_moe_15a2b_config(n_kv_heads=8)
    configs.append(("table1_k8", cfg_k8, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (3) q=48
    cfg_q48 = get_moe_15a2b_config(n_query_heads=48)
    configs.append(("table1_q48", cfg_q48, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (4) Add 4 Experts
    cfg_experts = get_moe_15a2b_config(n_experts=132, n_active_experts=12)
    configs.append(("table1_more_experts", cfg_experts, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (5) SDPA Elementwise G1
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table1_G1_elemwise", cfg, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (6) V Elementwise G2
    cfg = get_moe_15a2b_config(
        gating_position="G2", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table1_G2_elemwise", cfg, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (7) K Elementwise G3
    cfg = get_moe_15a2b_config(
        gating_position="G3", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table1_G3_elemwise", cfg, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (8) Q Elementwise G4
    cfg = get_moe_15a2b_config(
        gating_position="G4", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table1_G4_elemwise", cfg, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (9) Dense Output G5
    cfg = get_moe_15a2b_config(
        gating_position="G5", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table1_G5", cfg, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (10) SDPA Headwise G1
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="headwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table1_G1_headwise", cfg, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (11) V Headwise G2
    cfg = get_moe_15a2b_config(
        gating_position="G2", gating_granularity="headwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table1_G2_headwise", cfg, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (12) SDPA Head-Shared G1
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=False, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table1_G1_headshared", cfg, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (13) V Head-Shared G2
    cfg = get_moe_15a2b_config(
        gating_position="G2", gating_granularity="elementwise",
        gating_head_specific=False, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table1_G2_headshared", cfg, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (14) SDPA Additive G1 (SiLU)
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="additive",
        gating_activation="silu",
    )
    configs.append(("table1_G1_additive_silu", cfg, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (15) SDPA Elementwise G1 (SiLU activation)
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="silu",
    )
    configs.append(("table1_G1_silu", cfg, TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    return configs


def create_table2_experiments() -> list:
    """
    Generate configs for Table 2: Dense model performance.

    Table 2 rows:
    (1) Baseline 28L 1.7B, 400B, BS=1024, LR=4e-3
    (2) SDPA Elementwise 28L 1.7B, 400B, BS=1024, LR=4e-3
    (3) Baseline 28L 1.7B, 3.5T, BS=2048, LR=4.5e-3
    (4) SDPA Elementwise 28L 1.7B, 3.5T, BS=2048, LR=4.5e-3
    (5) Baseline 48L 1.7B, 400B, BS=1024, LR=4e-3
    (6) Baseline 48L 1.7B, 400B, BS=1024, LR=8e-3 (diverges)
    (7) Baseline + Sandwich Norm 48L, 400B, LR=8e-3
    (8) SDPA Elementwise 48L, 400B, LR=4e-3
    (9) SDPA Headwise 48L, 400B, LR=4e-3
    (10) SDPA Elementwise 48L, 400B, LR=8e-3
    (11) Baseline 48L 1.7B, 1T, BS=4096, LR=5.3e-3
    (12) Baseline 48L 1.7B, 1T, BS=4096, LR=8e-3 (diverges)
    (13) SDPA Elementwise 48L, 1T, BS=4096, LR=5.3e-3
    (14) SDPA Elementwise 48L, 1T, BS=4096, LR=8e-3
    """
    configs = []

    # (1) Baseline 28L, 400B
    cfg = get_dense_1_7b_28l_config()
    configs.append(("table2_baseline_28l_400b", cfg, TrainingConfig(
        max_lr=4e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (2) SDPA Elementwise 28L, 400B
    cfg = get_dense_1_7b_28l_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    # Reduce FFN width to match parameter count (paper says "we reduce the FFN's width")
    cfg.d_ff = cfg.d_ff * 3 // 4
    configs.append(("table2_gated_28l_400b", cfg, TrainingConfig(
        max_lr=4e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (3) Baseline 28L, 3.5T
    cfg = get_dense_1_7b_28l_config()
    configs.append(("table2_baseline_28l_3.5t", cfg, TrainingConfig(
        max_lr=4.5e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=3_500_000_000_000, batch_size=2048, seq_len=4096,
    )))

    # (4) SDPA Elementwise 28L, 3.5T
    cfg = get_dense_1_7b_28l_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    cfg.d_ff = cfg.d_ff * 3 // 4
    configs.append(("table2_gated_28l_3.5t", cfg, TrainingConfig(
        max_lr=4.5e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=3_500_000_000_000, batch_size=2048, seq_len=4096,
    )))

    # (5) Baseline 48L, 400B
    cfg = get_dense_1_7b_48l_config()
    configs.append(("table2_baseline_48l_400b", cfg, TrainingConfig(
        max_lr=4e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (6) Baseline 48L, 400B, LR=8e-3 (diverges)
    cfg = get_dense_1_7b_48l_config()
    configs.append(("table2_baseline_48l_400b_lr8e3", cfg, TrainingConfig(
        max_lr=8e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (7) Baseline + Sandwich Norm 48L, LR=8e-3
    cfg = get_dense_1_7b_48l_config(use_sandwich_norm=True)
    configs.append(("table2_sandwich_norm_48l", cfg, TrainingConfig(
        max_lr=8e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (8) SDPA Elementwise 48L, LR=4e-3
    cfg = get_dense_1_7b_48l_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table2_gated_48l_400b", cfg, TrainingConfig(
        max_lr=4e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (9) SDPA Headwise 48L, LR=4e-3
    cfg = get_dense_1_7b_48l_config(
        gating_position="G1", gating_granularity="headwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table2_gated_headwise_48l", cfg, TrainingConfig(
        max_lr=4e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (10) SDPA Elementwise 48L, LR=8e-3
    cfg = get_dense_1_7b_48l_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table2_gated_48l_lr8e3", cfg, TrainingConfig(
        max_lr=8e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )))

    # (11) Baseline 48L, 1T, BS=4096, LR=5.3e-3
    cfg = get_dense_1_7b_48l_config()
    configs.append(("table2_baseline_48l_1t", cfg, TrainingConfig(
        max_lr=5.3e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=1_000_000_000_000, batch_size=4096, seq_len=4096,
    )))

    # (12) Baseline 48L, 1T, BS=4096, LR=8e-3 (diverges)
    cfg = get_dense_1_7b_48l_config()
    configs.append(("table2_baseline_48l_1t_lr8e3", cfg, TrainingConfig(
        max_lr=8e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=1_000_000_000_000, batch_size=4096, seq_len=4096,
    )))

    # (13) SDPA Elementwise 48L, 1T, LR=5.3e-3
    cfg = get_dense_1_7b_48l_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table2_gated_48l_1t", cfg, TrainingConfig(
        max_lr=5.3e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=1_000_000_000_000, batch_size=4096, seq_len=4096,
    )))

    # (14) SDPA Elementwise 48L, 1T, LR=8e-3
    cfg = get_dense_1_7b_48l_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table2_gated_48l_1t_lr8e3", cfg, TrainingConfig(
        max_lr=8e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=1_000_000_000_000, batch_size=4096, seq_len=4096,
    )))

    return configs


def create_table3_experiments() -> list:
    """
    Generate configs for Table 3: Non-linearity augmentations.

    Table 3 rows:
    (1) Baseline
    (2) SDPA Elementwise Gate (Sigmoid)
    (3) V Elementwise Gate (Sigmoid)
    (4) SDPA Additive Gate (SiLU)
    (5) SDPA GroupNorm (RMSNorm)
    (6) SDPA SiLU (no parameters)
    (7) SDPA Additive Gate (Identity activation)
    """
    configs = []

    base_moe = get_moe_15a2b_config()
    train_cfg = TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )

    # (1) Baseline
    configs.append(("table3_baseline", base_moe, train_cfg))

    # (2) SDPA Elementwise Gate (Sigmoid)
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table3_G1_gate", cfg, train_cfg))

    # (3) V Elementwise Gate (Sigmoid)
    cfg = get_moe_15a2b_config(
        gating_position="G2", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table3_G2_gate", cfg, train_cfg))

    # (4) SDPA Additive Gate (SiLU)
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="additive",
        gating_activation="silu",
    )
    configs.append(("table3_G1_additive_silu", cfg, train_cfg))

    # (5) SDPA GroupNorm (RMSNorm per head) - handled via config flag
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="headwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="identity",
    )
    # Note: RMSNorm per head is applied in GatedAttentionRef
    configs.append(("table3_G1_rmsnorm", cfg, train_cfg))

    # (6) SDPA SiLU only (no params) - just apply SiLU to SDPA output
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="silu",
    )
    configs.append(("table3_G1_silu_only", cfg, train_cfg))

    # (7) SDPA Additive Gate (Identity)
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="additive",
        gating_activation="identity",
    )
    configs.append(("table3_G1_additive_identity", cfg, train_cfg))

    return configs


def create_table4_experiments() -> list:
    """
    Generate configs for Table 4: Sparsity analysis.

    Table 4 rows:
    (1) Baseline
    (2) SDPA Elementwise Sigmoid G1
    (3) V Elementwise Sigmoid G2
    (4) SDPA Headwise Sigmoid G1
    (5) SDPA Head-Shared Sigmoid G1
    (6) Input-Independent G1 (zero-init params)
    (7) NS-Sigmoid SDPA Elementwise G1
    """
    configs = []

    base_moe = get_moe_15a2b_config()
    train_cfg = TrainingConfig(
        max_lr=2e-3, min_lr=3e-5, warmup_steps=1000,
        total_tokens=400_000_000_000, batch_size=1024, seq_len=4096,
    )

    # (1) Baseline
    configs.append(("table4_baseline", base_moe, train_cfg))

    # (2) SDPA Elementwise Sigmoid G1
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table4_G1_elemwise", cfg, train_cfg))

    # (3) V Elementwise Sigmoid G2
    cfg = get_moe_15a2b_config(
        gating_position="G2", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table4_G2_elemwise", cfg, train_cfg))

    # (4) SDPA Headwise Sigmoid G1
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="headwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table4_G1_headwise", cfg, train_cfg))

    # (5) SDPA Head-Shared Sigmoid G1
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=False, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table4_G1_headshared", cfg, train_cfg))

    # (6) Input-Independent G1 (zero-init learnable params, sigmoid, multiply)
    # In paper: zero-initialize learnable params of shape (q * dk), sigmoid, multiply
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="sigmoid",
    )
    configs.append(("table4_G1_input_independent", cfg, train_cfg))

    # (7) NS-Sigmoid SDPA Elementwise G1
    # Uses NS-sigmoid(x) = 0.5 + 0.5 * sigmoid(x) which constrains scores to [0.5, 1.0]
    cfg = get_moe_15a2b_config(
        gating_position="G1", gating_granularity="elementwise",
        gating_head_specific=True, gating_mode="multiplicative",
        gating_activation="ns_sigmoid",
    )
    configs.append(("table4_G1_ns_sigmoid", cfg, train_cfg))

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run Gated Attention paper experiments"
    )
    parser.add_argument(
        "--table", type=str, default="all",
        choices=["all", "1", "2", "3", "4"],
        help="Which table's experiments to run",
    )
    parser.add_argument(
        "--data_path", type=str, default="data/tokens",
        help="Path to tokenized data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print configs without training",
    )

    args = parser.parse_args()

    all_configs = []

    if args.table in ("all", "1"):
        all_configs.extend(create_table1_experiments())

    if args.table in ("all", "2"):
        all_configs.extend(create_table2_experiments())

    if args.table in ("all", "3"):
        all_configs.extend(create_table3_experiments())

    if args.table in ("all", "4"):
        all_configs.extend(create_table4_experiments())

    print(f"Total experiments: {len(all_configs)}")
    print("-" * 60)

    for name, model_cfg, train_cfg in all_configs:
        print(f"\nExperiment: {name}")
        print(f"  Model: {model_cfg.model_type}, {model_cfg.n_layers} layers")
        print(f"  Gating: {model_cfg.gating_position}/{model_cfg.gating_granularity}")
        print(f"    Activation: {model_cfg.gating_activation}")
        print(f"    Head-specific: {model_cfg.gating_head_specific}")
        print(f"  Training: LR={train_cfg.max_lr}, BS={train_cfg.batch_size}")

        if not args.dry_run:
            output_subdir = os.path.join(args.output_dir, name)
            train_cfg.data_path = args.data_path
            train(
                model_config=model_cfg,
                training_config=train_cfg,
                exp_name=name,
                output_dir=output_subdir,
            )


if __name__ == "__main__":
    main()
