# olmoe/config.py

class OLMoEConfig:
    # Model parameters (inferred and directly stated from paper)
    vocab_size = 50257  # Common vocabulary size, not explicitly stated in paper's main text
    d_model = 2048      # Embedding dimension, inferred from parameter counts (similar to OLMo-1B for active params)
    num_layers = 24     # Number of Transformer layers, inferred to match total parameter count (approx 6.9B)
    num_heads = 16      # Number of attention heads (d_model / head_dim, e.g., 2048/128)
    d_ff = 1024         # FFN intermediate dimension for *each* expert (Section 4.1.2, Figure 4 legend)
    num_experts = 64    # Total number of experts per MoE layer (Table 1, Figure 5)
    num_experts_per_token = 8 # Number of activated experts per token (Table 1, Figure 5)
    dropout = 0.1       # Standard dropout rate, not explicitly stated but common

    # Training parameters (placeholders for now)
    learning_rate = 1e-4
    batch_size = 8
    num_epochs = 1

    # Loss weights (Section 2, 4.1.6, 4.1.7)
    load_balancing_loss_weight = 0.01 # alpha in paper
    router_z_loss_weight = 0.001      # beta in paper

    # Data paths (placeholders)
    train_data_path = "data/olmoe_mix_train.jsonl"
    eval_data_path = "data/olmoe_mix_eval.jsonl"

    # Output paths (placeholders)
    output_dir = "olmoe_checkpoints"

    # Other configurations
    seed = 42
