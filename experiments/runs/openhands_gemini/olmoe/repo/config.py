
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    # Architecture
    model_type: str = "olmoe"
    dimension: int = 2048 # Model dimension (d_model)
    activation_fn: str = "swiglu" # SwigGLU activation
    ffn_dimension: int = 1024 # FFN dimension for each expert
    vocab_size: int = 50304 # Vocabulary size
    num_attention_heads: int = 16 # Number of attention heads
    num_layers: int = 16 # Number of transformer layers
    layer_norm_type: str = "rmsnorm" # RMSNorm layer normalization
    layer_norm_epsilon: float = 1e-5 # Epsilon for layer normalization
    use_qk_norm: bool = True # Whether to use QK normalization
    pos_embedding_type: str = "rope" # Rotary Position Embedding
    rope_theta: float = 10000.0 # RoPE theta value
    attention_variant: str = "full" # Full attention variant
    use_bias: bool = False # Whether to use bias in linear layers (paper states '-')

    # Initialization
    init_method: str = "truncated_normal" # Truncated normal initialization
    init_std: float = 0.02 # Standard deviation for initialization
    init_trunc_cutoff: float = 0.06 # Cut-off for truncated normal (3 * init_std)

    # MoE Specific
    moe_layers_interval: int = 1 # Every layer is an MoE layer
    num_experts: int = 64 # Total number of experts per MoE layer
    num_activated_experts: int = 8 # Number of activated experts per token
    moe_layer_type: str = "dropless_token_choice" # Dropless MoE with token choice routing

    # Parameter counts (derived, not direct config)
    # Vocab params: 103M (for 50304 vocab size and 2048 dimension)
    # Active params: ~1.3B (base params + 8 * ffn_dimension * dimension * 2 * num_layers)
    # Total params: ~6.9B (base params + 64 * ffn_dimension * dimension * 2 * num_layers)

@dataclass
class TrainingConfig:
    # Global Training
    seed: int = 42 # Random seed
    mixed_precision: str = "bf16" # Mixed precision training (bfloat16)
    gradient_clipping: float = 1.0 # Gradient clipping value
    gradient_reduce_dtype: str = "fp32" # Gradient reduction dtype for FSDP
    optimizer_state_dtype: str = "fp32" # Optimizer state dtype for FSDP

    # Optimizer (AdamW)
    optimizer_type: str = "adamw"
    learning_rate: float = 4.0e-4 # Peak learning rate
    min_learning_rate: float = 4.0e-5 # Minimum learning rate after decay
    adam_beta1: float = 0.9 # AdamW beta1
    adam_beta2: float = 0.95 # AdamW beta2
    adam_epsilon: float = 1.0e-8 # AdamW epsilon

    # LR Schedule (Cosine Decay)
    lr_schedule: str = "cosine"
    warmup_steps: int = 2500 # Warmup steps

    # Weight Decay
    weight_decay: float = 0.1 # Weight decay value
    decay_rmsnorm_params: bool = True # Decay RMSNorm parameters
    decay_embedding_params: bool = True # Decay embedding parameters

    # Loss Specific
    use_load_balancing_loss: bool = True # Whether to use auxiliary load balancing loss
    load_balancing_loss_weight: float = 0.01 # Weight for load balancing loss
    use_router_z_loss: bool = True # Whether to use auxiliary router Z-loss
    router_z_loss_weight: float = 0.001 # Weight for router Z-loss

    # Training Duration
    pretraining_tokens_billions: float = 5000.0 # Total pretraining tokens (5T)
    annealing_tokens_billions: float = 100.0 # Annealing phase tokens
    annealing_schedule: str = "linear" # Linear decay for annealing
    annealing_min_lr: float = 0.0 # Minimum LR during annealing

    # Batching (Example values, actual will depend on hardware and FSDP)
    global_batch_size_samples: int = 1024
    global_batch_size_tokens: int = 4 * 1024 * 1024 # ~4M tokens (batch size * seq_len)
    sequence_length: int = 4096 # Sequence length

@dataclass
class DataConfig:
    # Pretraining Data
    dataset_name: str = "olmoe-mix" # Name of the dataset mix
    data_sources: List[str] = field(default_factory=lambda: [
        "dclm-baseline", "starcoder", "pes2o", "arxiv", "openwebmath",
        "algebraic-stack", "english-wikipedia-wikibooks"
    ])
    data_urls: List[str] = field(default_factory=lambda: [
        "https://hf.co/datasets/allenai/OLMoE-mix-0924", # Main OLMoE-Mix link
        # Specific component links from Table 2 and Appendix A
        "DCLM-Baseline [90]",
        "StarCoder [92, 84]",
        "peS2o [164, 163]",
        "arXiv [36]",
        "OpenWebMath [131]",
        "Algebraic Stack [11]",
        "English Wikipedia & Wikibooks [163]"
    ])
    # Filtering parameters
    ngram_filter_length: int = 32 # Removes docs with >= this many repeated n-grams
    ngram_filter_min_ngram_size: int = 1 # Min n-gram size for filter
    ngram_filter_max_ngram_size: int = 13 # Max n-gram size for filter

    # StarCoder specific filters
    starcoder_min_github_stars: int = 2
    starcoder_max_most_frequent_word_ratio: float = 0.3
    starcoder_max_top2_frequent_words_ratio: float = 0.5

    # Adaptation Data (Instruction Tuning and Preference Tuning)
    instruction_tuning_datasets: List[str] = field(default_factory=lambda: [
        "tulu-2-sft-mix", "no-robots", "codefeedback-filtered-instruction",
        "metamathqa", "advanced-daring-anteater-subset"
    ])
    preference_tuning_datasets: List[str] = field(default_factory=lambda: [
        "ultrafeedback-binarized-filtered"
    ])
    # Data URLs for adaptation (from Table 3 and Appendix A)
    sft_data_url: str = "https://hf.co/datasets/allenai/tulu-v3.1-mix-preview-4096-OLMoE"
    dpo_kto_data_url: str = "https://hf.co/datasets/allenai/ultrafeedback_binarized_cleaned"

    # Adaptation specific settings
    sft_max_seq_len: int = 4096 # Max sequence length for SFT samples
    sft_loss_aggregation: str = "token_level" # Aggregate loss at token level
    sft_epochs: int = 2
    sft_learning_rate: float = 2.0e-5
    sft_global_batch_size: int = 128
    sft_grad_accum_steps: int = 2 # Assuming 4 nodes * 8 GPUs/node = 32 devices, 128 global / 32 devices / 2 acc = 2 per device bs
    
    dpo_epochs: int = 3
    dpo_learning_rate: float = 5.0e-7
    dpo_beta: float = 0.1
    dpo_global_batch_size: int = 32
    dpo_grad_accum_steps: int = 1 # Assuming 4 nodes * 8 GPUs/node = 32 devices, 32 global / 32 devices / 1 acc = 1 per device bs

    # Checkpoint for adaptation
    adaptation_checkpoint_type: str = "post_annealing" # Use post-annealing checkpoint

@dataclass
class HardwareConfig:
    num_gpus_pretraining: int = 256 # Number of H100 GPUs for pretraining
    num_gpus_adaptation: int = 32 # Number of H100 GPUs for adaptation
    num_gpus_kto: int = 8 # Number of H100 GPUs for KTO adaptation
    pretraining_gpu_type: str = "H100"
    interconnect_type_gpus: str = "NVlink" # Interconnect within nodes
    interconnect_type_nodes: str = "InfiniBand" # Interconnect across nodes for pretraining
    interconnect_type_nodes_experiments: str = "GCP_TCPx" # Interconnect for experiments
    pretraining_duration_days: int = 10 # Approx duration

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def __post_init__(self):
        # Calculate tokens per second per GPU based on paper's reported throughput
        # This is for internal calculations/reference, not a direct config
        # MoE: 23,600 tokens/sec/GPU (Figure 4)
        # Dense: 37,500 tokens/sec/GPU (Figure 4)
        pass

# Instantiate the main config object
config = Config()
