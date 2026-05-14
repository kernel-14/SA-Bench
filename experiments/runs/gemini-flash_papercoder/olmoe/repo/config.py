import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ModelConfig:
    """
    Configuration for the OLMoE model architecture.
    """
    d_model: int = 2048
    num_layers: int = 16
    num_attention_heads: int = 16
    vocab_size: int = 50304
    ffn_dim_expert: int = 1024
    num_experts: int = 64
    num_activated_experts: int = 8
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    init_trunc_cutoff: float = 0.06
    use_qk_norm: bool = True
    attn_variant: str = "full"
    layer_norm_type: str = "RMSNorm"
    weight_tying: bool = False
    activation_function: str = "SwigGLU"

    def __post_init__(self):
        """Perform basic validation after initialization."""
        if not (0 < self.num_activated_experts <= self.num_experts):
            raise ValueError(
                f"num_activated_experts ({self.num_activated_experts}) must be "
                f"between 1 and num_experts ({self.num_experts})."
            )
        if self.init_trunc_cutoff < self.init_std:
            raise ValueError("init_trunc_cutoff should typically be greater than or equal to init_std.")

@dataclass
class DataConfig:
    """
    Configuration for datasets, data loading, and preprocessing.
    """
    pretraining_data_path: str = "https://hf.co/datasets/allenai/OLMoE-mix-0924"
    sft_data_paths: List[str] = field(default_factory=lambda: ["https://hf.co/datasets/allenai/tulu-v3.1-mix-preview-4096-OLMoE"])
    dpo_data_path: str = "https://hf.co/datasets/allenai/ultrafeedback_binarized_cleaned"
    max_seq_len: int = 4096
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    pretrain_mix_weights: Dict[str, float] = field(default_factory=lambda: {
        "dclm_baseline": 3860.0,
        "starcoder": 101.0,
        "pes2o": 57.2,
        "arxiv": 21.1,
        "openwebmath": 12.7,
        "algebraic_stack": 12.6,
        "wikipedia_wikibooks": 3.69
    })
    pretrain_filter_ngram_len: int = 13
    pretrain_filter_ngram_count: int = 32
    pretrain_starcoder_min_stars: int = 2
    pretrain_starcoder_max_freq_word_ratio: float = 0.3
    pretrain_starcoder_max_top2_freq_words_ratio: float = 0.5

    def __post_init__(self):
        """Perform basic validation after initialization."""
        for name, weight in self.pretrain_mix_weights.items():
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"Pretrain mix weight for '{name}' must be a non-negative number.")
        if not (0 <= self.pretrain_starcoder_max_freq_word_ratio <= 1):
            raise ValueError("pretrain_starcoder_max_freq_word_ratio must be between 0 and 1.")
        if not (0 <= self.pretrain_starcoder_max_top2_freq_words_ratio <= 1):
            raise ValueError("pretrain_starcoder_max_top2_freq_words_ratio must be between 0 and 1.")

@dataclass
class TrainingConfig:
    """
    Configuration for training parameters (pretraining, SFT, DPO).
    """
    project_name: str = "OLMoE_Reproducibility"
    run_name: str = "OLMoE-1B-7B_Pretrain_Run"
    total_tokens: float = 5.133e12
    global_batch_size_samples: int = 1024
    per_device_batch_size_samples: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate_peak: float = 4e-4
    learning_rate_min: float = 4e-5
    warmup_steps: int = 2500
    annealing_tokens: float = 1.0e11
    annealing_min_lr_final: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    gradient_clipping_norm: float = 1.0
    lbl_weight: float = 0.01
    rz_loss_weight: float = 0.001
    precision: str = "bf16"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100
    eval_interval: int = 5000
    device: str = "cuda"
    num_gpus: int = 256 # Number of GPUs for pretraining

    # Adaptation (SFT) settings
    sft_epochs: int = 2
    sft_lr: float = 2e-5
    sft_global_batch_size_samples: int = 128
    sft_per_device_batch_size_samples: int = 2
    sft_gradient_accumulation_steps: int = 2
    sft_num_gpus: int = 32 # Number of GPUs for SFT/DPO

    # Adaptation (DPO) settings
    dpo_epochs: int = 3
    dpo_lr: float = 5e-7
    dpo_beta: float = 0.1
    dpo_global_batch_size_samples: int = 32
    dpo_per_device_batch_size_samples: int = 1
    dpo_gradient_accumulation_steps: int = 1

    def __post_init__(self):
        """Perform consistency checks for batch sizes and GPU count."""
        # Pretraining batch consistency check
        expected_global_batch = (
            self.per_device_batch_size_samples * self.num_gpus * self.gradient_accumulation_steps
        )
        if self.global_batch_size_samples != expected_global_batch:
            print(f"Warning: Pretraining global_batch_size_samples ({self.global_batch_size_samples}) "
                  f"does not match calculated value ({expected_global_batch}) "
                  f"from per_device_batch_size_samples * num_gpus * gradient_accumulation_steps.")

        # SFT batch consistency check
        expected_sft_global_batch = (
            self.sft_per_device_batch_size_samples * self.sft_num_gpus * self.sft_gradient_accumulation_steps
        )
        if self.sft_global_batch_size_samples != expected_sft_global_batch:
            print(f"Warning: SFT sft_global_batch_size_samples ({self.sft_global_batch_size_samples}) "
                  f"does not match calculated value ({expected_sft_global_batch}) "
                  f"from sft_per_device_batch_size_samples * sft_num_gpus * sft_gradient_accumulation_steps.")

        # DPO batch consistency check
        expected_dpo_global_batch = (
            self.dpo_per_device_batch_size_samples * self.sft_num_gpus * self.dpo_gradient_accumulation_steps
        )
        if self.dpo_global_batch_size_samples != expected_dpo_global_batch:
            print(f"Warning: DPO dpo_global_batch_size_samples ({self.dpo_global_batch_size_samples}) "
                  f"does not match calculated value ({expected_dpo_global_batch}) "
                  f"from dpo_per_device_batch_size_samples * sft_num_gpus (assuming same as SFT) * dpo_gradient_accumulation_steps.")


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation benchmarks.
    """
    pretrain_eval_tasks: List[str] = field(default_factory=lambda: [
        "ARC-C", "ARC-E", "BoolQ", "COPA", "CSQA", "HellaSwag", "MMLU",
        "MMLU_Var", "OBQA", "PIQA", "SciQ", "SocialIQA", "Winogrande",
        "Paloma_Books", "Paloma_Reddit", "Paloma_Stack"
    ])
    post_pretrain_eval_tasks: List[str] = field(default_factory=lambda: [
        "MMLU", "HellaSwag", "ARC-C", "ARC-E", "PIQA", "WinoGrande",
        "DCLM_Core", "DCLM_Extended"
    ])
    adapt_eval_tasks: List[str] = field(default_factory=lambda: [
        "MMLU", "GSM8k", "BBH", "HumanEval", "AlpacaEval_1.0", "XSTest", "IFEval"
    ])
    few_shot_settings: Dict[str, Any] = field(default_factory=lambda: {
        "default_pretrain_progress": 0,
        "default_post_pretrain": 5,
        "MMLU_pretrain_progress": "0-5",
        "MMLU_adapted": 0,
        "GSM8k_adapted": 8,
        "BBH_adapted": 3,
        "HumanEval_adapted": 0,
        "AlpacaEval_adapted": 0,
        "XSTest_adapted": 0,
        "IFEval_adapted": 0
    })
    olmes_config_path: str = "configs/olmes_tasks.yaml"
    dclm_eval_repo_path: str = "path/to/dclm-repo"
    evaluation_base_model_path: Optional[str] = "allenai/OLMo-7B-0724-hf"

@dataclass
class Config:
    """
    Main configuration class that holds all sub-configurations.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @staticmethod
    def load_from_yaml(path: str) -> "Config":
        """
        Loads configuration from a YAML file.

        Args:
            path: The path to the YAML configuration file.

        Returns:
            A Config object populated with values from the YAML file.
        """
        try:
            with open(path, 'r') as f:
                yaml_config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {path}")
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML configuration file: {exc}")

        # Extract and instantiate sub-configs
        model_config = ModelConfig(**yaml_config.get("model", {}))
        data_config = DataConfig(**yaml_config.get("data", {}))
        training_config = TrainingConfig(**yaml_config.get("training", {}))
        evaluation_config = EvaluationConfig(**yaml_config.get("evaluation", {}))

        return Config(
            model=model_config,
            data=data_config,
            training=training_config,
            evaluation=evaluation_config
        )

if __name__ == '__main__':
    # Example usage and validation
    config_path = 'config.yaml' # Assuming config.yaml is in the same directory
    try:
        config = Config.load_from_yaml(config_path)
        print("Configuration loaded successfully!")
        print(f"Model d_model: {config.model.d_model}")
        print(f"Pretraining data path: {config.data.pretraining_data_path}")
        print(f"Learning rate peak: {config.training.learning_rate_peak}")
        print(f"Eval tasks (pretrain): {config.evaluation.pretrain_eval_tasks}")

        # Demonstrate access to derived warnings
        _ = TrainingConfig() # Just to trigger post_init warnings if defaults are inconsistent

        # Example of manually creating a Config object (without YAML loading)
        custom_model_config = ModelConfig(num_experts=10, num_activated_experts=2)
        custom_data_config = DataConfig(max_seq_len=2048)
        custom_training_config = TrainingConfig(num_gpus=1, global_batch_size_samples=1,
                                                per_device_batch_size_samples=1, gradient_accumulation_steps=1)
        custom_eval_config = EvaluationConfig(pretrain_eval_tasks=["some_task"])
        custom_config = Config(model=custom_model_config, data=custom_data_config,
                               training=custom_training_config, evaluation=custom_eval_config)
        print("\nCustom configuration created manually:")
        print(f"Custom model num_experts: {custom_config.model.num_experts}")

    except Exception as e:
        print(f"Error loading or validating configuration: {e}")

