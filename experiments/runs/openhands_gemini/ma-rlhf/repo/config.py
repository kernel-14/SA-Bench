
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class SFTConfig:
    """
    Configuration for the Supervised Fine-Tuning (SFT) stage.
    Appendix B.2 - SFT Training
    """
    batch_size: int = 64 # Varies by dataset/model: WebGPT=64, others=512 for 2B; 128 for 7B/27B
    epochs: int = 3 # Varies by dataset/model: WebGPT=5 for 7B, others=1
    learning_rate: float = 5e-5 # Varies by dataset/model: WebGPT=1e-4 for 2B; 2e-5 for 7B; 5e-6 for 27B
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1 # Varies by model: 0 for CodeGemma
    max_seq_length: int = 512 # Assumed default, not explicitly stated for SFT, but common for LLMs
    # For CodeGemma:
    code_batch_size: int = 16 # 2B
    code_learning_rate: float = 5e-6 # 2B
    code_epochs: int = 1
    code_warmup_ratio: float = 0.0


@dataclass
class RMConfig:
    """
    Configuration for the Reward Modeling (RM) stage.
    Appendix B.2 - Reward Modeling
    """
    batch_size: int = 64 # Varies by dataset/model: WebGPT=32; 128 for 7B TL;DR, 64 for 7B HH-RLHF
    epochs: int = 1 # Varies by model: 32 for 7B WebGPT
    learning_rate: float = 1e-5 # Varies by dataset/model: WebGPT=2e-5 for 2B; 1e-6 for 7B; 8e-6 for 27B
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_seq_length: int = 512 # Assumed similar to SFT for consistency
    # Code generation omits this stage


@dataclass
class PPOConfig:
    """
    Configuration for the PPO (RLHF) stage.
    Appendix B.2 - PPO Training
    """
    batch_size: int = 256 # Varies by model: 16 for CodeGemma
    policy_learning_rate: float = 1.5e-5 # Varies by model: 1e-6 for 7B; 7e-7 for 27B
    critic_learning_rate: float = 1.5e-5 # Varies by model: 1e-6 for 7B; 1e-6 for 27B
    epochs: int = 1 # Varies by dataset/model: WebGPT=4; others=1
    ppo_epochs: int = 1 # Not explicitly stated, but common in PPO for multiple passes over collected data
    rollout_per_device: int = 1 # Not explicitly stated, but common in PPO for number of rollouts
    clip_ratio: float = 0.2
    gamma: float = 1.0 # Discount factor for GAE, set to 1 in experiments
    lam: float = 0.95 # Lambda for GAE
    kl_coefficient: float = 0.05 # Varies by model: 0.1 for WebGPT; 0.01 for 7B TL;DR
    max_prompt_length: int = 512 # Varies by model: 600 for CodeGemma
    max_response_length: int = 512
    warmup_steps: int = 200 # Varies by model: 0 for CodeGemma 2B; 20 for CodeGemma 7B
    temperature: float = 0.8 # Varies by model: 1.0 for CodeGemma
    top_p: float = 1.0
    top_k: int = 50 # Varies by model: 5 for CodeGemma
    # For CodeGemma:
    code_batch_size: int = 16
    code_policy_learning_rate: float = 5e-7
    code_critic_learning_rate: float = 5e-5
    code_kl_coefficient: float = 0.05
    code_max_prompt_length: int = 600
    code_max_response_length: int = 512
    code_warmup_steps: int = 20
    code_temperature: float = 1.0
    code_top_k: int = 5


@dataclass
class MAConfig:
    """
    Configuration for Macro Action specifics.
    Section 3.2.1, Appendix D.1
    """
    termination_condition: str = "fixed_ngram" # "fixed_ngram", "randomized_ngram", "parsing", "perplexity"
    n_gram: Optional[int] = 5 # Default for fixed n-gram, or max length for randomized
    randomized_ngram_lengths: List[int] = field(default_factory=lambda: [2, 3, 5, 10]) # For randomized n-gram
    parsing_cutoff: int = 5 # C for parsing-based termination
    # Perplexity-based termination uses log-likelihoods from reference model, no direct hyperparam here
    value_estimation_assignment: str = "equal" # "equal", "unit", "position_decayed"


@dataclass
class ModelConfig:
    """
    Configuration for the base language model.
    Section 4.1 - Base Models and Training Details
    """
    model_name_or_path: str = "google/gemma-2b" # e.g., "google/gemma-2b", "google/gemma-7b", "google/gemma-2-27b", "codegemma/codegemma-1.1-2b-it", "codegemma/codegemma-1.1-7b-it"
    tokenizer_name_or_path: Optional[str] = None
    # LoRA / QLoRA related settings (not explicitly mentioned if used, but common for fine-tuning LLMs)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None # typically for Llama-based models


@dataclass
class DataConfig:
    """
    Configuration for datasets.
    Section 4.1 - Tasks and Datasets, Appendix B.1
    """
    tldr_dataset_path: str = "tldr_dataset" # Placeholder
    hh_rlhf_dataset_path: str = "hh_rlhf_dataset" # Placeholder
    webgpt_dataset_path: str = "webgpt_dataset" # Placeholder
    apps_dataset_path: str = "apps_dataset" # Placeholder
    # Data splits mentioned in Appendix B.2
    sft_data_ratio: float = 0.2
    rm_data_ratio: float = 0.4
    ppo_data_ratio: float = 0.4 # Remaining data after SFT and RM

    # For program synthesis, 80% for PPO, RM stage omitted
    apps_sft_data_ratio: float = 0.2
    apps_ppo_data_ratio: float = 0.8


@dataclass
class GeneralConfig:
    """
    General training and evaluation configurations.
    """
    project_name: str = "ma-rlhf"
    experiment_name: str = "default_experiment"
    seed: int = 42
    output_dir: str = "./outputs"
    num_train_epochs: int = 1 # Overall training epochs (RLHF stages run for specific steps)
    gradient_accumulation_steps: int = 1 # Not explicitly mentioned, but crucial for memory
    gradient_checkpointing: bool = True # Not explicitly mentioned, but common for LLM training
    fp16: bool = True # Not explicitly mentioned, but common for LLM training
    bf16: bool = False
    max_grad_norm: float = 1.0 # Not explicitly mentioned, but common for stability
    save_total_limit: int = 1 # Number of checkpoints to keep
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    # Evaluation metrics
    rm_eval_samples: int = 2000 # 2k for TL;DR and HH-RLHF, default for WebGPT
    gpt4_eval_samples: int = 50
    human_eval_samples: int = 50
    # GPT-4 model for evaluation
    gpt4_model_name: str = "gpt-4o-05-13"


@dataclass
class Config:
    sft: SFTConfig = field(default_factory=SFTConfig)
    rm: RMConfig = field(default_factory=RMConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    ma: MAConfig = field(default_factory=MAConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)

    def __post_init__(self):
        # Adjust batch sizes and learning rates based on model size and task if needed
        # This can be done dynamically based on self.model.model_name_or_path
        # For simplicity, these are set to the 2B model defaults, and would need to be updated
        # for 7B/27B or CodeGemma explicitly in a script or via command-line args.

        # Example for Gemma 7B adjustment (based on Table 5)
        if "gemma-7b" in self.model.model_name_or_path:
            self.sft.batch_size = 128
            self.sft.learning_rate = 2e-5
            self.sft.epochs = 1 # General, WebGPT would be 5
            self.rm.batch_size = 128 # TL;DR, HH-RLHF would be 64
            self.rm.epochs = 1 # General, WebGPT would be 32
            self.rm.learning_rate = 1e-6
            self.ppo.policy_learning_rate = 1e-6
            self.ppo.critic_learning_rate = 1e-6
            self.ppo.epochs = 1 # General, WebGPT would be 4
            self.ppo.kl_coefficient = 0.05 # Default, TL;DR specific adjustment to 0.01 mentioned

        elif "gemma-2-27b" in self.model.model_name_or_path:
            self.sft.batch_size = 128
            self.sft.learning_rate = 5e-6
            self.sft.epochs = 3
            self.rm.batch_size = 128
            self.rm.epochs = 1
            self.rm.learning_rate = 8e-6
            self.ppo.policy_learning_rate = 7e-7
            self.ppo.critic_learning_rate = 1e-6
            self.ppo.epochs = 1
            self.ppo.kl_coefficient = 0.1 # WebGPT specific
        
        elif "codegemma" in self.model.model_name_or_path:
            # SFT
            self.sft.batch_size = self.sft.code_batch_size
            self.sft.learning_rate = self.sft.code_learning_rate
            self.sft.epochs = self.sft.code_epochs
            self.sft.warmup_ratio = self.sft.code_warmup_ratio
            # RM stage is omitted, so RM configs are not directly used, but could be set to default/None
            # PPO
            self.ppo.batch_size = self.ppo.code_batch_size
            self.ppo.policy_learning_rate = self.ppo.code_policy_learning_rate
            self.ppo.critic_learning_rate = self.ppo.code_critic_learning_rate
            self.ppo.kl_coefficient = self.ppo.code_kl_coefficient
            self.ppo.max_prompt_length = self.ppo.code_max_prompt_length
            self.ppo.max_response_length = self.ppo.code_max_response_length
            self.ppo.warmup_steps = self.ppo.code_warmup_steps
            self.ppo.temperature = self.ppo.code_temperature
            self.ppo.top_k = self.ppo.code_top_k


