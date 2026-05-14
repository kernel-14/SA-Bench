import math
import warnings
from typing import Optional, List, Dict, Any, Union

class Config:
    """
    Configuration class to manage all hyperparameters and model settings for Gated Attention LLM.
    It loads configurations from a dictionary (typically parsed from a YAML file) and provides
    validation and derivation of essential parameters.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initializes the Config object from a dictionary.

        Args:
            config_dict: A dictionary containing configuration parameters.
        """
        # General experiment configuration
        self.experiment_name: str = config_dict.get("experiment_name", "gated_attention_llm_reproduction")
        self.seed: int = config_dict.get("seed", 42)
        self.output_dir: str = config_dict.get("output_dir", "./experiments")

        # Model Configuration
        model_cfg = config_dict.get("model", {})
        self.model_type: str = model_cfg.get("type", "dense")
        self.total_parameters_target: float = model_cfg.get("total_parameters_target", 1.7e9)
        self.num_layers: int = model_cfg.get("num_layers", 28)
        self.d_model: int = model_cfg.get("d_model", 2048)
        self.vocab_size: int = model_cfg.get("vocab_size", 32000)
        self.max_seq_len: int = model_cfg.get("max_seq_len", 4096)

        # Attention specific
        self.q_heads: int = model_cfg.get("q_heads", 32)
        self.kv_heads: int = model_cfg.get("kv_heads", 4)
        self.head_dim: int = model_cfg.get("head_dim", 128)
        self.attn_dropout: float = model_cfg.get("attn_dropout", 0.1)
        self.rope_base: float = model_cfg.get("rope_base", 10000.0)

        # Feedforward specific
        self.d_ff: int = model_cfg.get("d_ff", 8192)
        self.ffn_activation: str = model_cfg.get("ffn_activation", "gelu")

        # MoE specific (if model.type == moe)
        moe_cfg = model_cfg.get("moe", {})
        self.moe_num_experts: Optional[int] = moe_cfg.get("num_experts", None)
        self.moe_top_k_experts: Optional[int] = moe_cfg.get("top_k_experts", None)
        self.moe_router_bias: Optional[bool] = moe_cfg.get("router_bias", False)
        self.moe_z_loss_coeff: Optional[float] = moe_cfg.get("z_loss_coeff", None)
        self.moe_load_balancing_loss_coeff: Optional[float] = moe_cfg.get("load_balancing_loss_coeff", None)

        # Gating Configuration
        gating_cfg = config_dict.get("gating", {})
        self.gating_enabled: bool = gating_cfg.get("enabled", True)
        self.gating_position: str = gating_cfg.get("position", "G1")
        self.gating_granularity: str = gating_cfg.get("granularity", "elementwise")
        self.gating_head_specific: bool = gating_cfg.get("head_specific", True)
        self.gating_type: str = gating_cfg.get("type", "multiplicative")
        self.gating_activation_fn: str = gating_cfg.get("activation_fn", "sigmoid")
        self.gating_ns_sigmoid_factor: Optional[float] = gating_cfg.get("ns_sigmoid_factor", None)

        # Training Configuration
        training_cfg = config_dict.get("training", {})
        self.total_train_tokens: float = training_cfg.get("total_train_tokens", 400e9)
        self.max_learning_rate: float = training_cfg.get("max_learning_rate", 0.004)
        self.min_learning_rate: float = training_cfg.get("min_learning_rate", 0.000003)
        self.warmup_steps: int = training_cfg.get("warmup_steps", 1000)
        self.global_batch_size: int = training_cfg.get("global_batch_size", 1024)
        self.gradient_accumulation_steps: int = training_cfg.get("gradient_accumulation_steps", 1)
        self.optimizer: str = training_cfg.get("optimizer", "adamw")
        self.adam_beta1: float = training_cfg.get("adam_beta1", 0.9)
        self.adam_beta2: float = training_cfg.get("adam_beta2", 0.999)
        self.adam_epsilon: float = training_cfg.get("adam_epsilon", 1.0e-8)
        self.weight_decay: float = training_cfg.get("weight_decay", 0.1)
        self.mixed_precision: str = training_cfg.get("mixed_precision", "bf16")
        self.checkpoint_interval_steps: int = training_cfg.get("checkpoint_interval_steps", 10000)
        self.eval_interval_steps: int = training_cfg.get("eval_interval_steps", 1000)
        self.num_training_steps: Optional[int] = None # Derived later

        # Data Configuration
        data_cfg = config_dict.get("data", {})
        self.tokenizer_path: str = data_cfg.get("tokenizer_path", "path/to/your/tokenizer")
        self.train_data_paths: List[str] = data_cfg.get("train_data_paths", [])
        self.eval_data_paths: List[str] = data_cfg.get("eval_data_paths", [])

        # Long-Context Extension Specifics
        lce_cfg = config_dict.get("long_context_extension", {})
        self.long_context_extension_enabled: bool = lce_cfg.get("enabled", False)
        self.long_context_initial_rope_base: float = lce_cfg.get("initial_rope_base", 10000.0)
        self.long_context_extended_rope_base: float = lce_cfg.get("extended_rope_base", 1000000.0)
        self.long_context_extended_seq_len: int = lce_cfg.get("extended_seq_len", 32768)
        self.long_context_additional_train_tokens: float = lce_cfg.get("additional_train_tokens_for_extension", 80e9)

        # Evaluation Configuration
        eval_cfg = config_dict.get("evaluation", {})
        self.eval_benchmarks: List[str] = eval_cfg.get("benchmarks", ["hellaswag", "mmlu", "gsm8k", "human_eval", "c_eval", "cmmlu"])
        self.eval_ruler_benchmark_enabled: bool = eval_cfg.get("ruler_benchmark_enabled", False)
        self.eval_attention_sink_analysis_enabled: bool = eval_cfg.get("attention_sink_analysis_enabled", True)
        self.eval_massive_activation_analysis_enabled: bool = eval_cfg.get("massive_activation_analysis_enabled", True)
        self.eval_gating_score_analysis_enabled: bool = eval_cfg.get("gating_score_analysis_enabled", True)

        self._validate_and_derive_params()

    def _validate_and_derive_params(self) -> None:
        """
        Validates the configuration parameters and derives additional ones.
        Raises ValueError for critical inconsistencies.
        """
        # Model dimensions validation
        if not (self.d_model > 0 and self.q_heads > 0 and self.kv_heads > 0 and self.head_dim > 0):
            raise ValueError("All model dimensions (d_model, q_heads, kv_heads, head_dim) must be positive.")
        if self.d_model % self.q_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by q_heads ({self.q_heads}).")
        if self.d_model != self.q_heads * self.head_dim:
            raise ValueError(
                f"d_model ({self.d_model}) must be equal to q_heads ({self.q_heads}) * head_dim ({self.head_dim})."
            )

        # MoE parameters validation
        if self.model_type == "moe":
            if not all(
                isinstance(p, (int, float)) and p is not None for p in
                [self.moe_num_experts, self.moe_top_k_experts, self.moe_z_loss_coeff, self.moe_load_balancing_loss_coeff]
            ):
                raise ValueError(
                    "MoE model type requires moe_num_experts, moe_top_k_experts, moe_z_loss_coeff, "
                    "and moe_load_balancing_loss_coeff to be set."
                )
            if not (0 < self.moe_top_k_experts <= self.moe_num_experts):
                raise ValueError(
                    f"moe_top_k_experts ({self.moe_top_k_experts}) must be between 1 and moe_num_experts ({self.moe_num_experts})."
                )
            if not (self.moe_z_loss_coeff >= 0 and self.moe_load_balancing_loss_coeff >= 0):
                raise ValueError("MoE loss coefficients must be non-negative.")
        elif any(
            p is not None for p in
            [self.moe_num_experts, self.moe_top_k_experts, self.moe_z_loss_coeff, self.moe_load_balancing_loss_coeff]
        ):
            warnings.warn("MoE-specific parameters are set but model_type is not 'moe'. These will be ignored.")
            self.moe_num_experts = None
            self.moe_top_k_experts = None
            self.moe_router_bias = None
            self.moe_z_loss_coeff = None
            self.moe_load_balancing_loss_coeff = None


        # Gating parameters validation
        if self.gating_enabled:
            valid_positions = ["G1", "G2", "G3", "G4", "G5"]
            valid_granularities = ["elementwise", "headwise"]
            valid_types = ["multiplicative", "additive"]
            valid_activation_fns = ["sigmoid", "silu", "identity", "ns_sigmoid"]

            if self.gating_position not in valid_positions:
                raise ValueError(f"Invalid gating_position: {self.gating_position}. Must be one of {valid_positions}.")
            if self.gating_granularity not in valid_granularities:
                raise ValueError(f"Invalid gating_granularity: {self.gating_granularity}. Must be one of {valid_granularities}.")
            if self.gating_type not in valid_types:
                raise ValueError(f"Invalid gating_type: {self.gating_type}. Must be one of {valid_types}.")
            if self.gating_activation_fn not in valid_activation_fns:
                raise ValueError(f"Invalid gating_activation_fn: {self.gating_activation_fn}. Must be one of {valid_activation_fns}.")
            if self.gating_activation_fn == "ns_sigmoid" and not (
                isinstance(self.gating_ns_sigmoid_factor, (int, float)) and self.gating_ns_sigmoid_factor is not None
            ):
                raise ValueError("gating_ns_sigmoid_factor must be set for 'ns_sigmoid' activation function.")

        # Training parameters validation
        if not (self.total_train_tokens > 0 and self.max_learning_rate > 0 and self.global_batch_size > 0 and self.max_seq_len > 0):
            raise ValueError(
                "total_train_tokens, max_learning_rate, global_batch_size, and max_seq_len must be positive."
            )
        if not (self.warmup_steps >= 0 and self.gradient_accumulation_steps >= 1):
            raise ValueError("warmup_steps must be non-negative and gradient_accumulation_steps must be at least 1.")
        
        # Data paths validation
        if not self.tokenizer_path or not isinstance(self.tokenizer_path, str):
            raise ValueError("tokenizer_path must be a valid string.")
        if not self.train_data_paths:
            warnings.warn("No training data paths specified in config. Training will likely fail.")
        if not self.eval_data_paths:
            warnings.warn("No evaluation data paths specified in config. Evaluation will not be possible.")
        
        # Long-context extension validation
        if self.long_context_extension_enabled:
            if not (self.long_context_initial_rope_base > 0 and self.long_context_extended_rope_base > 0):
                raise ValueError("RoPE bases for long-context extension must be positive.")
            if not (self.long_context_extended_seq_len > self.max_seq_len):
                raise ValueError(
                    f"extended_seq_len ({self.long_context_extended_seq_len}) must be greater than max_seq_len ({self.max_seq_len})."
                )
            if not (self.long_context_additional_train_tokens >= 0):
                raise ValueError("additional_train_tokens_for_extension must be non-negative.")

        # Derived parameters
        # Calculate num_training_steps based on effective global batch size
        # effective_global_batch_size = self.global_batch_size * self.gradient_accumulation_steps
        # The config's global_batch_size is already defined as effective_global_batch_size
        tokens_per_step = self.global_batch_size * self.max_seq_len
        self.num_training_steps = math.ceil(self.total_train_tokens / tokens_per_step)


    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Config object to a dictionary, excluding private attributes.

        Returns:
            A dictionary representation of the configuration.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

