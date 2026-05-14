import yaml
from dataclasses import dataclass, field, replace
from typing import List, Optional, Dict, Any, Callable

from dataclasses_json import dataclass_json, config as dataclasses_json_config, mm_field


def list_field(default_list: List[Any]) -> List[Any]:
  """Helper to provide mutable default lists safely in dataclasses."""
  return field(default_factory=lambda: default_list.copy())


@dataclass_json
@dataclass
class UnetConfig:
  """Configuration for the Flow Matching UNet model."""
  sample_size: int = 64
  in_channels: int = 4
  out_channels: int = 4
  down_block_types: List[str] = list_field([
      "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
      "DownBlock2D"
  ])
  up_block_types: List[str] = list_field([
      "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
      "CrossAttnUpBlock2D"
  ])
  block_out_channels: List[int] = list_field([320, 640, 1280, 1280])
  layers_per_block: int = 2
  cross_attention_dim: int = 768
  attention_head_dim: int = 8


@dataclass_json
@dataclass
class TextEncoderConfig:
  """Configuration for the text encoder (e.g., CLIP)."""
  model_name: str = "openai/clip-vit-large-patch14"
  max_length: int = 77


@dataclass_json
@dataclass
class ModelConfig:
  """Grouped configuration for the generative model."""
  unet_config: UnetConfig = field(default_factory=UnetConfig)
  pretrained_base_model_path: str = (
      "path/to/your/pretrained_flow_matching_unet.pt")
  text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)


@dataclass_json
@dataclass
class RewardModelConfig:
  """Configuration for the reward model."""
  name: str = "ImageReward"
  model_path: str = "path/to/your/ImageReward_model.pt"


@dataclass_json
@dataclass
class DataConfig:
  """Configuration for dataset paths."""
  fine_tuning_prompts_path: str = "data/fine_tuning_prompts_40k.txt"
  eval_prompts_path: str = "data/eval_prompts_1k.txt"


@dataclass_json
@dataclass
class OptimizerConfig:
  """Configuration for the optimizer."""
  name: str = "Adam"
  learning_rate: float = 2e-5
  betas: List[float] = list_field([0.95, 0.999])
  eps: float = 1e-8
  weight_decay: float = 1e-2
  gradient_norm_clip: float = 1.0


@dataclass_json
@dataclass
class FineTuningConfig:
  """Configuration for the fine-tuning process."""
  method: str = "AdjointMatching"
  num_timesteps: int = 40
  num_fine_tune_iterations: int = 1000
  lambda_reward: float = 12500.0
  lct_value_scale: float = 1.6
  lct_value_scale_cont_adjoint: float = 1600.0
  
  # Fine-tuning sigma type will generally be 'memoryless' for AdjointMatching
  # But some baselines might use other sigma types for fine-tuning.
  fine_tuning_sigma_type: str = "memoryless" 
  
  # These are the sigma types used during sampling for evaluation *after* fine-tuning.
  # The default order is as per Table 2, 3, etc. for 'AdjointMatching' results.
  evaluation_sampling_sigma_types: List[str] = list_field(["memoryless", "ode"]) 

  optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
  batch_size: int = 20
  precision: str = "bfloat16"

  @property
  def h_timestep(self) -> float:
    """Calculates the discrete timestep size."""
    return 1.0 / self.num_timesteps

  @property
  def lct_value(self) -> float:
    """Calculates the Loss Clipping Threshold (LCT) based on method and lambda."""
    if self.method == "AdjointMatching":
      return self.lct_value_scale * (self.lambda_reward**2)
    elif self.method == "ContinuousAdjoint":
      return self.lct_value_scale_cont_adjoint * (self.lambda_reward**2)
    return 0.0  # Default or for baselines where LCT is not specified


@dataclass_json
@dataclass
class EvaluationConfig:
  """Configuration for the evaluation process."""
  sampling_sigma_types: List[str] = list_field(["memoryless", "ode"])
  cfg_weights: List[float] = list_field([0.0, 1.0, 4.0])
  num_inference_timesteps: int = 40
  num_samples_per_prompt: int = 40
  num_eval_prompts: int = 25
  eval_frequency: int = 100
  save_samples: bool = True
  save_metrics: bool = True


@dataclass_json
@dataclass
class BaselineSpecificConfig:
  """Specific overrides for a baseline method."""
  num_fine_tune_iterations: Optional[int] = None
  fine_tuning_sigma_type: Optional[str] = None
  evaluation_sampling_sigma_types: Optional[List[str]] = None
  learning_rate: Optional[float] = None


@dataclass_json
@dataclass
class BaselinesConfig:
  """Container for all baseline-specific configurations."""
  # Using mm_field to map YAML keys with hyphens to valid Python attribute names
  draft_1: Optional[BaselineSpecificConfig] = mm_field(
      metadata=dataclasses_json_config(mm_field=dict(data_key="DRaFT-1")),
      default=None)
  dpo: Optional[BaselineSpecificConfig] = mm_field(
      metadata=dataclasses_json_config(mm_field=dict(data_key="DPO")),
      default=None)
  refl: Optional[BaselineSpecificConfig] = mm_field(
      metadata=dataclasses_json_config(mm_field=dict(data_key="ReFL")),
      default=None)
  continuous_adjoint: Optional[BaselineSpecificConfig] = mm_field(
      metadata=dataclasses_json_config(
          mm_field=dict(data_key="ContinuousAdjoint")),
      default=None)
  discrete_adjoint: Optional[BaselineSpecificConfig] = mm_field(
      metadata=dataclasses_json_config(
          mm_field=dict(data_key="DiscreteAdjoint")),
      default=None)

  def get_baseline_specific_overrides(
      self, method_name: str) -> Optional[BaselineSpecificConfig]:
    """Retrieves baseline-specific overrides for a given method name."""
    if method_name == "AdjointMatching":
      return None  # AdjointMatching uses the default fine_tuning config
    elif method_name == "DRaFT-1":
      return self.draft_1
    elif method_name == "DPO":
      return self.dpo
    elif method_name == "ReFL":
      return self.refl
    elif method_name == "ContinuousAdjoint":
      return self.continuous_adjoint
    elif method_name == "DiscreteAdjoint":
      return self.discrete_adjoint
    else:
      raise ValueError(f"Unknown fine-tuning method: {method_name}")


@dataclass_json
@dataclass
class GeneralConfig:
  """General project-level configuration."""
  project_name: str = "adjoint_matching_reproduction"
  run_name: str = "adjoint_matching_lambda_12500"
  device: str = "cuda"
  seed: int = 42
  output_dir: str = "outputs"


@dataclass_json
@dataclass
class Config:
  """Root configuration for the entire project."""
  general: GeneralConfig = field(default_factory=GeneralConfig)
  model: ModelConfig = field(default_factory=ModelConfig)
  reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
  data: DataConfig = field(default_factory=DataConfig)
  fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)
  evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
  baselines: BaselinesConfig = field(default_factory=BaselinesConfig)

  @classmethod
  def from_yaml(cls, path: str) -> "Config":
    """Loads configuration from a YAML file."""
    with open(path, "r") as f:
      yaml_config = yaml.safe_load(f)
    return cls.from_dict(yaml_config)

  def get_effective_fine_tuning_config(self, method_name: str) -> FineTuningConfig:
    """
    Generates an effective FineTuningConfig for the given method name,
    applying baseline-specific overrides if applicable.
    """
    # Start with the base fine_tuning configuration
    effective_ft_config = replace(self.fine_tuning, method=method_name)

    # Get baseline-specific overrides
    baseline_override = self.baselines.get_baseline_specific_overrides(
        method_name)

    if baseline_override:
      # Apply overrides for FineTuningConfig fields
      if baseline_override.num_fine_tune_iterations is not None:
        effective_ft_config.num_fine_tune_iterations = (
            baseline_override.num_fine_tune_iterations)
      if baseline_override.fine_tuning_sigma_type is not None:
        effective_ft_config.fine_tuning_sigma_type = (
            baseline_override.fine_tuning_sigma_type)
      if baseline_override.evaluation_sampling_sigma_types is not None:
        effective_ft_config.evaluation_sampling_sigma_types = (
            baseline_override.evaluation_sampling_sigma_types)

      # Apply overrides for OptimizerConfig fields nested within FineTuningConfig
      if baseline_override.learning_rate is not None:
        effective_ft_config.optimizer.learning_rate = (
            baseline_override.learning_rate)

    return effective_ft_config

