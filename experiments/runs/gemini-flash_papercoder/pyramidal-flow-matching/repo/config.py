import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

@dataclass
class TrainingStageConfig:
    """
    Configuration for a single training stage.
    """
    name: str
    global_batch_size: int
    learning_rate: float
    training_steps: int
    warmup_steps: int
    optimizer_beta1: float
    optimizer_beta2: float
    optimizer_epsilon: float
    gradient_clipping: float
    weight_decay: float
    dataset_type: str  # "image" or "video"
    dataset_names: List[str]
    dataset_paths: Dict[str, str] = field(default_factory=dict)
    image_data_proportion: Optional[float] = None  # For video stages
    history_condition_noise_strength: Optional[List[float]] = None  # For video stages

@dataclass
class DitParamsConfig:
    """
    Configuration for the Diffusion Transformer (DiT) backbone.
    """
    num_layers: int = 24
    num_attention_heads: int = 16  # Common for 2B models, placeholder
    hidden_size: int = 1024  # Common for 2B models, placeholder
    mlp_ratio: int = 4  # Common, placeholder
    in_channels: int = 4  # VAE output channels
    out_channels: int = 4  # Velocity field output channels (same as in_channels for latents)
    block_wise_causal_attention: bool = True

@dataclass
class TextEncoderConfig:
    """
    Configuration for text encoders (T5 and CLIP).
    """
    t5_model_name: str = "google/t5-v1_1-large"  # Placeholder
    clip_model_name: str = "openai/clip-vit-large-patch14"  # Placeholder
    max_text_length: int = 77

@dataclass
class VaeConfig:
    """
    Configuration for the Video Variational Autoencoder (VAE).
    """
    name: str = "VideoVAE"
    compression_rate: List[int] = field(default_factory=lambda: [8, 8, 8])  # [Spatial, Spatial, Temporal]
    latent_channels: int = 4  # Placeholder, common for latent diffusion
    base_channels: int = 128  # Placeholder
    num_res_blocks: int = 2  # Placeholder
    attn_resolutions: List[int] = field(default_factory=lambda: [16])  # Placeholder
    use_causal_conv3d: bool = True
    kl_regularization: bool = True

@dataclass
class ModelConfig:
    """
    Overall model configuration.
    """
    name: str = "PyramidalFlowMatching"
    dit_backbone_name: str = "SD3_Medium"
    dit_params: DitParamsConfig = field(default_factory=DitParamsConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    vae: VaeConfig = field(default_factory=VaeConfig)
    pyramid_stages: int = 3
    # This will be derived dynamically based on pyramid_stages
    spatial_pyramid_time_windows: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class InferenceConfig:
    """
    Configuration for video generation inference.
    """
    guidance_scale: float = 7.0
    num_inference_steps: int = 50
    ode_solver: str = "dpm_solver"
    output_resolution: Tuple[int, int] = field(default_factory=lambda: (768, 768))
    output_fps: int = 24
    output_duration: int = 5  # seconds
    max_output_duration: int = 10  # seconds
    seed: int = 42

@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation metrics and paths.
    """
    vbench_path: str = "/path/to/VBENCH_scripts"
    evalcrafter_path: str = "/path/to/EVALCRAFTER_scripts"
    evaluation_prompts_path: str = "/data/evaluation_prompts.txt"
    generated_video_output_dir: str = "./generated_videos"

@dataclass
class ComputeConfig:
    """
    Configuration for computational resources.
    """
    device: str = "cuda"
    num_gpus: int = 128
    mixed_precision: str = "bfloat16"

@dataclass
class DataPathsConfig:
    """
    Configuration for data and model weights paths.
    """
    image_data_root: str = "/data/images"
    video_data_root: str = "/data/videos"
    vae_weights: str = "./weights/vae.pth"
    model_weights: str = "./weights/pyramidal_flow_matching_model.pth"

@dataclass
class Config:
    """
    Main configuration class for the entire project.
    """
    training: Dict[int, TrainingStageConfig] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    data_paths: DataPathsConfig = field(default_factory=DataPathsConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[Path, str]) -> 'Config':
        """
        Loads configuration from a YAML file and instantiates the Config dataclass.
        Derives spatial_pyramid_time_windows based on `pyramid_stages`.
        """
        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)

        with open(yaml_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)

        # Instantiate nested dataclasses
        training_stages_dict = {}
        if 'training' in cfg_dict and 'stages' in cfg_dict['training']:
            for i, stage_data in enumerate(cfg_dict['training']['stages']):
                training_stages_dict[i + 1] = TrainingStageConfig(**stage_data)
        
        model_config = ModelConfig(**cfg_dict.get('model', {}))
        inference_config = InferenceConfig(**cfg_dict.get('inference', {}))
        evaluation_config = EvaluationConfig(**cfg_dict.get('evaluation', {}))
        compute_config = ComputeConfig(**cfg_dict.get('compute', {}))
        data_paths_config = DataPathsConfig(**cfg_dict.get('data_paths', {}))

        # --- Derive spatial_pyramid_time_windows ---
        K = model_config.pyramid_stages
        derived_time_windows = []

        if K > 0:
            # e_k values: End time of stage k (k=0 is finest, k=K-1 is coarsest)
            # e_0 = 1.0 (finest stage ends at 1)
            # e_k = (K - k) / K for k in [1, K-1]
            e_values: Dict[int, float] = {}
            e_values[0] = 1.0
            for k_idx in range(1, K):
                e_values[k_idx] = (K - k_idx) / K
            
            # s_k values: Start time of stage k
            # s_{K-1} = 0.0 (coarsest stage starts at 0)
            # s_{k-1} = e_k / (2 - e_k) for k in [1, K-1]
            s_values: Dict[int, float] = {}
            s_values[K-1] = 0.0
            for k_idx in range(K-1, 0, -1): # Iterate k_idx from K-1 down to 1
                # Calculate s_{k_idx-1} from e_{k_idx}
                s_val = e_values[k_idx] / (2.0 - e_values[k_idx])
                s_values[k_idx-1] = s_val

            # Assemble (s_k, e_k) for each stage, ordered from coarsest (k=K-1) to finest (k=0)
            for k_idx in range(K - 1, -1, -1):
                s_k = s_values.get(k_idx, 0.0) # Should be present, but default for safety
                e_k = e_values.get(k_idx, 0.0) # Should be present, but default for safety
                derived_time_windows.append((s_k, e_k))
        
        model_config.spatial_pyramid_time_windows = derived_time_windows
        # --- End derivation ---

        return cls(
            training=training_stages_dict,
            model=model_config,
            inference=inference_config,
            evaluation=evaluation_config,
            compute=compute_config,
            data_paths=data_paths_config
        )

# Example Usage (for testing and demonstration)
if __name__ == "__main__":
    # Create a dummy config.yaml for testing
    dummy_yaml_content = """
training:
  stages:
    - name: "Stage 1: Image Training"
      global_batch_size: 1536
      learning_rate: 1e-4
      training_steps: 50000
      warmup_steps: 1000
      optimizer_beta1: 0.9
      optimizer_beta2: 0.999
      optimizer_epsilon: 1e-6
      gradient_clipping: 1.0
      weight_decay: 1e-4
      dataset_type: "image"
      dataset_names: ["LAION-5B-Aesthetic-Subset"]
      dataset_paths:
        LAION-5B-Aesthetic-Subset: "/data/laion_aesthetic"
    - name: "Stage 2: Low-Resolution Video Training"
      global_batch_size: 768
      learning_rate: 1e-4
      training_steps: 200000
      warmup_steps: 1000
      optimizer_beta1: 0.9
      optimizer_beta2: 0.95
      optimizer_epsilon: 1e-6
      gradient_clipping: 1.0
      weight_decay: 1e-4
      dataset_type: "video"
      dataset_names: ["WebVid-10M"]
      dataset_paths:
        WebVid-10M: "/data/webvid10m"
      image_data_proportion: 0.125
      history_condition_noise_strength: [0.0, 0.33333333]
    - name: "Stage 3: High-Resolution Video Training"
      global_batch_size: 384
      learning_rate: 5e-5
      training_steps: 50000
      warmup_steps: 1000
      optimizer_beta1: 0.9
      optimizer_beta2: 0.95
      optimizer_epsilon: 1e-6
      gradient_clipping: 1.0
      weight_decay: 1e-4
      dataset_type: "video"
      dataset_names: ["WebVid-10M"]
      dataset_paths:
        WebVid-10M: "/data/webvid10m"
      history_condition_noise_strength: [0.0, 0.33333333]

model:
  name: "PyramidalFlowMatching"
  dit_backbone_name: "SD3_Medium"
  dit_params:
    num_layers: 24
    num_attention_heads: 16
    hidden_size: 1024
    mlp_ratio: 4
    in_channels: 4
    out_channels: 4
    block_wise_causal_attention: True
  text_encoder:
    t5_model_name: "google/t5-v1_1-large"
    clip_model_name: "openai/clip-vit-large-patch14"
    max_text_length: 77
  vae:
    name: "VideoVAE"
    compression_rate: [8, 8, 8]
    latent_channels: 4
    base_channels: 128
    num_res_blocks: 2
    attn_resolutions: [16]
    use_causal_conv3d: True
    kl_regularization: True
  pyramid_stages: 3

inference:
  guidance_scale: 7.0
  num_inference_steps: 50
  ode_solver: "dpm_solver"
  output_resolution: [768, 768]
  output_fps: 24
  output_duration: 5
  max_output_duration: 10
  seed: 42

evaluation:
  vbench_path: "/path/to/VBENCH_scripts"
  evalcrafter_path: "/path/to/EVALCRAFTER_scripts"
  evaluation_prompts_path: "/data/evaluation_prompts.txt"
  generated_video_output_dir: "./generated_videos"

compute:
  device: "cuda"
  num_gpus: 128
  mixed_precision: "bfloat16"

data_paths:
  image_data_root: "/data/images"
  video_data_root: "/data/videos"
  vae_weights: "./weights/vae.pth"
  model_weights: "./weights/pyramidal_flow_matching_model.pth"
    """
    
    config_path = Path("test_config.yaml")
    with open(config_path, "w") as f:
        f.write(dummy_yaml_content)

    config = Config.from_yaml(config_path)

    print("--- Loaded Configuration ---")
    print(f"Model Name: {config.model.name}")
    print(f"Pyramid Stages (K): {config.model.pyramid_stages}")
    print(f"Derived Spatial Pyramid Time Windows (coarsest to finest):")
    for s_k, e_k in config.model.spatial_pyramid_time_windows:
        print(f"  (s_k={s_k:.4f}, e_k={e_k:.4f})")
    
    assert config.model.pyramid_stages == 3
    # Expected: [(0.0, 0.3333), (0.2, 0.6667), (0.5, 1.0)]
    expected_windows = [(0.0, 1/3), (0.2, 2/3), (0.5, 1.0)]
    for i, (s, e) in enumerate(config.model.spatial_pyramid_time_windows):
        assert abs(s - expected_windows[i][0]) < 1e-4
        assert abs(e - expected_windows[i][1]) < 1e-4

    print("\nTraining Stage 1:")
    print(f"  Name: {config.training[1].name}")
    print(f"  Learning Rate: {config.training[1].learning_rate}")

    print("\nCompute Device:", config.compute.device)
    print("\nConfiguration loaded and derived successfully!")

    config_path.unlink() # Clean up dummy config file
