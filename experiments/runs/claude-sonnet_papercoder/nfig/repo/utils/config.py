## utils/config.py
"""Configuration management for the NFIG project.

Loads config.yaml into strongly typed dataclasses for all training phases.
All hyperparameters from the paper are accessed exclusively through this module.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class FRVAEConfig:
    """Configuration for the Frequency-guided Residual-quantized VAE.

    All values sourced from config.yaml frvae section and paper Section 4.1.
    """

    # Encoder settings
    encoder_model: str = "vit_base_patch14_dinov2"
    pretrained_encoder: bool = True
    image_size: int = 256
    latent_spatial_size: int = 16  # H' = W' = 16
    latent_channels: int = 768  # DINOv2-base output channels

    # Residual Quantizer settings
    codebook_size: int = 4096  # K: number of codebook entries
    codebook_dim: int = 768  # C: codebook vector dimension

    # Frequency band scale factors (paper Section 4.1)
    # Token counts: 1+4+9+16+25+36+64+100+169+256 = 680 total
    scale_factors: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    )
    total_tokens: int = 680
    num_frequency_bands: int = 10

    # Loss weights (paper Appendix B.1)
    # L = ||I - I_hat||^2 + ||f_hat - f_hat_reconstructed||^2 + L_p(I) + 0.5*L_g(I)
    gan_loss_weight: float = 0.5
    lpips_weight: float = 1.0
    reconstruction_weight: float = 1.0
    freq_quantization_weight: float = 1.0
    commitment_loss_weight: float = 0.25  # Standard VQ beta

    # Discriminator
    use_dino_discriminator: bool = True

    # Training hyperparameters (VQ-GAN defaults; not specified in paper for FR-VAE)
    learning_rate: float = 1.0e-4
    disc_learning_rate: float = 1.0e-4
    batch_size: int = 8
    epochs: int = 100
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # Target reconstruction FID (paper Section 1 and Table 2)
    target_rfid: float = 0.85


@dataclass
class NFIGConfig:
    """Configuration for the NFIG Transformer (autoregressive generator).

    Key values explicitly stated in paper Section 4.1:
    - learning_rate: 8e-5
    - batch_size: 768
    - epochs: 350
    - cfg_scale: 4.5
    - top_k: 990
    - depth: 16
    """

    # Model architecture (paper Section 4.1: "VAR Transformer backbone with depth 16")
    depth: int = 16
    hidden_dim: int = 1024  # Inferred from VAR-d16 ~310M params
    num_heads: int = 16
    ffn_ratio: int = 4
    dropout: float = 0.0

    # Vocabulary and sequence settings
    codebook_size: int = 4096
    scale_factors: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    )
    total_tokens: int = 680
    num_frequency_bands: int = 10

    # Class conditioning
    num_classes: int = 1000  # ImageNet 1k
    null_class_id: int = 1000  # Extra null class for CFG unconditional training

    # CFG training dropout (standard practice; not specified in paper)
    cfg_dropout_prob: float = 0.1

    # Training hyperparameters (explicitly from paper Section 4.1)
    learning_rate: float = 8.0e-5
    batch_size: int = 768
    epochs: int = 350
    warmup_steps: int = 5000
    grad_clip: float = 1.0
    weight_decay: float = 0.0

    # Inference settings (explicitly from paper Section 4.1)
    cfg_scale: float = 4.5
    top_k: int = 990
    num_generation_steps: int = 10  # One step per frequency band

    # Target generation FID (paper Table 2)
    target_gfid: float = 2.81


@dataclass
class DataConfig:
    """Configuration for the ImageNet dataset and dataloaders.

    Based on paper Section 4.1: ImageNet ILSVRC 2012, 256x256 resolution.
    """

    dataset: str = "imagenet"
    train_dir: str = "data/imagenet/train"
    val_dir: str = "data/imagenet/val"
    image_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True

    # Normalization to [-1, 1] (standard for VQ-GAN)
    mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])

    # Augmentation
    random_horizontal_flip: bool = True
    center_crop: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training infrastructure.

    Hardware: NVIDIA H100 (paper Section 4.1).
    """

    device: str = "cuda"
    hardware: str = "H100"

    # Distributed training
    distributed: bool = True
    backend: str = "nccl"

    # Mixed precision (bf16 preferred on H100)
    mixed_precision: bool = True
    precision: str = "bf16"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_epochs: int = 10
    keep_last_n_checkpoints: int = 5

    # Logging
    log_dir: str = "logs"
    log_every_steps: int = 100
    tensorboard: bool = True

    # Evaluation
    eval_every_epochs: int = 10
    num_eval_samples: int = 50000  # Standard for FID on ImageNet


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation.

    Metrics from paper Section 4.1: FID, IS, Precision, Recall.
    Inference settings: CFG=4.5, top_k=990 (paper Section 4.1).
    """

    metrics: List[str] = field(
        default_factory=lambda: ["gfid", "rfid", "is", "precision", "recall"]
    )
    num_samples: int = 50000
    cfg_scale: float = 4.5
    top_k: int = 990
    output_dir: str = "generated_samples"
    fid_reference: str = "data/imagenet_val_fid_stats.npz"


@dataclass
class AblationStageConfig:
    """Configuration for a single ablation study stage.

    Corresponds to one row in paper Table 5.
    """

    name: str = "baseline_ar"
    sequence_length: int = 256
    use_fr_quantizer: bool = False
    use_dino_disc: bool = False
    use_cfg: bool = False
    use_topk: bool = False


@dataclass
class AblationConfig:
    """Configuration for the ablation study (paper Table 5).

    Tracks incremental component additions from baseline to full NFIG.
    """

    baseline_sequence_length: int = 256
    stages: List[AblationStageConfig] = field(default_factory=list)


@dataclass
class ScalingModelConfig:
    """Configuration for a single model in the scaling study.

    Corresponds to one row in paper Table 3.
    """

    name: str = "NFIG-310M"
    depth: int = 16
    hidden_dim: int = 1024
    num_heads: int = 16
    target_fid: float = 5.47
    target_is: float = 224.20


@dataclass
class ScalingConfig:
    """Configuration for the scaling study (paper Table 3).

    Both models trained for 55 epochs under computational constraints.
    """

    short_run_epochs: int = 55
    models: List[ScalingModelConfig] = field(default_factory=list)


@dataclass
class Config:
    """Root configuration composing all sub-configs for the NFIG project.

    Usage:
        config = Config.from_yaml("config.yaml")
        print(config.nfig.learning_rate)  # 8e-5
    """

    frvae: FRVAEConfig = field(default_factory=FRVAEConfig)
    nfig: NFIGConfig = field(default_factory=NFIGConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Fully populated and validated Config instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If configuration validation fails.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw: Dict = yaml.safe_load(f)

        frvae_config = cls._build_dataclass(FRVAEConfig, raw.get("frvae", {}))
        nfig_config = cls._build_dataclass(NFIGConfig, raw.get("nfig", {}))
        data_config = cls._build_dataclass(DataConfig, raw.get("data", {}))
        training_config = cls._build_dataclass(TrainingConfig, raw.get("training", {}))
        evaluation_config = cls._build_dataclass(
            EvaluationConfig, raw.get("evaluation", {})
        )

        ablation_config = cls._build_ablation_config(raw.get("ablation", {}))
        scaling_config = cls._build_scaling_config(raw.get("scaling", {}))

        config = cls(
            frvae=frvae_config,
            nfig=nfig_config,
            data=data_config,
            training=training_config,
            evaluation=evaluation_config,
            ablation=ablation_config,
            scaling=scaling_config,
        )
        config._validate()
        return config

    @staticmethod
    def _build_dataclass(cls_type, raw_dict: Dict):
        """Construct a dataclass from a dict, ignoring unknown keys.

        Args:
            cls_type: The dataclass type to instantiate.
            raw_dict: Dictionary of values from YAML.

        Returns:
            An instance of cls_type populated with values from raw_dict.
        """
        valid_keys = {f.name for f in dataclasses.fields(cls_type)}
        filtered = {k: v for k, v in raw_dict.items() if k in valid_keys}
        return cls_type(**filtered)

    @staticmethod
    def _build_ablation_config(raw_dict: Dict) -> AblationConfig:
        """Build AblationConfig including nested AblationStageConfig list.

        Args:
            raw_dict: Dictionary from YAML ablation section.

        Returns:
            Populated AblationConfig instance.
        """
        stages_raw = raw_dict.get("stages", [])
        stages = [
            Config._build_dataclass(AblationStageConfig, stage)
            for stage in stages_raw
        ]

        valid_keys = {f.name for f in dataclasses.fields(AblationConfig)}
        filtered = {
            k: v for k, v in raw_dict.items() if k in valid_keys and k != "stages"
        }
        ablation = AblationConfig(**filtered)
        ablation.stages = stages
        return ablation

    @staticmethod
    def _build_scaling_config(raw_dict: Dict) -> ScalingConfig:
        """Build ScalingConfig including nested ScalingModelConfig list.

        Args:
            raw_dict: Dictionary from YAML scaling section.

        Returns:
            Populated ScalingConfig instance.
        """
        models_raw = raw_dict.get("models", [])
        models = [
            Config._build_dataclass(ScalingModelConfig, model)
            for model in models_raw
        ]

        valid_keys = {f.name for f in dataclasses.fields(ScalingConfig)}
        filtered = {
            k: v for k, v in raw_dict.items() if k in valid_keys and k != "models"
        }
        scaling = ScalingConfig(**filtered)
        scaling.models = models
        return scaling

    def _validate(self) -> None:
        """Validate configuration consistency across all sub-configs.

        Raises:
            ValueError: If any validation check fails.
        """
        # --- FR-VAE validations ---
        expected_total_tokens = sum(s * s for s in self.frvae.scale_factors)
        if self.frvae.total_tokens != expected_total_tokens:
            raise ValueError(
                f"frvae.total_tokens={self.frvae.total_tokens} does not match "
                f"sum(s*s for s in scale_factors)={expected_total_tokens}. "
                f"scale_factors={self.frvae.scale_factors}"
            )

        if self.frvae.num_frequency_bands != len(self.frvae.scale_factors):
            raise ValueError(
                f"frvae.num_frequency_bands={self.frvae.num_frequency_bands} does not "
                f"match len(scale_factors)={len(self.frvae.scale_factors)}"
            )

        if self.frvae.codebook_dim != self.frvae.latent_channels:
            raise ValueError(
                f"frvae.codebook_dim={self.frvae.codebook_dim} must equal "
                f"frvae.latent_channels={self.frvae.latent_channels}"
            )

        if self.frvae.scale_factors[-1] != self.frvae.latent_spatial_size:
            raise ValueError(
                f"frvae.scale_factors[-1]={self.frvae.scale_factors[-1]} must equal "
                f"frvae.latent_spatial_size={self.frvae.latent_spatial_size}. "
                "The highest-frequency band must cover the full feature map resolution."
            )

        # --- NFIG Transformer validations ---
        nfig_expected_tokens = sum(s * s for s in self.nfig.scale_factors)
        if self.nfig.total_tokens != nfig_expected_tokens:
            raise ValueError(
                f"nfig.total_tokens={self.nfig.total_tokens} does not match "
                f"sum(s*s for s in scale_factors)={nfig_expected_tokens}. "
                f"scale_factors={self.nfig.scale_factors}"
            )

        if self.nfig.null_class_id != self.nfig.num_classes:
            raise ValueError(
                f"nfig.null_class_id={self.nfig.null_class_id} must equal "
                f"nfig.num_classes={self.nfig.num_classes} "
                "(null class is appended after the last real class)"
            )

        if self.nfig.num_generation_steps != self.nfig.num_frequency_bands:
            raise ValueError(
                f"nfig.num_generation_steps={self.nfig.num_generation_steps} must equal "
                f"nfig.num_frequency_bands={self.nfig.num_frequency_bands}"
            )

        if self.nfig.hidden_dim % self.nfig.num_heads != 0:
            raise ValueError(
                f"nfig.hidden_dim={self.nfig.hidden_dim} must be divisible by "
                f"nfig.num_heads={self.nfig.num_heads}"
            )

        # --- Cross-config consistency ---
        if self.nfig.scale_factors != self.frvae.scale_factors:
            raise ValueError(
                f"nfig.scale_factors={self.nfig.scale_factors} must match "
                f"frvae.scale_factors={self.frvae.scale_factors}. "
                "Both configs must agree on the frequency band scale sequence."
            )

        if self.nfig.codebook_size != self.frvae.codebook_size:
            raise ValueError(
                f"nfig.codebook_size={self.nfig.codebook_size} must equal "
                f"frvae.codebook_size={self.frvae.codebook_size}. "
                "The transformer vocabulary must match the FR-VAE codebook."
            )

    def to_dict(self) -> Dict:
        """Serialize the full configuration to a plain dictionary.

        Useful for logging to TensorBoard or saving alongside checkpoints
        to ensure full experiment reproducibility.

        Returns:
            Nested dictionary representation of all configuration values.
        """
        return dataclasses.asdict(self)
