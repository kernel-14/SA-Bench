import dataclasses
import yaml
from typing import Optional, Dict, Any


@dataclasses.dataclass
class GlobalConfig:
    seed: int
    output_dir: str
    image_res_high: int
    image_res_low: int
    latent_dim: int
    vae_downsample: int
    vae_path: str


@dataclasses.dataclass
class DataConfig:
    imagenet_root: str
    coco_root: str
    coco_ann_file: str
    clip_model: str


@dataclasses.dataclass
class DiffusionHead1Config:
    num_layers: int
    hidden_size: int


@dataclasses.dataclass
class DiffusionHead2Config:
    num_layers: int
    hidden_size: int
    num_heads: int


@dataclasses.dataclass
class ModelConfig:
    variant: str
    num_layers: int
    hidden_size: int
    mlp_ratio: int
    diffusion_head_1: DiffusionHead1Config
    diffusion_head_2: DiffusionHead2Config


@dataclasses.dataclass
class ImageNetTrainConfig:
    optimizer: str
    weight_decay: float
    beta1: float
    beta2: float
    learning_rate: float
    lr_schedule: str
    warmup_epochs: int
    epochs: int
    batch_size: Optional[int] = None
    ema_momentum: float = 0.9999
    mixed_precision: str = "bf16"
    gradient_clip: Optional[float] = None


@dataclasses.dataclass
class CocoTrainConfig:
    optimizer: str
    weight_decay: float
    beta1: float
    beta2: float
    learning_rate: float
    lr_schedule: str
    warmup_steps: int
    total_steps: Optional[int] = None
    batch_size: Optional[int] = None
    ema_momentum: float = 0.9999
    mixed_precision: str = "bf16"


@dataclasses.dataclass
class TrainingConfig:
    imagenet: ImageNetTrainConfig
    coco: CocoTrainConfig


@dataclasses.dataclass
class PhaseMaskConfig:
    ratio_distribution: str
    ratio_min: Optional[float] = None
    ratio_max: Optional[float] = None
    beta_alpha: Optional[float] = None
    beta_beta: Optional[float] = None
    schedule: str = "cosine"


@dataclasses.dataclass
class MaskConfig:
    phase1: PhaseMaskConfig
    phase2_imagenet: PhaseMaskConfig
    phase2_coco: PhaseMaskConfig


@dataclasses.dataclass
class InferencePhaseConfig:
    phase1_steps: int
    phase2_steps: int
    inner_diffusion_steps: int
    cfg_scale: float
    confidence_metric: str


@dataclasses.dataclass
class InferenceConfig:
    imagenet: InferencePhaseConfig
    coco: InferencePhaseConfig


@dataclasses.dataclass
class Config:
    global_config: GlobalConfig
    model: ModelConfig
    training: TrainingConfig
    masking: MaskConfig
    inference: InferenceConfig
    data: DataConfig


def _resolve_model_config(raw_model: Dict[str, Any]) -> ModelConfig:
    variant = raw_model["variant"]
    if variant not in ("base", "large", "huge"):
        raise ValueError(f"Unknown model variant: {variant}")

    variant_data = raw_model[variant]
    num_layers = variant_data["num_layers"]
    hidden_size = variant_data["hidden_size"]
    mlp_ratio = variant_data.get("mlp_ratio", 4)

    head1_data = raw_model["diffusion_head_1"][variant]
    head1 = DiffusionHead1Config(
        num_layers=head1_data["num_layers"],
        hidden_size=head1_data["hidden_size"],
    )

    head2_data = raw_model["diffusion_head_2"][variant]
    head2 = DiffusionHead2Config(
        num_layers=head2_data["num_layers"],
        hidden_size=head2_data["hidden_size"],
        num_heads=head2_data.get("num_heads", hidden_size // 64),
    )

    return ModelConfig(
        variant=variant,
        num_layers=num_layers,
        hidden_size=hidden_size,
        mlp_ratio=mlp_ratio,
        diffusion_head_1=head1,
        diffusion_head_2=head2,
    )


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Global
    global_cfg = GlobalConfig(
        seed=raw["global"]["seed"],
        output_dir=raw["global"]["output_dir"],
        image_res_high=raw["global"]["image_res_high"],
        image_res_low=raw["global"]["image_res_low"],
        latent_dim=raw["global"]["latent_dim"],
        vae_downsample=raw["global"]["vae_downsample"],
        vae_path=raw["global"]["vae_path"],
    )

    # Data
    data_cfg = DataConfig(
        imagenet_root=raw["data"]["imagenet_root"],
        coco_root=raw["data"]["coco_root"],
        coco_ann_file=raw["data"]["coco_ann_file"],
        clip_model=raw["data"]["clip_model"],
    )

    # Model (resolved)
    model_cfg = _resolve_model_config(raw["model"])

    # Training
    imgnet = raw["training"]["imagenet"]
    imagenet_train = ImageNetTrainConfig(
        optimizer=imgnet["optimizer"],
        weight_decay=imgnet["weight_decay"],
        beta1=imgnet["beta1"],
        beta2=imgnet["beta2"],
        learning_rate=imgnet["learning_rate"],
        lr_schedule=imgnet["lr_schedule"],
        warmup_epochs=imgnet["warmup_epochs"],
        epochs=imgnet["epochs"],
        batch_size=imgnet.get("batch_size", None),
        ema_momentum=imgnet.get("ema_momentum", 0.9999),
        mixed_precision=imgnet.get("mixed_precision", "bf16"),
        gradient_clip=imgnet.get("gradient_clip", None),
    )

    c = raw["training"]["coco"]
    coco_train = CocoTrainConfig(
        optimizer=c["optimizer"],
        weight_decay=c["weight_decay"],
        beta1=c["beta1"],
        beta2=c.get("beta2", 0.95),   # default same as imagenet if missing
        learning_rate=c["learning_rate"],
        lr_schedule=c["lr_schedule"],
        warmup_steps=c["warmup_steps"],
        total_steps=c.get("total_steps", None),
        batch_size=c.get("batch_size", None),
        ema_momentum=c.get("ema_momentum", 0.9999),
        mixed_precision=c.get("mixed_precision", "bf16"),
    )

    training_cfg = TrainingConfig(
        imagenet=imagenet_train,
        coco=coco_train,
    )

    # Masking
    def _parse_phase_mask(d: Dict[str, Any]) -> PhaseMaskConfig:
        return PhaseMaskConfig(
            ratio_distribution=d["ratio_distribution"],
            ratio_min=d.get("ratio_min", None),
            ratio_max=d.get("ratio_max", None),
            beta_alpha=d.get("beta_alpha", None),
            beta_beta=d.get("beta_beta", None),
            schedule=d.get("schedule", "cosine"),
        )

    phase1 = _parse_phase_mask(raw["masking"]["phase1"])
    phase2_imagenet = _parse_phase_mask(raw["masking"]["imagenet_phase2"])
    phase2_coco = _parse_phase_mask(raw["masking"]["coco_phase2"])

    mask_cfg = MaskConfig(
        phase1=phase1,
        phase2_imagenet=phase2_imagenet,
        phase2_coco=phase2_coco,
    )

    # Inference
    def _parse_inference_phase(d: Dict[str, Any]) -> InferencePhaseConfig:
        return InferencePhaseConfig(
            phase1_steps=d["phase1_steps"],
            phase2_steps=d["phase2_steps"],
            inner_diffusion_steps=d["inner_diffusion_steps"],
            cfg_scale=d["cfg_scale"],
            confidence_metric=d["confidence_metric"],
        )

    inf_imagenet = _parse_inference_phase(raw["inference"]["imagenet"])
    inf_coco = _parse_inference_phase(raw["inference"]["coco"])

    inference_cfg = InferenceConfig(
        imagenet=inf_imagenet,
        coco=inf_coco,
    )

    return Config(
        global_config=global_cfg,
        model=model_cfg,
        training=training_cfg,
        masking=mask_cfg,
        inference=inference_cfg,
        data=data_cfg,
    )
