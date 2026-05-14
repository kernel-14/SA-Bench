## config.py

import os
import copy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union, get_type_hints

import yaml


# ---------------------------------------------------------------------------
# Helper: recursively construct dataclass instances from dictionaries
# ---------------------------------------------------------------------------

def _dict_to_dataclass(cls, data: Any):
    """Recursively convert a dictionary (or list, scalar) to the given dataclass type.

    Handles Optional, List, and nested dataclasses.
    """
    if data is None:
        return None

    # If cls is a generic alias (e.g. Optional[SomeClass]), extract the origin and args.
    origin = getattr(cls, "__origin__", None)
    if origin is Union:
        # For Optional[X] (Union[X, None]), try to match non-None type.
        # We assume at most one non-None type and that data is not None.
        args = cls.__args__
        for arg in args:
            if arg is type(None):
                continue
            return _dict_to_dataclass(arg, data)
        raise TypeError(f"Cannot convert {data} to {cls}")

    if origin is list or origin is List:
        args = cls.__args__
        item_cls = args[0] if args else Any
        return [_dict_to_dataclass(item_cls, item) for item in data]

    if not isinstance(data, dict):
        # primitive value
        return data

    # We have a concrete dataclass (e.g., Config, DataConfig, ...)
    if not hasattr(cls, "__dataclass_fields__"):
        # Possibly a plain dict type passed as a field – remain as dict
        return data

    field_types = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        key = f.name
        if key in data:
            value = data[key]
            f_type = field_types.get(key, Any)
            kwargs[key] = _dict_to_dataclass(f_type, value)
        elif f.default is not field.MISSING:
            kwargs[key] = f.default
        elif f.default_factory is not field.MISSING:
            kwargs[key] = f.default_factory()
        # else: field is required, let __init__ raise.
    return cls(**kwargs)


def _apply_overrides(data: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge command-line overrides (dot-separated keys) into the parsed YAML dict."""
    for key, value in overrides.items():
        parts = key.split(".")
        d = data
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return data


# ---------------------------------------------------------------------------
# Sub-configuration dataclasses (all frozen, immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    dataset: str            # "internvid", "ucf101", "sky_timelapse"
    data_root: str
    resolution: int       # spatial resolution (must be divisible by 8)
    num_frames_per_sample: Optional[int] = None

    @property
    def latent_size(self) -> int:
        return self.resolution // 8


@dataclass(frozen=True)
class VideoConfig:
    chunk_size: int           # l
    max_prefix: int           # P_max
    train_max_len: int        # L_train
    first_stage_max_len: int  # used only for T2V stage1
    p_prime: int = 3


@dataclass(frozen=True)
class TransformerConfig:
    hidden_dim: int = 1152
    num_heads: int = 16
    num_layers: int = 28
    mlp_ratio: float = 4.0


@dataclass(frozen=True)
class ModelConfig:
    vae_model: str            # e.g., "stabilityai/sd-vae-ft-mse"
    text_encoder: str         # e.g., "google/t5-v1_1-xxl"
    dtype: str                # "bfloat16", "float16", "float32"
    transformer: TransformerConfig


@dataclass(frozen=True)
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "linear"


@dataclass(frozen=True)
class Stage1Config:
    enabled: bool
    batch_size: int
    steps: int
    num_frames: int
    use_prefix: bool = False


@dataclass(frozen=True)
class Stage2Config:
    enabled: bool
    batch_size: int
    steps: int
    prefix_length_sampler: str = "random_multiples"    # "random_multiples" or "fixed"
    loss_vlb_weight: float = 0.001


@dataclass(frozen=True)
class VideoPredictionConfig:
    batch_size: int
    steps: int
    chunk_size: int
    max_prefix: int
    train_max_len: int


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str = "adamw"
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    mixed_precision: str = "bf16"
    seed: int = 42
    stage1: Optional[Stage1Config] = None
    stage2: Optional[Stage2Config] = None
    video_prediction: Optional[VideoPredictionConfig] = None


@dataclass(frozen=True)
class FVDConfig:
    i3d_model_path: Optional[str] = None
    num_generated_videos: int = 2048
    chunk_size: int = 16
    chunks_per_video: int = 3


@dataclass(frozen=True)
class EvaluationConfig:
    fvd: FVDConfig
    metrics: List[str] = field(default_factory=lambda: ["fvd"])


@dataclass(frozen=True)
class InferenceConfig:
    denoising_steps: int = 100
    guidance_scale: float = 7.5
    generate_frames: int = 80
    initial_frame_source: str = "first"   # "first" or "given_image"


@dataclass(frozen=True)
class SystemConfig:
    device: str = "cuda"
    num_gpus: int = 1
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    task: str                      # "t2v" or "video_prediction"
    model_variant: str             # "ca2", "os_fix", "os_ext"
    data: DataConfig
    video: VideoConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    training: TrainingConfig
    inference: InferenceConfig
    evaluation: EvaluationConfig
    system: SystemConfig
    seed: int = 42                 # global random seed

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "Config":
        """Load a Config from a YAML file, possibly overriding leaf values."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # 1. apply dot‑seperated overrides
        raw = _apply_overrides(raw, overrides)

        # 2. extract seed from training (if not present at top level)
        if "seed" not in raw:
            raw["seed"] = raw.get("training", {}).get("seed", 42)
        # ensure training.seed equals config.seed
        raw.setdefault("training", {})
        raw["training"]["seed"] = raw["seed"]

        # 3. derive video parameters according to task and variant
        raw = cls._derive_video_params(raw)

        # 4. recursively instantiate dataclasses
        config = _dict_to_dataclass(cls, raw)

        # 5. validate cross‑field consistency
        config._validate()

        return config

    @staticmethod
    def _derive_video_params(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust the video section based on task and model_variant to ensure consistency."""
        task = raw.get("task", "t2v")
        variant = raw.get("model_variant", "ca2")
        video = raw.setdefault("video", {})

        # chunk_size if not provided defaults:
        if task == "video_prediction":
            chunk_size = video.get("chunk_size", 8)
        else:
            chunk_size = video.get("chunk_size", 16)

        video["chunk_size"] = chunk_size

        if variant in ("ca2", "os_ext"):
            max_prefix = 1 + 3 * chunk_size
            train_max_len = max_prefix + chunk_size
            first_stage_max_len = video.get("first_stage_max_len", 32)  # T2V default
        else:  # os_fix
            # OS‑Fix uses fixed prefix length equal to chunk_size (half of clip length)
            max_prefix = chunk_size
            train_max_len = 2 * chunk_size
            first_stage_max_len = 0   # not used

        video["max_prefix"] = max_prefix
        video["train_max_len"] = train_max_len
        video["first_stage_max_len"] = first_stage_max_len
        video.setdefault("p_prime", 3)

        # Also ensure training sub‑configs are consistent with task
        training = raw.setdefault("training", {})
        if task == "video_prediction":
            # Use video_prediction sub‑config if present; otherwise build from video
            if "video_prediction" not in training:
                training["video_prediction"] = {
                    "batch_size": training.get("batch_size", 8),
                    "steps": training.get("steps", 11000),
                    "chunk_size": chunk_size,
                    "max_prefix": max_prefix,
                    "train_max_len": train_max_len,
                }
            # disable stage1/stage2
            training.pop("stage1", None)
            training.pop("stage2", None)
        else:
            # T2V: ensure stage1/stage2 exist
            if "stage1" not in training:
                training["stage1"] = {
                    "enabled": True,
                    "batch_size": 288,
                    "steps": 32000,
                    "num_frames": 32,
                    "use_prefix": False,
                }
            if "stage2" not in training:
                training["stage2"] = {
                    "enabled": True,
                    "batch_size": 144,
                    "steps": 21000,
                    "prefix_length_sampler": "random_multiples",
                    "loss_vlb_weight": 0.001,
                }
            training.pop("video_prediction", None)

        return raw

    def _validate(self):
        """Run consistency checks on the assembled configuration."""
        # task
        if self.task not in ("t2v", "video_prediction"):
            raise ValueError(f"task must be 't2v' or 'video_prediction', got '{self.task}'")
        # model_variant
        if self.model_variant not in ("ca2", "os_fix", "os_ext"):
            raise ValueError(f"model_variant must be 'ca2', 'os_fix', or 'os_ext', got '{self.model_variant}'")

        # video consistency
        if self.model_variant in ("ca2", "os_ext"):
            expected_max_prefix = 1 + 3 * self.video.chunk_size
            expected_train_max_len = expected_max_prefix + self.video.chunk_size
            if self.video.max_prefix != expected_max_prefix:
                raise ValueError(
                    f"For variant '{self.model_variant}', max_prefix should be {expected_max_prefix} "
                    f"(got {self.video.max_prefix})"
                )
            if self.video.train_max_len != expected_train_max_len:
                raise ValueError(
                    f"For variant '{self.model_variant}', train_max_len should be {expected_train_max_len} "
                    f"(got {self.video.train_max_len})"
                )
        else:  # os_fix
            if self.video.max_prefix != self.video.chunk_size:
                raise ValueError(
                    f"For os_fix, max_prefix must equal chunk_size (got max_prefix={self.video.max_prefix}, "
                    f"chunk_size={self.video.chunk_size})"
                )
            if self.video.train_max_len != 2 * self.video.chunk_size:
                raise ValueError(
                    f"For os_fix, train_max_len must equal 2*chunk_size "
                    f"(got {self.video.train_max_len}, expected {2*self.video.chunk_size})"
                )

        # training stages
        if self.task == "t2v":
            if self.training.stage1 is None or self.training.stage2 is None:
                raise ValueError("For T2V task, stage1 and stage2 must be defined in training config.")
            if self.training.video_prediction is not None:
                raise ValueError("For T2V task, video_prediction sub‑config must be absent.")
            # stage1 frames should be <= train_max_len
            if self.training.stage1.num_frames > self.video.train_max_len:
                raise ValueError(
                    f"Stage1 num_frames ({self.training.stage1.num_frames}) exceeds train_max_len ({self.video.train_max_len})"
                )
        else:  # video_prediction
            if self.training.video_prediction is None:
                raise ValueError("For video_prediction task, video_prediction sub‑config must be defined.")
            if self.training.stage1 is not None or self.training.stage2 is not None:
                raise ValueError("For video_prediction task, stage1/stage2 must be absent.")

        # latent_size consistency
        if self.data.resolution % 8 != 0:
            raise ValueError(f"Resolution must be divisible by 8 (got {self.data.resolution})")
        expected_latent = self.data.resolution // 8
        # (We only have a property; no stored field to check)

        # p_prime must be positive and less than chunk_size
        if self.video.p_prime <= 0:
            raise ValueError(f"p_prime must be positive (got {self.video.p_prime})")
        if self.video.p_prime > self.video.chunk_size:
            raise ValueError(
                f"p_prime ({self.video.p_prime}) cannot exceed chunk_size ({self.video.chunk_size})"
            )

        # guidance_scale is only meaningful for T2V
        if self.task == "video_prediction" and self.inference.guidance_scale != 1.0:
            # We could force to 1.0, but here just warn
            pass

        # seed alignment
        if self.seed != self.training.seed:
            raise ValueError("Config.seed and TrainingConfig.seed must be equal.")

        # dtype check
        if self.model.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Unsupported dtype '{self.model.dtype}'")

        # check that log_dir and checkpoint_dir exist or can be created later (no immediate check)

        # All good
        return True
