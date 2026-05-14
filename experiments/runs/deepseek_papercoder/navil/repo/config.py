# config.py

"""
Central configuration module for NaViL reproduction.

Defines a strict hierarchy of Python dataclasses that mirror the structure of
the provided config.yaml.  Provides a single ``load_config`` entry point that
reads a YAML file, layers the built‑in defaults, applies 2B‑to‑9B variant
overrides, performs lightweight validation, and returns an immutable‑like
``Config`` object used throughout the pipeline.

All default values are taken directly from the NaViL paper Tables 6–8 and the
architectural descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import omegaconf


# ---------------------------------------------------------------------------
# Nested configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MultiScaleConfig:
    """Configuration for visual multi‑scale packing."""

    enabled: bool = True
    tau: float = 0.7071  # sqrt(2) / 2  – down‑sampling rate
    min_scale_area: int = 256  # stop when image area falls below this threshold (pixels²)


@dataclass
class SpecialTokensConfig:
    """Special tokens inserted into the multimodal token sequence."""

    begin_of_image: str = "<begin_of_image>"
    end_of_image: str = "<end_of_image>"
    end_of_line: str = "<end_of_line>"
    end_of_scale: str = "<end_of_scale>"


@dataclass
class VisualEncoderConfig:
    """Architecture of the visual encoder (Vision Transformer with bidirectional attention)."""

    depth: int = 24  # number of transformer layers (2B default)
    width: int = 1472  # hidden dimension
    mlp_width: int = 5888  # FFN intermediate dimension
    num_attention_heads: int = 23
    patch_size: int = 16
    rope_type: str = "2d"  # 2D rotary position embeddings


@dataclass
class ConnectorConfig:
    """Connector between visual encoder and LLM (pixel shuffle + MLP projection)."""

    pixel_shuffle_ratio: int = 2  # spatial down‑sampling factor in each dimension
    mlp_hidden_dim: Optional[int] = None  # resolved at runtime to LLM hidden size


@dataclass
class LLMConfig:
    """Base language model with modality‑specific MoE extensions."""

    base_model: str = "internlm2-1_8b"  # HuggingFace identifier or local checkpoint path
    num_experts: int = 2  # per modality (visual + linguistic)
    attention_experts: bool = True  # MHA‑MMoE enabled
    ffn_experts: bool = True  # FFN‑MMoE enabled
    activation_per_expert: int = 1  # top‑1 gating


@dataclass
class ModelConfig:
    """Top‑level model configuration, combining all architectural components."""

    variant: str = "2B"
    visual_encoder: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    multi_scale: MultiScaleConfig = field(default_factory=MultiScaleConfig)
    special_tokens: SpecialTokensConfig = field(default_factory=SpecialTokensConfig)


@dataclass
class DataStagePaths:
    """Paths for S1.1 data – 500M noisy image‑text pairs."""

    raw_image_text: List[str] = field(default_factory=list)
    synthetic_captions: str = ""
    total_size: int = 500_000_000


@dataclass
class DataStagePathsHighQual:
    """Paths for S1.2 data – 185M high‑quality multimodal + language data."""

    multimodal: str = ""
    pure_language: str = ""
    total_size: int = 185_000_000


@dataclass
class SFTDataPaths:
    """Paths for S2 supervised fine‑tuning data – 68M instructions."""

    sft_data: str = ""
    total_size: int = 68_000_000


@dataclass
class DataConfig:
    """Data loading configuration shared across all stages."""

    s1_1: DataStagePaths = field(default_factory=DataStagePaths)
    s1_2: DataStagePathsHighQual = field(default_factory=DataStagePathsHighQual)
    s2: SFTDataPaths = field(default_factory=SFTDataPaths)

    image_size: int = 512  # padded image size (multiple of 32)
    max_patches_per_sample: int = 4096  # default; overridden per stage
    batch_size: int = 1  # per‑device batch size
    num_workers: int = 8
    tokenizer_name: str = "internlm2-1_8b"  # matches base LLM; overridden for 9B


@dataclass
class StageConfig:
    """Hyperparameter set for a single training stage (S1.1, S1.2, S2)."""

    description: str = ""
    data_source: str = "s1_1"  # reference to DataConfig block ("s1_1", "s1_2", "s2")
    max_patches: int = 4096
    steps: int = 70000
    global_batch_size: int = 7000
    weight_decay: float = 0.05
    learning_rate: float = 5.0e-5
    lr_schedule: str = "constant_with_warmup"  # or "cosine_decay"
    warmup_steps: int = 200
    freeze_pattern: List[str] = field(
        default_factory=list
    )  # substrings of parameter names to freeze
    multi_scale: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration (AdamW)."""

    name: str = "AdamW"
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1.0e-8


def _default_stages() -> Dict[str, StageConfig]:
    """Return default training stage configurations for the 2B model."""
    return {
        "s1_1": StageConfig(
            description="Multi-modal generative pre-training (vision‑only trainable)",
            data_source="s1_1",
            max_patches=4096,
            steps=70000,
            global_batch_size=7000,
            weight_decay=0.05,
            learning_rate=5.0e-5,
            lr_schedule="constant_with_warmup",
            warmup_steps=200,
            freeze_pattern=["linguistic"],
            multi_scale=True,
        ),
        "s1_2": StageConfig(
            description="High‑quality alignment (unfreeze attention)",
            data_source="s1_2",
            max_patches=12188,
            steps=40000,
            global_batch_size=4614,
            weight_decay=0.1,
            learning_rate=5.0e-5,
            lr_schedule="constant_with_warmup",
            warmup_steps=200,
            freeze_pattern=["linguistic.ffn"],
            multi_scale=True,
        ),
        "s2": StageConfig(
            description="Supervised fine‑tuning (all parameters)",
            data_source="s2",
            max_patches=24576,
            steps=30000,
            global_batch_size=2234,
            weight_decay=0.01,
            learning_rate=2.0e-5,
            lr_schedule="cosine_decay",
            warmup_steps=200,
            freeze_pattern=[],
            multi_scale=True,
        ),
    }


@dataclass
class TrainingConfig:
    """Overall training configuration, including stage‑specific settings."""

    seed: int = 42
    precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    max_seq_length: int = 16384
    stages: Dict[str, StageConfig] = field(default_factory=_default_stages)


@dataclass
class EvalConfig:
    """Evaluation configuration – benchmarks and output settings."""

    benchmarks: List[str] = field(
        default_factory=lambda: [
            "MMVet",
            "MMMU_val",
            "MMBench_EN_test",
            "MME",
            "MathVista_MINI",
            "OCRBench",
            "CCBench",
            "TextVQA_val",
            "ScienceQA_IMG_test",
            "GQA_testdev",
            "DocVQA_test",
            "AI2D_test",
            "ChartQA_test",
            "InfographicVQA_test",
        ]
    )
    batch_size: int = 1
    output_dir: str = "./eval_results"


@dataclass
class Config:
    """Root configuration aggregating all sub‑configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> Config:
    """
    Load a YAML configuration file, layer defaults from the dataclass schemas,
    apply variant‑specific overrides (2B → 9B), and return a validated Config.

    Args:
        config_path: Path to the YAML configuration file (e.g., ``config.yaml``).

    Returns:
        Config: A fully populated configuration object that should be treated
        as immutable after loading.
    """
    # 1. Build default structured config from the Config dataclass.
    schema = omegaconf.OmegaConf.structured(Config)

    # 2. Load the user‑supplied YAML.
    user_cfg = omegaconf.OmegaConf.load(config_path)

    # 3. Merge user overrides on top of defaults.
    #    Temporarily allow extra keys (e.g., model_9B) in the schema.
    omegaconf.OmegaConf.set_struct(schema, False)
    merged_cfg = omegaconf.OmegaConf.merge(schema, user_cfg)

    # 4. Variant override: if the requested variant is "9B", apply the
    #    model_9B block to model and training sections, then discard it.
    variant = merged_cfg.model.variant.lower()
    if variant == "9b" and "model_9B" in merged_cfg:
        override = merged_cfg.model_9B

        # --- Model component overrides ---
        if "visual_encoder" in override:
            merged_cfg.model.visual_encoder = omegaconf.OmegaConf.merge(
                merged_cfg.model.visual_encoder, override.visual_encoder
            )
        if "llm" in override:
            merged_cfg.model.llm = omegaconf.OmegaConf.merge(
                merged_cfg.model.llm, override.llm
            )

        # Update tokenizer name – the 9B variant uses Qwen3‑8B's tokenizer.
        if "tokenizer_name" in override:
            merged_cfg.data.tokenizer_name = override.tokenizer_name
        else:
            merged_cfg.data.tokenizer_name = "qwen3-8b"

        # --- Training stage overrides ---
        if "training" in override and "stages" in override.training:
            for stage_name, stage_cfg in override.training.stages.items():
                if stage_name in merged_cfg.training.stages:
                    merged_cfg.training.stages[stage_name] = omegaconf.OmegaConf.merge(
                        merged_cfg.training.stages[stage_name], stage_cfg
                    )

        # Remove the temporary section so it does not pollute the final object.
        del merged_cfg["model_9B"]

    # 5. Ensure tokenizer_name is never empty; default to LLM's base model.
    if not merged_cfg.data.tokenizer_name:
        merged_cfg.data.tokenizer_name = merged_cfg.model.llm.base_model

    # 6. Convert the final OmegaConf DictConfig to a real dataclass instance.
    config: Config = omegaconf.OmegaConf.to_object(merged_cfg)

    # 7. Post‑processing: if global multi_scale is disabled, force it off for
    #    every stage (this reinforces the invariant).
    if not config.model.multi_scale.enabled:
        for stage in config.training.stages.values():
            stage.multi_scale = False

    return config
