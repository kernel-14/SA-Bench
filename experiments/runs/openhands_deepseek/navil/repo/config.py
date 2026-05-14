"""Configuration classes for NaViL model and training."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class VisualEncoderConfig:
    """Visual encoder architecture config.

    Default: NaViL-2B optimal encoder (600M params).
    Approx param count: 12 * depth * width^2.
    """
    depth: int = 24
    width: int = 1472
    mlp_width: int = 5888
    n_heads: int = 23
    patch_size: int = 16
    max_image_size: int = 4096
    dropout: float = 0.0


@dataclass
class LLMConfig:
    """MoE-extended LLM config.

    Default: NaViL-2B (1.8B activated params).
    """
    depth: int = 24
    dim: int = 2048
    mlp_dim: int = 8192
    n_heads: int = 16
    vocab_size: int = 92544
    max_seq_len: int = 16384
    num_experts: int = 2
    dropout: float = 0.0


@dataclass
class ConnectorConfig:
    visual_dim: int = 1472
    llm_dim: int = 2048
    pixel_shuffle_scale: int = 2
    mlp_hidden_mult: int = 4


@dataclass
class NaViLConfig:
    """Full NaViL model configuration."""
    visual_encoder: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    use_moe: bool = True
    special_tokens: dict = field(default_factory=lambda: {
        "begin_of_image": "<begin_of_image>",
        "end_of_image": "<end_of_image>",
        "end_of_line": "<end_of_line>",
        "end_of_scale": "<end_of_scale>",
    })

    @property
    def total_params(self) -> int:
        enc_cfg = self.visual_encoder
        llm_cfg = self.llm
        enc_params = 12 * enc_cfg.depth * enc_cfg.width ** 2
        llm_params = 12 * llm_cfg.depth * llm_cfg.dim ** 2
        if self.use_moe:
            llm_params *= 2
        return enc_params + llm_params


@dataclass
class StageConfig:
    """Training stage hyperparameters."""
    max_image_patches: int = 4096
    steps: int = 70000
    global_batch_size: int = 7000
    weight_decay: float = 0.05
    peak_lr: float = 5e-5
    lr_schedule: str = "constant_with_warmup"
    visual_multiscale_packing: bool = True
    llm_max_seq_len: int = 16384
    warmup_steps: int = 200
    gradient_accumulation: int = 1

    freeze_text_params: bool = False
    freeze_attention_text: bool = False
    freeze_ffn_text: bool = False


@dataclass
class TrainingConfig:
    """Full training recipe configuration."""

    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    precision: str = "bfloat16"

    stage1_1: StageConfig = field(default_factory=lambda: StageConfig(
        max_image_patches=4096,
        steps=70000,
        global_batch_size=7000,
        weight_decay=0.05,
        peak_lr=5e-5,
        lr_schedule="constant_with_warmup",
        visual_multiscale_packing=True,
        llm_max_seq_len=16384,
        warmup_steps=200,
        gradient_accumulation=1,
        freeze_text_params=True,
    ))

    stage1_2: StageConfig = field(default_factory=lambda: StageConfig(
        max_image_patches=12188,
        steps=40000,
        global_batch_size=4614,
        weight_decay=0.1,
        peak_lr=5e-5,
        lr_schedule="constant_with_warmup",
        visual_multiscale_packing=True,
        llm_max_seq_len=16384,
        warmup_steps=200,
        gradient_accumulation=1,
        freeze_ffn_text=True,
    ))

    stage2: StageConfig = field(default_factory=lambda: StageConfig(
        max_image_patches=24576,
        steps=30000,
        global_batch_size=2234,
        weight_decay=0.01,
        peak_lr=2e-5,
        lr_schedule="cosine_decay",
        visual_multiscale_packing=True,
        llm_max_seq_len=16384,
        warmup_steps=200,
        gradient_accumulation=1,
    ))

    multiscale_downsample_rate: float = 0.7071067811865476  # sqrt(2)/2

    pretrain_data_size: int = 500_000_000
    high_quality_data_size: int = 185_000_000
    sft_data_size: int = 68_000_000


NAVIL_2B_CONFIG = NaViLConfig()
NAVIL_9B_CONFIG = NaViLConfig(
    visual_encoder=VisualEncoderConfig(
        depth=32,
        width=1792,
        mlp_width=7168,
        n_heads=28,
    ),
    llm=LLMConfig(
        depth=36,
        dim=4096,
        mlp_dim=12288,
        n_heads=32,
    ),
    connector=ConnectorConfig(
        visual_dim=1792,
        llm_dim=4096,
    ),
)

NAVIL_2B_TRAINING = TrainingConfig()
NAVIL_9B_TRAINING = TrainingConfig(
    stage1_1=StageConfig(
        max_image_patches=4096,
        steps=50000,
        global_batch_size=10300,
        weight_decay=0.05,
        peak_lr=5e-5,
        lr_schedule="constant_with_warmup",
        visual_multiscale_packing=False,
        freeze_text_params=True,
    ),
    stage1_2=StageConfig(
        max_image_patches=12188,
        steps=33000,
        global_batch_size=1792,
        weight_decay=0.1,
        peak_lr=5e-5,
        lr_schedule="constant_with_warmup",
        visual_multiscale_packing=True,
        freeze_ffn_text=True,
    ),
    stage2=StageConfig(
        max_image_patches=24576,
        steps=6000,
        global_batch_size=3520,
        weight_decay=0.01,
        peak_lr=2e-5,
        lr_schedule="cosine_decay",
        visual_multiscale_packing=True,
    ),
)
