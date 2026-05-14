from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import math


@dataclass
class VisualEncoderConfig:
    depth: int = 24
    width: int = 1472
    mlp_width: int = 5888
    num_heads: int = 23
    patch_size: int = 16
    image_size: int = 448
    dropout: float = 0.0
    use_2d_rope: bool = True
    # Parameter count ≈ 12 * depth * width^2
    # NaViL-2B: 0.6B  (depth=24, width=1472)
    # NaViL-9B: 1.2B  (depth=32, width=1792)


@dataclass
class LLMConfig:
    depth: int = 24
    width: int = 2048
    mlp_width: int = 8192
    num_heads: int = 16
    num_kv_heads: int = 8          # GQA, matches InternLM2-1.8B
    vocab_size: int = 92544        # InternLM2 tokenizer
    max_seq_len: int = 16384
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = False
    # NaViL-2B: 1.8B activated (depth=24, width=2048)
    # NaViL-9B: 8.0B activated (depth=36, width=4096)


@dataclass
class MoEConfig:
    num_experts: int = 2           # visual + linguistic
    num_activated_experts: int = 1 # modality-based selection, no routing
    # Modality indices
    VISUAL_EXPERT: int = 0
    LINGUISTIC_EXPERT: int = 1


@dataclass
class ConnectorConfig:
    pixel_shuffle_factor: int = 2  # downsampling factor for pixel shuffle
    input_dim: int = 1472          # visual encoder width
    output_dim: int = 2048         # LLM width


@dataclass
class NaViLConfig:
    visual_encoder: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)

    # Visual multi-scale packing
    multiscale_downsample_rate: float = math.sqrt(2) / 2   # τ = √2/2
    multiscale_min_area: int = 32 * 32                      # stop downsampling threshold
    max_image_patches: int = 4096

    # Special token IDs (set after tokenizer initialization)
    begin_of_image_token_id: int = -1
    end_of_image_token_id: int = -1
    end_of_line_token_id: int = -1
    end_of_scale_token_id: int = -1
    image_patch_token_id: int = -1

    model_name: str = "navil_2b"
    llm_pretrained: str = "internlm/internlm2-1_8b"


# ── Preset configurations ──────────────────────────────────────────────────────

def get_navil_2b_config() -> NaViLConfig:
    """NaViL-2B: 0.6B visual encoder + 1.8B LLM (InternLM2-1.8B base)."""
    visual = VisualEncoderConfig(
        depth=24,
        width=1472,
        mlp_width=5888,
        num_heads=23,
        patch_size=16,
    )
    llm = LLMConfig(
        depth=24,
        width=2048,
        mlp_width=8192,
        num_heads=16,
        num_kv_heads=8,
        vocab_size=92544,
        max_seq_len=16384,
        rope_theta=1000000.0,
        rms_norm_eps=1e-5,
    )
    connector = ConnectorConfig(
        pixel_shuffle_factor=2,
        input_dim=1472,
        output_dim=2048,
    )
    return NaViLConfig(
        visual_encoder=visual,
        llm=llm,
        connector=connector,
        model_name="navil_2b",
        llm_pretrained="internlm/internlm2-1_8b",
    )


def get_navil_9b_config() -> NaViLConfig:
    """NaViL-9B: 1.2B visual encoder + 8.0B LLM (Qwen3-8B base)."""
    visual = VisualEncoderConfig(
        depth=32,
        width=1792,
        mlp_width=7168,
        num_heads=28,
        patch_size=16,
    )
    llm = LLMConfig(
        depth=36,
        width=4096,
        mlp_width=12288,
        num_heads=32,
        num_kv_heads=8,
        vocab_size=151936,   # Qwen3 tokenizer
        max_seq_len=16384,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
    )
    connector = ConnectorConfig(
        pixel_shuffle_factor=2,
        input_dim=1792,
        output_dim=4096,
    )
    return NaViLConfig(
        visual_encoder=visual,
        llm=llm,
        connector=connector,
        model_name="navil_9b",
        llm_pretrained="Qwen/Qwen3-8B",
    )


# ── Training hyperparameters ───────────────────────────────────────────────────

@dataclass
class TrainingStageConfig:
    name: str = "stage1_1"
    max_steps: int = 70000
    global_batch_size: int = 7000
    peak_lr: float = 5e-5
    weight_decay: float = 0.05
    warmup_steps: int = 200
    lr_schedule: str = "constant_with_warmup"   # or "cosine"
    max_seq_len: int = 16384
    max_image_patches: int = 4096
    multiscale_packing: bool = True
    # Frozen / trainable parameter groups
    freeze_llm_text: bool = True       # Stage 1.1: freeze LLM text params
    freeze_llm_attn: bool = True       # Stage 1.1: also freeze attention text
    freeze_visual_encoder: bool = False
    freeze_connector: bool = False
    freeze_moe_visual: bool = False
    freeze_moe_linguistic: bool = True  # Stage 1.1: linguistic MoE frozen


@dataclass
class TrainingConfig:
    model_config_name: str = "navil_2b"
    output_dir: str = "./checkpoints"
    seed: int = 42
    num_workers: int = 8
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 50
    deepspeed_config: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    # Stage definitions
    stages: List[TrainingStageConfig] = field(default_factory=lambda: [
        # Stage 1.1: 500M image-text pairs, freeze text params
        TrainingStageConfig(
            name="stage1_1",
            max_steps=70000,
            global_batch_size=7000,
            peak_lr=5e-5,
            weight_decay=0.05,
            warmup_steps=200,
            lr_schedule="constant_with_warmup",
            max_image_patches=4096,
            multiscale_packing=True,
            freeze_llm_text=True,
            freeze_llm_attn=True,
            freeze_visual_encoder=False,
            freeze_connector=False,
            freeze_moe_visual=False,
            freeze_moe_linguistic=True,
        ),
        # Stage 1.2: 185M high-quality data, unfreeze attention text params
        TrainingStageConfig(
            name="stage1_2",
            max_steps=40000,
            global_batch_size=4614,
            peak_lr=2e-5,
            weight_decay=0.1,
            warmup_steps=200,
            lr_schedule="cosine",
            max_image_patches=4096,
            multiscale_packing=True,
            freeze_llm_text=True,       # freeze non-attention text params
            freeze_llm_attn=False,      # unfreeze attention text params
            freeze_visual_encoder=False,
            freeze_connector=False,
            freeze_moe_visual=False,
            freeze_moe_linguistic=False,
        ),
        # Stage 2: 68M high-quality data, all params unfrozen
        TrainingStageConfig(
            name="stage2_sft",
            max_steps=30000,
            global_batch_size=2340,
            peak_lr=2e-5,
            weight_decay=0.01,
            warmup_steps=200,
            lr_schedule="cosine",
            max_image_patches=24576,    # higher resolution for SFT
            multiscale_packing=True,
            freeze_llm_text=False,
            freeze_llm_attn=False,
            freeze_visual_encoder=False,
            freeze_connector=False,
            freeze_moe_visual=False,
            freeze_moe_linguistic=False,
        ),
    ])


# ── NaViL-9B training config ───────────────────────────────────────────────────

def get_navil_9b_training_config() -> TrainingConfig:
    return TrainingConfig(
        model_config_name="navil_9b",
        stages=[
            TrainingStageConfig(
                name="stage1_1",
                max_steps=50000,
                global_batch_size=10300,
                peak_lr=5e-5,
                weight_decay=0.05,
                warmup_steps=200,
                lr_schedule="constant_with_warmup",
                max_image_patches=4096,
                multiscale_packing=False,   # disabled in 9B stage 1.1 for speed
                freeze_llm_text=True,
                freeze_llm_attn=True,
                freeze_visual_encoder=False,
                freeze_connector=False,
                freeze_moe_visual=False,
                freeze_moe_linguistic=True,
            ),
            TrainingStageConfig(
                name="stage1_2",
                max_steps=33000,
                global_batch_size=1792,
                peak_lr=5e-5,
                weight_decay=0.1,
                warmup_steps=200,
                lr_schedule="constant_with_warmup",
                max_image_patches=4096,
                multiscale_packing=True,
                freeze_llm_text=True,
                freeze_llm_attn=False,
                freeze_visual_encoder=False,
                freeze_connector=False,
                freeze_moe_visual=False,
                freeze_moe_linguistic=False,
            ),
            TrainingStageConfig(
                name="stage2_sft",
                max_steps=6000,
                global_batch_size=3520,
                peak_lr=2e-5,
                weight_decay=0.01,
                warmup_steps=200,
                lr_schedule="cosine",
                max_image_patches=24576,
                multiscale_packing=True,
                freeze_llm_text=False,
                freeze_llm_attn=False,
                freeze_visual_encoder=False,
                freeze_connector=False,
                freeze_moe_visual=False,
                freeze_moe_linguistic=False,
            ),
        ],
    )


# ── Ablation configurations (Sec. 3.2) ────────────────────────────────────────

# Visual encoder depth/width ablation (fixed 600M param budget, fixed 600M LLM)
VISUAL_ENCODER_ABLATIONS: List[Dict[str, Any]] = [
    {"depth": 3,  "width": 4096, "mlp_width": 16384, "num_heads": 32},
    {"depth": 6,  "width": 2880, "mlp_width": 11520, "num_heads": 24},
    {"depth": 12, "width": 2048, "mlp_width": 8192,  "num_heads": 16},
    {"depth": 24, "width": 1472, "mlp_width": 5888,  "num_heads": 23},  # optimal
    {"depth": 48, "width": 1024, "mlp_width": 4096,  "num_heads": 16},
]

# LLM scaling ablation (fixed 600M visual encoder)
LLM_SCALING_SIZES: List[str] = ["0.5B", "1.8B", "7B"]

# Visual encoder scaling ablation (fixed LLM)
VISUAL_ENCODER_SCALING_SIZES_M: List[int] = [75, 150, 300, 600, 1200, 2400]


# ── Evaluation benchmark configs ──────────────────────────────────────────────

BENCHMARK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mmvet": {
        "metric": "score",
        "split": "test",
        "requires_gpt_eval": True,
    },
    "mmmu": {
        "metric": "accuracy",
        "split": "val",
        "num_choices": 4,
    },
    "mmbench": {
        "metric": "accuracy",
        "split": "test",
        "language": "en",
    },
    "mme": {
        "metric": "score",
        "split": "test",
        "subscores": ["perception", "cognition"],
    },
    "mathvista": {
        "metric": "accuracy",
        "split": "mini",
    },
    "ocrbench": {
        "metric": "score",
        "split": "test",
    },
    "ccbench": {
        "metric": "accuracy",
        "split": "test",
    },
    "textvqa": {
        "metric": "accuracy",
        "split": "val",
    },
    "scienceqa": {
        "metric": "accuracy",
        "split": "test",
        "image_only": True,
    },
    "gqa": {
        "metric": "accuracy",
        "split": "testdev",
    },
    "docvqa": {
        "metric": "anls",
        "split": "test",
    },
    "ai2d": {
        "metric": "accuracy",
        "split": "test",
    },
    "chartqa": {
        "metric": "relaxed_accuracy",
        "split": "test",
    },
    "infovqa": {
        "metric": "anls",
        "split": "test",
    },
}
