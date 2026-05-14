from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class NoiseScheduleConfig:
    # Flow Matching reference flow: X_t = beta_t * X_0 + alpha_t * X_1
    # alpha_t = t, beta_t = 1 - t  (linear interpolation)
    alpha_type: str = "linear"   # "linear" -> alpha_t = t
    beta_type: str = "linear"    # "linear" -> beta_t = 1 - t

    # Diffusion model schedule (for DDIM/DDPM)
    # alpha_bar_t: cumulative product schedule
    diffusion_schedule: str = "cosine"  # "cosine" or "linear"
    diffusion_beta_start: float = 0.0001
    diffusion_beta_end: float = 0.02

    # Sampling noise schedule during inference
    # "memoryless" -> sigma(t) = sqrt(2*eta_t)
    # "zero"       -> sigma(t) = 0  (ODE / DDIM deterministic)
    # "ddpm"       -> sigma(t) = sqrt(alpha_bar_dot_t / alpha_bar_t)
    sampling_sigma_type: str = "zero"

    # Fine-tuning noise schedule (must be "memoryless" per Theorem 1)
    finetuning_sigma_type: str = "memoryless"

    # Small offset added to avoid division by zero in sigma(t) = sqrt(2*(1-t+h)/(t+h))
    # Paper uses h = 1/K where K = 40
    sigma_offset_h: float = 0.025  # = 1/40


@dataclass
class ModelConfig:
    # Latent diffusion / flow matching setup
    image_size: int = 512
    latent_size: int = 64          # 512 / 8 (VAE downsampling factor)
    latent_channels: int = 4
    model_channels: int = 320
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (4, 2, 1)
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_heads: int = 8
    context_dim: int = 768         # text embedding dimension (CLIP)
    use_spatial_transformer: bool = True
    transformer_depth: int = 1

    # Classifier-free guidance
    # v(x, t | y, w) = (1+w)*v(x,t|y) - w*v(x,t)
    cfg_scale: float = 1.0         # w=0 means no guidance


@dataclass
class TrainingConfig:
    # Optimizer (Adam)
    learning_rate: float = 2e-5
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_norm_clip: float = 1.0

    # Batch / compute
    batch_size: int = 20           # per GPU
    num_gpus: int = 2
    effective_batch_size: int = 40  # = batch_size * num_gpus
    precision: str = "bfloat16"

    # Timesteps
    num_timesteps: int = 40        # K = 40 discretization steps
    # h = 1/K = 0.025

    # Fine-tuning iterations
    num_iterations: int = 1000     # default; varies per method

    # Reward scaling lambda (controls KL vs reward tradeoff)
    reward_lambda: float = 12500.0

    # Loss clipping threshold (LCT)
    # LCT = 1.6 * lambda^2 for Adjoint Matching
    # LCT = 1600 * lambda^2 for Continuous Adjoint
    lct_factor_adj_match: float = 1.6
    lct_factor_cont_adj: float = 1600.0

    # Gradient evaluation timesteps
    # Sample 10 uniformly from [0, 0.725] + always include last 10 steps [0.75, ..., 0.975]
    num_grad_timesteps_early: int = 10
    early_timestep_max: float = 0.725
    num_grad_timesteps_late: int = 10   # always include [0.75, ..., 0.975]

    # Seed
    seed: int = 42


@dataclass
class DataConfig:
    # Training prompts
    train_prompt_dataset: str = "licensed_text_image_pairs"
    num_train_prompts: int = 40000
    total_prompt_pool: int = 100000

    # Evaluation
    num_eval_prompts: int = 1000
    num_eval_runs: int = 3

    # Image generation for evaluation
    num_diversity_samples_per_prompt: int = 40
    num_diversity_prompts: int = 25


@dataclass
class RewardConfig:
    # Reward model: ImageReward (Xu et al., 2023)
    reward_model_name: str = "ImageReward-v1.0"
    reward_lambda: float = 12500.0  # scaling factor

    # For DPO: preference data
    dpo_beta: float = 5000.0       # beta_tilde in paper (range [4000, 10000])


@dataclass
class EvalConfig:
    # Metrics
    clip_model: str = "ViT-H-14"
    clip_pretrained: str = "laion2b_s32b_b79k"
    pickscore_model: str = "yuvalkirstain/PickScore_v1"
    hps_model: str = "ViT-H-14"
    dreamsim_model: str = "ensemble"

    # Classifier-free guidance weights to evaluate
    cfg_scales: List[float] = field(default_factory=lambda: [0.0, 1.0, 4.0])


@dataclass
class AdjointMatchingConfig:
    noise_schedule: NoiseScheduleConfig = field(default_factory=NoiseScheduleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Fine-tuning method: "adjoint_matching", "cont_adjoint", "disc_adjoint",
    #                     "draft_1", "draft_40", "refl", "dpo"
    method: str = "adjoint_matching"

    # Base model checkpoint path
    base_model_path: str = "checkpoints/flow_matching_base"
    output_dir: str = "outputs"
    log_dir: str = "logs"


def get_default_config() -> AdjointMatchingConfig:
    return AdjointMatchingConfig()


def get_config_for_method(method: str, reward_lambda: float = 12500.0) -> AdjointMatchingConfig:
    cfg = AdjointMatchingConfig()
    cfg.method = method
    cfg.reward.reward_lambda = reward_lambda
    cfg.training.reward_lambda = reward_lambda

    if method == "adjoint_matching":
        cfg.noise_schedule.finetuning_sigma_type = "memoryless"
        cfg.noise_schedule.sampling_sigma_type = "zero"
        cfg.training.num_iterations = 1000

    elif method == "cont_adjoint":
        cfg.noise_schedule.finetuning_sigma_type = "memoryless"
        cfg.noise_schedule.sampling_sigma_type = "zero"
        cfg.training.num_iterations = 750

    elif method == "disc_adjoint":
        cfg.noise_schedule.finetuning_sigma_type = "memoryless"
        cfg.noise_schedule.sampling_sigma_type = "zero"
        cfg.training.num_iterations = 1000

    elif method == "draft_1":
        cfg.noise_schedule.finetuning_sigma_type = "memoryless"
        cfg.noise_schedule.sampling_sigma_type = "zero"
        cfg.training.num_iterations = 4000

    elif method == "draft_40":
        cfg.noise_schedule.finetuning_sigma_type = "memoryless"
        cfg.noise_schedule.sampling_sigma_type = "zero"
        cfg.training.num_iterations = 500

    elif method == "refl":
        cfg.noise_schedule.finetuning_sigma_type = "memoryless"
        cfg.noise_schedule.sampling_sigma_type = "zero"
        cfg.training.num_iterations = 1500

    elif method == "dpo":
        cfg.noise_schedule.finetuning_sigma_type = "memoryless"
        cfg.noise_schedule.sampling_sigma_type = "zero"
        cfg.training.num_iterations = 1000

    return cfg
