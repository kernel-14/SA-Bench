## config.py
"""Configuration dataclass for Adjoint Matching experiments.

This module defines the Config dataclass that serves as the single source of
truth for all hyperparameters. All values default to the primary experimental
setup from the paper (Appendix G, Tables 2-8): Adjoint Matching with λ=12500,
K=40 timesteps, Adam lr=2e-5.

All derived fields (h, lct, etc.) are computed in __post_init__ to ensure
internal consistency. No external dependencies beyond Python stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ALGORITHM_ITERATIONS: Dict[str, int] = {
    "adjoint_matching": 1000,
    "draft_1": 4000,
    "draft_40": 500,
    "refl": 1500,
    "dpo": 1000,
    "cont_adjoint": 750,
    "disc_adjoint": 1000,
}
"""Default iteration counts per algorithm from Tables 3/4 and Appendix G."""

VALID_ALGORITHMS: List[str] = list(ALGORITHM_ITERATIONS.keys())
VALID_SIGMA_SCHEDULES: List[str] = ["memoryless", "constant", "zero"]
VALID_PRECISIONS: List[str] = ["bfloat16", "float16", "float32"]

# Mapping from config.yaml key names to Config field names where they differ.
_YAML_TO_FIELD_MAP: Dict[str, str] = {
    "learning_rate": "lr",
    "epsilon": "eps",
    "schedule_type": "sigma_schedule",
    "offset": "sigma_offset",
    "reward_model": "reward_model_name",
    "lct_constant": "lct_constant",
    "lct_constant_cont_adjoint": "lct_constant_cont_adjoint",
    "prompts_file": "prompts_file",
    "total_prompts": "total_prompts",
    "num_train_prompts": "num_train_prompts",
    "num_test_prompts": "num_eval_prompts",
    "num_images_per_prompt": "num_images_per_prompt_diversity",
    "num_prompts_for_diversity": "num_prompts_for_diversity",
    "eval_interval": "eval_interval",
    "num_images_per_prompt_eval": "num_images_per_prompt_eval",
    "num_steps": "num_steps",
    "guidance_weight": "guidance_weight",
    "wandb_project": "wandb_project",
    "wandb_entity": "wandb_entity",
    "model_id": "model_id",
    "latent_channels": "latent_channels",
    "latent_height": "latent_height",
    "latent_width": "latent_width",
    "vae_scale_factor": "vae_scale_factor",
    "num_train_timesteps": "num_train_timesteps",
    "alpha_type": "alpha_type",
    "beta_type": "beta_type",
    "K": "K",
    "inference_sigma": "inference_sigma",
    "num_early_samples": "num_early_timestep_samples",
    "early_t_max": "early_t_max",
    "late_t_min": "late_t_min",
    "optimizer": "optimizer",
    "beta1": "beta1",
    "beta2": "beta2",
    "weight_decay": "weight_decay",
    "grad_clip": "grad_clip",
    "batch_size": "batch_size",
    "per_gpu_batch_size": "per_gpu_batch_size",
    "precision": "precision",
    "num_gpus": "num_gpus",
    "seed": "seed",
    "num_runs": "num_runs",
    "lambda_reward": "lambda_reward",
    "beta_dpo": "beta_dpo",
    "K_backprop": "K_backprop",
    "output_dir": "output_dir",
    "checkpoint_dir": "checkpoint_dir",
    "results_dir": "results_dir",
    "log_dir": "log_dir",
    "device": "device",
    "algorithm": "algorithm",
    "num_iterations": "num_iterations",
    "clipscore_model": "clipscore_model",
    "clipscore_pretrained": "clipscore_pretrained",
    "pickscore_model": "pickscore_model",
    "disc_adjoint_lr": "disc_adjoint_lr",
}


def _flatten_config_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """Recursively flatten a nested config dict into a flat dict.

    Nested keys are not concatenated; inner keys overwrite outer keys of the
    same name. This matches the yaml structure where e.g. ``training.lr``
    should map to the field ``lr``.

    Args:
        d: Possibly nested dictionary (e.g. loaded from config.yaml).
        parent_key: Unused; kept for signature compatibility.

    Returns:
        Flat dictionary with all leaf values.
    """
    result: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            # Recurse and merge; inner keys take precedence.
            inner = _flatten_config_dict(v)
            result.update(inner)
        else:
            result[k] = v
    return result


@dataclass
class Config:
    """All hyperparameters for Adjoint Matching experiments.

    Defaults correspond to the primary experimental setup:
    - Algorithm: Adjoint Matching
    - λ = 12500
    - K = 40 timesteps
    - Adam lr = 2e-5, β₁ = 0.95, β₂ = 0.999
    - Memoryless noise schedule
    - SD1.5 base model

    Derived fields (h, lct, lct_cont_adjoint, sigma_offset) are computed
    automatically in __post_init__ and should not be set manually.
    """

    # ------------------------------------------------------------------
    # Model fields
    # ------------------------------------------------------------------
    model_id: str = "runwayml/stable-diffusion-v1-5"
    """HuggingFace model ID for the base generative model."""

    latent_channels: int = 4
    """Number of latent channels in the VAE (SD1.5: 4)."""

    latent_height: int = 64
    """Spatial height of the latent representation (SD1.5: 64)."""

    latent_width: int = 64
    """Spatial width of the latent representation (SD1.5: 64)."""

    vae_scale_factor: float = 0.18215
    """VAE latent scaling factor (standard for SD1.5)."""

    num_train_timesteps: int = 1000
    """Number of training timesteps for the diffusers UNet convention."""

    # ------------------------------------------------------------------
    # Flow Matching fields
    # ------------------------------------------------------------------
    alpha_type: str = "linear"
    """Type of alpha schedule: 'linear' means alpha_t = t."""

    beta_type: str = "linear"
    """Type of beta schedule: 'linear' means beta_t = 1 - t."""

    # ------------------------------------------------------------------
    # Noise schedule fields
    # ------------------------------------------------------------------
    sigma_schedule: str = "memoryless"
    """Noise schedule type. One of: 'memoryless', 'constant', 'zero'.

    - 'memoryless': sigma(t) = sqrt(2*(1-t+h)/(t+h)) [proposed, Table 1]
    - 'constant': sigma(t) = 1 [biased baseline, Table 7]
    - 'zero': sigma(t) = 0 [ODE / noiseless]
    """

    sigma_offset: float = field(default=0.025, init=True)
    """Offset h added to numerator and denominator of sigma(t) to avoid
    division by zero at t=0 (Appendix G.1). Derived from K in __post_init__
    but can be overridden. Default matches h=1/40=0.025."""

    # ------------------------------------------------------------------
    # Sampling fields
    # ------------------------------------------------------------------
    K: int = 40
    """Number of discretization timesteps (Appendix G, K=40)."""

    h: float = field(default=0.025, init=True)
    """Step size h = 1/K. Derived from K in __post_init__; do not set manually."""

    inference_sigma: float = 0.0
    """Sigma used at inference time. 0.0 = ODE (primary), or use memoryless SDE."""

    # ------------------------------------------------------------------
    # Timestep subset fields (Appendix G.2)
    # ------------------------------------------------------------------
    num_early_timestep_samples: int = 10
    """Number of timesteps sampled uniformly from [0, early_t_max] for gradient
    evaluation (Appendix G.2)."""

    early_t_max: float = 0.725
    """Upper bound for early timestep sampling region (Appendix G.2)."""

    late_t_min: float = 0.75
    """Lower bound for always-included late timesteps (Appendix G.2).
    All timesteps >= late_t_min are always included (last 10 of 40 steps)."""

    # ------------------------------------------------------------------
    # Training / optimizer fields (Appendix G)
    # ------------------------------------------------------------------
    optimizer: str = "adam"
    """Optimizer type. Currently only 'adam' is supported."""

    lr: float = 2.0e-5
    """Adam learning rate (Appendix G: 2e-5)."""

    beta1: float = 0.95
    """Adam β₁ (Appendix G: 0.95)."""

    beta2: float = 0.999
    """Adam β₂ (Appendix G: 0.999)."""

    eps: float = 1.0e-8
    """Adam ε (Appendix G: 1e-8)."""

    weight_decay: float = 1.0e-2
    """Adam weight decay (Appendix G: 1e-2)."""

    grad_clip: float = 1.0
    """Gradient norm clipping value (Appendix G: 1.0)."""

    batch_size: int = 40
    """Effective batch size across all GPUs (Appendix G: 40 = 20 per GPU × 2)."""

    per_gpu_batch_size: int = 20
    """Per-GPU batch size (Appendix G: 20 per 80GB A100)."""

    precision: str = "bfloat16"
    """Training precision. One of: 'bfloat16', 'float16', 'float32'."""

    num_gpus: int = 2
    """Number of GPUs for distributed training (Appendix G: 2 × 80GB A100)."""

    seed: int = 42
    """Random seed for reproducibility."""

    num_runs: int = 3
    """Number of independent runs per data point (Appendix G: 3 runs)."""

    # ------------------------------------------------------------------
    # Reward fields
    # ------------------------------------------------------------------
    reward_model_name: str = "ImageReward-v1.0"
    """Name/path of the reward model (Section 7: ImageReward)."""

    lambda_reward: float = 12500.0
    """Reward scaling factor λ. r(x) = λ * RewardModel(x).
    Primary value: 12500. Also tested: 1000, 2500 (Table 2)."""

    # ------------------------------------------------------------------
    # Loss / clipping fields (Appendix G.3)
    # ------------------------------------------------------------------
    lct_constant: float = 1.6
    """Constant c such that LCT = c * λ² for Adjoint Matching (Appendix G.3)."""

    lct_constant_cont_adjoint: float = 1600.0
    """Constant c such that LCT = c * λ² for Continuous Adjoint (Appendix G.3).
    Regular adjoint states have much larger magnitude than lean adjoint states."""

    lct: float = field(default=0.0, init=True)
    """Loss Clipping Threshold for Adjoint Matching: lct_constant * λ².
    Derived in __post_init__; do not set manually."""

    lct_cont_adjoint: float = field(default=0.0, init=True)
    """Loss Clipping Threshold for Continuous Adjoint: lct_constant_cont_adjoint * λ².
    Derived in __post_init__; do not set manually."""

    # ------------------------------------------------------------------
    # Algorithm fields
    # ------------------------------------------------------------------
    algorithm: str = "adjoint_matching"
    """Fine-tuning algorithm. One of: 'adjoint_matching', 'draft_1', 'draft_40',
    'refl', 'dpo', 'cont_adjoint', 'disc_adjoint'."""

    num_iterations: int = 0
    """Number of fine-tuning iterations. If 0, derived from ALGORITHM_ITERATIONS
    in __post_init__. Can be overridden explicitly."""

    K_backprop: int = 1
    """For DRaFT-K: number of denoising steps to backpropagate through.
    DRaFT-1: K_backprop=1; DRaFT-40: K_backprop=40."""

    beta_dpo: float = 5000.0
    """DPO β parameter (Wallace et al., 2023a, Sec. 5.1 recommends [4000, 10000])."""

    disc_adjoint_lr: float = 1.0e-5
    """Learning rate override for Discrete Adjoint due to instability (Table 6).
    Trainer uses this when algorithm == 'disc_adjoint'."""

    # ------------------------------------------------------------------
    # Data fields
    # ------------------------------------------------------------------
    prompts_file: str = "data/prompts.txt"
    """Path to text file with one prompt per line (Appendix G)."""

    total_prompts: int = 100000
    """Total prompt pool size (Appendix G: 100k prompts)."""

    num_train_prompts: int = 40000
    """Training prompts per run, sampled from total pool (Appendix G: 40k)."""

    num_eval_prompts: int = 1000
    """Test prompts per run, held-out and different per run (Appendix G: 1k)."""

    # ------------------------------------------------------------------
    # Evaluation fields (Appendix G.4)
    # ------------------------------------------------------------------
    clipscore_model: str = "ViT-L-14"
    """CLIP model architecture for ClipScore computation."""

    clipscore_pretrained: str = "openai"
    """CLIP pretrained weights identifier for open_clip."""

    pickscore_model: str = "yuvalkirstain/PickScore_v1"
    """HuggingFace model ID for PickScore (Kirstain et al., 2023)."""

    num_images_per_prompt_diversity: int = 25
    """Number of images per prompt for DreamSim diversity computation
    (Appendix G.4: 25 images × 40 prompts)."""

    num_prompts_for_diversity: int = 40
    """Number of prompts used for diversity computation (Appendix G.4: 40 prompts)."""

    eval_interval: int = 100
    """Evaluation frequency during training (iterations between evaluations)."""

    num_images_per_prompt_eval: int = 1
    """Number of images per prompt for reward/consistency metrics evaluation."""

    # ------------------------------------------------------------------
    # Inference fields
    # ------------------------------------------------------------------
    num_steps: int = 40
    """Number of inference timesteps (primary: 40, same as fine-tuning)."""

    guidance_weight: float = 0.0
    """Classifier-Free Guidance weight w. 0.0 = no guidance (primary evaluation).
    CFG formula: v_guided = (1+w)*v_cond - w*v_uncond."""

    # ------------------------------------------------------------------
    # Path fields
    # ------------------------------------------------------------------
    output_dir: str = "outputs"
    """Root output directory for all experiment artifacts."""

    checkpoint_dir: str = "outputs/checkpoints"
    """Directory for saving model checkpoints."""

    results_dir: str = "outputs/results"
    """Directory for saving evaluation results (JSON)."""

    log_dir: str = "outputs/logs"
    """Directory for training logs."""

    wandb_project: str = "adjoint-matching"
    """Weights & Biases project name."""

    wandb_entity: str = ""
    """Weights & Biases entity (username or team). Empty string = default."""

    device: str = "cuda"
    """PyTorch device string ('cuda', 'cpu', 'cuda:0', etc.)."""

    # ------------------------------------------------------------------
    # __post_init__: derive computed fields and validate
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Derive computed fields and validate all settings.

        Derived fields:
            h = 1.0 / K
            sigma_offset = h  (kept in sync with h per Appendix G.1)
            lct = lct_constant * lambda_reward ** 2
            lct_cont_adjoint = lct_constant_cont_adjoint * lambda_reward ** 2
            num_iterations = ALGORITHM_ITERATIONS[algorithm]  (if 0)

        Raises:
            ValueError: If any field has an invalid value.
        """
        # --- Validate algorithm ---
        if self.algorithm not in VALID_ALGORITHMS:
            raise ValueError(
                f"Invalid algorithm '{self.algorithm}'. "
                f"Must be one of: {VALID_ALGORITHMS}"
            )

        # --- Validate sigma_schedule ---
        if self.sigma_schedule not in VALID_SIGMA_SCHEDULES:
            raise ValueError(
                f"Invalid sigma_schedule '{self.sigma_schedule}'. "
                f"Must be one of: {VALID_SIGMA_SCHEDULES}"
            )

        # --- Validate precision ---
        if self.precision not in VALID_PRECISIONS:
            raise ValueError(
                f"Invalid precision '{self.precision}'. "
                f"Must be one of: {VALID_PRECISIONS}"
            )

        # --- Validate positive integers ---
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.lambda_reward <= 0.0:
            raise ValueError(
                f"lambda_reward must be positive, got {self.lambda_reward}"
            )

        # --- Derive h from K (Appendix G: h = 1/K = 0.025 for K=40) ---
        self.h = 1.0 / float(self.K)

        # --- Keep sigma_offset in sync with h (Appendix G.1) ---
        self.sigma_offset = self.h

        # --- Derive LCT values (Appendix G.3) ---
        self.lct = self.lct_constant * (self.lambda_reward ** 2)
        self.lct_cont_adjoint = (
            self.lct_constant_cont_adjoint * (self.lambda_reward ** 2)
        )

        # --- Derive num_iterations from algorithm if not explicitly set ---
        if self.num_iterations == 0:
            self.num_iterations = ALGORITHM_ITERATIONS[self.algorithm]

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Construct a Config from a (possibly nested) dictionary.

        Handles the nested structure of config.yaml by flattening it first,
        then applying the YAML-to-field name mapping, and finally filtering
        to only valid Config field names.

        Args:
            d: Dictionary loaded from config.yaml (may be nested).

        Returns:
            Config instance with all specified values and derived fields
            computed by __post_init__.

        Example:
            >>> import yaml
            >>> with open("config.yaml") as f:
            ...     raw = yaml.safe_load(f)
            >>> config = Config.from_dict(raw)
        """
        # Step 1: Flatten nested dict
        flat: Dict[str, Any] = _flatten_config_dict(d)

        # Step 2: Apply YAML key → field name mapping
        remapped: Dict[str, Any] = {}
        for yaml_key, value in flat.items():
            field_name = _YAML_TO_FIELD_MAP.get(yaml_key, yaml_key)
            remapped[field_name] = value

        # Step 3: Get valid field names for Config
        valid_field_names: set = {f.name for f in fields(cls)}

        # Step 4: Filter to only valid fields
        filtered: Dict[str, Any] = {
            k: v for k, v in remapped.items() if k in valid_field_names
        }

        # Step 5: Construct Config (triggers __post_init__)
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all Config fields (including derived) to a flat dict.

        Includes derived fields (h, lct, lct_cont_adjoint, sigma_offset) so
        the saved config is fully self-contained for experiment reproducibility.

        Returns:
            Flat dictionary of all field names to their values.
        """
        result: Dict[str, Any] = {}
        for f in fields(self):
            result[f.name] = getattr(self, f.name)
        return result

    def get_lambda_values(self) -> List[float]:
        """Return the list of lambda values for ablation studies (Table 2).

        Returns:
            List of lambda values: [1000.0, 2500.0, 12500.0].
        """
        return [1000.0, 2500.0, 12500.0]

    def get_guidance_weights(self) -> List[float]:
        """Return the list of CFG guidance weights for ablation (Table 5).

        Returns:
            List of guidance weights: [0.0, 1.0, 4.0].
        """
        return [0.0, 1.0, 4.0]

    def get_noise_schedule_ablation_values(self) -> List[str]:
        """Return sigma schedule names for Table 7 ablation.

        Returns:
            List of schedule names: ['memoryless', 'constant', 'zero'].
        """
        return ["memoryless", "constant", "zero"]

    def get_inference_steps_ablation(self) -> List[int]:
        """Return inference step counts for Table 8 ablation.

        Returns:
            List of step counts: [10, 20, 40, 100, 200].
        """
        return [10, 20, 40, 100, 200]

    def clone_with(self, **kwargs: Any) -> "Config":
        """Create a copy of this Config with specified fields overridden.

        Useful for ablation studies where a single field changes.

        Args:
            **kwargs: Field name → new value pairs to override.

        Returns:
            New Config instance with overrides applied and derived fields
            recomputed.

        Example:
            >>> base_config = Config()
            >>> config_1000 = base_config.clone_with(lambda_reward=1000.0)
            >>> config_1000.lct  # Automatically recomputed: 1.6 * 1000^2
            1600000.0
        """
        current = self.to_dict()
        current.update(kwargs)
        # Remove derived fields so __post_init__ recomputes them cleanly.
        # h, sigma_offset, lct, lct_cont_adjoint are always re-derived.
        for derived in ("h", "sigma_offset", "lct", "lct_cont_adjoint"):
            current.pop(derived, None)
        # Reset num_iterations to 0 if algorithm changed, so it gets re-derived.
        if "algorithm" in kwargs and "num_iterations" not in kwargs:
            current["num_iterations"] = 0
        return Config(**current)

    def __repr__(self) -> str:
        """Human-readable representation showing key hyperparameters."""
        return (
            f"Config("
            f"algorithm={self.algorithm!r}, "
            f"lambda_reward={self.lambda_reward}, "
            f"K={self.K}, "
            f"h={self.h:.4f}, "
            f"lr={self.lr:.2e}, "
            f"sigma_schedule={self.sigma_schedule!r}, "
            f"lct={self.lct:.2e}, "
            f"num_iterations={self.num_iterations}, "
            f"batch_size={self.batch_size}"
            f")"
        )
