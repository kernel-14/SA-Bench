```python
## trainer.py
"""Main Trainer class for Adjoint Matching fine-tuning experiments.

This module orchestrates the entire fine-tuning process, wiring together:
- Model loading (SD1.5 UNet as v_theta/v_base, VAE, CLIP text encoder)
- Optimization (Adam with config-specified hyperparameters)
- Trajectory sampling (memoryless Flow Matching SDE)
- Lean adjoint computation (backward ODE integration)
- Loss computation (Adjoint Matching + baselines)
- Training loop with logging and checkpointing

Configuration alignment (config.yaml):
    model.model_id: "runwayml/stable-diffusion-v1-5"
    model.vae_scale_factor: 0.18215
    model.latent_channels: 4
    model.latent_height: 64
    model.latent_width: 64
    model.num_train_timesteps: 1000
    training.learning_rate: 2e-5
    training.beta1: 0.95
    training.beta2: 0.999
    training.epsilon: 1e-8
    training.weight_decay: 1e-2
    training.grad_clip: 1.0
    training.batch_size: 40
    training.precision: "bfloat16"
    training.seed: 42
    sampling.K: 40
    sampling.h: 0.025
    reward.lambda_reward: 12500
    loss.lct_constant: 1.6
    loss.lct_constant_cont_adjoint: 1600.0
    algorithms.disc_adjoint.learning_rate_override: 1e-5
    algorithms.dpo.beta_dpo: 5000.0
    algorithms.draft_1.K_backprop: 1
    algorithms.draft_40.K_backprop: 40
    paths.checkpoint_dir: "outputs/checkpoints"
    paths.output_dir: "outputs"

Dependencies:
    - config.py: Config
    - noise_schedule.py: NoiseSchedule
    - trajectory_sampler.py: TrajectorySampler
    - lean_adjoint.py: LeanAdjointSolver
    - losses.py: AdjointMatchingLoss, continuous_adjoint_loss
    - baselines.py: BaselineLoss
    - reward_models.py: ImageRewardModel
    - prompt_dataset.py: PromptDataset
    - utils.py: set_seed, compute_grad_norm, setup_wandb, latents_to_pil
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image as PILImage
from tqdm import tqdm

from config import Config
from lean_adjoint import LeanAdjointSolver
from losses import AdjointMatchingLoss, continuous_adjoint_loss
from baselines import BaselineLoss
from noise_schedule import NoiseSchedule
from prompt_dataset import PromptDataset
from reward_models import ImageRewardModel
from trajectory_sampler import TrajectorySampler
from utils import (
    compute_grad_norm,
    latents_to_pil,
    set_seed,
    setup_wandb,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Number of UNet training timesteps (config.yaml model.num_train_timesteps: 1000)
_NUM_TRAIN_TIMESTEPS: int = 1000

# VAE latent scaling factor (config.yaml model.vae_scale_factor: 0.18215)
_VAE_SCALE_FACTOR: float = 0.18215

# Logging interval (every N iterations)
_LOG_INTERVAL: int = 10

# Checkpoint interval (every N iterations)
_CHECKPOINT_INTERVAL: int = 100


class Trainer:
    """Orchestrates the full Adjoint Matching fine-tuning process.

    Loads all model components, builds the optimizer, and runs the training
    loop for the specified algorithm (adjoint_matching, draft_1, draft_40,
    refl, dpo, cont_adjoint, disc_adjoint).

    The training loop follows Algorithm 1 from the paper:
    1. Sample prompts from the dataset
    2. Encode text with CLIP
    3. Sample trajectory using memoryless SDE (stop-grad)
    4. Compute reward gradient at noiseless terminal state
    5. Solve lean adjoint ODE backwards
    6. Compute Adjoint Matching loss (grad through v_theta only)
    7. Backpropagate and update v_theta

    Attributes:
        config: Full experiment configuration.
        v_theta: Fine-tuned UNet velocity field (trainable).
        v_base: Frozen copy of the base UNet (never updated).
        vae: Frozen AutoencoderKL for latent↔pixel conversion.
        text_encoder: Frozen CLIPTextModel for text conditioning.
        tokenizer: CLIPTokenizer for text tokenization.
        reward_model: ImageRewardModel for reward computation.
        noise_schedule: NoiseSchedule providing σ(t), κ_t, timesteps.
        trajectory_sampler: TrajectorySampler for forward SDE integration.
        lean_adjoint_solver: LeanAdjointSolver for backward ODE integration.
        adj_loss: AdjointMatchingLoss for the main algorithm.
        baseline_loss: BaselineLoss for comparison methods.
        optimizer: Adam optimizer on v_theta.parameters() only.
        dataset: PromptDataset for training prompt batches.
        device: PyTorch device (cuda or cpu).
        dtype: Training precision dtype (bfloat16).
        lct: Loss Clipping Threshold = lct_constant * λ².
        lct_cont_adjoint: LCT for Continuous Adjoint = lct_constant_cont * λ².
        global_step: Current training iteration counter.
        wandb_run: Weights & Biases run object (or no-op mock).
    """

    def __init__(
        self,
        config: Config,
        dataset: PromptDataset,
    ) -> None:
        """Initialize the trainer with config and dataset.

        Args:
            config: Fully initialized Config object with all hyperparameters.
                Sourced from config.yaml via Config.from_dict().
            dataset: PromptDataset with train/test split already performed.
                Must have self.train_prompts populated via
                dataset.get_train_test_split() before calling train().
        """
        self.config: Config = config
        self.dataset: PromptDataset = dataset

        # Device and dtype setup
        self.device: torch.device = torch.device(config.device)
        # config.yaml training.precision: "bfloat16"
        self.dtype: torch.dtype = (
            torch.bfloat16
            if config.precision == "bfloat16"
            else (
                torch.float16
                if config.precision == "float16"
                else torch.float32
            )
        )

        # Set random seed for reproducibility (Appendix G: 3 independent runs)
        set_seed(config.seed)

        # Training state
        self.global_step: int = 0

        # Loss Clipping Thresholds (Appendix G.3)
        # LCT = lct_constant * λ² for Adjoint Matching
        self.lct: float = config.lct_constant * (config.lambda_reward ** 2)
        # LCT = lct_constant_cont_adjoint * λ² for Continuous Adjoint
        self.lct_cont_adjoint: float = (
            config.lct_constant_cont_adjoint * (config.lambda_reward ** 2)
        )

        logger.info(
            "Trainer initializing: algorithm=%s, lambda=%.1f, K=%d, "
            "device=%s, dtype=%s, lct=%.2e",
            config.algorithm,
            config.lambda_reward,
            config.K,
            config.device,
            str(self.dtype),
            self.lct,
        )

        # Build all model components
        self._build_models()

        # Build optimizer (after models are built)
        self._build_optimizer()

        # Initialize Weights & Biases logging
        self.wandb_run: Any = setup_wandb(config)

        logger.info(
            "Trainer initialized: v_theta params=%d, "
            "num_iterations=%d, batch_size=%d",
            sum(p.numel() for p in self.v_theta.parameters()),
            config.num_iterations,
            config.batch_size,
        )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_models(self) -> None:
        """Load and configure all neural network components.

        Loads from HuggingFace using config.model_id (default:
        "runwayml/stable-diffusion-v1-5"):
            - CLIPTokenizer (frozen)
            - CLIPTextModel (frozen, bfloat16)
            - AutoencoderKL (frozen, float32 for numerical stability)
            - UNet2DConditionModel as v_theta (trainable, bfloat16)
            - Deep copy of UNet as v_base (frozen, bfloat16)

        Also instantiates:
            - NoiseSchedule, TrajectorySampler, LeanAdjointSolver
            - AdjointMatchingLoss, BaselineLoss
            - ImageRewardModel

        Raises:
            ImportError: If diffusers or transformers are not installed.
            OSError: If the model_id cannot be loaded from HuggingFace.
        """
        try:
            from diffusers import AutoencoderKL, UNet2DConditionModel
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as exc:
            raise ImportError(
                "diffusers and transformers are required. "
                "Install with: pip install diffusers==0.25.0 transformers==4.36.0"
            ) from exc

        model_id: str = self.config.model_id
        logger.info("Loading models from '%s'...", model_id)

        # ------------------------------------------------------------------
        # 1. Tokenizer (no device placement needed — CPU only)
        # ------------------------------------------------------------------
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
        )
        logger.info("CLIPTokenizer loaded.")

        # ------------------------------------------------------------------
        # 2. Text encoder (frozen, bfloat16)
        # config.yaml: training.precision = "bfloat16"
        # ------------------------------------------------------------------
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
        )
        self.text_encoder = self.text_encoder.to(
            device=self.device, dtype=self.dtype
        )
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        logger.info("CLIPTextModel loaded and frozen.")

        # ------------------------------------------------------------------
        # 3. VAE (frozen, float32 for numerical stability in reward gradient)
        # Shared Knowledge item 7: VAE stays in float32 even in bfloat16 training.
        # config.yaml model.vae_scale_factor: 0.18215
        # ------------------------------------------------------------------
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
        )
        # VAE must be float32 for the reward gradient path through vae.decode()
        self.vae = self.vae.to(device=self.device, dtype=torch.float32)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad_(False)
        logger.info("AutoencoderKL loaded and frozen (float32).")

        # ------------------------------------------------------------------
        # 4. UNet as v_theta (trainable, bfloat16)
        # This is the model being fine-tuned.
        # ------------------------------------------------------------------
        self.v_theta: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
        )
        self.v_theta = self.v_theta.to(device=self.device, dtype=self.dtype)
        self.v_theta.train()
        # Ensure all parameters are trainable
        for param in self.v_theta.parameters():
            param.requires_grad_(True)
        logger.info(
            "UNet2DConditionModel (v_theta) loaded: %d parameters.",
            sum(p.numel() for p in self.v_theta.parameters()),
        )

        # ------------------------------------------------------------------
        # 5. v_base: deep copy of v_theta, permanently frozen
        # Must be a completely independent copy — not a reference.
        # copy.deepcopy ensures updates to v_theta don't affect v_base.
        # ------------------------------------------------------------------
        self.v_base: UNet2DConditionModel = copy.deepcopy(self.v_theta)
        self.v_base.eval()
        for param in self.v_base.parameters():
            param.requires_grad_(False)
        logger.info("UNet2DConditionModel (v_base) created as frozen deep copy.")

        # ------------------------------------------------------------------
        # 6. Noise schedule (config.yaml sampling.h: 0.025 for K=40)
        # ------------------------------------------------------------------
        self.noise_schedule: NoiseSchedule = NoiseSchedule(h=self.config.h)

        # ------------------------------------------------------------------
        # 7. Trajectory sampler
        # ------------------------------------------------------------------
        self.trajectory_sampler: TrajectorySampler = TrajectorySampler(
            noise_schedule=self.noise_schedule,
            device=str(self.device),
        )

        # ------------------------------------------------------------------
        # 8. Lean adjoint solver
        # ------------------------------------------------------------------
        self.lean_adjoint_solver: LeanAdjointSolver = LeanAdjointSolver(
            noise_schedule=self.noise_schedule,
            device=str(self.device),
        )

        # ------------------------------------------------------------------
        # 9. Adjoint Matching loss
        # ------------------------------------------------------------------
        self.adj_loss: AdjointMatchingLoss = AdjointMatchingLoss(
            noise_schedule=self.noise_schedule,
        )

        # ------------------------------------------------------------------
        # 10. Baseline loss (requires vae for DRaFT/ReFL reward computation)
        # ------------------------------------------------------------------
        self.baseline_loss: BaselineLoss = BaselineLoss(
            noise_schedule=self.noise_schedule,
            vae=self.vae,
            device=str(self.device),
        )

        # ------------------------------------------------------------------
        # 11. Reward model (ImageReward, config.yaml reward.reward_model)
        # ------------------------------------------------------------------
        self.reward_model: ImageRewardModel = ImageRewardModel(
            device=str(self.device),
        )
        logger.info("ImageRewardModel loaded.")

        logger.info("All models built successfully.")

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> None:
        """Create the Adam optimizer on v_theta parameters only.

        Uses hyperparameters from config.yaml training section:
            learning_rate: 2e-5
            beta1: 0.95
            beta2: 0.999
            epsilon: 1e-8
            weight_decay: 1e-2

        Special case for Discrete Adjoint (Table 6, Appendix G):
            Uses learning_rate_override = 1e-5 due to training instability.
            config.yaml algorithms.disc_adjoint.learning_rate_override: 1e-5
        """
        # Determine effective learning rate
        # config.yaml algorithms.disc_adjoint.learning_rate_override: 1e-5
        effective_lr: float = self.config.lr
        if self.config.algorithm == "disc_adjoint":
            effective_lr = self.config.disc_adjoint_lr
            logger.info(
                "Discrete Adjoint: using reduced learning rate %.2e "
                "(default %.2e) due to instability (Table 6).",
                effective_lr,
                self.config.lr,
            )

        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            self.v_theta.parameters(),
            lr=effective_lr,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

        logger.info(
            "Adam optimizer built: lr=%.2e, beta1=%.3f, beta2=%.3f, "
            "eps=%.2e, weight_decay=%.2e",
            effective_lr,
            self.config.beta1,
            self.config.beta2,
            self.config.eps,
            self.config.weight_decay,
        )

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts to CLIP encoder hidden states.

        Produces both conditional (text) and unconditional (empty string)
        embeddings for Classifier-Free Guidance (CFG) support.

        The text encoder is always called with torch.no_grad() since it is
        frozen and never fine-tuned.

        Args:
            prompts: List of text prompt strings, length = batch_size.
                From config.yaml: training.batch_size = 40.

        Returns:
            Tuple (text_emb, uncond_emb) where:
                text_emb: Conditional embeddings of shape
                    (batch_size, seq_len, hidden_dim).
                    For SD1.5 CLIP: (B, 77, 768).
                uncond_emb: Unconditional embeddings (empty string) of same shape.
                    Used for CFG: v_guided = (1+w)*v_cond - w*v_uncond.
            Both tensors are on self.device in self.dtype (bfloat16).
        """
        batch_size: int = len(prompts)

        # Tokenize conditional prompts
        # config.yaml model.num_train_timesteps: 1000 (not directly relevant here)
        # SD1.5 CLIP tokenizer max_length: 77
        cond_tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Tokenize unconditional (empty string) prompts for CFG
        uncond_tokens = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Encode with frozen text encoder (no gradient)
        with torch.no_grad():
            text_emb: torch.Tensor = self.text_encoder(cond_tokens).last_hidden_state
            uncond_emb: torch.Tensor = self.text_encoder(uncond_tokens).last_hidden_state

        # Ensure correct dtype (bfloat16 for training)
        text_emb = text_emb.to(dtype=self.dtype)
        uncond_emb = uncond_emb.to(dtype=self.dtype)

        return text_emb, uncond_emb

    # ------------------------------------------------------------------
    # Latent decoding
    # ------------------------------------------------------------------

    def decode_latents(
        self,
        latents: torch.Tensor,
    ) -> List[PILImage.Image]:
        """Decode latent tensors to PIL images via the VAE decoder.

        Performs the full pipeline:
            latents (UNet space) → scale → VAE decode → clamp → PIL

        The VAE decode is performed in float32 for numerical stability
        (Shared Knowledge item 7). The VAE was loaded in float32 in
        _build_models().

        Args:
            latents: Tensor of shape (B, 4, 64, 64) in UNet latent space.
                May be in bfloat16; cast to float32 internally.
                config.yaml model.latent_channels: 4,
                model.latent_height: 64, model.latent_width: 64.

        Returns:
            List of B PIL.Image.Image objects in RGB format, each 512×512
            for SD1.5.
        """
        # Use the utility function from utils.py which handles the full pipeline
        # config.yaml model.vae_scale_factor: 0.18215
        return latents_to_pil(
            latents=latents,
            vae=self.vae,
            vae_scale_factor=self.config.vae_scale_factor,
        )

    # ------------------------------------------------------------------
    # Training step: Adjoint Matching (Algorithm 1)
    # ------------------------------------------------------------------

    def train_step_adjoint_matching(
        self,
        prompts: List[str],
    ) -> torch.Tensor:
        """Execute one Adjoint Matching training step (Algorithm 1).

        Implements the full Algorithm 1 from the paper:
        1. Encode text prompts to CLIP embeddings
        2. Sample initial noise X_0 ~ N(0, I)
        3. Sample trajectory using memoryless SDE (stop-grad)
        4. Compute noiseless terminal state X̂_1
        5. Compute reward gradient: ã_1 = -λ * ∇_{X̂_1} r(X̂_1)
        6. Solve lean adjoint ODE backwards
        7. Select timestep subset (10 early + 10 late)
        8. Compute clipped Adjoint Matching loss

        The gradient flows ONLY through v_theta(X_t.detach(), t, text_emb)
        in step 8. All other quantities are detached constants.

        Args:
            prompts: List of text prompt strings, length = batch_size.
                From dataset.get_batch(config.batch_size).

        Returns:
            Scalar loss tensor with gradient graph through v_theta.parameters().
            Value is the mean clipped squared norm from equation (42).
        """
        batch_size: int = len(prompts)

        # ------------------------------------------------------------------
        # Step 1: Encode text prompts
        # ------------------------------------------------------------------
        text_emb, uncond_emb = self.encode_text(prompts)
        # text_emb: (B, 77, 768) bfloat16

        # ------------------------------------------------------------------
        # Step 2: Sample initial noise X_0 ~ N(0, I)
        # config.yaml model.latent_channels: 4, latent_height: 64, latent_width: 64
        # ------------------------------------------------------------------
        X0: torch.Tensor = torch.randn(
            batch_size,
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
            device=self.device,
            dtype=self.dtype,
        )

        # ------------------------------------------------------------------
        # Step 3: Sample trajectory using memoryless SDE (stop-grad)
        # Algorithm 1, equation (40):
        #   X_{t+h} = X_t + h*(2*v_theta(X_t,t) - κ_t*X_t) + sqrt(h)*σ(t)*ε
        # All trajectory tensors are detached after each step.
        # ------------------------------------------------------------------
        timesteps: List[float] = self.noise_schedule.get_timesteps(
            K=self.config.K
        )

        with torch.no_grad():
            trajectory: List[torch.Tensor] = (
                self.trajectory_sampler.sample_trajectory(
                    v_theta=self.v_theta,
                    v_base=self.v_base,
                    X0=X0,
                    timesteps=timesteps,
                    text_emb=text_emb,
                )
            )
        # trajectory: List[Tensor(B,4,64,64)] × (K+1), all detached

        # ------------------------------------------------------------------
        # Step 4: Compute noiseless terminal state X̂_1 (Appendix G.1)
        # X̂_1 = X_{1-h} + h * v_base(X_{1-h}, 1-h)  [no noise]
        # This avoids gradient distortion from the final noise injection.
        # ------------------------------------------------------------------
        with torch.no_grad():
            X_hat_1: torch.Tensor = (
                self.trajectory_sampler.get_noiseless_terminal(
                    trajectory=trajectory,
                    v_base=self.v_base,
                    text_emb=text_emb,
                )
            )
        # X_hat_1: (B, 4, 64, 64), detached

        # ------------------------------------------------------------------
        # Step 5: Compute reward gradient (terminal condition for lean adjoint)
        # ã(1; X) = ∇_x g(X̂_1) = -λ * ∇_{X̂_1} r(X̂_1)
        # The gradient flows from pixel space through the VAE decoder to latent.
        # config.yaml reward.lambda_reward: 12500
        # ------------------------------------------------------------------
        terminal_grad: torch.Tensor = self.reward_model.gradient(
            X_latent=X_hat_1,
            prompts=prompts,
            vae=self.vae,
            lambda_r=self.config.lambda_reward,
        )
        # terminal_grad: (B, 4, 64, 64), detached, float32
        # Represents -λ * d(r)/dX_latent (negative because g = -r)

        # ------------------------------------------------------------------
        # Step 6: Solve lean adjoint ODE backwards (equations 38-39)
        # Euler backward step (Algorithm 1, equation 41):
        #   ã_{t-h} = ã_t + h * ã_tᵀ ∇_{X_t}(2*v_base(X_t,t) - κ_t*X_t)
        # All adjoint states are detached.
        # ------------------------------------------------------------------
        with torch.no_grad():
            adjoints: Dict[float, torch.Tensor] = (
                self.lean_adjoint_solver.solve(
                    trajectory=trajectory,
                    v_base=self.v_base,
                    terminal_grad=terminal_grad,
                    timesteps=timesteps,
                    text_emb=text_emb,
                )
            )
        # adjoints: Dict[float → Tensor(B,4,64,64)], all detached

        # ------------------------------------------------------------------
        # Step 7: Select timestep subset for gradient evaluation (Appendix G.2)
        # 10 random from [0, 0.725] + all from [0.75, 0.975] = ~20 steps
        # ------------------------------------------------------------------
        timestep_subset: List[float] = (
            self.trajectory_sampler.select_timestep_subset(timesteps)
        )

        # ------------------------------------------------------------------
        # Step 8: Compute clipped Adjoint Matching loss (equation 42)
        # L̂_AdjMatch(θ) = Σ_{t∈κ} min{LCT, ||(2/σ)*(v_θ-v_base) + σ*ã_t||²}
        # Gradient flows ONLY through v_theta(X_t.detach(), t, text_emb).
        # config.yaml loss.lct_constant: 1.6 → LCT = 1.6 * λ²
        # ------------------------------------------------------------------
        loss: torch.Tensor = self.adj_loss.compute(
            v_theta=self.v_theta,
            v_base=self.v_base,
            trajectory=trajectory,
            adjoints=adjoints,
            timestep_subset=timestep_subset,
            text_emb=text_emb,
            lct=self.lct,
        )

        return loss

    # ------------------------------------------------------------------
    # Training step: DRaFT-K baseline
    # ------------------------------------------------------------------

    def train_step_draft(
        self,
        prompts: List[str],
    ) -> torch.Tensor:
        """Execute one DRaFT-K training step (Clark et al., 2024).

        DRaFT-K backpropagates the reward through the last K_backprop
        denoising steps. Uses ODE (σ=0) during fine-tuning (Table 2).

        Configuration:
            DRaFT-1: config.K_backprop = 1 (algorithms.draft_1.K_backprop: 1)
            DRaFT-40: config.K_backprop = 40 (algorithms.draft_40.K_backprop: 40)

        Args:
            prompts: List of text prompt strings, length = batch_size.

        Returns:
            Scalar loss tensor with gradient through v_theta.parameters().
        """
        batch_size: int = len(prompts)

        # Encode text
        text_emb, _ = self.encode_text(prompts)

        # Sample initial noise
        X0: torch.Tensor = torch.randn(
            batch_size,
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
            device=self.device,
            dtype=self.dtype,
        )

        # Delegate to baseline loss
        # config.yaml algorithms.draft_1.K_backprop: 1
        # config.yaml algorithms.draft_40.K_backprop: 40
        loss: torch.Tensor = self.baseline_loss.draft_loss(
            v_theta=self.v_theta,
            X0=X0,
            reward_fn=self.reward_model,
            text_emb=text_emb,
            prompts=prompts,
            K_backprop=self.config.K_backprop,
            lambda_r=self.config.lambda_reward,
        )

        return loss

    # ------------------------------------------------------------------
    # Training step: ReFL baseline
    # ------------------------------------------------------------------

    def train_step_refl(
        self,
        prompts: List[str],
    ) -> torch.Tensor:
        """Execute one ReFL training step adapted to Flow Matching (Appendix F.1).

        ReFL maximizes the reward on the denoised prediction X̂_1(x,t) at a
        random timestep. The denoiser map for α_t=t, β_t=1-t:
            X̂_1(x, t) = v(x,t)*(1-t) + x

        Args:
            prompts: List of text prompt strings, length = batch_size.

        Returns:
            Scalar loss tensor with gradient through v_theta.parameters().
        """
        batch_size: int = len(prompts)

        # Encode text
        text_emb, _ = self.encode_text(prompts)

        # Sample initial noise
        X0: torch.Tensor = torch.randn(
            batch_size,
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
            device=self.device,
            dtype=self.dtype,
        )

        # Sample trajectory (stop-grad) for ReFL
        timesteps: List[float] = self.noise_schedule.get_timesteps(
            K=self.config.K
        )
        with torch.no_grad():
            trajectory: List[torch.Tensor] = (
                self.trajectory_sampler.sample_trajectory(
                    v_theta=self.v_theta,
                    v_base=self.v_base,
                    X0=X0,
                    timesteps=timesteps,
                    text_emb=text_emb