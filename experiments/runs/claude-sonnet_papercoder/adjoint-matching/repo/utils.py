## utils.py
"""Shared utility functions for Adjoint Matching experiments.

This module provides foundational utilities used across all other modules.
It has NO dependencies on other project files — only standard library and
third-party packages (torch, numpy, random, PIL, wandb, math, typing).

Key utilities:
    - set_seed: Reproducibility across 3 independent runs (Appendix G)
    - get_unet_timestep: Bridge between FM continuous time and diffusers UNet
    - latents_to_pil: Latent tensor → PIL image conversion
    - pil_to_tensor: PIL image → normalized tensor conversion
    - compute_grad_norm: Gradient health monitoring
    - AverageMeter: Running average tracker for training metrics
    - setup_wandb: Weights & Biases initialization
    - log_images: Image grid logging to wandb
    - make_grid_image: PIL image grid construction
"""

from __future__ import annotations

import math
import random
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility across 3 independent runs.

    The paper (Appendix G) performs 3 independent runs per data point, each
    with a different 40k prompt subset. Callers should use
    ``set_seed(base_seed + run_idx)`` for each run.

    Sets seeds for:
        - Python's built-in ``random`` module (used in timestep subset selection)
        - NumPy (used in prompt dataset sampling)
        - PyTorch CPU and all CUDA devices (paper uses 2× A100)
        - cuDNN deterministic mode for full reproducibility

    Args:
        seed: Integer seed value. Paper default is 42 (config.yaml seed: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# UNet timestep conversion
# ---------------------------------------------------------------------------


def get_unet_timestep(
    t_continuous: float,
    num_train_timesteps: int = 1000,
) -> int:
    """Convert Flow Matching continuous time to diffusers UNet integer timestep.

    The paper uses Flow Matching convention: t=0 is noise, t=1 is clean data.
    The diffusers SD1.5 UNet uses DDPM convention: timestep=0 is clean,
    timestep=999 is pure noise. This function bridges the two conventions.

    Conversion formula:
        timestep_int = int((1.0 - t_continuous) * num_train_timesteps)

    Examples:
        - t=0.025 (first FM step, most noisy) → int(0.975 * 1000) = 975
        - t=0.975 (last FM step, nearly clean) → int(0.025 * 1000) = 25
        - t=1.0 (terminal, clean) → 0

    Args:
        t_continuous: Continuous time in [0, 1] from Flow Matching convention.
            t=0 corresponds to noise X_0 ~ N(0, I).
            t=1 corresponds to clean data X_1.
        num_train_timesteps: Total number of training timesteps in the diffusers
            UNet (config.yaml num_train_timesteps: 1000 for SD1.5).

    Returns:
        Integer timestep in [0, num_train_timesteps - 1] for the diffusers UNet.
    """
    raw: float = (1.0 - t_continuous) * float(num_train_timesteps)
    timestep_int: int = int(raw)
    # Clamp to valid range to handle floating point edge cases at t=0.0 and t=1.0
    timestep_int = max(0, min(timestep_int, num_train_timesteps - 1))
    return timestep_int


# ---------------------------------------------------------------------------
# Tensor ↔ PIL image conversion
# ---------------------------------------------------------------------------


def latents_to_pil(
    latents: torch.Tensor,
    vae: Any,
    vae_scale_factor: float = 0.18215,
) -> List[Image.Image]:
    """Decode latent tensors to PIL images via the VAE decoder.

    Performs the full pipeline:
        latents (UNet space) → scale → VAE decode → clamp → uint8 → PIL

    The VAE decode is performed in float32 for numerical stability, even when
    training in bfloat16 (per Shared Knowledge in task spec). This is critical
    for the reward gradient computation path in reward_models.py.

    Args:
        latents: Tensor of shape (B, 4, 64, 64) in UNet latent space.
            These are the raw latents as output by the UNet, before VAE scaling.
        vae: Diffusers AutoencoderKL instance. Must have a .decode() method
            returning an object with a .sample attribute.
        vae_scale_factor: VAE latent scaling factor (config.yaml vae_scale_factor:
            0.18215 for SD1.5). Latents are divided by this before decoding.

    Returns:
        List of B PIL.Image.Image objects in RGB format, each of size
        (512, 512) for SD1.5.
    """
    # Cast to float32 for VAE decode stability (bfloat16 can cause artifacts)
    latents_f32: torch.Tensor = latents.float()

    # Scale latents: UNet outputs are in scaled space; VAE expects unscaled
    scaled_latents: torch.Tensor = latents_f32 / vae_scale_factor

    # Decode via VAE: output shape (B, 3, H, W) in range approximately [-1, 1]
    with torch.no_grad():
        decoded = vae.decode(scaled_latents).sample

    # Clamp to [-1, 1] for safety (VAE can occasionally produce out-of-range values)
    decoded = decoded.clamp(-1.0, 1.0)

    # Rescale from [-1, 1] to [0, 1]
    decoded = (decoded + 1.0) / 2.0

    # Clamp to [0, 1] after rescaling
    decoded = decoded.clamp(0.0, 1.0)

    # Convert to uint8: (B, 3, H, W) float → (B, 3, H, W) uint8
    decoded_uint8: torch.Tensor = (decoded * 255.0).round().clamp(0, 255).to(torch.uint8)

    # Move to CPU and convert to numpy: (B, 3, H, W) → (B, H, W, 3)
    decoded_np: np.ndarray = decoded_uint8.permute(0, 2, 3, 1).cpu().numpy()

    # Convert each image to PIL
    pil_images: List[Image.Image] = []
    for i in range(decoded_np.shape[0]):
        pil_img = Image.fromarray(decoded_np[i], mode="RGB")
        pil_images.append(pil_img)

    return pil_images


def pil_to_tensor(
    images: List[Image.Image],
    target_size: Optional[tuple] = None,
) -> torch.Tensor:
    """Convert a list of PIL images to a normalized float tensor.

    Produces a tensor in [0, 1] range in standard PyTorch (B, C, H, W) format.
    Note: Different evaluation models (CLIP, PickScore, DreamSim) apply their
    own preprocessing on top of this generic conversion.

    Args:
        images: List of PIL.Image.Image objects. All images should be the same
            size for batching. If sizes differ, they are resized to target_size
            or the size of the first image.
        target_size: Optional (width, height) tuple for resizing. If None,
            uses the size of the first image in the list.

    Returns:
        Float tensor of shape (B, 3, H, W) in [0, 1] range on CPU.
        Returns empty tensor of shape (0, 3, 1, 1) if images list is empty.
    """
    if len(images) == 0:
        return torch.zeros(0, 3, 1, 1, dtype=torch.float32)

    # Determine target size from first image if not specified
    if target_size is None:
        first_img = images[0].convert("RGB")
        target_size = first_img.size  # (width, height)

    arrays: List[np.ndarray] = []
    for img in images:
        # Ensure RGB format
        img_rgb = img.convert("RGB")
        # Resize if needed
        if img_rgb.size != target_size:
            img_rgb = img_rgb.resize(target_size, Image.LANCZOS)
        # Convert to numpy (H, W, 3) uint8
        arr = np.array(img_rgb, dtype=np.uint8)
        arrays.append(arr)

    # Stack to (B, H, W, 3) numpy array
    stacked: np.ndarray = np.stack(arrays, axis=0)

    # Convert to float tensor (B, H, W, 3) in [0, 1]
    tensor_hwc: torch.Tensor = torch.from_numpy(stacked).float() / 255.0

    # Permute to (B, 3, H, W) — standard PyTorch image format
    tensor_chw: torch.Tensor = tensor_hwc.permute(0, 3, 1, 2).contiguous()

    return tensor_chw


# ---------------------------------------------------------------------------
# Gradient monitoring
# ---------------------------------------------------------------------------


def compute_grad_norm(model: nn.Module) -> float:
    """Compute the total L2 gradient norm across all model parameters.

    Matches PyTorch's internal computation in ``clip_grad_norm_``, so the
    logged value directly corresponds to what gets clipped during training.
    Useful for detecting instability in Discrete Adjoint (which the paper
    notes requires lower LR due to instability, Table 6).

    Args:
        model: PyTorch module whose gradient norm to compute.

    Returns:
        Total L2 gradient norm as a Python float. Returns 0.0 if no
        parameters have gradients (e.g., before the first backward pass).
    """
    total_norm_sq: float = 0.0
    has_grad: bool = False

    for param in model.parameters():
        if param.grad is not None:
            param_norm: float = param.grad.data.norm(2).item()
            total_norm_sq += param_norm ** 2
            has_grad = True

    if not has_grad:
        return 0.0

    return math.sqrt(total_norm_sq)


# ---------------------------------------------------------------------------
# Running average tracker
# ---------------------------------------------------------------------------


class AverageMeter:
    """Track running averages of scalar metrics during training.

    Used in trainer.py to track: loss per iteration, ImageReward score,
    control cost (KL term), ClipScore during training — matching the
    training curves shown in Figure 6 of the paper.

    Attributes:
        name: Human-readable name for this metric (used in __str__).
        val: Most recently updated value.
        avg: Running average over all updates.
        sum: Running sum of (val * n) over all updates.
        count: Total number of samples accumulated.

    Example:
        >>> meter = AverageMeter("loss")
        >>> meter.update(0.5, n=40)  # batch of 40
        >>> meter.update(0.3, n=40)
        >>> print(meter.avg)  # 0.4
        >>> print(meter)      # "loss: 0.4000"
    """

    def __init__(self, name: str) -> None:
        """Initialize the meter with a given name.

        Args:
            name: Human-readable name for this metric (e.g., 'loss', 'reward').
        """
        self.name: str = name
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def reset(self) -> None:
        """Reset all accumulated statistics to zero.

        Call at the start of each epoch or evaluation period.
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update the meter with a new value.

        Args:
            val: The new value to incorporate. For batch metrics, this should
                be the mean over the batch (not the sum).
            n: Number of samples this value represents (typically batch_size).
                The sum is updated as ``sum += val * n``.
        """
        self.val = val
        self.sum += val * float(n)
        self.count += n
        self.avg = self.sum / float(self.count) if self.count > 0 else 0.0

    def __str__(self) -> str:
        """Return formatted string showing the running average.

        Returns:
            String of the form ``"name: avg_value"`` with 4 decimal places.
        """
        return f"{self.name}: {self.avg:.4f}"

    def __repr__(self) -> str:
        """Detailed representation including val, avg, count."""
        return (
            f"AverageMeter(name={self.name!r}, "
            f"val={self.val:.4f}, avg={self.avg:.4f}, "
            f"sum={self.sum:.4f}, count={self.count})"
        )


# ---------------------------------------------------------------------------
# Weights & Biases utilities
# ---------------------------------------------------------------------------


class _NoOpWandbRun:
    """No-op mock for wandb.run when wandb is unavailable or disabled.

    Provides the same interface as a real wandb run so that training code
    does not need to check for None before calling .log() or .finish().
    """

    def log(self, *args: Any, **kwargs: Any) -> None:
        """No-op log method."""
        pass

    def finish(self, *args: Any, **kwargs: Any) -> None:
        """No-op finish method."""
        pass

    def __bool__(self) -> bool:
        """Returns False so callers can check ``if wandb_run:``."""
        return False


def setup_wandb(config: Any) -> Any:
    """Initialize Weights & Biases logging for experiment tracking.

    Creates a wandb run with the experiment configuration. The run name
    encodes the algorithm, lambda, and seed for easy identification in
    the wandb dashboard across multiple ablation runs.

    If wandb is not installed or ``config.wandb_project`` is empty/None,
    returns a no-op mock object so the training loop continues uninterrupted.

    Args:
        config: Config object (duck-typed). Accesses the following attributes:
            - wandb_project (str): W&B project name
            - wandb_entity (str): W&B entity (username/team), empty = default
            - output_dir (str): Local directory for wandb files
            - algorithm (str): Fine-tuning algorithm name
            - lambda_reward (float): Reward scaling factor
            - seed (int): Random seed for this run
            - to_dict() method: Returns flat dict of all config fields

    Returns:
        wandb.run object if wandb is available and configured, otherwise
        a _NoOpWandbRun instance.
    """
    try:
        import wandb  # type: ignore[import]
    except ImportError:
        return _NoOpWandbRun()

    # Check if wandb project is configured
    project: str = getattr(config, "wandb_project", "") or ""
    if not project:
        return _NoOpWandbRun()

    entity: str = getattr(config, "wandb_entity", "") or ""
    output_dir: str = getattr(config, "output_dir", "outputs")
    algorithm: str = getattr(config, "algorithm", "unknown")
    lambda_reward: float = getattr(config, "lambda_reward", 0.0)
    seed: int = getattr(config, "seed", 42)

    # Construct descriptive run name for dashboard identification
    run_name: str = f"{algorithm}_lambda{int(lambda_reward)}_seed{seed}"

    # Get full config dict for wandb config panel
    config_dict: dict = {}
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()

    try:
        run = wandb.init(
            project=project,
            entity=entity if entity else None,
            config=config_dict,
            name=run_name,
            dir=output_dir,
            reinit=True,  # Allow multiple runs in same process (for ablations)
        )
        return run
    except Exception:
        # Gracefully fall back to no-op if wandb init fails
        return _NoOpWandbRun()


def log_images(
    images: List[Image.Image],
    prompts: List[str],
    step: int,
    wandb_run: Any,
    max_images: int = 8,
    nrow: int = 4,
) -> None:
    """Log a grid of generated images to Weights & Biases.

    Creates a grid from up to ``max_images`` images and logs it with
    associated prompt captions. Called every ``eval_interval`` iterations
    in trainer.py for qualitative monitoring (matching Figures 8-11).

    Args:
        images: List of PIL.Image.Image objects to log.
        prompts: List of text prompts corresponding to each image.
        step: Current training iteration (used as x-axis in wandb).
        wandb_run: wandb.run object or _NoOpWandbRun. If falsy, silently skips.
        max_images: Maximum number of images to include in the grid (default 8).
        nrow: Number of images per row in the grid (default 4).
    """
    if not wandb_run:
        return

    try:
        import wandb  # type: ignore[import]
    except ImportError:
        return

    if not images:
        return

    # Limit to max_images
    display_images: List[Image.Image] = images[:max_images]
    display_prompts: List[str] = prompts[:max_images]

    # Create grid image
    grid_image: Image.Image = make_grid_image(display_images, nrow=nrow)

    # Build caption from prompts (truncated for readability)
    caption_parts: List[str] = []
    for i, prompt in enumerate(display_prompts):
        truncated: str = prompt[:50] + "..." if len(prompt) > 50 else prompt
        caption_parts.append(f"[{i}] {truncated}")
    caption: str = " | ".join(caption_parts)

    try:
        wandb_run.log(
            {
                "generated_images": wandb.Image(grid_image, caption=caption),
                "step": step,
            }
        )
    except Exception:
        # Silently ignore logging failures to not interrupt training
        pass


# ---------------------------------------------------------------------------
# Image grid construction
# ---------------------------------------------------------------------------


def make_grid_image(
    images: List[Image.Image],
    nrow: int = 4,
    padding: int = 2,
    background_color: tuple = (255, 255, 255),
) -> Image.Image:
    """Create a single grid PIL image from a list of images.

    Arranges images in a grid with ``nrow`` columns. All images are assumed
    to be the same size (the size of the first image is used as reference).

    Args:
        images: List of PIL.Image.Image objects to arrange in a grid.
        nrow: Number of images per row (default 4).
        padding: Pixel padding between images in the grid (default 2).
        background_color: RGB tuple for the grid background (default white).

    Returns:
        Single PIL.Image.Image containing all input images arranged in a grid.
        Returns a small blank 1×1 white image if ``images`` is empty.
    """
    if not images:
        return Image.new("RGB", (1, 1), color=background_color)

    # Use size of first image as reference
    img_w: int
    img_h: int
    img_w, img_h = images[0].size

    n_images: int = len(images)
    ncols: int = min(nrow, n_images)
    nrows: int = math.ceil(n_images / ncols)

    # Compute canvas size including padding
    canvas_w: int = ncols * img_w + (ncols + 1) * padding
    canvas_h: int = nrows * img_h + (nrows + 1) * padding

    # Create blank canvas
    canvas: Image.Image = Image.new("RGB", (canvas_w, canvas_h), color=background_color)

    # Paste each image at the correct grid position
    for idx, img in enumerate(images):
        row: int = idx // ncols
        col: int = idx % ncols

        # Ensure image is RGB and correct size
        img_rgb: Image.Image = img.convert("RGB")
        if img_rgb.size != (img_w, img_h):
            img_rgb = img_rgb.resize((img_w, img_h), Image.LANCZOS)

        # Compute paste position with padding
        x: int = padding + col * (img_w + padding)
        y: int = padding + row * (img_h + padding)

        canvas.paste(img_rgb, (x, y))

    return canvas
