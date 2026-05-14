"""
Main training loop for Adjoint Matching fine-tuning.

Implements the complete fine-tuning pipeline from Section 7:
- Loads pre-trained Flow Matching model
- Sets up reward model (ImageReward)
- Runs fine-tuning with specified method
- Evaluates periodically and saves checkpoints

Hyperparameters from the paper (Appendix G):
- K = 40 timesteps
- Adam: lr=2e-5, beta1=0.95, beta2=0.999, eps=1e-8, weight_decay=1e-2
- Gradient norm clipping: 1.0
- Effective batch size: 40 (2 GPUs x 20)
- bfloat16 precision
- 40k training prompts, 1k test prompts
- 3 independent runs per method
"""

import os
import json
import time
import copy
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from config import AdjointMatchingConfig, get_config_for_method
from noise_schedules import FlowMatchingSchedule
from model import FlowMatchingModel, build_unet
from baselines import build_finetuner
from data import (
    build_infinite_dataloader,
    get_synthetic_prompts,
    load_prompts_from_file,
    create_train_eval_split,
)
from evaluate import (
    ClipScoreMetric,
    PickScoreMetric,
    HPSv2Metric,
    DreamSimDiversityMetric,
    EvaluationRunner,
    format_metrics_table,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward function setup
# ---------------------------------------------------------------------------

def build_reward_fn(
    reward_model_name: str,
    reward_lambda: float,
    device: torch.device,
    prompts_ref: Optional[List[str]] = None,
):
    """
    Build differentiable reward function.

    r(x) = lambda * ImageReward(x, prompt)

    The reward function takes latent images and returns scalar rewards.
    For gradient-based methods, it must be differentiable w.r.t. x.
    """
    try:
        import ImageReward as RM
        reward_model = RM.load(reward_model_name, device=str(device))
        reward_model.eval()
    except ImportError:
        logger.warning("ImageReward not available, using dummy reward")
        reward_model = None

    # Closure that captures current prompts
    current_prompts = {"prompts": prompts_ref or []}

    def reward_fn(images: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled reward: lambda * r(x).
        images: [B, C, H, W] latent or pixel space
        """
        if reward_model is None:
            # Dummy reward for testing
            return torch.randn(images.shape[0], device=images.device)

        prompts = current_prompts["prompts"]
        if not prompts:
            prompts = [""] * images.shape[0]

        # Decode latents to pixel space if needed
        # (handled by the model wrapper in practice)
        scores = []
        from torchvision.transforms.functional import to_pil_image
        imgs_pixel = images
        if imgs_pixel.min() < 0:
            imgs_pixel = (imgs_pixel + 1.0) / 2.0
        imgs_pixel = imgs_pixel.clamp(0, 1)

        for img, prompt in zip(imgs_pixel, prompts[:len(imgs_pixel)]):
            pil_img = to_pil_image(img.detach().cpu())
            score = reward_model.score(prompt, pil_img)
            scores.append(score)

        return reward_lambda * torch.tensor(scores, device=images.device, dtype=images.dtype)

    return reward_fn, current_prompts


# ---------------------------------------------------------------------------
# Optimizer setup
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    lr: float = 2e-5,
    beta1: float = 0.95,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
) -> torch.optim.AdamW:
    """Build Adam optimizer with paper's hyperparameters."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    config: AdjointMatchingConfig,
    finetune_model: FlowMatchingModel,
    base_model: FlowMatchingModel,
    reward_fn,
    current_prompts: Dict,
    train_prompts: List[str],
    eval_prompts: List[str],
    text_encoder,
    eval_runner: Optional[EvaluationRunner],
    device: torch.device,
    output_dir: Path,
    run_id: int = 0,
) -> Dict:
    """
    Main fine-tuning loop.

    Args:
        config: Training configuration
        finetune_model: Model to fine-tune (initialized from base)
        base_model: Frozen base model
        reward_fn: Reward function r(x) -> [B]
        current_prompts: Mutable dict for passing prompts to reward_fn
        train_prompts: Training prompts
        eval_prompts: Evaluation prompts
        text_encoder: CLIP text encoder
        eval_runner: Evaluation runner
        device: Target device
        output_dir: Output directory
        run_id: Run index (0, 1, 2 for 3 runs)

    Returns:
        Dict of final metrics
    """
    cfg = config.training
    schedule = FlowMatchingSchedule(
        num_timesteps=cfg.num_timesteps,
        sigma_offset_h=config.noise_schedule.sigma_offset_h,
    )

    # Build optimizer
    optimizer = build_optimizer(
        finetune_model.unet,
        lr=cfg.learning_rate,
        beta1=cfg.adam_beta1,
        beta2=cfg.adam_beta2,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    # Build fine-tuner
    finetuner = build_finetuner(
        method=config.method,
        model=finetune_model.unet,
        base_model=base_model.unet,
        reward_fn=reward_fn,
        schedule=schedule,
        optimizer=optimizer,
        reward_lambda=cfg.reward_lambda,
        device=device,
        grad_timestep_selector=schedule.select_grad_timesteps,
    )

    # Mixed precision
    use_amp = cfg.precision == "bfloat16"
    scaler = GradScaler() if use_amp and cfg.precision == "float16" else None

    # Infinite dataloader
    prompt_loader = build_infinite_dataloader(
        train_prompts,
        batch_size=cfg.effective_batch_size,
        seed=cfg.seed + run_id,
    )

    # Training state
    metrics_history = []
    best_reward = float("-inf")
    start_time = time.time()

    logger.info(f"Starting fine-tuning: method={config.method}, "
                f"lambda={cfg.reward_lambda}, run={run_id}")

    for iteration, batch_prompts in enumerate(prompt_loader):
        if iteration >= cfg.num_iterations:
            break

        # Update current prompts for reward function
        current_prompts["prompts"] = batch_prompts

        # Encode text
        with torch.no_grad():
            if text_encoder is not None:
                text_emb = text_encoder.encode(batch_prompts).to(device)
            else:
                text_emb = None

        # Sample initial noise
        B = len(batch_prompts)
        x0 = torch.randn(
            B,
            config.model.latent_channels,
            config.model.latent_size,
            config.model.latent_size,
            device=device,
            dtype=torch.bfloat16 if use_amp else torch.float32,
        )

        # Fine-tuning step
        if use_amp:
            with autocast(dtype=torch.bfloat16):
                step_metrics = finetuner.step(x0, text_emb)
        else:
            step_metrics = finetuner.step(x0, text_emb)

        # Logging
        if iteration % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Iter {iteration}/{cfg.num_iterations} | "
                f"Loss: {step_metrics['loss']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

        # Evaluation
        if eval_runner is not None and iteration % 200 == 0 and iteration > 0:
            finetune_model.unet.eval()
            eval_metrics = run_evaluation(
                model=finetune_model,
                eval_prompts=eval_prompts[:200],  # Quick eval
                text_encoder=text_encoder,
                eval_runner=eval_runner,
                schedule=schedule,
                config=config,
                device=device,
            )
            finetune_model.unet.train()

            eval_metrics["iteration"] = iteration
            eval_metrics["elapsed_time"] = time.time() - start_time
            metrics_history.append(eval_metrics)

            logger.info(
                f"Eval @ iter {iteration}: "
                f"ClipScore={eval_metrics.get('clip_score_mean', 0):.2f}, "
                f"PickScore={eval_metrics.get('pick_score_mean', 0):.2f}"
            )

            # Save checkpoint
            if eval_metrics.get("image_reward_mean", 0) > best_reward:
                best_reward = eval_metrics.get("image_reward_mean", 0)
                save_checkpoint(
                    finetune_model, optimizer, iteration, eval_metrics,
                    output_dir / f"best_checkpoint_run{run_id}.pt"
                )

    # Final evaluation
    finetune_model.unet.eval()
    if eval_runner is not None:
        final_metrics = run_evaluation(
            model=finetune_model,
            eval_prompts=eval_prompts,
            text_encoder=text_encoder,
            eval_runner=eval_runner,
            schedule=schedule,
            config=config,
            device=device,
        )
    else:
        final_metrics = {}

    final_metrics["total_time"] = time.time() - start_time
    final_metrics["num_iterations"] = cfg.num_iterations

    # Save final checkpoint
    save_checkpoint(
        finetune_model, optimizer, cfg.num_iterations, final_metrics,
        output_dir / f"final_checkpoint_run{run_id}.pt"
    )

    # Save metrics history
    with open(output_dir / f"metrics_run{run_id}.json", "w") as f:
        json.dump({"history": metrics_history, "final": final_metrics}, f, indent=2)

    return final_metrics


def run_evaluation(
    model: FlowMatchingModel,
    eval_prompts: List[str],
    text_encoder,
    eval_runner: EvaluationRunner,
    schedule: FlowMatchingSchedule,
    config: AdjointMatchingConfig,
    device: torch.device,
    batch_size: int = 8,
    sigma_type: str = "zero",
) -> Dict:
    """Run evaluation on a set of prompts."""

    def generate_fn(prompts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            if text_encoder is not None:
                text_emb = text_encoder.encode(prompts).to(device)
            else:
                text_emb = None

            x0 = torch.randn(
                len(prompts),
                config.model.latent_channels,
                config.model.latent_size,
                config.model.latent_size,
                device=device,
            )

            # Generate latents
            if sigma_type == "zero":
                from sde_utils import sample_fm_ode
                z = sample_fm_ode(
                    lambda x, t, _: model.velocity(x, t, text_emb),
                    x0, schedule
                )
            else:
                from sde_utils import sample_fm_sde_memoryless
                z = sample_fm_sde_memoryless(
                    lambda x, t, _: model.velocity(x, t, text_emb),
                    x0, schedule
                )

            # Decode to pixel space
            images = model.decode_latent(z)
        return images

    return eval_runner.evaluate_full(
        generate_fn=generate_fn,
        eval_prompts=eval_prompts,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: FlowMatchingModel,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    metrics: Dict,
    path: Path,
):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "metrics": metrics,
    }, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: FlowMatchingModel,
    optimizer: Optional[torch.optim.Optimizer],
    path: Path,
    device: torch.device,
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.unet.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("metrics", {})


# ---------------------------------------------------------------------------
# Multi-run experiment (3 runs as in paper)
# ---------------------------------------------------------------------------

def run_experiment(
    config: AdjointMatchingConfig,
    all_prompts: List[str],
    device: torch.device,
    output_dir: Path,
    num_runs: int = 3,
    text_encoder=None,
    eval_runner: Optional[EvaluationRunner] = None,
) -> List[Dict]:
    """
    Run the full experiment with multiple independent runs.

    Paper uses 3 runs per method, each with different prompt subsets.
    Reports mean ± standard error across runs.
    """
    all_run_metrics = []

    for run_id in range(num_runs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {run_id + 1}/{num_runs} for method: {config.method}")
        logger.info(f"{'='*60}")

        # Sample different prompt subsets for each run
        train_prompts, eval_prompts = create_train_eval_split(
            all_prompts,
            num_train=config.data.num_train_prompts,
            num_eval=config.data.num_eval_prompts,
            seed=config.training.seed + run_id * 1000,
        )

        # Build models
        unet_finetune = build_unet(config.model).to(device)
        unet_base = build_unet(config.model).to(device)

        # Load pre-trained weights
        if Path(config.base_model_path).exists():
            checkpoint = torch.load(config.base_model_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            unet_finetune.load_state_dict(state_dict)
            unet_base.load_state_dict(state_dict)
        else:
            logger.warning(f"Base model not found at {config.base_model_path}, "
                           "using random initialization")

        schedule = FlowMatchingSchedule(
            num_timesteps=config.training.num_timesteps,
            sigma_offset_h=config.noise_schedule.sigma_offset_h,
        )

        finetune_model = FlowMatchingModel(unet=unet_finetune, schedule=schedule).to(device)
        base_model = FlowMatchingModel(unet=unet_base, schedule=schedule).to(device)
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad_(False)

        # Build reward function
        reward_fn, current_prompts = build_reward_fn(
            reward_model_name=config.reward.reward_model_name,
            reward_lambda=config.reward.reward_lambda,
            device=device,
        )

        # Run training
        run_output_dir = output_dir / f"run_{run_id}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        metrics = train(
            config=config,
            finetune_model=finetune_model,
            base_model=base_model,
            reward_fn=reward_fn,
            current_prompts=current_prompts,
            train_prompts=train_prompts,
            eval_prompts=eval_prompts,
            text_encoder=text_encoder,
            eval_runner=eval_runner,
            device=device,
            output_dir=run_output_dir,
            run_id=run_id,
        )

        all_run_metrics.append(metrics)
        logger.info(f"Run {run_id + 1} complete: {metrics}")

    return all_run_metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Adjoint Matching fine-tuning")
    parser.add_argument("--method", type=str, default="adjoint_matching",
                        choices=["adjoint_matching", "cont_adjoint", "disc_adjoint",
                                 "draft_1", "draft_40", "refl", "dpo"],
                        help="Fine-tuning method")
    parser.add_argument("--reward_lambda", type=float, default=12500.0,
                        help="Reward scaling factor lambda")
    parser.add_argument("--num_iterations", type=int, default=None,
                        help="Number of fine-tuning iterations (overrides config)")
    parser.add_argument("--base_model_path", type=str, default="checkpoints/flow_matching_base",
                        help="Path to pre-trained base model")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Path to prompt file (txt or json)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of independent runs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Per-GPU batch size")
    parser.add_argument("--num_timesteps", type=int, default=40,
                        help="Number of discretization timesteps")
    parser.add_argument("--sampling_sigma", type=str, default="zero",
                        choices=["zero", "memoryless"],
                        help="Noise schedule for sampling (inference)")
    parser.add_argument("--no_eval", action="store_true",
                        help="Skip evaluation during training")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Build config
    config = get_config_for_method(args.method, args.reward_lambda)
    config.base_model_path = args.base_model_path
    config.output_dir = args.output_dir
    config.training.seed = args.seed
    config.training.batch_size = args.batch_size
    config.training.effective_batch_size = args.batch_size  # single GPU
    config.training.num_timesteps = args.num_timesteps
    config.noise_schedule.sampling_sigma_type = args.sampling_sigma

    if args.num_iterations is not None:
        config.training.num_iterations = args.num_iterations

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / args.method / f"lambda_{args.reward_lambda}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # Load prompts
    if args.prompt_file and Path(args.prompt_file).exists():
        all_prompts = load_prompts_from_file(args.prompt_file)
        logger.info(f"Loaded {len(all_prompts)} prompts from {args.prompt_file}")
    else:
        logger.warning("No prompt file provided, using synthetic prompts")
        all_prompts = get_synthetic_prompts(n=config.data.total_prompt_pool, seed=args.seed)

    # Setup text encoder (optional, skip if not available)
    text_encoder = None
    try:
        from data import CLIPTextEncoder
        text_encoder = CLIPTextEncoder(device=device)
        logger.info("CLIP text encoder loaded")
    except Exception as e:
        logger.warning(f"Could not load CLIP text encoder: {e}")

    # Setup evaluation runner (optional)
    eval_runner = None
    if not args.no_eval:
        try:
            clip_metric = ClipScoreMetric(device=device)
            eval_runner = EvaluationRunner(clip_metric=clip_metric, device=device)
            logger.info("Evaluation runner initialized")
        except Exception as e:
            logger.warning(f"Could not initialize evaluation runner: {e}")

    # Run experiment
    all_metrics = run_experiment(
        config=config,
        all_prompts=all_prompts,
        device=device,
        output_dir=output_dir,
        num_runs=args.num_runs,
        text_encoder=text_encoder,
        eval_runner=eval_runner,
    )

    # Aggregate and report results
    from evaluate import aggregate_metrics_across_runs
    aggregated = aggregate_metrics_across_runs(all_metrics)

    logger.info("\nFinal Results:")
    for key, (mean, std_err) in aggregated.items():
        logger.info(f"  {key}: {mean:.4f} ± {std_err:.4f}")

    # Save aggregated results
    with open(output_dir / "aggregated_results.json", "w") as f:
        json.dump({k: {"mean": v[0], "std_err": v[1]} for k, v in aggregated.items()},
                  f, indent=2)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
