"""Main training script for pyramidal flow matching.

Three-stage training procedure:
  Stage 1: Image training (50k steps, batch=1536, lr=1e-4)
  Stage 2: Low-resolution video training (200k steps, batch=768, lr=1e-4)
  Stage 3: High-resolution video training (50k steps, batch=384, lr=5e-5)

Usage:
    # Single GPU
    python train.py --stage 1 --output_dir outputs/

    # Multi-GPU (128 A100s as in paper)
    torchrun --nproc_per_node=8 train.py --stage 1 --output_dir outputs/
"""

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset

from config import ModelConfig, get_default_config
from data.dataset import (
    ImageDataset,
    MixedDataset,
    VideoDataset,
    build_dataloader,
)
from data.text_encoder import TextEncoder
from model.dit import MMDiT
from model.vae import VideoVAE
from training.trainer import PyramidFlowTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def build_stage1_dataset(config: ModelConfig) -> ImageDataset:
    """Build image dataset for stage 1 training."""
    datasets = []
    data_cfg = config.data

    for path in [
        data_cfg.laion_path,
        data_cfg.cc12m_path,
        data_cfg.sa1b_path,
        data_cfg.journeydb_path,
        data_cfg.synthetic_path,
    ]:
        if Path(path).exists():
            datasets.append(ImageDataset(path))

    if not datasets:
        logger.warning("No image datasets found. Using dummy dataset.")
        datasets.append(ImageDataset.__new__(ImageDataset))
        datasets[0].samples = [{"image_path": "", "caption": "test image"}] * 1000

    if len(datasets) == 1:
        return datasets[0]

    return ConcatDataset(datasets)


def build_stage2_dataset(config: ModelConfig) -> MixedDataset:
    """Build mixed image+video dataset for stage 2 training."""
    data_cfg = config.data

    image_datasets = []
    for path in [data_cfg.laion_path, data_cfg.cc12m_path]:
        if Path(path).exists():
            image_datasets.append(ImageDataset(path))

    video_datasets = []
    for path in [data_cfg.webvid_path, data_cfg.openvid_path, data_cfg.opensora_path]:
        if Path(path).exists():
            video_datasets.append(VideoDataset(path, num_frames=49))  # 2s at 24fps

    if not image_datasets:
        image_datasets = [ImageDataset.__new__(ImageDataset)]
        image_datasets[0].samples = [{"image_path": "", "caption": "test"}] * 100

    if not video_datasets:
        video_datasets = [VideoDataset.__new__(VideoDataset)]
        video_datasets[0].samples = [{"video_path": "", "caption": "test"}] * 100
        video_datasets[0].num_frames = 49
        video_datasets[0].fps = 24
        video_datasets[0].min_size = 256
        video_datasets[0].max_size = 1024
        video_datasets[0].buckets = {}

    return MixedDataset(
        image_datasets=image_datasets,
        video_datasets=video_datasets,
        image_ratio=config.training.image_data_ratio,
    )


def build_stage3_dataset(config: ModelConfig) -> MixedDataset:
    """Build high-resolution video dataset for stage 3 training."""
    data_cfg = config.data

    image_datasets = []
    for path in [data_cfg.laion_path]:
        if Path(path).exists():
            image_datasets.append(ImageDataset(path))

    video_datasets = []
    for path in [data_cfg.webvid_path, data_cfg.openvid_path, data_cfg.opensora_path]:
        if Path(path).exists():
            # 5-10 second videos at 24fps = 121-241 frames
            video_datasets.append(VideoDataset(path, num_frames=121))

    if not image_datasets:
        image_datasets = [ImageDataset.__new__(ImageDataset)]
        image_datasets[0].samples = [{"image_path": "", "caption": "test"}] * 100

    if not video_datasets:
        video_datasets = [VideoDataset.__new__(VideoDataset)]
        video_datasets[0].samples = [{"video_path": "", "caption": "test"}] * 100
        video_datasets[0].num_frames = 121
        video_datasets[0].fps = 24
        video_datasets[0].min_size = 256
        video_datasets[0].max_size = 1024
        video_datasets[0].buckets = {}

    return MixedDataset(
        image_datasets=image_datasets,
        video_datasets=video_datasets,
        image_ratio=config.training.image_data_ratio,
    )


def train_stage(
    stage: int,
    config: ModelConfig,
    trainer: PyramidFlowTrainer,
    rank: int,
    world_size: int,
    resume_from: str = None,
):
    """Run a single training stage."""
    train_cfg = config.training

    # Stage-specific hyperparameters
    if stage == 1:
        lr = train_cfg.stage1_lr
        beta1 = train_cfg.stage1_beta1
        beta2 = train_cfg.stage1_beta2
        eps = train_cfg.stage1_eps
        total_steps = train_cfg.stage1_steps
        warmup_steps = train_cfg.stage1_warmup_steps
        batch_size = train_cfg.stage1_batch_size // world_size
        dataset = build_stage1_dataset(config)
        is_video_stage = False
    elif stage == 2:
        lr = train_cfg.stage2_lr
        beta1 = train_cfg.stage2_beta1
        beta2 = train_cfg.stage2_beta2
        eps = train_cfg.stage2_eps
        total_steps = train_cfg.stage2_steps
        warmup_steps = train_cfg.stage2_warmup_steps
        batch_size = train_cfg.stage2_batch_size // world_size
        dataset = build_stage2_dataset(config)
        is_video_stage = True
    else:  # stage == 3
        lr = train_cfg.stage3_lr
        beta1 = train_cfg.stage3_beta1
        beta2 = train_cfg.stage3_beta2
        eps = train_cfg.stage3_eps
        total_steps = train_cfg.stage3_steps
        warmup_steps = train_cfg.stage3_warmup_steps
        batch_size = train_cfg.stage3_batch_size // world_size
        dataset = build_stage3_dataset(config)
        is_video_stage = True

    # Setup optimizer
    optimizer, scheduler = trainer.setup_optimizer(
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=train_cfg.weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # Resume from checkpoint if provided
    if resume_from:
        trainer.load_checkpoint(resume_from)

    # Build dataloader
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        distributed=world_size > 1,
    )

    if rank == 0:
        logger.info(f"Starting Stage {stage} training:")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Learning rate: {lr}")
        logger.info(f"  Batch size per GPU: {batch_size}")
        logger.info(f"  Global batch size: {batch_size * world_size}")

    # Training loop
    step = trainer.global_step
    data_iter = iter(dataloader)
    start_time = time.time()

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        metrics = trainer.train_step(
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=train_cfg.grad_clip,
            cfg_dropout_prob=train_cfg.cfg_dropout_prob,
        )
        step = trainer.global_step

        # Logging
        if rank == 0 and step % 100 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = 100 / elapsed
            start_time = time.time()
            logger.info(
                f"Stage {stage} | Step {step}/{total_steps} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"LR: {metrics['lr']:.2e} | "
                f"Grad norm: {metrics['grad_norm']:.3f} | "
                f"Steps/s: {steps_per_sec:.2f}"
            )

        # Save checkpoint
        if rank == 0 and step % config.save_every == 0:
            trainer.save_checkpoint(config.output_dir, step)

    if rank == 0:
        trainer.save_checkpoint(config.output_dir, step)
        logger.info(f"Stage {stage} training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train pyramidal flow matching model")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="Training stage (1=image, 2=low-res video, 3=high-res video)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--pretrained_dit", type=str, default=None,
                        help="Path to pretrained DiT weights (SD3 Medium)")
    parser.add_argument("--pretrained_vae", type=str, default=None,
                        help="Path to pretrained VAE weights")
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load config
    config = get_default_config()
    config.output_dir = args.output_dir

    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Build models
    if rank == 0:
        logger.info("Building models...")

    dit = MMDiT(
        hidden_size=config.dit.hidden_size,
        num_layers=config.dit.num_layers,
        num_heads=config.dit.num_heads,
        mlp_ratio=config.dit.mlp_ratio,
        in_channels=config.dit.in_channels,
        patch_size=config.dit.patch_size,
        context_dim=config.dit.context_dim,
        qk_norm=config.dit.qk_norm,
        dropout=config.dit.dropout,
        use_causal_attention=config.dit.use_causal_attention,
    )

    # Load pretrained SD3 Medium weights if provided
    if args.pretrained_dit and Path(args.pretrained_dit).exists():
        if rank == 0:
            logger.info(f"Loading pretrained DiT from {args.pretrained_dit}")
        state_dict = torch.load(args.pretrained_dit, map_location="cpu")
        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        dit.load_state_dict(state_dict, strict=False)

    vae = VideoVAE(
        in_channels=config.vae.in_channels,
        out_channels=config.vae.out_channels,
        latent_channels=config.vae.latent_channels,
        base_channels=config.vae.base_channels,
        channel_multipliers=tuple(config.vae.channel_multipliers),
        num_res_blocks=config.vae.num_res_blocks,
        kl_weight=config.vae.kl_weight,
    )

    if args.pretrained_vae and Path(args.pretrained_vae).exists():
        if rank == 0:
            logger.info(f"Loading pretrained VAE from {args.pretrained_vae}")
        vae.load_state_dict(torch.load(args.pretrained_vae, map_location="cpu"))

    text_encoder = TextEncoder(
        t5_model_name=config.t5_model,
        clip_model_name=config.clip_model,
        max_length=config.max_text_length,
    )

    # Build trainer
    trainer = PyramidFlowTrainer(
        config=config,
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    if rank == 0:
        total_params = sum(p.numel() for p in dit.parameters())
        logger.info(f"DiT parameters: {total_params / 1e9:.2f}B")

    # Run training stage
    train_stage(
        stage=args.stage,
        config=config,
        trainer=trainer,
        rank=rank,
        world_size=world_size,
        resume_from=args.resume_from,
    )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
