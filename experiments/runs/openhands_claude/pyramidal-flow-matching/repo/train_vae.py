"""VAE training script.

The 3D VAE is trained from scratch on WebVid-10M and SA-1B images
with 8x8x8 compression ratio. Uses KL regularization.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.cuda.amp import GradScaler

from config import get_default_config
from data.dataset import ImageDataset, VideoDataset, build_dataloader
from model.vae import VideoVAE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def perceptual_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple L1 + L2 reconstruction loss."""
    return F.l1_loss(pred, target) + 0.5 * F.mse_loss(pred, target)


def train_vae(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    config = get_default_config()

    vae = VideoVAE(
        in_channels=config.vae.in_channels,
        out_channels=config.vae.out_channels,
        latent_channels=config.vae.latent_channels,
        base_channels=config.vae.base_channels,
        channel_multipliers=tuple(config.vae.channel_multipliers),
        num_res_blocks=config.vae.num_res_blocks,
        kl_weight=config.vae.kl_weight,
    ).to(device)

    if world_size > 1:
        vae = DDP(vae, device_ids=[local_rank])

    optimizer = AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()

    # Build dataset
    datasets = []
    if Path(config.data.webvid_path).exists():
        datasets.append(VideoDataset(config.data.webvid_path, num_frames=17))
    if Path(config.data.sa1b_path).exists():
        datasets.append(ImageDataset(config.data.sa1b_path))

    if not datasets:
        logger.warning("No datasets found. Using dummy data.")
        from torch.utils.data import TensorDataset
        dummy = TensorDataset(torch.randn(100, 3, 256, 256))
        datasets = [dummy]

    from torch.utils.data import ConcatDataset
    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    dataloader = build_dataloader(dataset, batch_size=4, distributed=world_size > 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            is_video = batch["is_video"][0] if isinstance(batch["is_video"], list) else False

            if not is_video:
                pixel_values = pixel_values.unsqueeze(2)  # (B, C, 1, H, W)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                recon, z, mean, logvar = (
                    vae.module if hasattr(vae, "module") else vae
                ).forward(pixel_values)

                recon_loss = perceptual_loss(recon, pixel_values)
                kl = (vae.module if hasattr(vae, "module") else vae).kl_loss(mean, logvar)
                loss = recon_loss + config.vae.kl_weight * kl

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if rank == 0 and step % 100 == 0:
                logger.info(
                    f"Step {step} | Recon: {recon_loss.item():.4f} | KL: {kl.item():.4f}"
                )

            if rank == 0 and step % 5000 == 0:
                model = vae.module if hasattr(vae, "module") else vae
                torch.save(model.state_dict(), output_dir / f"vae_step{step}.pt")

    if rank == 0:
        model = vae.module if hasattr(vae, "module") else vae
        torch.save(model.state_dict(), output_dir / "vae_final.pt")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/vae")
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()
    train_vae(args)
