import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict
import argparse
from tqdm import tqdm

from config import Config, get_t2v_config, get_video_prediction_config
from model import Ca2VDM, Ca2VDM_Bidirectional
from data import VideoDataset, VideoPredictionDataset, get_dataloader


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    mixed_precision: bool = False,
) -> Dict[str, float]:
    """Single training step."""
    latent = batch["latent"].to(device)  # (B, L, C, H, W)
    P = batch["prefix_len"].item() if isinstance(batch["prefix_len"], int) else batch["prefix_len"][0].item()
    tpe_offset = batch["tpe_offset"].item() if isinstance(batch["tpe_offset"], int) else batch["tpe_offset"][0].item()
    
    B, L, C, H, W = latent.shape
    
    # Sample timestep uniformly
    t = torch.randint(0, 1000, (B,), device=device)
    
    # Sample noise
    noise = torch.randn_like(latent)
    
    # Get text embeddings if available
    encoder_hidden_states = batch.get("text_embed", None)
    if encoder_hidden_states is not None:
        encoder_hidden_states = encoder_hidden_states.to(device)

    if mixed_precision:
        with autocast():
            loss_dict = model.get_loss(latent, noise, t, P, encoder_hidden_states, tpe_offset)
            loss = loss_dict["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_dict = model.get_loss(latent, noise, t, P, encoder_hidden_states, tpe_offset)
        loss = loss_dict["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    optimizer.zero_grad()

    return {
        "loss": loss.item(),
        "simple_loss": loss_dict.get("simple_loss", loss).item(),
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
    }


def train(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    config: Config,
    save_dir: str,
    start_step: int = 0,
    total_steps: int = 10000,
):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
    )

    dataloader = get_dataloader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
    )

    use_amp = config.training.mixed_precision == "fp16"
    scaler = GradScaler() if use_amp else None

    os.makedirs(save_dir, exist_ok=True)

    step = start_step
    pbar = tqdm(total=total_steps, initial=start_step, desc="Training")

    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break

            metrics = train_step(
                model, batch, optimizer, device, scaler, use_amp
            )

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "simple": f"{metrics['simple_loss']:.4f}",
            })
            pbar.update(1)
            step += 1

            # Logging
            if step % 100 == 0:
                print(f"Step {step}: loss={metrics['loss']:.4f}, simple_loss={metrics['simple_loss']:.4f}")

            # Save checkpoint
            if step % 5000 == 0:
                ckpt_path = os.path.join(save_dir, f"checkpoint_{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    pbar.close()

    # Final save
    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)
    print(f"Training complete. Model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Ca2-VDM")
    parser.add_argument("--task", type=str, default="t2v", choices=["t2v", "video_prediction"])
    parser.add_argument("--model_type", type=str, default="ca2_vdm", choices=["ca2_vdm", "os_fix", "os_ext"])
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.task == "t2v":
        config = get_t2v_config()
    else:
        config = get_video_prediction_config()

    # Build model
    if args.model_type == "ca2_vdm":
        model = Ca2VDM(
            hidden_size=config.model.hidden_size,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            spatial_head_dim=config.model.spatial_attn_head_dim,
            temporal_head_dim=config.model.temporal_attn_head_dim,
            cross_head_dim=config.model.cross_attn_head_dim,
            cross_attn_dim=config.model.text_encoder_dim,
            prefix_len_enhance=config.model.prefix_len_enhance,
            max_train_len=config.training.max_train_len,
            patch_size=config.model.patch_size,
            latent_channels=config.model.latent_channels,
            spatial_size=config.model.spatial_size,
            learn_sigma=True,
        )
    elif args.model_type in ("os_fix", "os_ext"):
        model = Ca2VDM_Bidirectional(
            hidden_size=config.model.hidden_size,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            head_dim=config.model.spatial_attn_head_dim,
            cross_attn_dim=config.model.text_encoder_dim,
            max_train_len=config.training.max_train_len,
            patch_size=config.model.patch_size,
            latent_channels=config.model.latent_channels,
            spatial_size=config.model.spatial_size,
            learn_sigma=True,
        )

    # Initialize from Open-Sora v1.0 pretrained weights if available
    # (placeholder — replace with actual weight loading)
    # _load_pretrained_weights(model, "open_sora_v1.0.pt")

    # Build dataset
    if args.task == "t2v":
        dataset = VideoDataset(
            video_dir=args.data_dir,
            vae=None,  # Placeholder — load actual VAE
            text_encoder=None,  # Placeholder — load T5
            resolution=config.training.video_resolution,
            chunk_length=config.training.video_frames_chunk,
            max_prefix_len=config.training.max_prefix_len,
            max_train_len=config.training.max_train_len,
            is_t2v=True,
            stage=args.stage,
        )
    else:
        dataset = VideoPredictionDataset(
            video_dir=args.data_dir,
            vae=None,
            resolution=config.training.video_resolution,
            chunk_length=config.training.video_frames_chunk,
            max_prefix_len=config.training.max_prefix_len,
            max_train_len=config.training.max_train_len,
        )

    # Determine training steps
    if args.stage == 1:
        total_steps = config.training.stage1_steps
    else:
        total_steps = config.training.stage2_steps

    # Resume if provided
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume} at step {start_step}")

    train(model, dataset, config, args.save_dir, start_step, total_steps)


if __name__ == "__main__":
    main()
