from __future__ import annotations

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    Ca2VDMConfig,
    ModelConfig,
    get_t2v_ca2vdm_config,
    get_t2v_osfix_config,
    get_vidpred_ca2vdm_config,
    get_vidpred_osext_config,
    get_vidpred_osfix_config,
)
from data import build_dataloader, build_dataset
from diffusion import GaussianDiffusion
from model import Ca2VDM, OSExt, OSFix, build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cyclic TPE Utilities
# ---------------------------------------------------------------------------

def sample_cyclic_offset(max_train_len: int) -> int:
    """Sample a random cyclic offset for TPE during training."""
    return random.randint(0, max_train_len - 1)


def get_tpe_indices_training(
    num_frames: int,
    max_train_len: int,
    cyclic_offset: int,
) -> torch.Tensor:
    """
    Get TPE indices for training with cyclic shift.

    Args:
        num_frames: L — total number of frames
        max_train_len: L_train = P_max + l
        cyclic_offset: random shift
    Returns:
        indices: (L,) in [0, max_train_len)
    """
    indices = torch.arange(num_frames)
    return (indices + cyclic_offset) % max_train_len


# ---------------------------------------------------------------------------
# Prefix Sampling
# ---------------------------------------------------------------------------

def sample_prefix_length(
    chunk_len: int,
    prefix_multiples: List[int],
    max_frames: int,
) -> int:
    """
    Sample prefix length P from {1, 1+l, 1+2l, ..., 1+nl=P_max}.

    The paper uses P in multiples of chunk length l:
    P ∈ {1, 1+l, ..., 1+nl} where P_max = 1+nl
    """
    n = random.choice(prefix_multiples)
    p = 1 + n * chunk_len
    return min(p, max_frames - 1)


# ---------------------------------------------------------------------------
# Batch Preparation
# ---------------------------------------------------------------------------

def prepare_batch(
    frames: torch.Tensor,
    prefix_frames: int,
    diffusion: GaussianDiffusion,
    device: torch.device,
    chunk_len: int,
    max_train_len: int,
    prefix_multiples: List[int],
) -> Dict[str, torch.Tensor]:
    """
    Prepare a training batch:
    1. Sample prefix length P
    2. Add noise to denoising target frames
    3. Build per-frame timestep vector t_vec
    4. Build loss mask m
    5. Assign cyclic TPE indices

    Args:
        frames: (B, L, C, H, W) — clean video frames in [-1, 1]
        prefix_frames: pre-sampled P (or -1 to sample randomly)
        diffusion: GaussianDiffusion instance
        device: target device
        chunk_len: l
        max_train_len: L_train
        prefix_multiples: list of n values for P = 1 + n*l
    Returns:
        batch dict with all training inputs
    """
    B, L, C, H, W = frames.shape
    frames = frames.to(device)

    # Sample prefix length if not provided
    if prefix_frames < 0:
        prefix_frames = sample_prefix_length(chunk_len, prefix_multiples, L)

    # Sample diffusion timestep t for denoising target
    t = torch.randint(0, diffusion.num_train_timesteps, (B,), device=device)

    # Sample noise for denoising target frames
    noise = torch.randn_like(frames[:, prefix_frames:])  # (B, L-P, C, H, W)

    # Add noise to denoising target
    t_expanded = t.reshape(B, 1, 1, 1, 1).expand(-1, L - prefix_frames, -1, -1, -1)
    t_flat = t.unsqueeze(1).expand(-1, L - prefix_frames).reshape(B * (L - prefix_frames))
    noisy_target = diffusion.q_sample(
        frames[:, prefix_frames:].reshape(B * (L - prefix_frames), C, H, W),
        t_flat,
        noise.reshape(B * (L - prefix_frames), C, H, W),
    ).reshape(B, L - prefix_frames, C, H, W)

    # Concatenate: [clean prefix | noisy target]
    z_input = torch.cat([frames[:, :prefix_frames], noisy_target], dim=1)  # (B, L, C, H, W)

    # Per-frame timestep vector: t_vec[i] = 0 if i < P else t
    t_vec = torch.zeros(B, L, dtype=torch.long, device=device)
    t_vec[:, prefix_frames:] = t.unsqueeze(1).expand(-1, L - prefix_frames)

    # Loss mask: m[i] = 1 if i >= P else 0
    loss_mask = torch.zeros(B, L, 1, 1, 1, device=device)
    loss_mask[:, prefix_frames:] = 1.0

    # Cyclic TPE indices
    cyclic_offset = sample_cyclic_offset(max_train_len)
    tpe_indices = get_tpe_indices_training(L, max_train_len, cyclic_offset).to(device)

    return {
        "z_input": z_input,
        "z_clean": frames,
        "noise": noise,
        "t": t,
        "t_vec": t_vec,
        "loss_mask": loss_mask,
        "prefix_frames": prefix_frames,
        "tpe_indices": tpe_indices,
    }


# ---------------------------------------------------------------------------
# Training Step
# ---------------------------------------------------------------------------

def training_step_ca2vdm(
    model: Ca2VDM,
    batch: Dict[str, torch.Tensor],
    diffusion: GaussianDiffusion,
    text_embeddings: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    One training step for Ca2-VDM.

    Loss: L_simple + L_vlb (Eq. 2 + Appendix B in paper)
    Applied only to denoising target frames (loss mask m).
    """
    z_input = batch["z_input"]
    z_clean = batch["z_clean"]
    noise = batch["noise"]
    t = batch["t"]
    t_vec = batch["t_vec"]
    loss_mask = batch["loss_mask"]
    prefix_frames = batch["prefix_frames"]
    tpe_indices = batch["tpe_indices"]

    B, L, C, H, W = z_input.shape

    # Forward pass
    model_output, _ = model(
        z=z_input,
        t_vec=t_vec,
        prefix_frames=prefix_frames,
        tpe_indices=tpe_indices,
        context=text_embeddings,
    )
    # model_output: (B, L, 2*C, H, W)

    # Compute losses only on denoising target
    target_output = model_output[:, prefix_frames:]  # (B, L-P, 2*C, H, W)
    target_clean = z_clean[:, prefix_frames:]
    target_noisy = z_input[:, prefix_frames:]
    target_mask = loss_mask[:, prefix_frames:]

    losses = diffusion.training_losses(
        model_output=target_output,
        x_start=target_clean,
        x_t=target_noisy,
        t=t,
        noise=noise,
        loss_mask=target_mask,
    )
    return losses


def training_step_osfix(
    model: OSFix,
    batch: Dict[str, torch.Tensor],
    diffusion: GaussianDiffusion,
    text_embeddings: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Training step for OS-Fix baseline (unified timestep embedding)."""
    z_input = batch["z_input"]
    z_clean = batch["z_clean"]
    noise = batch["noise"]
    t = batch["t"]
    t_vec = batch["t_vec"]
    loss_mask = batch["loss_mask"]
    prefix_frames = batch["prefix_frames"]
    tpe_indices = batch["tpe_indices"]

    B, L, C, H, W = z_input.shape

    model_output = model(
        z=z_input,
        t=t,
        prefix_frames=prefix_frames,
        tpe_indices=tpe_indices,
        context=text_embeddings,
    )

    target_output = model_output[:, prefix_frames:]
    target_clean = z_clean[:, prefix_frames:]
    target_noisy = z_input[:, prefix_frames:]
    target_mask = loss_mask[:, prefix_frames:]

    losses = diffusion.training_losses(
        model_output=target_output,
        x_start=target_clean,
        x_t=target_noisy,
        t=t,
        noise=noise,
        loss_mask=target_mask,
    )
    return losses


def training_step_osext(
    model: OSExt,
    batch: Dict[str, torch.Tensor],
    diffusion: GaussianDiffusion,
    text_embeddings: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Training step for OS-Ext baseline."""
    z_input = batch["z_input"]
    z_clean = batch["z_clean"]
    noise = batch["noise"]
    t = batch["t"]
    t_vec = batch["t_vec"]
    loss_mask = batch["loss_mask"]
    prefix_frames = batch["prefix_frames"]
    tpe_indices = batch["tpe_indices"]

    B, L, C, H, W = z_input.shape

    model_output = model(
        z=z_input,
        t_vec=t_vec,
        prefix_frames=prefix_frames,
        tpe_indices=tpe_indices,
        context=text_embeddings,
    )

    target_output = model_output[:, prefix_frames:]
    target_clean = z_clean[:, prefix_frames:]
    target_noisy = z_input[:, prefix_frames:]
    target_mask = loss_mask[:, prefix_frames:]

    losses = diffusion.training_losses(
        model_output=target_output,
        x_start=target_clean,
        x_t=target_noisy,
        t=t,
        noise=noise,
        loss_mask=target_mask,
    )
    return losses


# ---------------------------------------------------------------------------
# Text Encoder (T5)
# ---------------------------------------------------------------------------

class T5TextEncoder(nn.Module):
    """T5 text encoder for text-to-video generation (PixArt-α style)."""

    def __init__(self, model_name: str = "google/t5-v1_1-xxl", max_length: int = 120):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.max_length = max_length

        # Freeze T5 encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Args:
            texts: list of B text strings
            device: target device
        Returns:
            embeddings: (B, max_length, 4096)
        """
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Main trainer for Ca2-VDM and baselines.

    Implements the two-stage training procedure described in the paper:
    - Stage 1 (T2V): 32-frame videos, no clean prefix, batch=288, 32k steps
    - Stage 2 (T2V): 65-frame videos, with clean prefix, batch=144, 21k steps
    - Video prediction: 33-frame videos, batch=8, 11k steps
    """

    def __init__(
        self,
        config: Ca2VDMConfig,
        task: str = "t2v",
        model_type: str = "ca2vdm",
        stage: int = 2,
    ):
        self.config = config
        self.task = task
        self.model_type = model_type
        self.stage = stage

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._setup_training_config()
        self._build_components()

    def _setup_training_config(self) -> None:
        if self.task == "t2v":
            tc = self.config.t2v_train
            if self.stage == 1:
                self.batch_size = tc.stage1_batch_size
                self.num_steps = tc.stage1_num_steps
                self.max_frames = tc.stage1_num_frames
                self.prefix_multiples = [0]  # No prefix in stage 1
                self.chunk_len = tc.chunk_len
                self.p_max = 0
            else:
                self.batch_size = tc.stage2_batch_size
                self.num_steps = tc.stage2_num_steps
                self.max_frames = tc.max_train_frames
                self.prefix_multiples = tc.prefix_multiples
                self.chunk_len = tc.chunk_len
                self.p_max = tc.p_max
            self.max_train_len = self.p_max + self.chunk_len if self.p_max > 0 else self.max_frames
            self.learning_rate = tc.learning_rate
            self.weight_decay = tc.weight_decay
            self.grad_clip = tc.grad_clip
            self.save_every = tc.save_every
            self.log_every = tc.log_every
            self.output_dir = Path(tc.output_dir)
            self.dataset_name = tc.dataset
            self.fixed_prefix = tc.fixed_prefix
        else:  # vidpred
            tc = self.config.vidpred_train
            self.batch_size = tc.batch_size
            self.num_steps = tc.num_steps
            self.max_frames = tc.max_train_frames
            self.prefix_multiples = tc.prefix_multiples
            self.chunk_len = tc.chunk_len
            self.p_max = tc.p_max
            self.max_train_len = self.p_max + self.chunk_len
            self.learning_rate = tc.learning_rate
            self.weight_decay = tc.weight_decay
            self.grad_clip = tc.grad_clip
            self.save_every = tc.save_every
            self.log_every = tc.log_every
            self.output_dir = Path(tc.output_dir)
            self.dataset_name = tc.dataset
            self.fixed_prefix = tc.fixed_prefix

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_components(self) -> None:
        mc = self.config.model
        dc = self.config.diffusion

        # Adjust model config for task
        if self.task == "vidpred":
            mc.use_text = False
            mc.context_dim = None

        # Build model
        model_config_obj = type("ModelCfg", (), {
            "in_channels": mc.in_channels,
            "patch_size": mc.patch_size,
            "hidden_dim": mc.hidden_dim,
            "num_layers": mc.num_layers,
            "num_heads": mc.num_heads,
            "context_dim": mc.context_dim if mc.use_text else None,
            "ff_mult": mc.ff_mult,
            "dropout": mc.dropout,
            "max_spatial_h": mc.max_spatial_h,
            "max_spatial_w": mc.max_spatial_w,
            "max_temporal_len": mc.max_temporal_len,
            "chunk_len": self.chunk_len,
            "p_max": self.p_max,
            "prefix_len": mc.prefix_len,
            "use_text": mc.use_text,
            "fixed_prefix": self.fixed_prefix,
        })()

        self.model = build_model(self.model_type, model_config_obj).to(self.device)

        # Text encoder (T2V only)
        self.text_encoder = None
        if self.task == "t2v" and mc.use_text:
            self.text_encoder = T5TextEncoder().to(self.device)

        # Diffusion
        self.diffusion = GaussianDiffusion(
            num_train_timesteps=dc.num_train_timesteps,
            beta_start=dc.beta_start,
            beta_end=dc.beta_end,
            schedule=dc.schedule,
            learn_variance=dc.learn_variance,
        )

        # Optimizer (AdamW)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.config.t2v_train.adam_beta1, self.config.t2v_train.adam_beta2),
            eps=self.config.t2v_train.adam_eps,
        )

        # Mixed precision
        self.scaler = GradScaler() if self.config.mixed_precision in ("fp16", "bf16") else None
        self.use_amp = self.config.mixed_precision != "no"
        self.amp_dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16

        # Dataset
        self.config.t2v_train.batch_size = self.batch_size
        self.config.vidpred_train.batch_size = self.batch_size
        dataset = build_dataset(self.dataset_name, self._get_dataset_config(), split="train")
        self.dataloader = build_dataloader(dataset, self._get_dataset_config(), shuffle=True)
        self.data_iter = iter(self.dataloader)

        self.global_step = 0

    def _get_dataset_config(self):
        if self.task == "t2v":
            tc = self.config.t2v_train
            return type("DataCfg", (), {
                "batch_size": self.batch_size,
                "num_workers": tc.num_workers,
                "internvid_root": tc.internvid_root,
                "max_train_frames": self.max_frames,
                "chunk_len": self.chunk_len,
                "resolution": tc.resolution,
            })()
        else:
            tc = self.config.vidpred_train
            return type("DataCfg", (), {
                "batch_size": self.batch_size,
                "num_workers": tc.num_workers,
                "skytimelapse_root": tc.skytimelapse_root,
                "max_train_frames": self.max_frames,
                "chunk_len": self.chunk_len,
                "resolution": tc.resolution,
            })()

    def _get_batch(self) -> Dict:
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            return next(self.data_iter)

    def _encode_text(self, captions: List[str]) -> Optional[torch.Tensor]:
        if self.text_encoder is None:
            return None
        # Classifier-free guidance: randomly drop text condition
        if random.random() < self.config.diffusion.cfg_dropout:
            captions = [""] * len(captions)
        return self.text_encoder(captions, self.device)

    def train(self) -> None:
        """Main training loop."""
        logging.basicConfig(level=logging.INFO)
        logger.info(f"Starting training: task={self.task}, model={self.model_type}, stage={self.stage}")
        logger.info(f"Steps: {self.num_steps}, batch_size: {self.batch_size}")

        self.model.train()
        if self.text_encoder is not None:
            self.text_encoder.eval()

        pbar = tqdm(range(self.global_step, self.num_steps), desc="Training")

        for step in pbar:
            raw_batch = self._get_batch()
            frames = raw_batch["frames"]  # (B, L, C, H, W)
            captions = raw_batch.get("captions", [""] * frames.shape[0])

            # Encode text
            text_emb = self._encode_text(captions)

            # Prepare training batch
            # For OS-Fix: fixed prefix length
            if self.model_type == "osfix":
                prefix_frames = self.fixed_prefix
            else:
                prefix_frames = -1  # sample randomly

            batch = prepare_batch(
                frames=frames,
                prefix_frames=prefix_frames,
                diffusion=self.diffusion,
                device=self.device,
                chunk_len=self.chunk_len,
                max_train_len=self.max_train_len,
                prefix_multiples=self.prefix_multiples,
            )

            # Forward + loss
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                if self.model_type == "ca2vdm":
                    losses = training_step_ca2vdm(self.model, batch, self.diffusion, text_emb)
                elif self.model_type == "osfix":
                    losses = training_step_osfix(self.model, batch, self.diffusion, text_emb)
                elif self.model_type == "osext":
                    losses = training_step_osext(self.model, batch, self.diffusion, text_emb)
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")

            total_loss = losses["total"]

            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.global_step += 1

            if step % self.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "simple": f"{losses['simple'].item():.4f}",
                    "vlb": f"{losses['vlb'].item():.4f}",
                    "P": batch["prefix_frames"],
                })

            if step % self.save_every == 0 and step > 0:
                self.save_checkpoint(step)

        self.save_checkpoint(self.num_steps, final=True)
        logger.info("Training complete.")

    def save_checkpoint(self, step: int, final: bool = False) -> None:
        suffix = "final" if final else f"step_{step:06d}"
        ckpt_path = self.output_dir / f"checkpoint_{suffix}.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

    def load_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["step"]
        logger.info(f"Loaded checkpoint from step {self.global_step}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ca2-VDM")
    parser.add_argument("--task", type=str, default="t2v", choices=["t2v", "vidpred"])
    parser.add_argument("--model_type", type=str, default="ca2vdm",
                        choices=["ca2vdm", "osfix", "osext"])
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2],
                        help="Training stage (1: no prefix, 2: with prefix)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Build config
    if args.task == "t2v":
        if args.model_type == "ca2vdm":
            config = get_t2v_ca2vdm_config()
        else:
            config = get_t2v_osfix_config()
    else:
        if args.model_type == "ca2vdm":
            config = get_vidpred_ca2vdm_config()
        elif args.model_type == "osfix":
            config = get_vidpred_osfix_config()
        else:
            config = get_vidpred_osext_config()

    config.seed = args.seed

    trainer = Trainer(config, task=args.task, model_type=args.model_type, stage=args.stage)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
