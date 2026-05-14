"""Training pipeline for Pyramidal Flow Matching.

Implements the three-stage training procedure:
1. Stage 1: Image training (50k steps, 128 A100 GPUs)
   - Pure image data from LAION, CC-12M, SA-1B, JourneyDB
   - Original aspect ratio bucketing
2. Stage 2: Low-resolution video training (200k steps)
   - 80k steps at 2s, 120k steps at 5s
   - Image proportion 12.5%
3. Stage 3: High-resolution video training (50k steps)
   - 5-10s videos
   - Fine-tuning at higher resolution

Uses AdamW optimizer with stage-specific betas.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, List, Tuple
import os
import math
import time
from tqdm import tqdm
import random
import numpy as np

from model import MMDiT
from vae import VideoVAE
from pyramidal_flow import (
    pyramidal_flow_matching_loss,
    get_stage_boundaries,
    generate_pyramidal_flow,
    temporal_pyramid_condition,
)
from data import (
    ImageDataset,
    VideoDataset,
    MixedImageVideoDataset,
    collate_fn_video,
)
from config import Config, TrainingConfig, PyramidConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PyramidalFlowTrainer:
    """Trainer for pyramidal flow matching video generation."""

    def __init__(
        self,
        config: Config,
        model: MMDiT,
        vae: Optional[VideoVAE] = None,
        t5_model: Any = None,
        t5_tokenizer: Any = None,
        clip_model: Any = None,
        clip_tokenizer: Any = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.config = config
        self.model = model
        self.vae = vae
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.device = device

        self.model = self.model.to(device)
        if self.vae is not None:
            self.vae = self.vae.to(device)
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False

        self.stage_boundaries = get_stage_boundaries(
            num_stages=config.pyramid.num_stages,
        )
        self.num_stages = config.pyramid.num_stages

    def _get_optimizer(
        self,
        lr: float,
        betas: Tuple[float, float],
        weight_decay: float,
        eps: float,
    ) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )

    def _get_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _encode_text(
        self,
        captions: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode text with T5 and CLIP."""
        with torch.no_grad():
            if self.t5_tokenizer is not None:
                t5_tokens = self.t5_tokenizer(
                    captions,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)
                t5_outputs = self.t5_model(**t5_tokens)
                context = t5_outputs.last_hidden_state
                pooled = t5_outputs.last_hidden_state.mean(dim=1)
            else:
                B = len(captions)
                context = torch.randn(B, 77, self.config.dit.context_dim, device=self.device)
                pooled = torch.randn(B, self.config.dit.pooled_text_dim, device=self.device)

            if self.clip_model is not None:
                clip_tokens = self.clip_tokenizer(
                    captions,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)
                clip_outputs = self.clip_model(**clip_tokens)
                clip_context = clip_outputs.last_hidden_state
            else:
                clip_context = None

        return context, pooled, clip_context

    def _apply_cfg_noise(
        self,
        context: torch.Tensor,
        pooled: torch.Tensor,
        clip_context: Optional[torch.Tensor],
        cfg_prob: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Randomly drop text conditioning for classifier-free guidance training."""
        B = context.shape[0]
        mask = (torch.rand(B, device=context.device) < cfg_prob).float()
        mask = mask.view(B, 1, 1)

        context_dropped = context * (1 - mask)
        pooled_dropped = pooled * (1 - mask.view(B, 1))

        clip_context_dropped = None
        if clip_context is not None:
            clip_context_dropped = clip_context * (1 - mask)

        return context_dropped, pooled_dropped, clip_context_dropped

    def train_stage1(self):
        """Stage 1: Image training.

        Train on pure image data for 50k steps.
        Uses original aspect ratio and bucketing.
        """
        cfg = self.config
        tc = cfg.training

        print("=" * 60)
        print("Stage 1: Image Training")
        print(f"Steps: {tc.stage1_steps}, Batch size: {tc.stage1_batch_size}")
        print(f"Learning rate: {tc.stage1_lr}, Betas: {tc.stage1_betas}")
        print("=" * 60)

        # Build image dataset
        image_paths = {
            "laion": cfg.data.laion_path,
            "cc12m": cfg.data.cc12m_path,
            "sa1b": cfg.data.sa1b_path,
            "journeydb": cfg.data.journeydb_path,
            "synthetic": cfg.data.synthetic_data_path,
        }

        train_dataset = ImageDataset(
            data_paths=image_paths,
            image_size=cfg.data.image_size,
            latent_size=cfg.data.latent_size,
            target_h=cfg.data.image_size,
            target_w=cfg.data.image_size,
            vae=self.vae,
            tokenizer_t5=self.t5_tokenizer,
            tokenizer_clip=self.clip_tokenizer,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=tc.stage1_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        optimizer = self._get_optimizer(
            lr=tc.stage1_lr,
            betas=tc.stage1_betas,
            weight_decay=tc.weight_decay,
            eps=tc.eps,
        )
        scheduler = self._get_lr_scheduler(optimizer, tc.warmup_steps, tc.stage1_steps)
        scaler = GradScaler(enabled=(tc.mixed_precision == "fp16"))

        self.model.train()
        global_step = 0

        pbar = tqdm(total=tc.stage1_steps, desc="Stage 1")
        while global_step < tc.stage1_steps:
            for batch in train_loader:
                if global_step >= tc.stage1_steps:
                    break

                latents = batch["latent"].to(self.device)
                captions = batch["caption"]

                context, pooled, clip_context = self._encode_text(captions)
                context, pooled, clip_context = self._apply_cfg_noise(
                    context, pooled, clip_context, tc.cfg_prob
                )

                optimizer.zero_grad()

                if tc.mixed_precision in ("fp16", "bf16"):
                    dtype = torch.float16 if tc.mixed_precision == "fp16" else torch.bfloat16
                    with autocast(device_type="cuda", dtype=dtype):
                        loss = pyramidal_flow_matching_loss(
                            model=self.model,
                            x1=latents,
                            stage_boundaries=self.stage_boundaries,
                            context=context,
                            pooled_text=pooled,
                            clip_context=clip_context,
                            coupled_sampling=cfg.pyramid.coupled_sampling,
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = pyramidal_flow_matching_loss(
                        model=self.model,
                        x1=latents,
                        stage_boundaries=self.stage_boundaries,
                        context=context,
                        pooled_text=pooled,
                        clip_context=clip_context,
                        coupled_sampling=cfg.pyramid.coupled_sampling,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.gradient_clip)
                    optimizer.step()

                scheduler.step()
                global_step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                if global_step % 1000 == 0:
                    self._save_checkpoint(global_step, "stage1")

        pbar.close()
        self._save_checkpoint(global_step, "stage1_final")

    def train_stage2(self):
        """Stage 2: Low-resolution video training.

        80k steps at 2s duration, then 120k steps at 5s duration.
        Includes 12.5% image data proportion.
        """
        cfg = self.config
        tc = cfg.training

        print("=" * 60)
        print("Stage 2: Low-Resolution Video Training")
        print(f"Steps: {tc.stage2_steps}, Batch size: {tc.stage2_batch_size}")
        print(f"Learning rate: {tc.stage2_lr}, Betas: {tc.stage2_betas}")
        print("=" * 60)

        image_paths = {
            "laion": cfg.data.laion_path,
            "cc12m": cfg.data.cc12m_path,
            "sa1b": cfg.data.sa1b_path,
            "journeydb": cfg.data.journeydb_path,
            "synthetic": cfg.data.synthetic_data_path,
        }
        video_paths = {
            "webvid": cfg.data.webvid_path,
            "openvid": cfg.data.openvid_path,
            "opensora_plan": cfg.data.opensora_plan_path,
        }

        stage2a_steps = 80000
        stage2b_steps = tc.stage2_steps - stage2a_steps

        # Stage 2a: 2-second videos
        print("\nStage 2a: 2-second video training (80k steps)")
        for duration, steps in [(2.0, stage2a_steps), (5.0, stage2b_steps)]:
            print(f"\nTraining on {duration}s videos for {steps} steps")

            video_dataset = VideoDataset(
                data_paths=video_paths,
                image_size=cfg.data.image_size,
                latent_size=cfg.data.latent_size,
                fps=cfg.data.fps,
                min_duration=duration,
                max_duration=duration,
                vae=self.vae,
                tokenizer_t5=self.t5_tokenizer,
                tokenizer_clip=self.clip_tokenizer,
                latent_temporal_compression=cfg.data.latent_temporal_compression,
                history_frames=3,
                history_noise_max=tc.history_noise_max,
            )

            image_dataset = ImageDataset(
                data_paths=image_paths,
                image_size=cfg.data.image_size,
                latent_size=cfg.data.latent_size,
                target_h=cfg.data.image_size,
                target_w=cfg.data.image_size,
                vae=self.vae,
                tokenizer_t5=self.t5_tokenizer,
                tokenizer_clip=self.clip_tokenizer,
            )

            train_dataset = MixedImageVideoDataset(
                image_dataset=image_dataset,
                video_dataset=video_dataset,
                image_proportion=tc.image_proportion,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=tc.stage2_batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
            )

            optimizer = self._get_optimizer(
                lr=tc.stage2_lr,
                betas=tc.stage2_betas,
                weight_decay=tc.weight_decay,
                eps=tc.eps,
            )
            scheduler = self._get_lr_scheduler(optimizer, tc.warmup_steps, steps)
            scaler = GradScaler(enabled=(tc.mixed_precision == "fp16"))

            self.model.train()
            global_step = 0
            pbar = tqdm(total=steps, desc=f"Video {duration}s")

            while global_step < steps:
                for batch in train_loader:
                    if global_step >= steps:
                        break

                    latents = batch["latent"].to(self.device)
                    captions = batch["caption"]

                    context, pooled, clip_context = self._encode_text(captions)
                    context, pooled, clip_context = self._apply_cfg_noise(
                        context, pooled, clip_context, tc.cfg_prob
                    )

                    if latents.dim() == 3:
                        latents = latents.unsqueeze(2)

                    optimizer.zero_grad()

                    if tc.mixed_precision in ("fp16", "bf16"):
                        dtype = torch.float16 if tc.mixed_precision == "fp16" else torch.bfloat16
                        with autocast(device_type="cuda", dtype=dtype):
                            loss = self._compute_video_loss(
                                latents, context, pooled, clip_context, batch
                            )
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.gradient_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss = self._compute_video_loss(
                            latents, context, pooled, clip_context, batch
                        )
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.gradient_clip)
                        optimizer.step()

                    scheduler.step()
                    global_step += 1
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                    if global_step % 1000 == 0:
                        self._save_checkpoint(global_step, f"stage2_{int(duration)}s")

            pbar.close()

        self._save_checkpoint(0, "stage2_final")

    def train_stage3(self):
        """Stage 3: High-resolution video training.

        50k steps on 5-10s videos at high resolution.
        """
        cfg = self.config
        tc = cfg.training

        print("=" * 60)
        print("Stage 3: High-Resolution Video Training")
        print(f"Steps: {tc.stage3_steps}, Batch size: {tc.stage3_batch_size}")
        print(f"Learning rate: {tc.stage3_lr}, Betas: {tc.stage3_betas}")
        print("=" * 60)

        video_paths = {
            "webvid": cfg.data.webvid_path,
            "openvid": cfg.data.openvid_path,
            "opensora_plan": cfg.data.opensora_plan_path,
        }

        video_dataset = VideoDataset(
            data_paths=video_paths,
            image_size=cfg.data.image_size,
            latent_size=cfg.data.latent_size,
            fps=cfg.data.fps,
            min_duration=cfg.data.min_duration,
            max_duration=cfg.data.max_duration,
            vae=self.vae,
            tokenizer_t5=self.t5_tokenizer,
            tokenizer_clip=self.clip_tokenizer,
            latent_temporal_compression=cfg.data.latent_temporal_compression,
            history_frames=3,
            history_noise_max=tc.history_noise_max,
        )

        train_loader = DataLoader(
            video_dataset,
            batch_size=tc.stage3_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        optimizer = self._get_optimizer(
            lr=tc.stage3_lr,
            betas=tc.stage3_betas,
            weight_decay=tc.weight_decay,
            eps=tc.eps,
        )
        scheduler = self._get_lr_scheduler(optimizer, tc.warmup_steps, tc.stage3_steps)
        scaler = GradScaler(enabled=(tc.mixed_precision == "fp16"))

        self.model.train()
        global_step = 0
        pbar = tqdm(total=tc.stage3_steps, desc="Stage 3")

        while global_step < tc.stage3_steps:
            for batch in train_loader:
                if global_step >= tc.stage3_steps:
                    break

                latents = batch["latent"].to(self.device)
                captions = batch["caption"]

                context, pooled, clip_context = self._encode_text(captions)
                context, pooled, clip_context = self._apply_cfg_noise(
                    context, pooled, clip_context, tc.cfg_prob
                )

                optimizer.zero_grad()

                if tc.mixed_precision in ("fp16", "bf16"):
                    dtype = torch.float16 if tc.mixed_precision == "fp16" else torch.bfloat16
                    with autocast(device_type="cuda", dtype=dtype):
                        loss = self._compute_video_loss(
                            latents, context, pooled, clip_context, batch
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = self._compute_video_loss(
                        latents, context, pooled, clip_context, batch
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.gradient_clip)
                    optimizer.step()

                scheduler.step()
                global_step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                if global_step % 1000 == 0:
                    self._save_checkpoint(global_step, "stage3")

        pbar.close()
        self._save_checkpoint(global_step, "stage3_final")

    def _compute_video_loss(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        pooled: torch.Tensor,
        clip_context: Optional[torch.Tensor],
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute pyramidal flow matching loss for video data.

        Handles temporal pyramid conditioning and autoregressive training.
        """
        B, T, C, H, W = latents.shape

        total_loss = torch.tensor(0.0, device=latents.device)

        # For each frame, build temporal pyramid condition and compute loss
        num_frames = T
        history_frames = 3

        for t in range(num_frames):
            frame_latent = latents[:, t]

            # Build temporal pyramid conditions
            history_conds = []
            for h in range(1, min(history_frames, t) + 1):
                hist_latent = latents[:, t - h]
                comp_factor = 2 ** h
                hist_latent = F.interpolate(
                    hist_latent,
                    scale_factor=1.0 / comp_factor,
                    mode="bilinear",
                    align_corners=False,
                )
                history_conds.append(hist_latent)

            # Compute frame loss
            loss = pyramidal_flow_matching_loss(
                model=self.model,
                x1=frame_latent,
                stage_boundaries=self.stage_boundaries,
                context=context,
                pooled_text=pooled,
                clip_context=clip_context,
                coupled_sampling=self.config.pyramid.coupled_sampling,
            )
            total_loss = total_loss + loss

        return total_loss / max(1, num_frames)

    def _save_checkpoint(self, step: int, stage: str):
        """Save model checkpoint."""
        save_dir = os.path.join(self.config.output_dir, "checkpoints", stage)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"checkpoint_{step}.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "step": step,
                "stage": stage,
                "config": self.config,
            },
            save_path,
        )
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint loaded from {checkpoint_path} (step {checkpoint['step']})")

    def train(self):
        """Run full training pipeline (Stages 1, 2, 3)."""
        self.train_stage1()
        self.train_stage2()
        self.train_stage3()


def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Pyramidal Flow Matching Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=0, help="Training stage (0=all)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    args = parser.parse_args()

    config = Config()

    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = MMDiT(
        num_layers=config.dit.num_layers,
        hidden_size=config.dit.hidden_size,
        num_heads=config.dit.num_heads,
        head_dim=config.dit.head_dim,
        ff_mult=config.dit.ff_mult,
        patch_size=config.dit.patch_size,
        in_channels=config.dit.in_channels,
        out_channels=config.dit.out_channels,
        pooled_text_dim=config.dit.pooled_text_dim,
        context_dim=config.dit.context_dim,
        clip_dim=config.dit.clip_dim,
        rope_theta=config.dit.rope_theta,
        dropout=config.dit.dropout,
        qk_norm=config.dit.qk_norm,
    )

    # Build VAE (optional, can be used externally)
    vae = None

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Text encoders (lazy loading)
    t5_model = None
    t5_tokenizer = None
    clip_model = None
    clip_tokenizer = None

    try:
        from transformers import T5EncoderModel, T5Tokenizer
        t5_model = T5EncoderModel.from_pretrained(config.data.t5_model).to(device)
        t5_tokenizer = T5Tokenizer.from_pretrained(config.data.t5_model)
        t5_model.eval()
        for p in t5_model.parameters():
            p.requires_grad = False
        print("T5 model loaded")
    except Exception:
        print("T5 model not available, using random embeddings")

    try:
        from transformers import CLIPTextModel, CLIPTokenizer
        clip_model = CLIPTextModel.from_pretrained(config.data.clip_model).to(device)
        clip_tokenizer = CLIPTokenizer.from_pretrained(config.data.clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        print("CLIP model loaded")
    except Exception:
        print("CLIP model not available")

    trainer = PyramidalFlowTrainer(
        config=config,
        model=model,
        vae=vae,
        t5_model=t5_model,
        t5_tokenizer=t5_tokenizer,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if args.stage == 1:
        trainer.train_stage1()
    elif args.stage == 2:
        trainer.train_stage2()
    elif args.stage == 3:
        trainer.train_stage3()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
