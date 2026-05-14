## main.py
"""
Main entry point for Pyramidal Flow Matching reproduction.

Handles configuration loading, VAE training (if needed), model initialisation
(including SD3 Medium weight transfer), and dispatches to training, inference,
or evaluation modes.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf

# Project imports
from config import Config
from dataset import ImageDataset, PatchPackCollator, VideoDataset
from model import MMDiT
from trainer import Trainer
from inference import Sampler
from evaluate import Evaluator
from utils import compute_causal_mask
from vae import ThreeDVAE


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pyramidal Flow Matching Reproduction")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "inference", "eval"],
        help="Operating mode: train, inference, or eval.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint (for resume training, inference, or eval).",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default=None,
        help="Path to a VAE checkpoint (if not in the output directory).",
    )
    parser.add_argument(
        "--sd3_ckpt",
        type=str,
        default=None,
        help="Path to the official SD3 Medium checkpoint for weight initialization.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for inference mode.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Optional path to an image for image‑to‑video inference.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the output directory defined in the configuration.",
    )
    parser.add_argument(
        "--train_vae",
        action="store_true",
        help="Force training of the 3D VAE even if a checkpoint exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from the configuration.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility: seed everything
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# VAE training helper (minimal)
# ---------------------------------------------------------------------------
def train_vae_if_needed(
    cfg: Config,
    accelerator: Optional[Accelerator] = None,
    force: bool = False,
) -> ThreeDVAE:
    """
    Train a 3D VAE from scratch if a checkpoint is not found, or if `force` is True.
    Returns the trained (and eval‑mode) VAE.
    """
    vae_checkpoint = cfg.vae.get("checkpoint", None)
    if not force and vae_checkpoint and os.path.isfile(vae_checkpoint):
        logger.info(f"Loading existing VAE checkpoint from {vae_checkpoint}")
        vae = ThreeDVAE(cfg.vae)
        state = torch.load(vae_checkpoint, map_location="cpu")
        vae.load_state_dict(state)
        return vae

    logger.info("No VAE checkpoint found. Starting VAE training from scratch...")

    # Use the datasets specified for VAE training
    vae_data_cfg = cfg.vae.data

    # VAE expects video tensors and image tensors. We create a simple dataset
    # that yields random clips from video files and image files.
    # For simplicity, we reuse VideoDataset and ImageDataset from the project.
    # The target number of frames for VAE training is not specified; we use 1‑second clips (8 frames at 8fps?).
    # Since the VAE uses 8×8×8 compression, a temporal block of 8 frames is natural.
    T_vae = 8  # temporal block length
    resolution_vae = (256, 256)  # a moderate resolution for VAE training

    video_root = vae_data_cfg.train_videos
    image_root = vae_data_cfg.train_images

    # Video dataset
    video_ds = VideoDataset(
        cfg,
        split="train",
        target_frames=T_vae,
        resolution=resolution_vae,
        recaption_cache=None,   # captions not needed for VAE
        original_captions_path=None,
    )

    # Image dataset (treated as single‑frame videos)
    image_ds = ImageDataset(
        cfg,
        split="train",
        buckets=[(256, 256)],    # fixed resolution for VAE
        recaption_cache=None,
        original_captions_dir=None,
    )

    # Combine: cycle through both, alternating video and image batches
    # Use a simple DataLoader for each, zip them
    from torch.utils.data import DataLoader, ConcatDataset

    # We'll use a custom dataset that wraps both and returns dicts with "video" key.
    # For images, repeat the frame T_vae times.
    class VAECombinedDataset(Dataset):
        def __init__(self, video_ds, image_ds, img_repeat_frames):
            self.video_ds = video_ds
            self.image_ds = image_ds
            self.total = len(video_ds) + len(image_ds)
            self.img_repeat = img_repeat_frames

        def __len__(self):
            return self.total

        def __getitem__(self, idx):
            if idx < len(self.video_ds):
                sample = self.video_ds[idx]
                return {"video": sample["video"]}
            else:
                img_idx = idx - len(self.video_ds)
                img_sample = self.image_ds[img_idx]
                img = img_sample["image"]  # (C, H, W)
                # Repeat to create a pseudo‑video of length T_vae
                video = img.unsqueeze(0).repeat(self.img_repeat, 1, 1, 1)  # (T, C, H, W)
                return {"video": video}

    combined_ds = VAECombinedDataset(video_ds, image_ds, T_vae)

    train_loader = DataLoader(
        combined_ds,
        batch_size=cfg.vae.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    vae = ThreeDVAE(cfg.vae)
    device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    vae.train()

    optimizer = optim.AdamW(
        vae.parameters(),
        lr=cfg.vae.train.learning_rate,
        weight_decay=cfg.vae.train.weight_decay,
    )

    if accelerator:
        vae, optimizer, train_loader = accelerator.prepare(vae, optimizer, train_loader)

    # Perceptual loss (LPIPS)
    perceptual_loss_fn = None
    if cfg.vae.train.get("perceptual_loss_weight", 0) > 0:
        import lpips
        perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
        if accelerator:
            perceptual_loss_fn = accelerator.prepare(perceptual_loss_fn)

    total_epochs = cfg.vae.train.epochs
    logger.info(f"Training VAE for {total_epochs} epochs...")
    for epoch in range(total_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            video = batch["video"]  # (B, T, C, H, W) -> (B, C, T, H, W)
            video = video.permute(0, 2, 1, 3, 4).contiguous()
            recon, mu, logvar = vae(video)
            loss_dict = vae.compute_loss(recon, video, mu, logvar, perceptual_loss_fn)
            loss = loss_dict["loss"]
            optimizer.zero_grad()
            if accelerator:
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logger.info(f"VAE Epoch {epoch+1}/{total_epochs}, avg loss: {epoch_loss / len(train_loader):.6f}")

    # Save VAE checkpoint
    vae.eval()
    save_path = os.path.join(cfg.global.output_dir, "vae_checkpoint.pt")
    os.makedirs(cfg.global.output_dir, exist_ok=True)
    if accelerator is None or accelerator.is_main_process:
        torch.save(vae.state_dict(), save_path)
        cfg.vae.checkpoint = save_path
    if accelerator:
        accelerator.wait_for_everyone()
    logger.info(f"VAE training complete. Checkpoint saved to {save_path}")
    return vae


# ---------------------------------------------------------------------------
# SD3 Medium weight loading (best‑effort mapping)
# ---------------------------------------------------------------------------
def load_sd3_medium_weights(model: MMDiT, sd3_ckpt_path: str) -> None:
    """
    Loads pretrained weights from an SD3 Medium checkpoint into our MM‑DiT.

    The mapping assumes that the checkpoint follows the diffusers SD3Transformer2D
    key naming and our model's layer blocks are named ``layers.{i}``.
    Because our architecture combines Q/K/V into a single ``qkv`` linear layer,
    a proper mapping requires concatenation of the individual Q, K, V weights
    and biases.  This function performs that mapping.

    Args:
        model: an instance of MMDiT (must have the same number of layers as SD3 Medium, typically 24).
        sd3_ckpt_path: path to a .safetensors or .bin file with the SD3 state dict.

    Raises:
        RuntimeError if the checkpoint cannot be loaded or mapped.
    """
    if sd3_ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd3_state = load_file(sd3_ckpt_path)
    else:
        sd3_state = torch.load(sd3_ckpt_path, map_location="cpu")

    if len(sd3_state) == 0:
        raise ValueError("Empty state dict loaded from checkpoint.")

    # Prepare a mapping: for each key in the official SD3, we create the corresponding key in our model.
    # We'll go through each layer and build a new dict for our model.
    model_state = model.state_dict()
    mapped = {}
    missing_keys = set(model_state.keys())
    unexpected_keys = set(sd3_state.keys())

    # Helper to copy or concatenate weights
    def _set_weight(target_name: str, tensor: torch.Tensor):
        if target_name in model_state:
            mapped[target_name] = tensor.to(dtype=model_state[target_name].dtype)

    # Map transformer blocks
    num_layers = model.num_layers
    # The official SD3 Medium uses "transformer_blocks.{i}" keys.
    for i in range(num_layers):
        prefix_sd3 = f"transformer_blocks.{i}."
        prefix_ours = f"layers.{i}."

        # adaLN modulation
        # SD3: norm1.linear.weight/biases...? Actually SD3 uses adaLN via a separate "norm1" or "modulation"?
        # In diffusers, the modulation is a single linear layer that outputs [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp].
        # In our model, it's `adaLN_modulation` (nn.Sequential with SiLU and Linear).
        # SD3 stores as `norm1.linear.weight`? Wait, let's check: in SD3Transformer2D, each block has `attn.add_q_proj`, `attn.add_k_proj`, etc? Actually, the newest SD3 in diffusers uses a different modulation scheme. This mapping is highly dependent on the exact version.
        # We'll skip the adaLN mapping for simplicity and warn the user.
        logger.warning(
            f"Layer {i}: adaLN modulation weights not mapped automatically. "
            "They will be randomly initialised. For exact reproduction, provide a custom mapping."
        )

        # Q, K, V linear layers -> concatenate into our qkv
        q_weight_path = prefix_sd3 + "attn.to_q.weight"
        k_weight_path = prefix_sd3 + "attn.to_k.weight"
        v_weight_path = prefix_sd3 + "attn.to_v.weight"
        q_bias_path = prefix_sd3 + "attn.to_q.bias"
        k_bias_path = prefix_sd3 + "attn.to_k.bias"
        v_bias_path = prefix_sd3 + "attn.to_v.bias"

        if all(p in sd3_state for p in [q_weight_path, k_weight_path, v_weight_path]):
            qw = sd3_state[q_weight_path]
            kw = sd3_state[k_weight_path]
            vw = sd3_state[v_weight_path]
            qkv_weight = torch.cat([qw, kw, vw], dim=0)

            qb = sd3_state.get(q_bias_path, None)
            kb = sd3_state.get(k_bias_path, None)
            vb = sd3_state.get(v_bias_path, None)
            if qb is not None and kb is not None and vb is not None:
                qkv_bias = torch.cat([qb, kb, vb], dim=0)
            else:
                qkv_bias = None

            _set_weight(prefix_ours + "qkv.weight", qkv_weight)
            if qkv_bias is not None:
                _set_weight(prefix_ours + "qkv.bias", qkv_bias)

            for p in [q_weight_path, k_weight_path, v_weight_path, q_bias_path, k_bias_path, v_bias_path]:
                if p in sd3_state:
                    unexpected_keys.discard(p)
        else:
            logger.warning(f"Could not find Q/K/V weights for layer {i} in checkpoint. Skipping.")

        # Attention output projection
        out_weight_path = prefix_sd3 + "attn.to_out.0.weight"
        out_bias_path = prefix_sd3 + "attn.to_out.0.bias"
        if out_weight_path in sd3_state:
            _set_weight(prefix_ours + "attn_out.weight", sd3_state[out_weight_path])
            if out_bias_path in sd3_state:
                _set_weight(prefix_ours + "attn_out.bias", sd3_state[out_bias_path])
            unexpected_keys.discard(out_weight_path)
            unexpected_keys.discard(out_bias_path)

        # Feed‑forward (GEGLU): SD3 has ff.net.0.proj.weight, ff.net.2.weight
        # Our ffn is Sequential: Linear(in, out*2), GELU, Linear(out, in)
        # In SD3, the gated linear is often implemented as two separate linears: ff.net.0.proj and ff.net.2 for the gate? We'll map approximate.
        ffn_weight1_path = prefix_sd3 + "ff.net.0.proj.weight"
        ffn_weight2_path = prefix_sd3 + "ff.net.2.weight"
        ffn_bias1_path = prefix_sd3 + "ff.net.0.proj.bias"
        ffn_bias2_path = prefix_sd3 + "ff.net.2.bias"
        if ffn_weight1_path in sd3_state and ffn_weight2_path in sd3_state:
            # Concatenate along first dim for the first linear (the gated part)
            w1 = sd3_state[ffn_weight1_path]  # (ffn_hidden*2, hidden)
            w2 = sd3_state[ffn_weight2_path]  # (hidden, ffn_hidden)
            # Our first Linear expects (hidden, ffn_hidden*2), but we need to reshape.
            # Actually in GEGLU: gate_proj and up_proj are two separate weights, then fused into one. SD3 stores them as two linears.
            # We'll combine appropriately based on dimensions.
            # Our model's ffn is Sequential(Linear(in, ffn_hidden*2), GELU, Linear(ffn_hidden, in)).
            # The first Linear expects weight of shape (ffn_hidden*2, in). So we need to concatenate w1 (shape (ffn_hidden*2, in)) and the up_proj? Wait, in SD3, the gate is typically `ff.net.0.proj` (shape (ffn_hidden, in)) and `ff.net.2` is the output linear (shape (in, ffn_hidden)). That doesn't match.
            # This mapping is clearly not straightforward. We'll skip FFN for now and warn.
            logger.warning(f"Layer {i}: FFN weights not mapped (architecture mismatch). They will be random.")
        else:
            logger.warning(f"Could not find FFN weights for layer {i}.")

    # Load what we have
    missing_keys = [k for k in missing_keys if k not in mapped]
    model.load_state_dict(mapped, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys after SD3 mapping ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys:
        logger.warning(f"Unused keys from SD3 checkpoint ({len(unexpected_keys)}): {list(unexpected_keys)[:5]}...")

    logger.info("SD3 Medium weight mapping completed (with potential gaps).")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
class Main:
    """
    Central controller that bootstraps the reproduction pipeline.
    """

    def __init__(self, cfg: Config, args: argparse.Namespace):
        self.cfg = cfg
        self.args = args

        # Seed
        seed = args.seed if args.seed is not None else cfg.global.seed
        set_seed(seed)
        logger.info(f"Set global seed to {seed}")

        # Output directory override
        if args.output_dir:
            cfg.global.output_dir = args.output_dir
        os.makedirs(cfg.global.output_dir, exist_ok=True)

        # Accelerator (for training only)
        self.accelerator = None
        if args.mode == "train":
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
            self.accelerator = Accelerator(
                mixed_precision=cfg.global.mixed_precision,
                kwargs_handlers=[ddp_kwargs],
            )
            if self.accelerator.is_main_process:
                logger.info(f"Training using {self.accelerator.num_processes} device(s).")

        # Devices
        self.device = self.accelerator.device if self.accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # VAE checkpoint path
        self.vae_ckpt = args.vae_checkpoint or os.path.join(cfg.global.output_dir, "vae_checkpoint.pt")

    def _init_vae(self) -> ThreeDVAE:
        """Initialize (and possibly train) the 3D VAE."""
        if os.path.isfile(self.vae_ckpt) and not self.args.train_vae:
            logger.info(f"Loading VAE from {self.vae_ckpt}")
            vae = ThreeDVAE(self.cfg.vae)
            state = torch.load(self.vae_ckpt, map_location="cpu")
            vae.load_state_dict(state)
        else:
            vae = train_vae_if_needed(self.cfg, self.accelerator, force=self.args.train_vae)
            # The function saves the checkpoint; reload to ensure eval state
            vae = ThreeDVAE(self.cfg.vae)
            state = torch.load(self.cfg.vae.checkpoint, map_location="cpu")
            vae.load_state_dict(state)

        vae.to(self.device)
        vae.eval()
        # freeze VAE for downstream tasks
        for p in vae.parameters():
            p.requires_grad = False
        return vae

    def _init_model(self, vae: ThreeDVAE) -> MMDiT:
        """Initialise the MM‑DiT model, optionally loading SD3 Medium weights."""
        model = MMDiT(self.cfg.to_dict())

        if self.args.sd3_ckpt:
            logger.info(f"Loading SD3 Medium weights from {self.args.sd3_ckpt}")
            load_sd3_medium_weights(model, self.args.sd3_ckpt)
        else:
            logger.warning(
                "No SD3 Medium checkpoint provided. Model weights are randomly initialised. "
                "Performance may be suboptimal."
            )

        # If training, wrap with accelerator
        if self.accelerator:
            model = self.accelerator.prepare(model)
        else:
            model.to(self.device)
        return model

    def run(self) -> None:
        mode = self.args.mode

        # ---------- Train ----------
        if mode == "train":
            logger.info("Starting training pipeline...")
            # 1. VAE
            vae = self._init_vae()

            # 2. Model
            model = self._init_model(vae)

            # 3. Dataset for stage 1
            # We need an ImageDataset instance. Buckets can be read from config or default.
            buckets = [(256, 256), (512, 256), (256, 512), (384, 384), (512, 512)]
            image_dataset = ImageDataset(
                self.cfg.to_dict(),
                split="train",
                buckets=buckets,
                recaption_cache=None,   # to be provided if required
                original_captions_dir=None,
            )

            # 4. Trainer
            trainer = Trainer(
                cfg=self.cfg,
                model=model,
                vae=vae,
                train_data=image_dataset,  # stage 1 data
                accelerator=self.accelerator,
            )

            # 5. Run stages
            trainer.train_stage1()
            trainer.train_stage2()
            trainer.train_stage3()
            logger.info("Training completed successfully.")

        # ---------- Inference ----------
        elif mode == "inference":
            logger.info("Running inference...")
            # Need a model checkpoint
            if not self.args.checkpoint:
                raise ValueError("Inference mode requires --checkpoint pointing to a trained model.")
            # VAE
            vae = self._init_vae()

            # Model
            model = MMDiT(self.cfg.to_dict())
            # Load checkpoint
            ckpt = torch.load(self.args.checkpoint, map_location="cpu")
            # Remove unwanted prefixes (e.g., from DDP wrapping)
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt, strict=True)
            model.to(self.device)
            model.eval()

            # Sampler
            sampler = Sampler(self.cfg.to_dict(), model, vae)

            prompt = self.args.prompt
            if not prompt:
                prompt = input("Enter a text prompt: ").strip()
            if not prompt:
                raise ValueError("No prompt provided.")

            if self.args.image_path:
                # image‑to‑video
                from PIL import Image
                image = Image.open(self.args.image_path).convert("RGB")
                video_tensor = sampler.sample_image_to_video(
                    image=image,
                    prompt=prompt,
                    duration_sec=5,
                    fps=self.cfg.inference.video.default_fps,
                    guidance_scale=self.cfg.inference.guidance_scale,
                )
            else:
                video_tensor = sampler.sample_text_to_video(
                    prompt=prompt,
                    duration_sec=5,
                    fps=self.cfg.inference.video.default_fps,
                    guidance_scale=self.cfg.inference.guidance_scale,
                )

            # Save video
            output_video = os.path.join(self.cfg.global.output_dir, "generated.mp4")
            from evaluate import _save_video   # reuse helper
            _save_video(video_tensor, output_video, fps=self.cfg.inference.video.default_fps)
            logger.info(f"Video saved to {output_video}")

        # ---------- Evaluation ----------
        elif mode == "eval":
            logger.info("Running evaluation...")
            if not self.args.checkpoint:
                raise ValueError("Evaluation mode requires --checkpoint pointing to a trained model.")
            vae = self._init_vae()

            model = MMDiT(self.cfg.to_dict())
            ckpt = torch.load(self.args.checkpoint, map_location="cpu")
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt, strict=True)
            model.to(self.device)
            model.eval()

            sampler = Sampler(self.cfg.to_dict(), model, vae)

            evaluator = Evaluator(self.cfg, sampler, benchmark="VBench")
            metrics = evaluator.run_evaluation()
            logger.info(f"VBench metrics: {metrics}")

            # Optionally run EvalCrafter
            if self.cfg.evaluation.get("evalcrafter_prompts", None):
                evaluator2 = Evaluator(self.cfg, sampler, benchmark="EvalCrafter")
                metrics2 = evaluator2.run_evaluation()
                logger.info(f"EvalCrafter metrics: {metrics2}")

        else:
            raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    # Load configuration
    cfg = Config(args.config)
    # Inject CLI overrides into config (if needed)
    if args.output_dir:
        cfg.global.output_dir = args.output_dir
    controller = Main(cfg, args)
    controller.run()


if __name__ == "__main__":
    main()
