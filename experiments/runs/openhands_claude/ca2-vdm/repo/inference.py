from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Ca2VDMConfig, InferenceConfig, get_t2v_ca2vdm_config, get_vidpred_ca2vdm_config
from diffusion import ImprovedDDPMSampler
from model import Ca2VDM, OSExt, OSFix, assign_cyclic_tpe, build_model
from modules import SpatialKVCache, TemporalKVCacheQueue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TPE Assignment for Inference
# ---------------------------------------------------------------------------

def get_inference_tpe_indices(
    ar_step: int,
    chunk_len: int,
    p_max: int,
    max_train_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Assign TPE indices for the denoising target at AR step k.

    When P_k < P_max: assign sequential TPEs [P_k, P_k+1, ..., P_k+l-1]
    When P_k >= P_max: apply Cyclic-TPE — wrap around from beginning.

    The denoising target always gets TPEs indexed from the beginning
    of the cyclic sequence (Figure 4(c) in paper).

    Args:
        ar_step: current AR step k (0-indexed)
        chunk_len: l
        p_max: P_max
        max_train_len: L_train = P_max + l
        device: target device
    Returns:
        tpe_indices: (l,) indices for the denoising target chunk
    """
    p_k = min(1 + ar_step * chunk_len, p_max)

    if p_k < p_max:
        # Sequential assignment: denoising target follows prefix
        start = p_k
        indices = torch.arange(start, start + chunk_len, device=device)
    else:
        # Cyclic-TPE: denoising target wraps to beginning
        # The prefix KV-cache already has TPEs bound; denoising target
        # gets TPEs from [0, l) cyclically
        indices = torch.arange(0, chunk_len, device=device)

    return indices % max_train_len


def get_full_tpe_indices_for_cache_write(
    ar_step: int,
    chunk_len: int,
    p_max: int,
    max_train_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Get TPE indices for the newly denoised chunk during cache writing.
    Same as the denoising target TPEs used during denoising.
    """
    return get_inference_tpe_indices(ar_step, chunk_len, p_max, max_train_len, device)


# ---------------------------------------------------------------------------
# Autoregressive Inference Engine
# ---------------------------------------------------------------------------

class AutoregressiveInference:
    """
    Autoregressive inference for Ca2-VDM with KV-cache queue and cache sharing.

    Algorithm (Section 3.3 in paper):
    For each AR step k:
      1. Denoising stage: run T denoising steps using shared KV-cache
         - Each step: model(noisy_chunk, t, cached_KV) -> denoised_chunk
      2. Cache writing stage: compute clean KV for denoised chunk
         - model(denoised_chunk, t=0) -> new KV-cache entries
         - Update temporal KV-cache queue (dequeue oldest if P_k >= P_max)
         - Update spatial KV-cache (overwrite with most recent chunk)
    """

    def __init__(
        self,
        model: Ca2VDM,
        config: InferenceConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device

        self.sampler = ImprovedDDPMSampler(
            num_train_timesteps=1000,
            num_inference_steps=config.num_inference_steps,
        )

        # KV-cache structures
        self.temporal_cache = TemporalKVCacheQueue(
            num_layers=self._get_num_layers(),
            max_frames=config.p_max,
        )
        self.spatial_cache = SpatialKVCache(
            num_layers=self._get_num_layers(),
        )

    def _get_num_layers(self) -> int:
        if hasattr(self.model, "blocks"):
            return len(self.model.blocks)
        return 28  # default Open-Sora v1.0

    def reset_cache(self) -> None:
        self.temporal_cache.reset()
        self.spatial_cache.reset()

    @torch.no_grad()
    def generate(
        self,
        first_frame: torch.Tensor,
        num_ar_steps: int,
        text_embeddings: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Generate a video autoregressively.

        Args:
            first_frame: (B, C, H, W) — first frame latent (clean)
            num_ar_steps: number of autoregression steps
            text_embeddings: (B, S, context_dim) — text conditioning
            cfg_scale: classifier-free guidance scale
            verbose: show progress bar
        Returns:
            generated_frames: (B, 1 + num_ar_steps * l, C, H, W) — all generated latents
        """
        self.reset_cache()
        self.model.eval()

        B, C, H, W = first_frame.shape
        l = self.config.chunk_len
        p_max = self.config.p_max
        max_train_len = self.config.max_train_len

        # Initialize with first frame
        generated = [first_frame.unsqueeze(1)]  # list of (B, 1, C, H, W)

        # Cache write for first frame
        self._cache_write_single_frame(first_frame, ar_step=0)

        ar_range = tqdm(range(num_ar_steps), desc="AR steps") if verbose else range(num_ar_steps)

        for ar_step in ar_range:
            p_k = min(1 + ar_step * l, p_max)

            # Denoising stage
            denoised_chunk = self._denoising_stage(
                ar_step=ar_step,
                p_k=p_k,
                B=B, C=C, H=H, W=W,
                text_embeddings=text_embeddings,
                cfg_scale=cfg_scale,
                max_train_len=max_train_len,
            )
            generated.append(denoised_chunk)

            # Cache writing stage
            self._cache_write_chunk(
                denoised_chunk=denoised_chunk,
                ar_step=ar_step + 1,  # next step's perspective
                p_k=p_k,
                max_train_len=max_train_len,
            )

        return torch.cat(generated, dim=1)  # (B, 1 + num_ar_steps*l, C, H, W)

    def _denoising_stage(
        self,
        ar_step: int,
        p_k: int,
        B: int, C: int, H: int, W: int,
        text_embeddings: Optional[torch.Tensor],
        cfg_scale: float,
        max_train_len: int,
    ) -> torch.Tensor:
        """
        Run T denoising steps for one AR chunk.

        The KV-cache is shared across all T denoising steps (cache sharing).
        """
        l = self.config.chunk_len

        # Initialize with pure noise
        z_t = torch.randn(B, l, C, H, W, device=self.device)

        # TPE indices for denoising target
        tpe_indices = get_inference_tpe_indices(
            ar_step, l, self.config.p_max, max_train_len, self.device
        )

        # Denoising loop (T steps, shared cache)
        for t_val in self.sampler.timesteps:
            t_tensor = torch.tensor([t_val] * B, device=self.device)

            # Per-frame timestep vector: all l frames have timestep t
            t_vec = torch.full((B, l), t_val, dtype=torch.long, device=self.device)

            # Model forward (denoising target only, prefix_frames=0 since
            # prefix is in KV-cache, not in z_t)
            model_output = self._model_forward_with_cache(
                z_noisy=z_t,
                t_vec=t_vec,
                tpe_indices=tpe_indices,
                text_embeddings=text_embeddings,
                prefix_frames=0,  # prefix is in KV-cache
            )

            # Classifier-free guidance
            if text_embeddings is not None and cfg_scale > 1.0:
                uncond_output = self._model_forward_with_cache(
                    z_noisy=z_t,
                    t_vec=t_vec,
                    tpe_indices=tpe_indices,
                    text_embeddings=None,
                    prefix_frames=0,
                )
                model_output = uncond_output + cfg_scale * (model_output - uncond_output)

            # Denoise one step
            # model_output: (B, l, 2*C, H, W)
            # Process each frame independently for the sampler
            z_t_new = []
            for frame_idx in range(l):
                frame_out = model_output[:, frame_idx]  # (B, 2*C, H, W)
                frame_noisy = z_t[:, frame_idx]         # (B, C, H, W)
                frame_denoised = self.sampler.step(frame_out, t_val, frame_noisy)
                z_t_new.append(frame_denoised)
            z_t = torch.stack(z_t_new, dim=1)  # (B, l, C, H, W)

        return z_t  # denoised chunk z_0^{P_k:P_k+l}

    def _model_forward_with_cache(
        self,
        z_noisy: torch.Tensor,
        t_vec: torch.Tensor,
        tpe_indices: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
        prefix_frames: int,
    ) -> torch.Tensor:
        """
        Model forward pass using KV-cache (denoising stage).
        The temporal and spatial KV-caches are passed to each block.
        """
        output, _ = self.model(
            z=z_noisy,
            t_vec=t_vec,
            prefix_frames=prefix_frames,
            tpe_indices=tpe_indices,
            context=text_embeddings,
            temporal_kv_cache=self.temporal_cache if self.config.use_kv_cache else None,
            spatial_kv_cache=self.spatial_cache if self.config.use_spatial_cache else None,
            cache_write_mode=False,
        )
        return output

    def _cache_write_single_frame(
        self,
        frame: torch.Tensor,
        ar_step: int,
    ) -> None:
        """
        Cache write for the initial first frame.
        Computes clean KV at t=0 for the first frame.
        """
        B, C, H, W = frame.shape
        max_train_len = self.config.max_train_len

        # First frame gets TPE index 0
        tpe_indices = torch.tensor([0], device=self.device)

        # t_vec = 0 (clean frame)
        t_vec = torch.zeros(B, 1, dtype=torch.long, device=self.device)

        _, new_caches = self.model(
            z=frame.unsqueeze(1),
            t_vec=t_vec,
            prefix_frames=1,  # entire input is "prefix" (clean)
            tpe_indices=tpe_indices,
            context=None,
            temporal_kv_cache=None,
            spatial_kv_cache=None,
            cache_write_mode=True,
        )

        if new_caches is not None:
            for layer_idx in range(len(new_caches["temporal_k"])):
                tk = new_caches["temporal_k"][layer_idx]
                tv = new_caches["temporal_v"][layer_idx]
                if tk is not None:
                    self.temporal_cache.update(layer_idx, tk, tv)

                sk = new_caches["spatial_k"][layer_idx]
                sv = new_caches["spatial_v"][layer_idx]
                if sk is not None:
                    self.spatial_cache.update(layer_idx, sk, sv)

    def _cache_write_chunk(
        self,
        denoised_chunk: torch.Tensor,
        ar_step: int,
        p_k: int,
        max_train_len: int,
    ) -> None:
        """
        Cache writing stage: compute clean KV for the denoised chunk.

        The denoised chunk z_0^{P_k:P_k+l} is input to the model at t=0
        to compute its clean spatial and temporal KV-caches.
        """
        B, l, C, H, W = denoised_chunk.shape

        # TPE indices for this chunk (same as used during denoising)
        tpe_indices = get_inference_tpe_indices(
            ar_step - 1, l, self.config.p_max, max_train_len, self.device
        )

        # t_vec = 0 for all frames (clean)
        t_vec = torch.zeros(B, l, dtype=torch.long, device=self.device)

        _, new_caches = self.model(
            z=denoised_chunk,
            t_vec=t_vec,
            prefix_frames=l,  # entire chunk is "prefix" (clean) for cache writing
            tpe_indices=tpe_indices,
            context=None,
            temporal_kv_cache=self.temporal_cache if self.config.use_kv_cache else None,
            spatial_kv_cache=self.spatial_cache if self.config.use_spatial_cache else None,
            cache_write_mode=True,
        )

        if new_caches is not None:
            for layer_idx in range(len(new_caches["temporal_k"])):
                tk = new_caches["temporal_k"][layer_idx]
                tv = new_caches["temporal_v"][layer_idx]
                if tk is not None:
                    self.temporal_cache.update(layer_idx, tk, tv)

                sk = new_caches["spatial_k"][layer_idx]
                sv = new_caches["spatial_v"][layer_idx]
                if sk is not None:
                    self.spatial_cache.update(layer_idx, sk, sv)


# ---------------------------------------------------------------------------
# OS-Ext Autoregressive Inference (Baseline, no KV-cache)
# ---------------------------------------------------------------------------

class OSExtInference:
    """
    Autoregressive inference for OS-Ext baseline.
    No KV-cache: re-computes all conditional frames at each AR step.
    Quadratic complexity w.r.t. AR step.
    """

    def __init__(
        self,
        model: OSExt,
        config: InferenceConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.sampler = ImprovedDDPMSampler(
            num_train_timesteps=1000,
            num_inference_steps=config.num_inference_steps,
        )

    @torch.no_grad()
    def generate(
        self,
        first_frame: torch.Tensor,
        num_ar_steps: int,
        text_embeddings: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5,
        verbose: bool = True,
    ) -> torch.Tensor:
        self.model.eval()
        B, C, H, W = first_frame.shape
        l = self.config.chunk_len
        p_max = self.config.p_max
        max_train_len = self.config.max_train_len

        generated_frames = [first_frame]  # list of (B, C, H, W)

        ar_range = tqdm(range(num_ar_steps), desc="AR steps (OS-Ext)") if verbose else range(num_ar_steps)

        for ar_step in ar_range:
            # Collect all generated frames as prefix (up to p_max)
            prefix_list = generated_frames[-p_max:]
            p_k = len(prefix_list)
            prefix = torch.stack(prefix_list, dim=1)  # (B, P_k, C, H, W)

            # Initialize noisy chunk
            z_t = torch.randn(B, l, C, H, W, device=self.device)

            # TPE indices for full sequence [prefix | target]
            total_len = p_k + l
            tpe_indices = torch.arange(total_len, device=self.device) % max_train_len

            for t_val in self.sampler.timesteps:
                t_vec = torch.zeros(B, total_len, dtype=torch.long, device=self.device)
                t_vec[:, p_k:] = t_val

                z_input = torch.cat([prefix, z_t], dim=1)  # (B, P_k+l, C, H, W)

                model_output = self.model(
                    z=z_input,
                    t_vec=t_vec,
                    prefix_frames=p_k,
                    tpe_indices=tpe_indices,
                    context=text_embeddings,
                )
                # Only use output for denoising target
                target_output = model_output[:, p_k:]  # (B, l, 2*C, H, W)

                z_t_new = []
                for frame_idx in range(l):
                    frame_out = target_output[:, frame_idx]
                    frame_noisy = z_t[:, frame_idx]
                    frame_denoised = self.sampler.step(frame_out, t_val, frame_noisy)
                    z_t_new.append(frame_denoised)
                z_t = torch.stack(z_t_new, dim=1)

            for frame_idx in range(l):
                generated_frames.append(z_t[:, frame_idx])

        return torch.stack(generated_frames, dim=1)  # (B, 1 + num_ar_steps*l, C, H, W)


# ---------------------------------------------------------------------------
# OS-Fix Autoregressive Inference (Baseline)
# ---------------------------------------------------------------------------

class OSFixInference:
    """
    Autoregressive inference for OS-Fix baseline.
    Fixed-length conditional frames (P = fixed_prefix).
    """

    def __init__(
        self,
        model: OSFix,
        config: InferenceConfig,
        device: torch.device,
        fixed_prefix: int = 8,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.fixed_prefix = fixed_prefix
        self.sampler = ImprovedDDPMSampler(
            num_train_timesteps=1000,
            num_inference_steps=config.num_inference_steps,
        )

    @torch.no_grad()
    def generate(
        self,
        first_frame: torch.Tensor,
        num_ar_steps: int,
        text_embeddings: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5,
        verbose: bool = True,
    ) -> torch.Tensor:
        self.model.eval()
        B, C, H, W = first_frame.shape
        l = self.config.chunk_len
        P = self.fixed_prefix
        max_train_len = P + l

        generated_frames = [first_frame]

        ar_range = tqdm(range(num_ar_steps), desc="AR steps (OS-Fix)") if verbose else range(num_ar_steps)

        for ar_step in ar_range:
            # Use last P frames as fixed-length prefix
            if len(generated_frames) >= P:
                prefix_list = generated_frames[-P:]
            else:
                # Pad with first frame if not enough frames yet
                pad_len = P - len(generated_frames)
                prefix_list = [generated_frames[0]] * pad_len + generated_frames
            prefix = torch.stack(prefix_list, dim=1)  # (B, P, C, H, W)

            z_t = torch.randn(B, l, C, H, W, device=self.device)
            tpe_indices = torch.arange(P + l, device=self.device) % max_train_len

            for t_val in self.sampler.timesteps:
                t_tensor = torch.tensor([t_val] * B, device=self.device)
                z_input = torch.cat([prefix, z_t], dim=1)

                model_output = self.model(
                    z=z_input,
                    t=t_tensor,
                    prefix_frames=P,
                    tpe_indices=tpe_indices,
                    context=text_embeddings,
                )
                target_output = model_output[:, P:]

                z_t_new = []
                for frame_idx in range(l):
                    frame_out = target_output[:, frame_idx]
                    frame_noisy = z_t[:, frame_idx]
                    frame_denoised = self.sampler.step(frame_out, t_val, frame_noisy)
                    z_t_new.append(frame_denoised)
                z_t = torch.stack(z_t_new, dim=1)

            for frame_idx in range(l):
                generated_frames.append(z_t[:, frame_idx])

        return torch.stack(generated_frames, dim=1)


# ---------------------------------------------------------------------------
# VAE Decode Utility
# ---------------------------------------------------------------------------

def decode_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    """
    Decode VAE latents to pixel space.

    Args:
        latents: (B, L, C, H, W) — latent frames
        vae: pretrained VAE decoder
    Returns:
        frames: (B, L, 3, H*8, W*8) — pixel frames in [0, 1]
    """
    B, L, C, H, W = latents.shape
    latents_flat = latents.reshape(B * L, C, H, W)
    with torch.no_grad():
        frames = vae.decode(latents_flat / 0.18215).sample
    frames = (frames.clamp(-1, 1) + 1) / 2  # [0, 1]
    return frames.reshape(B, L, 3, H * 8, W * 8)


# ---------------------------------------------------------------------------
# Save Video Utility
# ---------------------------------------------------------------------------

def save_video(frames: torch.Tensor, output_path: str, fps: int = 8) -> None:
    """
    Save frames as a video file.

    Args:
        frames: (L, 3, H, W) float tensor in [0, 1]
        output_path: output file path
        fps: frames per second
    """
    import imageio
    frames_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    imageio.mimwrite(output_path, frames_np, fps=fps, quality=8)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoregressive inference with Ca2-VDM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="ca2vdm",
                        choices=["ca2vdm", "osfix", "osext"])
    parser.add_argument("--task", type=str, default="t2v", choices=["t2v", "vidpred"])
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean")
    parser.add_argument("--first_frame", type=str, default=None,
                        help="Path to first frame image (optional)")
    parser.add_argument("--num_ar_steps", type=int, default=6)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and model
    if args.task == "t2v":
        config = get_t2v_ca2vdm_config()
    else:
        config = get_vidpred_ca2vdm_config()

    inf_config = config.inference
    inf_config.num_ar_steps = args.num_ar_steps
    inf_config.output_dir = args.output_dir

    # Build model
    model_cfg = type("ModelCfg", (), {
        "in_channels": config.model.in_channels,
        "patch_size": config.model.patch_size,
        "hidden_dim": config.model.hidden_dim,
        "num_layers": config.model.num_layers,
        "num_heads": config.model.num_heads,
        "context_dim": config.model.context_dim if config.model.use_text else None,
        "ff_mult": config.model.ff_mult,
        "dropout": 0.0,
        "max_spatial_h": config.model.max_spatial_h,
        "max_spatial_w": config.model.max_spatial_w,
        "max_temporal_len": config.model.max_temporal_len,
        "chunk_len": inf_config.chunk_len,
        "p_max": inf_config.p_max,
        "prefix_len": config.model.prefix_len,
        "use_text": config.model.use_text,
        "fixed_prefix": config.t2v_train.fixed_prefix,
    })()

    model = build_model(args.model_type, model_cfg).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded model from {args.checkpoint}")

    # Prepare first frame (random noise if not provided)
    B = 1
    C = config.model.in_channels
    H = inf_config.latent_h
    W = inf_config.latent_w

    if args.first_frame is not None:
        from PIL import Image
        from torchvision import transforms
        img = Image.open(args.first_frame).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((inf_config.resolution, inf_config.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        first_frame_pixel = transform(img).unsqueeze(0).to(device)
        # Encode with VAE (placeholder: use random latent if VAE not available)
        first_frame_latent = torch.randn(B, C, H, W, device=device)
    else:
        first_frame_latent = torch.randn(B, C, H, W, device=device)

    # Text embeddings
    text_emb = None
    if args.task == "t2v" and config.model.use_text:
        from train import T5TextEncoder
        text_encoder = T5TextEncoder().to(device)
        text_emb = text_encoder([args.prompt], device)

    # Run inference
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if args.model_type == "ca2vdm":
        engine = AutoregressiveInference(model, inf_config, device)
        generated = engine.generate(
            first_frame=first_frame_latent,
            num_ar_steps=args.num_ar_steps,
            text_embeddings=text_emb,
            cfg_scale=args.cfg_scale,
        )
    elif args.model_type == "osfix":
        engine = OSFixInference(model, inf_config, device, fixed_prefix=config.t2v_train.fixed_prefix)
        generated = engine.generate(
            first_frame=first_frame_latent,
            num_ar_steps=args.num_ar_steps,
            text_embeddings=text_emb,
            cfg_scale=args.cfg_scale,
        )
    else:
        engine = OSExtInference(model, inf_config, device)
        generated = engine.generate(
            first_frame=first_frame_latent,
            num_ar_steps=args.num_ar_steps,
            text_embeddings=text_emb,
            cfg_scale=args.cfg_scale,
        )

    elapsed = time.time() - start_time
    total_frames = generated.shape[1]
    logger.info(f"Generated {total_frames} frames in {elapsed:.1f}s ({total_frames/elapsed:.1f} fps)")

    # Save latents (VAE decode would be done here with actual VAE)
    output_path = output_dir / "generated_latents.pt"
    torch.save(generated.cpu(), output_path)
    logger.info(f"Saved latents to {output_path}")


if __name__ == "__main__":
    main()
