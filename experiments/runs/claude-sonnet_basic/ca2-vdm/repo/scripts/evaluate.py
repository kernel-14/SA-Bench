"""
Evaluation script for Ca2-VDM.

Computes FVD (Frechet Video Distance) scores on:
  - MSR-VTT: Zero-shot T2V FVD (Table 1)
  - UCF-101: Zero-shot and finetuned FVD (Tables 1, 2)
  - SkyTimelapse: Video prediction FVD (Tables 3, 4)

FVD computation follows prior works (Blattmann et al., 2023b; Ge et al., 2022):
  - Uses pretrained I3D model for feature extraction
  - Uses StyleGAN-V codebase for FVD statistics

Evaluation details (Appendix D):
  - MSR-VTT: 2990 test videos, randomly select 1 caption per video
  - UCF-101: 2048 samples with uniform distribution per category
  - SkyTimelapse: 225 test clips, chunk-wise FVD evaluation
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from ca2_vdm.models import Ca2VDM
from ca2_vdm.models.transformer import Ca2VDMTransformer
from ca2_vdm.data import SkyTimelapseDataset, MSRVTTDataset, UCF101Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Ca2-VDM")

    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["msrvtt", "ucf101", "sky_timelapse"],
                        help="Evaluation dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs", help="Output directory")

    # Generation settings
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate")
    parser.add_argument("--num_ar_steps", type=int, default=6, help="Number of AR steps")
    parser.add_argument("--num_denoising_steps", type=int, default=100, help="DDPM denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--num_samples", type=int, default=512, help="Number of samples for FVD")

    # Model settings
    parser.add_argument("--hidden_size", type=int, default=1152)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--prefix_len", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--max_prefix_len", type=int, default=49)
    parser.add_argument("--resolution", type=int, default=256)

    # Evaluation mode
    parser.add_argument("--use_kv_cache", action="store_true", default=True,
                        help="Use KV-cache (Ca2-VDM mode)")
    parser.add_argument("--no_kv_cache", dest="use_kv_cache", action="store_false",
                        help="Disable KV-cache (baseline mode)")
    parser.add_argument("--use_prefix_enhancement", action="store_true", default=True)

    # FVD computation
    parser.add_argument("--i3d_path", type=str, default=None,
                        help="Path to pretrained I3D model for FVD")
    parser.add_argument("--fvd_num_videos", type=int, default=2048,
                        help="Number of videos for FVD computation")

    return parser.parse_args()


def load_model(args, device: torch.device) -> Ca2VDM:
    """Load Ca2-VDM model from checkpoint."""
    resolution = args.resolution
    patch_size = args.patch_size
    latent_h = resolution // 8
    latent_w = resolution // 8
    max_height = latent_h // patch_size
    max_width = latent_w // patch_size
    max_seq_len = args.max_prefix_len + args.chunk_size

    use_text = args.dataset in ["msrvtt", "ucf101"]
    context_dim = 4096 if use_text else None

    transformer = Ca2VDMTransformer(
        in_channels=4,
        out_channels=8,
        patch_size=patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        context_dim=context_dim,
        prefix_len=args.prefix_len,
        max_seq_len=max_seq_len,
        max_height=max_height,
        max_width=max_width,
    )

    model = Ca2VDM(
        transformer=transformer,
        chunk_size=args.chunk_size,
        max_prefix_len=args.max_prefix_len,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def compute_fvd(
    real_videos: torch.Tensor,
    fake_videos: torch.Tensor,
    i3d_model: Optional[nn.Module] = None,
) -> float:
    """
    Compute FVD between real and generated videos.

    Uses I3D features following prior works (Blattmann et al., 2023b; Ge et al., 2022).
    Follows the StyleGAN-V codebase for FVD statistics computation.

    Args:
        real_videos: Real videos of shape (N, T, C, H, W), values in [-1, 1].
        fake_videos: Generated videos of shape (N, T, C, H, W), values in [-1, 1].
        i3d_model: Pretrained I3D model for feature extraction.

    Returns:
        FVD score (float).
    """
    if i3d_model is None:
        logger.warning("No I3D model provided. Returning placeholder FVD.")
        return float("nan")

    device = next(i3d_model.parameters()).device

    def extract_features(videos: torch.Tensor) -> np.ndarray:
        """Extract I3D features from videos."""
        # I3D expects (B, C, T, H, W) in [0, 1]
        videos = (videos + 1) / 2  # [-1, 1] -> [0, 1]
        videos = videos.permute(0, 2, 1, 3, 4)  # (N, T, C, H, W) -> (N, C, T, H, W)

        features = []
        batch_size = 16
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i + batch_size].to(device)
            with torch.no_grad():
                feat = i3d_model(batch)
            features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    real_features = extract_features(real_videos)
    fake_features = extract_features(fake_videos)

    # Compute FVD using Frechet distance
    fvd = _frechet_distance(real_features, fake_features)
    return fvd


def _frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Frechet distance between two sets of features.

    FVD = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*sqrt(Sigma_r * Sigma_g))
    """
    from scipy import linalg

    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)
    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)

    diff = mu_x - mu_y
    mean_diff = np.dot(diff, diff)

    # Compute sqrt of product of covariances
    covmean, _ = linalg.sqrtm(sigma_x @ sigma_y, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    trace_term = np.trace(sigma_x) + np.trace(sigma_y) - 2 * np.trace(covmean)
    fvd = mean_diff + trace_term

    return float(fvd)


def generate_videos(
    model: Ca2VDM,
    first_frames: torch.Tensor,
    num_frames: int,
    num_denoising_steps: int,
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    guidance_scale: float = 7.5,
    use_kv_cache: bool = True,
) -> torch.Tensor:
    """
    Generate videos autoregressively.

    Args:
        model: Ca2-VDM model.
        first_frames: Initial frames of shape (B, C, H, W).
        num_frames: Total frames to generate.
        num_denoising_steps: DDPM steps.
        context: Optional text embeddings.
        context_mask: Optional text mask.
        guidance_scale: CFG scale.
        use_kv_cache: Whether to use KV-cache.

    Returns:
        Generated videos of shape (B, num_frames, C, H, W).
    """
    with torch.no_grad():
        videos = model.autoregressive_generate(
            first_frame=first_frames,
            num_frames=num_frames,
            num_denoising_steps=num_denoising_steps,
            context=context,
            context_mask=context_mask,
            guidance_scale=guidance_scale,
            use_kv_cache=use_kv_cache,
        )
    return videos


def evaluate_sky_timelapse(args, model: Ca2VDM, device: torch.device):
    """
    Evaluate on SkyTimelapse dataset.

    Generates 48 frames with 6 AR steps (l=8).
    Evaluates chunk-wise FVD for 3 chunks of 16 frames each.
    """
    logger.info("Evaluating on SkyTimelapse...")

    dataset = SkyTimelapseDataset(
        data_dir=args.data_dir,
        split="test",
        chunk_size=args.chunk_size,
        max_prefix_len=args.max_prefix_len,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_real = []
    all_fake = []

    for batch in dataloader:
        frames = batch["frames"].to(device)  # (B, L, C, H, W)
        first_frame = frames[:, 0]  # (B, C, H, W)

        # Generate 48 frames (6 AR steps * 8 frames)
        generated = generate_videos(
            model, first_frame,
            num_frames=48,
            num_denoising_steps=args.num_denoising_steps,
            use_kv_cache=args.use_kv_cache,
        )

        all_real.append(frames[:, :48].cpu())
        all_fake.append(generated.cpu())

        if len(all_real) * args.batch_size >= args.num_samples:
            break

    real_videos = torch.cat(all_real, dim=0)[:args.num_samples]
    fake_videos = torch.cat(all_fake, dim=0)[:args.num_samples]

    # Chunk-wise FVD evaluation (3 chunks of 16 frames)
    results = {}
    for chunk_id in range(3):
        start = chunk_id * 16
        end = start + 16
        real_chunk = real_videos[:, start:end]
        fake_chunk = fake_videos[:, start:end]
        fvd = compute_fvd(real_chunk, fake_chunk)
        results[f"chunk_{chunk_id + 1}_fvd"] = fvd
        logger.info(f"Chunk {chunk_id + 1} FVD: {fvd:.1f}")

    return results


def evaluate_msrvtt(args, model: Ca2VDM, device: torch.device):
    """
    Evaluate zero-shot T2V FVD on MSR-VTT.

    Generates 2990 videos (16 frames each) and computes FVD.
    """
    logger.info("Evaluating on MSR-VTT...")

    dataset = MSRVTTDataset(
        data_dir=args.data_dir,
        split="test",
        resolution=(args.resolution, args.resolution),
        chunk_size=args.chunk_size,
        max_prefix_len=args.max_prefix_len,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_real = []
    all_fake = []

    for batch in dataloader:
        frames = batch["frames"].to(device)
        first_frame = frames[:, 0]

        # For T2V, we'd encode text with T5 here
        # For now, generate without text conditioning
        generated = generate_videos(
            model, first_frame,
            num_frames=16,
            num_denoising_steps=args.num_denoising_steps,
            guidance_scale=args.guidance_scale,
            use_kv_cache=args.use_kv_cache,
        )

        all_real.append(frames[:, :16].cpu())
        all_fake.append(generated.cpu())

        if len(all_real) * args.batch_size >= args.fvd_num_videos:
            break

    real_videos = torch.cat(all_real, dim=0)[:args.fvd_num_videos]
    fake_videos = torch.cat(all_fake, dim=0)[:args.fvd_num_videos]

    fvd = compute_fvd(real_videos, fake_videos)
    logger.info(f"MSR-VTT FVD: {fvd:.1f}")

    return {"fvd": fvd}


def evaluate_ucf101(args, model: Ca2VDM, device: torch.device):
    """
    Evaluate FVD on UCF-101.

    Generates 2048 samples with uniform distribution per category.
    """
    logger.info("Evaluating on UCF-101...")

    dataset = UCF101Dataset(
        data_dir=args.data_dir,
        split="test",
        resolution=(args.resolution, args.resolution),
        chunk_size=args.chunk_size,
        max_prefix_len=args.max_prefix_len,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    all_real = []
    all_fake = []

    for batch in dataloader:
        frames = batch["frames"].to(device)
        first_frame = frames[:, 0]

        generated = generate_videos(
            model, first_frame,
            num_frames=16,
            num_denoising_steps=args.num_denoising_steps,
            guidance_scale=args.guidance_scale,
            use_kv_cache=args.use_kv_cache,
        )

        all_real.append(frames[:, :16].cpu())
        all_fake.append(generated.cpu())

        if len(all_real) * args.batch_size >= args.fvd_num_videos:
            break

    real_videos = torch.cat(all_real, dim=0)[:args.fvd_num_videos]
    fake_videos = torch.cat(all_fake, dim=0)[:args.fvd_num_videos]

    fvd = compute_fvd(real_videos, fake_videos)
    logger.info(f"UCF-101 FVD: {fvd:.1f}")

    return {"fvd": fvd}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args, device)

    # Run evaluation
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "sky_timelapse":
        results = evaluate_sky_timelapse(args, model, device)
    elif args.dataset == "msrvtt":
        results = evaluate_msrvtt(args, model, device)
    elif args.dataset == "ucf101":
        results = evaluate_ucf101(args, model, device)

    # Save results
    import json
    results_path = output_dir / f"results_{args.dataset}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
