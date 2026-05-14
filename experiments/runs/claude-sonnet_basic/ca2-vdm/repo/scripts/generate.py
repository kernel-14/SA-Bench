"""
Video generation script for Ca2-VDM.

Generates videos autoregressively using the Ca2-VDM model.
Supports both text-to-video and video prediction modes.

Usage:
  # Video prediction (SkyTimelapse)
  python scripts/generate.py \\
    --checkpoint outputs/model_final.pt \\
    --first_frame path/to/first_frame.png \\
    --num_frames 80 \\
    --output_path generated_video.mp4

  # Text-to-video
  python scripts/generate.py \\
    --checkpoint outputs/model_final.pt \\
    --text "A beautiful sunset over the ocean" \\
    --num_frames 80 \\
    --output_path generated_video.mp4
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torchvision.io as tvio
import torchvision.transforms as T
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ca2_vdm.models import Ca2VDM
from ca2_vdm.models.transformer import Ca2VDMTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos with Ca2-VDM")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--first_frame", type=str, default=None,
                        help="Path to first frame image (for video prediction)")
    parser.add_argument("--text", type=str, default=None,
                        help="Text prompt (for T2V generation)")
    parser.add_argument("--output_path", type=str, default="generated_video.mp4")
    parser.add_argument("--num_frames", type=int, default=80)
    parser.add_argument("--num_denoising_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--max_prefix_len", type=int, default=49)
    parser.add_argument("--hidden_size", type=int, default=1152)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--prefix_len", type=int, default=3)
    parser.add_argument("--use_kv_cache", action="store_true", default=True)
    parser.add_argument("--no_kv_cache", dest="use_kv_cache", action="store_false")
    parser.add_argument("--fps", type=int, default=8, help="Output video FPS")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_first_frame(path: str, resolution: int) -> torch.Tensor:
    """Load and preprocess first frame image."""
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
    ])
    return transform(img).unsqueeze(0)  # (1, C, H, W)


def save_video(frames: torch.Tensor, output_path: str, fps: int = 8):
    """
    Save generated frames as video.

    Args:
        frames: Video frames of shape (T, C, H, W) in [-1, 1].
        output_path: Output file path.
        fps: Frames per second.
    """
    # Convert from [-1, 1] to [0, 255]
    frames = ((frames + 1) / 2 * 255).clamp(0, 255).byte()
    # frames: (T, C, H, W) -> (T, H, W, C) for torchvision
    frames = frames.permute(0, 2, 3, 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tvio.write_video(
        str(output_path),
        frames,
        fps=fps,
        video_codec="libx264",
    )
    logger.info(f"Saved video to {output_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    resolution = args.resolution
    patch_size = args.patch_size
    latent_h = resolution // 8
    latent_w = resolution // 8
    max_height = latent_h // patch_size
    max_width = latent_w // patch_size
    max_seq_len = args.max_prefix_len + args.chunk_size

    use_text = args.text is not None
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

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from {args.checkpoint}")

    # Prepare first frame
    if args.first_frame:
        first_frame = load_first_frame(args.first_frame, args.resolution).to(device)
    else:
        # Random first frame for unconditional generation
        first_frame = torch.randn(1, 4, latent_h, latent_w, device=device)

    # Prepare text context
    context = None
    context_mask = None
    if use_text and args.text:
        # In practice, encode text with T5 here
        # For now, use None
        logger.warning("Text encoding not implemented in this version. Generating without text.")

    # Generate video
    logger.info(f"Generating {args.num_frames} frames...")
    with torch.no_grad():
        generated = model.autoregressive_generate(
            first_frame=first_frame,
            num_frames=args.num_frames,
            num_denoising_steps=args.num_denoising_steps,
            context=context,
            context_mask=context_mask,
            guidance_scale=args.guidance_scale if use_text else 1.0,
            use_kv_cache=args.use_kv_cache,
        )

    # generated: (1, num_frames, C, H, W)
    frames = generated[0]  # (num_frames, C, H, W)

    # In practice, decode with VAE here
    # For now, save directly
    save_video(frames, args.output_path, fps=args.fps)


if __name__ == "__main__":
    main()
