"""Generation script for pyramidal flow matching.

Supports text-to-image, text-to-video, and image-to-video generation.

Usage:
    # Text-to-image
    python generate.py --mode image --prompt "A beautiful sunset" --output outputs/

    # Text-to-video (5s, 768p, 24fps)
    python generate.py --mode video --prompt "A steam train crossing a viaduct" \\
        --height 768 --width 768 --num_frames 121 --output outputs/

    # Text-to-video (10s)
    python generate.py --mode video --prompt "..." --num_frames 241 --output outputs/

    # Image-to-video
    python generate.py --mode i2v --prompt "..." --image_path input.jpg --output outputs/
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from config import get_default_config
from inference.pipeline import PyramidFlowPipeline


def load_prompts(prompt_or_file: str) -> List[str]:
    """Load prompts from a string or file."""
    path = Path(prompt_or_file)
    if path.exists():
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return [data.get("prompt", "")]
        elif path.suffix == ".txt":
            with open(path) as f:
                return [line.strip() for line in f if line.strip()]
    return [prompt_or_file]


def main():
    parser = argparse.ArgumentParser(description="Generate images/videos with pyramidal flow matching")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--mode", type=str, default="video",
                        choices=["image", "video", "i2v"],
                        help="Generation mode")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape",
                        help="Text prompt or path to prompt file")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Input image for image-to-video generation")
    parser.add_argument("--output", type=str, default="outputs/generated",
                        help="Output directory")
    parser.add_argument("--height", type=int, default=768,
                        help="Output height (384 or 768)")
    parser.add_argument("--width", type=int, default=768,
                        help="Output width")
    parser.add_argument("--num_frames", type=int, default=121,
                        help="Number of frames (121=5s, 241=10s at 24fps)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Output video FPS")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="Number of ODE steps per pyramid stage")
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print(f"Loading pipeline from {args.checkpoint_dir}...")
    config = get_default_config()
    pipeline = PyramidFlowPipeline.from_pretrained(
        args.checkpoint_dir, config=config, device=device
    )

    # Load prompts
    prompts = load_prompts(args.prompt)
    print(f"Generating {len(prompts)} sample(s) in {args.mode} mode...")

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt[:80]}...")

        if args.mode == "image":
            images = pipeline.generate_image(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed + i,
            )
            for j, img in enumerate(images):
                out_path = output_dir / f"image_{i:04d}_{j:02d}.png"
                img.save(out_path)
                print(f"  Saved: {out_path}")

        elif args.mode == "video":
            videos = pipeline.generate_video(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed + i,
            )
            for j, frames in enumerate(videos):
                out_path = output_dir / f"video_{i:04d}_{j:02d}.mp4"
                pipeline.save_video(frames, str(out_path), fps=args.fps)
                print(f"  Saved: {out_path} ({len(frames)} frames)")

        elif args.mode == "i2v":
            if args.image_path is None:
                print("  Error: --image_path required for i2v mode")
                continue

            first_frame = Image.open(args.image_path).convert("RGB")
            first_frame = first_frame.resize((args.width, args.height))
            import torchvision.transforms as T
            frame_tensor = T.ToTensor()(first_frame).unsqueeze(0) * 2 - 1
            frame_tensor = frame_tensor.to(device)

            videos = pipeline.generate_video(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed + i,
                first_frame=frame_tensor,
            )
            for j, frames in enumerate(videos):
                out_path = output_dir / f"i2v_{i:04d}_{j:02d}.mp4"
                pipeline.save_video(frames, str(out_path), fps=args.fps)
                print(f"  Saved: {out_path} ({len(frames)} frames)")

    print(f"\nGeneration complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
