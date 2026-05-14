## main.py

"""
Ca2‑VDM reproduction entry point.

Usage:
    python main.py --mode train --config config.yaml
    python main.py --mode inference --config config.yaml --checkpoint path/to/model.pt --output_dir results/
    python main.py --mode eval --config config.yaml --checkpoint checkpoints/ca2_final.pt

The script handles the full pipeline: loading configuration, setting up datasets,
training the causal video diffusion model (including two‑stage training for T2V),
running autoregressive inference with KV‑cache sharing, and evaluating generated
videos using FVD.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Imports from the project (must be in Python path)
# ---------------------------------------------------------------------------
from config import Config
from data.dataset import (
    InternVidDataset,
    SkyTimelapseDataset,
    UCF101Dataset,
    collate_fn,
)
from data.preprocess import VideoProcessor
from model.ca2_vdm import Ca2VDM
from trainer import Trainer
from inference import InferenceEngine
from evaluate import Evaluator

# ---------------------------------------------------------------------------
# Logging & seed utilities
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> None:
    """Configure logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "experiment.log")

    format_str = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=format_str,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Logging to %s", log_file)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic operations when possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Ca2‑VDM reproduction")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "inference", "eval"],
        help="Execution mode",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint (required for inference/eval)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for generated videos (inference/eval)",
    )
    parser.add_argument(
        "--first_frame",
        type=str,
        default=None,
        help="Path to an image file used as the first frame for inference (optional)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides, e.g., model.dtype=float32 training.stage1.batch_size=128",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset helper for inference/evaluation
# ---------------------------------------------------------------------------

def _get_first_frame_latent_from_path(
    model: Ca2VDM,
    image_path: str,
) -> torch.Tensor:
    """Load an image, preprocess, and encode to latent."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    # Resize to model resolution and center crop (mirrors VideoProcessor)
    # Using transforms similar to VideoProcessor
    res = model.config.data.resolution
    img = img.resize((res, res), Image.BILINEAR)
    # Convert to tensor in [-1, 1]
    frame_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
    frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)
    latents = model.encode_latents(frame_tensor.to(model.vae.device)).squeeze(0)  # (1, C, H_l, W_l)
    return latents


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def run_training(
    config: Config,
    resume_checkpoint: Optional[str] = None,
) -> None:
    """
    Run the training loop according to the configuration.

    Supports:
    - Two‑stage T2V training (stage1 → stage2).
    - Single‑stage video prediction training.
    """
    device = torch.device(config.system.device if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------
    model = Ca2VDM(config)
    model = model.to(device)
    if resume_checkpoint:
        logging.info("Resuming from checkpoint: %s", resume_checkpoint)
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # ------------------------------------------------------------------
    # Determine task and create appropriate dataset & loader
    # ------------------------------------------------------------------
    if config.task == "t2v":
        # For T2V, we have two possible stages.
        if config.training.stage1 is not None and config.training.stage1.enabled:
            logging.info("--- Stage 1: causal pretraining (no clean prefix) ---")
            dataset_s1 = InternVidDataset(
                config,
                split="train",      # InternVid training split
                stage=1,
            )
            loader_s1 = DataLoader(
                dataset_s1,
                batch_size=config.training.stage1.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            trainer_s1 = Trainer(
                model=model,
                config=config,
                train_loader=loader_s1,
                val_loader=None,
            )
            # Optionally override max steps & batch size (already handled in Trainer)
            trainer_s1.run_training()
            # Save stage1 checkpoint
            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(config.system.checkpoint_dir, "stage1_final.pt"),
            )

        if config.training.stage2 is not None and config.training.stage2.enabled:
            logging.info("--- Stage 2: training with extendable clean prefix ---")
            # If stage1 was run, model already contains updated weights.
            dataset_s2 = InternVidDataset(
                config,
                split="train",
                stage=2,
            )
            loader_s2 = DataLoader(
                dataset_s2,
                batch_size=config.training.stage2.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            trainer_s2 = Trainer(
                model=model,
                config=config,
                train_loader=loader_s2,
                val_loader=None,
            )
            trainer_s2.run_training()
            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(config.system.checkpoint_dir, "stage2_final.pt"),
            )
    else:   # video_prediction
        logging.info("--- Video prediction training ---")
        # Dataset for video prediction (SkyTimelapse)
        dataset = SkyTimelapseDataset(config, split="train")
        loader = DataLoader(
            dataset,
            batch_size=config.training.video_prediction.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=loader,
            val_loader=None,
        )
        trainer.run_training()
        torch.save(
            {"model_state_dict": model.state_dict()},
            os.path.join(config.system.checkpoint_dir, "final.pt"),
        )

    logging.info("Training completed.")


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def run_inference(
    config: Config,
    checkpoint_path: str,
    output_dir: str,
    first_frame_path: Optional[str] = None,
) -> None:
    """
    Load a trained model and generate a video autoregressively.

    The first frame can be provided as an image file; otherwise a random
    first frame from the training set is used (or a blank frame).
    """
    device = torch.device(config.system.device if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = Ca2VDM(config)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # ------------------------------------------------------------------
    # Acquire first frame latent
    # ------------------------------------------------------------------
    if first_frame_path:
        latents = _get_first_frame_latent_from_path(model, first_frame_path)
        logging.info("Using first frame from %s", first_frame_path)
    else:
        # Fallback: use a latent of zeros (not meaningful, but shows pipeline)
        logging.warning("No first frame provided. Using latent of zeros; results will be meaningless.")
        B = 1
        C = model.latent_channels
        H = W = config.data.latent_size
        latents = torch.zeros(1, C, H, W, device=device)

    # ------------------------------------------------------------------
    # Prepare text prompt (if T2V)
    # ------------------------------------------------------------------
    text_prompt: Optional[str] = None
    if config.task == "t2v":
        # Use a default prompt or ask user; for demo, use generic
        text_prompt = "A beautiful sunset over water."
        logging.info("Using text prompt: %s", text_prompt)

    # ------------------------------------------------------------------
    # Inference engine
    # ------------------------------------------------------------------
    engine = InferenceEngine(model, config)

    # Generate video
    num_chunks = config.inference.generate_frames // config.video.chunk_size
    if num_chunks < 1:
        num_chunks = 1
    logging.info("Generating %d chunks (approx. %d frames)",
                 num_chunks, 1 + num_chunks * config.video.chunk_size)
    video_pixel = engine.autoregressive_generate(
        first_frame=latents,       # already latent
        text_prompt=text_prompt,
        num_chunks=num_chunks,
    )   # shape (T, 3, H, W), uint8

    # ------------------------------------------------------------------
    # Save video using torchvision or OpenCV
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "generated_video.mp4")
    # Convert tensor to numpy array (T, H, W, 3) uint8
    video_np = video_pixel.cpu().permute(0, 2, 3, 1).numpy()  # (T, H, W, 3)

    # Write with OpenCV
    try:
        import cv2
        H, W = video_np.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 8  # arbitrary, can be configurable
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        for i in range(video_np.shape[0]):
            writer.write(cv2.cvtColor(video_np[i], cv2.COLOR_RGB2BGR))
        writer.release()
        logging.info("Video saved to %s", out_path)
    except Exception as e:
        logging.error("Failed to write video with OpenCV: %s. Saving frames as PNGs.", e)
        # Fallback: save individual frames
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        from PIL import Image
        for i in range(video_np.shape[0]):
            img = Image.fromarray(video_np[i])
            img.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
        logging.info("Frames saved to %s", frames_dir)


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    config: Config,
    checkpoint_path: str,
    output_dir: str,
) -> None:
    """
    Compute quantitative metrics (FVD) as reported in the paper.

    Supports:
    - Zero‑shot T2V FVD on MSR‑VTT and UCF‑101.
    - Fine‑tuned FVD on UCF‑101 (if finetuning was done).
    - Chunk‑wise FVD for autoregressive consistency (SkyTimelapse).
    """
    device = torch.device(config.system.device if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = Ca2VDM(config)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # ------------------------------------------------------------------
    # Create evaluator
    # ------------------------------------------------------------------
    evaluator = Evaluator(config, video_processor=VideoProcessor(config))

    # ------------------------------------------------------------------
    # Determine which evaluation to run based on the dataset specified
    # ------------------------------------------------------------------
    dataset_name = config.data.dataset
    if dataset_name == "internvid" or dataset_name == "msr_vtt":
        # Zero‑shot FVD on MSR‑VTT
        logging.info("Evaluating zero‑shot FVD on MSR‑VTT...")
        msrvtt_metadata_path = os.path.join(config.data.data_root, "msr_vtt", "test_metadata.json")
        if not os.path.isfile(msrvtt_metadata_path):
            raise FileNotFoundError(f"MSR-VTT test metadata not found at {msrvtt_metadata_path}")
        # We'll build a simple dataset that yields (first_frame, caption)
        # from the metadata.
        class MSRVTTSampleDataset(torch.utils.data.Dataset):
            def __init__(self, metadata_path, video_root):
                import json
                with open(metadata_path, "r") as f:
                    self.samples = json.load(f)
                self.video_root = video_root
                self.vp = VideoProcessor(config)

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                item = self.samples[idx]
                video_path = os.path.join(self.video_root, item["video"])
                caption = item["caption"]      # randomly selected one
                # Load only the first frame (T=1)
                first_frame = self.vp.load_video(
                    video_path, num_frames=1, start_frame=0, uniform=False
                )   # (1, 3, H, W)
                first_frame = first_frame.squeeze(0)  # (3, H, W)
                return first_frame, caption

        gen_videos_dir = os.path.join(output_dir, "msrvtt_generated")
        os.makedirs(gen_videos_dir, exist_ok=True)
        dataset = MSRVTTSampleDataset(msrvtt_metadata_path, os.path.join(config.data.data_root, "msr_vtt", "videos"))

        # Generate one 16‑frame clip per sample
        for i, (first_frame, caption) in enumerate(tqdm(dataset, desc="Generating MSR-VTT")):
            with torch.no_grad():
                engine = InferenceEngine(model, config)
                # Encode first frame to latent
                first_frame = first_frame.unsqueeze(0).unsqueeze(0)   # (1,1,3,H,W)
                latent = model.encode_latents(first_frame.to(device)).squeeze(0)  # (1, C, H_l, W_l)
                # Generate only one chunk (the chunk contains first frame + generated)
                # The inference engine expects first frame latent shape (1, C, H, W)
                video_pixel = engine.autoregressive_generate(
                    first_frame=latent,
                    text_prompt=caption,
                    num_chunks=1,
                )  # (L+1, ...) but we only want the generated frames? Actually it includes first frame.
                # The FVD evaluation expects 16‑frame clips; we can extract frames 1‑16.
                # Since first frame is included, we take entire video (first frame + generated chunk) and trim to 16 frames.
                video_pixel = video_pixel[:16]  # ensure 16 frames
                # Save as a .mp4 file
                out_path = os.path.join(gen_videos_dir, f"sample_{i:05d}.mp4")
                import cv2
                video_np = video_pixel.cpu().permute(0, 2, 3, 1).numpy()
                H, W = video_np.shape[1:3]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 8, (W, H))
                for frame in video_np:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()

        # Compute FVD between generated videos and a real set (need a directory of real 16‑frame clips)
        real_clips_dir = os.path.join(config.data.data_root, "msr_vtt", "real_clips")
        if not os.path.isdir(real_clips_dir):
            raise FileNotFoundError(f"Real MSR-VTT clips directory not found: {real_clips_dir}")
        fvd = evaluator.compute_fvd(gen_videos_dir, real_clips_dir)
        logging.info("MSR-VTT FVD: %.4f", fvd)

    elif dataset_name == "ucf101":
        # Zero‑shot FVD on UCF‑101
        logging.info("Evaluating zero‑shot FVD on UCF‑101...")
        # We'll generate 2048 videos, 16 frames each, conditioned on the first frame and class prompt.
        # Use UCF101Dataset with test split and zero_shot=True
        test_dataset = UCF101Dataset(config, split="test", zero_shot=True)
        gen_videos_dir = os.path.join(output_dir, "ucf101_generated")
        os.makedirs(gen_videos_dir, exist_ok=True)

        # For reproducibility, generate exactly 2048 videos (config.evaluation.fvd.num_generated_videos)
        num_to_generate = min(config.evaluation.fvd.num_generated_videos, len(test_dataset))
        for i in tqdm(range(num_to_generate), desc="Generating UCF-101"):
            sample = test_dataset[i]
            first_frame = sample["latent_frames"][0]  # first frame latent
            first_frame = first_frame.unsqueeze(0)    # (1, C, H, W)
            text_emb = sample["text_embeddings"]       # comes as (1, seq_len, hidden)
            # Decode text to prompt? We need the prompt string. The dataset stored embeddings, but we need the string to pass to InferenceEngine. For simplicity, we'll reconstruct from class label.
            # We can store a mapping from class name to prompt. Since UCF101Dataset already has PROMPTS dict, we can use that.
            class_name = test_dataset.class_labels[i]
            prompt = test_dataset.PROMPTS.get(class_name, ["A video of an action."])[0]

            with torch.no_grad():
                engine = InferenceEngine(model, config)
                # Generate 16 frames (1 chunk)
                video_pixel = engine.autoregressive_generate(
                    first_frame=first_frame.to(device),
                    text_prompt=prompt,
                    num_chunks=1,
                )  # (L+1, 3, H, W)
                video_pixel = video_pixel[:16]  # ensure 16 frames
                out_path = os.path.join(gen_videos_dir, f"sample_{i:05d}.mp4")
                import cv2
                video_np = video_pixel.cpu().permute(0, 2, 3, 1).numpy()
                H, W = video_np.shape[1:3]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 8, (W, H))
                for frame in video_np:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()

        real_clips_dir = os.path.join(config.data.data_root, "ucf101", "test_clips")   # user must prepare
        if not os.path.isdir(real_clips_dir):
            raise FileNotFoundError(f"Real UCF-101 test clips directory not found: {real_clips_dir}")
        fvd = evaluator.compute_fvd(gen_videos_dir, real_clips_dir)
        logging.info("UCF-101 FVD: %.4f", fvd)

    elif dataset_name == "sky_timelapse":
        # Chunk‑wise FVD for video prediction (SkyTimelapse)
        logging.info("Evaluating chunk‑wise FVD on SkyTimelapse...")
        # Generate 48‑frame videos (6 AR steps of 8 frames each) for each test sample
        test_dataset = SkyTimelapseDataset(config, split="test")
        gen_videos = []
        for i in tqdm(range(len(test_dataset)), desc="Generating SkyTimelapse"):
            sample = test_dataset[i]
            # The dataset returns full clip of length L, but for inference we only need first frame.
            first_latent = sample["latent_frames"][0].unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                engine = InferenceEngine(model, config)
                # Generate 6 AR steps → total 1+6*8 = 49 frames; we can take 48.
                video_pixel = engine.autoregressive_generate(
                    first_frame=first_latent,
                    text_prompt=None,
                    num_chunks=6,
                )
                # Extract 48 frames starting from second frame (skip first)
                video_48 = video_pixel[1:49, ...]  # now (48, 3, H, W)
                gen_videos.append(video_48)

        # For real frames, we need 16‑frame clips from the test set (the dataset yields full videos; we can take the first 16 frames).
        real_clips = []
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            latent = sample["latent_frames"][:16, ...]  # (16, C, H_l, W_l)
            # Decode to pixel
            pixel = model.decode_latents(latent.unsqueeze(0))  # (1, 16, 3, H, W)
            real_clips.append(pixel.squeeze(0))  # (16, 3, H, W)
        # Convert list to tensor -> not needed for Evaluator's chunkwise method; it expects lists of tensors.
        chunkwise_fvd = evaluator.compute_chunkwise_fvd(gen_videos, real_clips)
        for k, v in chunkwise_fvd.items():
            logging.info("SkyTimelapse %s FVD: %.4f", k, v)

    else:
        raise ValueError(f"Unknown dataset for evaluation: {dataset_name}")

    logging.info("Evaluation complete.")


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load configuration with overrides
    # ------------------------------------------------------------------
    config = Config.from_yaml(
        args.config,
        **{kv.split("=")[0]: kv.split("=")[1] for kv in args.override},
    )

    # ------------------------------------------------------------------
    # Setup logging and seed
    # ------------------------------------------------------------------
    os.makedirs(config.system.log_dir, exist_ok=True)
    setup_logging(config.system.log_dir)
    set_seed(config.training.seed)
    logging.info("Starting Ca2‑VDM reproduction in mode: %s", args.mode)

    # ------------------------------------------------------------------
    # Dispatch mode
    # ------------------------------------------------------------------
    if args.mode == "train":
        run_training(
            config,
            resume_checkpoint=args.checkpoint,   # None if not provided
        )
    elif args.mode == "inference":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for inference mode.")
        run_inference(
            config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            first_frame_path=args.first_frame,
        )
    elif args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for evaluation mode.")
        run_evaluation(
            config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    logging.info("Done.")

if __name__ == "__main__":
    main()

