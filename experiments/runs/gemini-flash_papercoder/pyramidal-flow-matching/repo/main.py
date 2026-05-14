import argparse
import os
import logging
import torch
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Project-specific imports
from config import Config
from utils import configure_distributed, create_optimizer_scheduler, load_checkpoint, get_default_device
from vae import VideoVAE
from pyramid_logic import PyramidFlowMatcher
from model import PyramidalFlowMatchingModel, TextEncoder
from data_loader import create_data_loaders
from trainer import Trainer
from inference import VideoGenerator
from evaluation import Evaluator

# Setup logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def main(args: argparse.Namespace) -> None:
    """
    Main function to orchestrate the Pyramidal Flow Matching application.
    Loads configuration, initializes components, and runs training, generation, or evaluation.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    try:
        # 1. Configuration Loading
        cfg = Config.from_yaml(args.config_path)
        logger.info(f"Configuration loaded from {args.config_path}")
        # Log derived spatial pyramid time windows for verification
        logger.info(f"Derived Spatial Pyramid Time Windows (coarsest to finest):")
        for i, (s_k, e_k) in enumerate(cfg.model.spatial_pyramid_time_windows):
            logger.info(f"  Stage k={i}: (s_k={s_k:.4f}, e_k={e_k:.4f})")

        # 2. Distributed Training Setup
        # configure_distributed() should be called early to set up process groups
        # if using accelerate it might handle this, but explicit call ensures it.
        # However, for accelerate, it's usually initialized by `accelerate launch`.
        # For this setup, we'll let `accelerate` handle DDP initialization when the Trainer is created.
        
        # Determine the primary device for model loading etc.
        # Accelerator will later manage device placement for distributed training.
        device = torch.device(cfg.compute.device)
        logger.info(f"Using device: {device}")

        # 3. Shared Component Initialization
        # Initialize VAE
        video_vae = VideoVAE(cfg).to(device)
        if Path(cfg.data_paths.vae_weights).exists():
            # For VAE, we usually load only model weights, not optimizer/scheduler states
            # load_checkpoint returns global_step, current_stage_idx
            _, _ = load_checkpoint(video_vae, None, None, cfg, cfg.data_paths.vae_weights)
            logger.info(f"Loaded VAE weights from {cfg.data_paths.vae_weights}")
        else:
            logger.warning(f"VAE weights not found at {cfg.data_paths.vae_weights}. VAE will be trained from scratch (if part of overall training) or used uninitialized.")
        video_vae.eval() # VAE should typically be in eval mode during main model training/inference

        # Initialize Text Encoders
        text_encoders = TextEncoder(cfg).to(device)
        text_encoders.eval() # Text encoders are typically in eval mode
        logger.info("Initialized Text Encoders.")

        # Initialize Pyramid Flow Matcher Logic
        pyramid_flow_matcher = PyramidFlowMatcher(cfg, video_vae)
        logger.info("Initialized PyramidFlowMatcher.")

        # 4. Mode-Specific Execution
        if args.mode == 'train':
            logger.info("Starting training mode...")

            # Initialize PyramidalFlowMatchingModel
            pfm_model = PyramidalFlowMatchingModel(cfg, text_encoders)
            
            # Load main model checkpoint if available
            start_global_step, start_stage_idx = load_checkpoint(
                pfm_model, None, None, cfg, cfg.data_paths.model_weights
            )
            pfm_model.to(device) # Move to device before passing to Trainer

            # Initialize Trainer (will handle Accelerator setup)
            # The trainer will handle its own optimizer/scheduler management per stage
            # We need to create dummy loaders for init, and update them later for each stage.
            dummy_train_img_loader = None
            dummy_train_vid_loader = None
            dummy_val_vid_loader = None
            
            # Initialize with dummy optimizer/scheduler for the first stage
            initial_optimizer, initial_scheduler = create_optimizer_scheduler(pfm_model, cfg, 1)

            trainer = Trainer(
                cfg, pfm_model, video_vae, initial_optimizer, initial_scheduler,
                pyramid_flow_matcher, dummy_train_img_loader, dummy_train_vid_loader, dummy_val_vid_loader
            )
            trainer.global_step = start_global_step # Set global step
            trainer.current_stage_idx = start_stage_idx # Set current stage

            for stage_idx_one_based, stage_config in cfg.training.items():
                if stage_idx_one_based < start_stage_idx:
                    logger.info(f"Skipping already completed stage {stage_idx_one_based}.")
                    continue

                logger.info(f"--- Entering Training Stage {stage_idx_one_based}: {stage_config.name} ---")

                # Create Optimizer and Scheduler for the current stage
                current_optimizer, current_scheduler = create_optimizer_scheduler(pfm_model, cfg, stage_idx_one_based)
                
                # Update trainer's optimizer and scheduler
                trainer.update_optimizer_scheduler(current_optimizer, current_scheduler)

                # Create Data Loaders for the current stage
                current_train_img_loader, current_train_vid_loader, current_val_vid_loader = create_data_loaders(
                    cfg, video_vae, text_encoders, stage_idx_one_based
                )
                
                # Update trainer's data loaders based on stage type
                trainer.update_data_loaders(current_train_img_loader, current_train_vid_loader, current_val_vid_loader)

                # Run the current training stage
                trainer.train_stage(stage_idx_one_based)
            
            logger.info("All training stages completed.")

        elif args.mode == 'generate':
            logger.info("Starting generation mode...")

            # Initialize PyramidalFlowMatchingModel
            pfm_model = PyramidalFlowMatchingModel(cfg, text_encoders)
            
            # Load model weights for generation
            _, _ = load_checkpoint(pfm_model, None, None, cfg, cfg.data_paths.model_weights)
            pfm_model.eval().to(device)
            video_vae.eval().to(device) # Ensure VAE is also in eval mode
            logger.info(f"Loaded PyramidalFlowMatchingModel for generation from {cfg.data_paths.model_weights}")

            # Initialize Video Generator
            video_generator = VideoGenerator(cfg, pfm_model, video_vae, pyramid_flow_matcher)
            
            # Example prompts for generation (can be loaded from a file or CLI)
            if cfg.evaluation.evaluation_prompts_path and Path(cfg.evaluation.evaluation_prompts_path).exists():
                with open(cfg.evaluation.evaluation_prompts_path, 'r', encoding='utf-8') as f:
                    generation_prompts = [line.strip() for line in f if line.strip()]
            else:
                generation_prompts = ["A majestic lion roaring in the savanna.", "A serene waterfall in a lush forest.", "A bustling city street at night."]
                logger.warning(f"No evaluation prompts file found at {cfg.evaluation.evaluation_prompts_path}. Using default prompts for generation.")

            logger.info(f"Generating {len(generation_prompts)} videos...")
            for i, prompt in enumerate(generation_prompts):
                logger.info(f"Generating video for prompt: '{prompt}'")
                generated_video_tensor = video_generator.generate_video(
                    prompt=prompt,
                    guidance_scale=cfg.inference.guidance_scale,
                    num_frames=cfg.inference.output_duration * cfg.inference.output_fps,
                    output_resolution=cfg.inference.output_resolution
                )
                
                # Save the generated video (assuming generated_video_tensor is (T, H, W, C) uint8)
                output_dir = Path(cfg.evaluation.generated_video_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = f"generated_video_{i:03d}_{prompt[:50].replace(' ', '_').replace('/', '')}.mp4"
                output_path = output_dir / filename
                from torchvision.io import write_video
                write_video(
                    filename=str(output_path),
                    video_array=generated_video_tensor,
                    fps=cfg.inference.output_fps,
                    video_codec="libx264"
                )
                logger.info(f"Saved generated video to {output_path}")

            logger.info("Video generation completed.")

        elif args.mode == 'evaluate':
            logger.info("Starting evaluation mode...")

            # Initialize PyramidalFlowMatchingModel
            pfm_model = PyramidalFlowMatchingModel(cfg, text_encoders)
            
            # Load model weights for evaluation
            _, _ = load_checkpoint(pfm_model, None, None, cfg, cfg.data_paths.model_weights)
            pfm_model.eval().to(device)
            video_vae.eval().to(device) # Ensure VAE is also in eval mode
            logger.info(f"Loaded PyramidalFlowMatchingModel for evaluation from {cfg.data_paths.model_weights}")

            # Initialize Video Generator
            video_generator = VideoGenerator(cfg, pfm_model, video_vae, pyramid_flow_matcher)
            logger.info("Initialized VideoGenerator for evaluation.")

            # Initialize Evaluator
            evaluator = Evaluator(cfg, video_generator)
            logger.info("Initialized Evaluator.")

            # Run Evaluation
            evaluation_results = evaluator.evaluate()
            logger.info("Evaluation results:")
            print(json.dumps(evaluation_results, indent=2))

        else:
            logger.error(f"Unknown mode: {args.mode}")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        # Optionally re-raise for external error handling
        # raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pyramidal Flow Matching for Efficient Video Generative Modeling")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['train', 'generate', 'evaluate'],
        default='train',
        help="Operation mode: 'train' for training, 'generate' for video generation, 'evaluate' for evaluation."
    )
    # Add any other command-line arguments needed (e.g., for specific generation prompts)
    # parser.add_argument("--prompt", type=str, help="Text prompt for generation mode.")

    args = parser.parse_args()
    main(args)
