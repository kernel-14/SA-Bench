import argparse
import logging
import os
import yaml
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from diffusers.models import AutoencoderKL
from transformers import CLIPTextModel, AutoTokenizer

# Import custom modules
from config import Config
from utils.logger import setup_logging
from utils.diffusion_schedulers import DiffusionScheduler
from utils.kv_cache_manager import KVCacheManager
from models.ca2_vdm_model import Ca2VDM
from data.video_dataset import VideoDataset
from pipelines.trainer import Trainer
from pipelines.inferrer import Inferrer
from utils.metrics import Evaluator

logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the Ca2-VDM reproduction system.
    Handles command-line arguments, configuration loading, component initialization,
    and execution of training, evaluation, or profiling tasks.
    """
    parser = argparse.ArgumentParser(description="Ca2-VDM Reproduction System")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "profile"],
                        help="Operation mode: 'train', 'eval', or 'profile'.")
    parser.add_argument("--config_path", type=str, default="config.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument("--task", type=str, default="t2v_internvid",
                        choices=["t2v_internvid", "vp_skytimelapse"],
                        help="Specific task to run (e.g., 't2v_internvid' for Text-to-Video).")
    parser.add_argument("--stage", type=str, default=None,
                        help="Specific stage within a task (e.g., 'ca2_vdm_stage1', 'ca2_vdm_stage2', 'os_fix_baseline' "
                             "for T2V InternVid; 'ca2_vdm', 'os_ext', 'os_fix_baseline' for VP SkyTimelapse).")
    args = parser.parse_args()

    # 1. Configuration Loading
    try:
        config = Config(args.config_path, args.task, args.stage)
        logger.info(f"Loaded configuration for task: '{args.task}', stage: '{args.stage}'")
    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        logger.error(f"Error loading or parsing configuration: {e}")
        exit(1)

    # 2. Logging Setup
    setup_logging(config)

    logger.info(f"Starting Ca2-VDM in '{args.mode}' mode.")
    logger.info(f"Configuration details: \n{config}")

    # 3. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 4. Component Initialization (Common to all modes)

    # VAE
    logger.info(f"Initializing VAE from '{config.vae_model_name}'...")
    vae = AutoencoderKL.from_pretrained(config.vae_model_name).to(device)
    vae.eval()
    vae.requires_grad_(False) # Freeze VAE parameters
    logger.info("VAE initialized and frozen.")

    # Text Encoder and Tokenizer
    logger.info(f"Initializing Text Encoder and Tokenizer from '{config.text_encoder_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_name)
    text_encoder = CLIPTextModel.from_pretrained(config.text_encoder_name).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False) # Freeze Text Encoder parameters
    logger.info("Text Encoder and Tokenizer initialized and frozen.")

    # Diffusion Scheduler
    logger.info("Initializing Diffusion Scheduler...")
    diffusion_scheduler = DiffusionScheduler(
        num_train_timesteps=config.diffusion_steps,
        beta_schedule=config.beta_schedule,
        beta_start=config.beta_start,
        beta_end=config.beta_end
    )
    logger.info("Diffusion Scheduler initialized.")

    # Ca2VDM Model
    logger.info(f"Initializing Ca2VDM model based on Open-Sora architecture from '{config.open_sora_config_path}'...")
    # As per ambiguity, Ca2VDM model expects a dict for Open-Sora config.
    # In a real Open-Sora integration, this would likely be a more complex config object or loaded directly.
    # For this reproduction, we provide a placeholder dict that Ca2VDM constructor will interpret.
    # The values here are illustrative; Ca2VDM class would ultimately parse its own internal structure.
    dummy_open_sora_config_dict = {
        "model_type": "transformer",
        "num_layers": 12,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "patch_size": [1, 2, 2],
        "vae_latent_channels": vae.config.latent_channels if vae.config.latent_channels else 4,
        "cross_attention_dim": text_encoder.config.hidden_size if text_encoder.config.hidden_size else 768,
        "in_channels": vae.config.latent_channels if vae.config.latent_channels else 4,
        "out_channels": vae.config.latent_channels * 2 if vae.config.latent_channels else 8, # For noise and variance
        "image_size": config.image_size // vae.config.scaling_factor if vae.config.scaling_factor else 32, # Latent spatial size
        "use_prefix_enhancement": config.use_prefix_enhancement,
        "prefix_enhancement_sub_len": config.prefix_enhancement_sub_len
    }

    model = Ca2VDM(open_sora_config=dummy_open_sora_config_dict, vae=vae, text_encoder=text_encoder).to(device)
    logger.info("Ca2VDM model initialized.")

    # --- Mode-Specific Execution ---

    if args.mode == "train":
        logger.info("Entering training mode.")
        # Ensure that the config has valid training parameters for the selected task/stage
        if config.batch_size is None or config.training_steps is None:
            logger.error("Batch size or training steps not configured for the selected task/stage.")
            exit(1)

        # VideoDataset for training
        train_dataset = VideoDataset(
            data_path=config.data_path,
            config=config,
            is_train=True,
            text_encoder_tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            task_name=args.task
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 2 if os.cpu_count() else 4, # Use half cores for workers, or 4 if unknown
            pin_memory=True
        )
        logger.info(f"Training DataLoader created with batch size {config.batch_size}.")

        # VideoDataset for validation (if needed for Trainer)
        # For simplicity, we'll create a small validation dataset, possibly using a subset of train data.
        # In a real scenario, a dedicated validation split would be loaded.
        val_dataset = VideoDataset(
            data_path=config.data_path,
            config=config,
            is_train=False, # Treated as validation set during training
            text_encoder_tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            task_name=args.task
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2 if os.cpu_count() else 4,
            pin_memory=True
        )
        logger.info(f"Validation DataLoader created with batch size {config.batch_size}.")


        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        logger.info(f"Optimizer (AdamW) initialized with learning rate {config.learning_rate}.")

        # Trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=diffusion_scheduler,
            vae=vae,
            text_encoder=text_encoder,
            config=config
        )
        logger.info("Trainer initialized. Starting training...")
        trainer.train()
        logger.info("Training completed.")

    elif args.mode == "eval" or args.mode == "profile":
        logger.info(f"Entering {args.mode} mode.")

        # Load a pre-trained model checkpoint
        checkpoint_path = os.path.join(config.save_path, "checkpoints", "ca2_vdm_checkpoint_final.pt")
        if not os.path.exists(checkpoint_path):
            logger.error(f"Model checkpoint not found at {checkpoint_path}. Cannot perform evaluation or profiling.")
            exit(1)
        
        logger.info(f"Loading model checkpoint from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info("Model loaded and set to evaluation mode.")

        # Inferrer
        inferrer = Inferrer(
            model=model,
            vae=vae,
            text_encoder=text_encoder,
            scheduler=diffusion_scheduler,
            config=config
        )
        logger.info("Inferrer initialized.")

        # Evaluator
        evaluator = Evaluator(
            config=config,
            inferrer=inferrer,
            device=device
        )
        logger.info("Evaluator initialized.")

        if args.mode == "eval":
            # VideoDataset for evaluation
            eval_dataset = VideoDataset(
                data_path=config.data_path,
                config=config,
                is_train=False,
                text_encoder_tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                task_name=args.task
            )
            logger.info("Evaluation Dataset created.")

            logger.info("Evaluating generation quality...")
            quality_results = evaluator.evaluate_generation_quality(eval_dataset, task_mode=args.task)
            logger.info(f"Generation Quality Results: {quality_results}")

            logger.info("Evaluating temporal consistency...")
            consistency_results = evaluator.evaluate_temporal_consistency(eval_dataset, task_mode=args.task)
            logger.info(f"Temporal Consistency Results: {consistency_results}")

        elif args.mode == "profile":
            logger.info("Entering profiling mode.")
            # For profiling, we need some dummy inputs or a single real sample.
            # Here, we create a dummy input for a single frame and text.
            dummy_first_frame_latents = torch.randn(
                1, vae.config.latent_channels, config.image_size // vae.config.scaling_factor, config.image_size // vae.config.scaling_factor
            ).to(device)
            dummy_text_prompt = "a video of a dog running"
            dummy_text_input = tokenizer(
                dummy_text_prompt,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)
            dummy_text_embeddings = text_encoder(dummy_text_input.input_ids, attention_mask=dummy_text_input.attention_mask)[0]
            
            # Using the configured chunk_length from the active task
            num_ar_steps_for_profile = 5 # Example number of AR steps for profiling
            
            # Define a lambda function to pass to profile_performance
            generate_func = lambda: inferrer.generate_video_autoregressive(
                first_frame_latents=dummy_first_frame_latents,
                text_prompt_embeddings=dummy_text_embeddings,
                num_ar_steps=num_ar_steps_for_profile,
                task_mode=args.task,
                uncond_text_embeddings=None # For profiling, keep it simple
            )

            logger.info(f"Profiling performance for {num_ar_steps_for_profile} AR steps...")
            performance_results = evaluator.profile_performance(generate_func)
            logger.info(f"Performance Profiling Results: {performance_results}")

    else:
        logger.error(f"Unknown mode: {args.mode}. Please choose from 'train', 'eval', or 'profile'.")
        exit(1)

if __name__ == "__main__":
    main()

