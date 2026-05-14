## main.py
import argparse
import logging
import os
import random
import sys
import yaml
import json

import numpy as np
import torch
import torch.optim as optim
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel

from config import Config
from data.dataset import TextPromptDataset
from diffusion.adjoint_solver import LeanAdjointSolver
from diffusion.noise_schedule import NoiseSchedule
from diffusion.sde_solver import SDESolver
from evaluation.evaluator import Evaluator
from models.flow_matching_unet import FlowMatchingUNet
from models.reward_model import RewardModel
from trainers.adjoint_matching_trainer import AdjointMatchingTrainer
from trainers.baseline_trainer import BaselineTrainer
from utils import helpers


def setup_environment(config: Config):
  """
  Sets up the computational environment, including random seeds,
  output directories, and logging.

  Args:
      config: The global configuration object.
  """
  # Set random seeds for reproducibility
  seed = config.general.seed
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  # Create output directories
  run_output_dir = os.path.join(config.general.output_dir, config.general.run_name)
  os.makedirs(run_output_dir, exist_ok=True)
  os.makedirs(os.path.join(run_output_dir, "checkpoints"), exist_ok=True)
  os.makedirs(os.path.join(run_output_dir, "evaluation"), exist_ok=True)

  # Setup logging
  log_file_path = os.path.join(run_output_dir, "log.txt")
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - %(message)s",
      handlers=[
          logging.FileHandler(log_file_path),
          logging.StreamHandler(sys.stdout),
      ],
  )
  logging.info(f"Environment setup complete. Output directory: {run_output_dir}")
  logging.info(f"Configuration:\n{json.dumps(config.to_dict(), indent=4)}")

  return run_output_dir


def main():
  """
  Main entry point of the Adjoint Matching reproduction pipeline.
  Handles argument parsing, configuration loading, component initialization,
  training, and evaluation.
  """
  parser = argparse.ArgumentParser(
      description="Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models"
  )
  parser.add_argument(
      "--config_path",
      type=str,
      default="config.yaml",
      help="Path to the configuration YAML file.",
  )
  parser.add_argument(
      "--mode",
      type=str,
      default="train_and_eval",
      choices=["train", "eval", "train_and_eval"],
      help="Operation mode: 'train', 'eval', or 'train_and_eval'.",
  )
  parser.add_argument(
      "--run_name",
      type=str,
      default=None,
      help="Override run_name specified in config.yaml.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=None,
      help="Override random seed specified in config.yaml.",
  )
  parser.add_argument(
      "--checkpoint_path",
      type=str,
      default=None,
      help="Path to a model checkpoint to load for evaluation mode.",
  )
  args = parser.parse_args()

  # Load configuration
  config = Config.from_yaml(args.config_path)
  if args.run_name:
    config.general.run_name = args.run_name
  if args.seed:
    config.general.seed = args.seed

  # Setup environment (seeds, directories, logging)
  run_output_dir = setup_environment(config)
  logging.info(f"Using device: {config.general.device}")

  # Determine the effective fine-tuning config (handles baseline overrides)
  effective_ft_config = config.get_effective_fine_tuning_config(
      config.fine_tuning.method
  )
  logging.info(
      f"Effective Fine-tuning Configuration for '{config.fine_tuning.method}':\n"
      f"{json.dumps(effective_ft_config.to_dict(), indent=4)}"
  )
  config.fine_tuning = effective_ft_config # Update global config with effective settings

  # --- Initialize Common Components ---
  # Text Encoder and Tokenizer (for FlowMatchingUNet and Datasets)
  logging.info("Initializing text encoder and tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder.model_name)
  text_encoder = CLIPTextModel.from_pretrained(
      config.model.text_encoder.model_name
  ).to(config.general.device)
  text_encoder.eval()
  for param in text_encoder.parameters():
    param.requires_grad = False
  logging.info(f"Text encoder: {config.model.text_encoder.model_name} loaded.")

  # VAE Decoder (for converting latents to pixel space for RewardModel and evaluation)
  logging.info("Initializing VAE decoder...")
  # Assuming Stable Diffusion's VAE for 512x512 latent space
  vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(
      config.general.device
  )
  vae.eval()
  for param in vae.parameters():
    param.requires_grad = False
  logging.info("VAE decoder loaded.")

  # Noise Schedule
  logging.info("Initializing noise schedule...")
  noise_schedule = NoiseSchedule(config.fine_tuning.num_timesteps)
  logging.info(f"Noise schedule initialized with K={config.fine_tuning.num_timesteps}.")

  # --- Initialize Models ---
  logging.info("Initializing generative models and reward model...")
  # Base Flow Matching Model (v_base) - frozen
  v_base = FlowMatchingUNet(
      unet_config=config.model.unet_config.to_dict(),
      text_encoder=text_encoder,
      noise_schedule=noise_schedule,
      pretrained_path=config.model.pretrained_base_model_path,
      device=config.general.device,
  )
  v_base.eval()
  for param in v_base.parameters():
    param.requires_grad = False
  logging.info("Base Flow Matching UNet (v_base) loaded and frozen.")
  config.base_flow_model = v_base # Store base model in config for Evaluator's access

  # Fine-tuned Flow Matching Model (v_finetune) - starts as copy of v_base
  v_finetune = FlowMatchingUNet(
      unet_config=config.model.unet_config.to_dict(),
      text_encoder=text_encoder,
      noise_schedule=noise_schedule,
      device=config.general.device,
  )
  v_finetune.load_state_dict(v_base.state_dict())  # Start from base model weights
  logging.info("Fine-tuned Flow Matching UNet (v_finetune) initialized from v_base.")

  # Reward Model
  reward_model = RewardModel(
      model_name=config.reward_model.name, device=config.general.device
  )
  reward_model.eval()
  logging.info(f"Reward Model: {config.reward_model.name} loaded and frozen.")

  # --- Initialize Datasets ---
  logging.info("Initializing datasets...")
  fine_tuning_dataset = TextPromptDataset(
      file_path=config.data.fine_tuning_prompts_path,
      tokenizer=tokenizer,
      text_encoder=text_encoder,
      max_length=config.model.text_encoder.max_length,
      device=config.general.device,
  )
  eval_dataset = TextPromptDataset(
      file_path=config.data.eval_prompts_path,
      tokenizer=tokenizer,
      text_encoder=text_encoder,
      max_length=config.model.text_encoder.max_length,
      device=config.general.device,
  )
  logging.info("Datasets initialized.")

  # --- Initialize Solvers ---
  logging.info("Initializing SDE and Lean Adjoint solvers...")
  sde_solver = SDESolver(
      generative_model=v_finetune,
      noise_schedule=noise_schedule,
      cfg_weight=0.0,  # CFG is for sampling, not fine-tuning SDE simulation directly
      device=config.general.device,
  )
  lean_adjoint_solver = LeanAdjointSolver(
      config=config,
      base_model=v_base,
      reward_model=reward_model,
      noise_schedule=noise_schedule,
      vae_decoder=vae,
  )
  logging.info("Solvers initialized.")

  # --- Initialize Optimizer ---
  logging.info("Initializing optimizer...")
  if config.fine_tuning.optimizer.name == "Adam":
      optimizer_class = optim.Adam
  elif config.fine_tuning.optimizer.name == "AdamW":
      optimizer_class = optim.AdamW
  else:
      logging.warning(f"Optimizer {config.fine_tuning.optimizer.name} not explicitly handled. Defaulting to AdamW.")
      optimizer_class = optim.AdamW

  optimizer = optimizer_class(
      v_finetune.parameters(),
      lr=config.fine_tuning.optimizer.learning_rate,
      betas=config.fine_tuning.optimizer.betas,
      eps=config.fine_tuning.optimizer.eps,
      weight_decay=config.fine_tuning.optimizer.weight_decay,
  )
  logging.info(f"Optimizer: {config.fine_tuning.optimizer.name} initialized.")

  # --- Training Phase ---
  if args.mode in ["train", "train_and_eval"]:
    logging.info("Starting training phase...")
    trainer = None
    if config.fine_tuning.method == "AdjointMatching":
      trainer = AdjointMatchingTrainer(
          config=config,
          flow_model=v_finetune,
          base_flow_model=v_base,
          reward_model=reward_model,
          dataset=fine_tuning_dataset,
          sde_solver=sde_solver,
          lean_adjoint_solver=lean_adjoint_solver,
          noise_schedule=noise_schedule,
          optimizer=optimizer,
      )
    else:  # Baseline methods
      trainer = BaselineTrainer(
          config=config,
          flow_model=v_finetune,
          base_flow_model=v_base,
          reward_model=reward_model,
          dataset=fine_tuning_dataset,
          sde_solver=sde_solver,
          noise_schedule=noise_schedule,
          optimizer=optimizer,
          baseline_type=config.fine_tuning.method,
          vae_decoder=vae,
          text_encoder=text_encoder,
          tokenizer=tokenizer,
      )
    v_finetune = trainer.train()
    logging.info("Training phase completed.")
    # The final checkpoint is saved within trainer.train()

  # --- Evaluation Phase ---
  if args.mode in ["eval", "train_and_eval"]:
    logging.info("Starting evaluation phase...")
    current_iteration = config.fine_tuning.num_fine_tune_iterations

    # If only evaluation mode, load the fine-tuned model checkpoint
    if args.mode == "eval" and args.checkpoint_path:
      checkpoint_path = args.checkpoint_path
      if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
      v_finetune.load_state_dict(torch.load(checkpoint_path, map_location=config.general.device))
      logging.info(f"Loaded model checkpoint from {checkpoint_path} for evaluation.")
      # For pure eval, we might not have a meaningful iteration number.
      # Use a placeholder like 0 or derive from filename if possible.
      try:
          current_iteration = int(os.path.basename(checkpoint_path).split('_')[-1].split('.')[0])
      except:
          current_iteration = 0 # Default if iteration cannot be parsed

    evaluator = Evaluator(
        config=config,
        flow_model=v_finetune,
        reward_model=reward_model,
        eval_dataset=eval_dataset,
        sde_solver=sde_solver,
        noise_schedule=noise_schedule,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # Evaluate the base model (iteration 0) if in train_and_eval mode,
    # or if explicitly evaluating a checkpoint (in which case current_iteration might be 0)
    # The evaluator will use `config.base_flow_model` for the `iteration=0` evaluation.
    # The main loop's `sde_solver` is initialized with `v_finetune`.
    # `Evaluator.evaluate` handles switching between `v_base` and `v_finetune` for metrics.
    # The `Evaluator.evaluate` expects `config.base_flow_model` to be set.
    # Also, `config.evaluation.cfg_weights` includes `0.0` for no-guidance evaluation.

    # Evaluate the fine-tuned model
    evaluator.evaluate(iteration=current_iteration)
    logging.info("Evaluation phase completed.")

  logging.info("Adjoint Matching pipeline finished.")


if __name__ == "__main__":
  main()
