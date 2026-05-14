import argparse
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local application imports
from config import Config
from utils.noise_schedules import NoiseScheduler
from utils.logger import setup_logger # Renamed to avoid confusion with `logging` module
from data.base_dataset import BaseDataset # Import BaseDataset for type hinting
from data.slimpajama_dataset import SlimPajamaDataset
from data.lo_naesat_dataset import LO_NAESAT_Dataset
from data.puzzle_dataset import PuzzleDataset
from models.transformer_mdm import TransformerMDM
from models.pi_learner_arm import PiLearnerARM
from training.trainer import Trainer
from inference.inferrer import Inferrer
from evaluation.evaluator import Evaluator

# Get the logger instance set up by utils.logger
logger = logging.getLogger("MDM_Project_Logger")


def set_seed(seed: int) -> None:
    """Sets the random seed for reproducibility across multiple libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed} for reproducibility.")


class DummyTokenizer:
    """
    A dummy tokenizer for datasets that handle integer IDs directly (e.g., puzzles).
    It primarily provides mask_token_id and vocab_size from the config.
    """
    def __init__(self, config: Config):
        self.mask_token_id: int = config.get('data.mask_token_id', 0)
        self.vocab_size: int = config.get('data.vocab_size', 10) # Default for Sudoku


def run_experiment(config: Config) -> None:
    """
    Orchestrates the entire experiment workflow: data loading, model initialization,
    training, inference, and evaluation based on the provided configuration.

    Args:
        config (Config): The configuration object containing all experiment parameters.
    """
    # 1. Device Setup
    device: torch.device = torch.device(config.get('general.device', 'cpu'))
    logger.info(f"Using device: {device}")
    config.set('general.device', str(device)) # Update config with actual device string

    # 2. Tokenizer Initialization
    tokenizer: Any
    dataset_type: str = config.get('data.dataset_type')
    if dataset_type == "slimpajama" or dataset_type == "llada_tasks":
        tokenizer_name: str = config.get('data.tokenizer_name', 'bert-base-uncased')
        # Using `padding_side='left'` for perplexity calculation compatible with some models
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side='right') 
        # Add pad token if not already present, often needed for batching.
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Added [PAD] as special token to tokenizer.")
        
        # Ensure mask_token_id is available
        if tokenizer.mask_token_id is None:
            logger.warning(f"Tokenizer {tokenizer_name} does not have a default mask_token_id. "
                           "Using config default (0). Ensure this is correct for your model.")
            config.set('data.mask_token_id', config.get('data.mask_token_id', 0))
        else:
            config.set('data.mask_token_id', tokenizer.mask_token_id)

        config.set('data.vocab_size', len(tokenizer))
        logger.info(f"Initialized HuggingFace tokenizer '{tokenizer_name}'. Vocab size: {len(tokenizer)}")
    elif dataset_type in ["lo_naesat", "sudoku", "zebra"]:
        tokenizer = DummyTokenizer(config) # Custom tokenizer will be used by dataset
        # Vocab size and mask token ID will be confirmed/set by the dataset's __init__ or prepare_data
        # For Sudoku, vocab size is 10 (0-9) and mask is 0.
        # For Zebra, vocab size must be specified in config.
        # For L&O-NAESAT, vocab size is 'm'.
        logger.info(f"Initialized DummyTokenizer for {dataset_type} data.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # 3. Noise Scheduler Initialization
    noise_scheduler: NoiseScheduler = NoiseScheduler(
        config.get('training.noise_schedule.type', 'linear'),
        config.get('training.noise_schedule.num_diffusion_steps', 1000)
    )
    logger.info(f"Noise scheduler initialized: {config.get('training.noise_schedule.type')} with {config.get('training.noise_schedule.num_diffusion_steps')} steps.")

    # 4. Dataset and DataLoader Initialization
    train_dataset: Optional[BaseDataset] = None
    val_dataset: Optional[BaseDataset] = None
    test_dataset: Optional[BaseDataset] = None

    if dataset_type == "slimpajama":
        full_dataset = SlimPajamaDataset(config, tokenizer)
        full_dataset.prepare_data()
        
        # Split into train/val/test. No explicit splits specified in paper, so use common ratios.
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
        
        logger.info(f"SlimPajama Dataset loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    elif dataset_type == "lo_naesat":
        # For L&O-NAESAT, we create distinct datasets for train, val, test
        # `prepare_data` generates data, so each instance should be separate
        train_dataset = LO_NAESAT_Dataset(config)
        train_dataset.prepare_data(split="train")
        val_dataset = LO_NAESAT_Dataset(config)
        val_dataset.prepare_data(split="val")
        test_dataset = LO_NAESAT_Dataset(config)
        test_dataset.prepare_data(split="test")
        
        # Update config with actual vocab_size and mask_token_id from dataset if determined by it
        config.set('data.vocab_size', train_dataset.vocab_size)
        config.set('data.mask_token_id', train_dataset.mask_token_id)

        logger.info(f"L&O-NAESAT Dataset generated. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    elif dataset_type in ["sudoku", "zebra"]:
        train_dataset = PuzzleDataset(config, tokenizer) # tokenizer is DummyTokenizer here
        train_dataset.prepare_data(split="train")
        val_dataset = PuzzleDataset(config, tokenizer) # Assuming a validation split from train data if not separate
        val_dataset.prepare_data(split="test") # Using test as val if no specific val set
        test_dataset = PuzzleDataset(config, tokenizer)
        test_dataset.prepare_data(split="test")

        # For hard Sudoku evaluation, a separate test dataset
        if dataset_type == "sudoku" and config.get('puzzles.hard_test_data_path'):
            hard_test_dataset = PuzzleDataset(config, tokenizer)
            hard_test_dataset.prepare_data(split="hard_test")
            logger.info(f"Sudoku Hard Test Dataset loaded: {len(hard_test_dataset)}")
            config.set('evaluator.hard_test_dataset', hard_test_dataset) # Store for evaluator access

        # Update config with actual vocab_size and mask_token_id from dataset
        config.set('data.vocab_size', train_dataset.vocab_size)
        config.set('data.mask_token_id', train_dataset.mask_token_id)
        logger.info(f"Puzzle Dataset loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    else:
        raise NotImplementedError(f"Dataset type '{dataset_type}' not yet implemented.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('training.batch_size', 16),
        shuffle=True,
        num_workers=config.get('data.num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('training.batch_size', 16),
        shuffle=False,
        num_workers=config.get('data.num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('training.batch_size', 16), # Batch size can be larger for evaluation
        shuffle=False,
        num_workers=config.get('data.num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 5. Model Initialization
    model: torch.nn.Module
    model_type: str = config.get('model.model_type')

    if model_type == "mdm_transformer":
        model = TransformerMDM(config).to(device)
    elif model_type == "arm_transformer":
        model = PiLearnerARM(config).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info(f"Model '{model_type}' initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M parameters.")

    # Load pretrained model weights if specified
    pretrained_path: Optional[str] = config.get('model.pretrained_mdm_path')
    if pretrained_path:
        if os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device)
            # Assuming 'model_state_dict' key in checkpoint for consistency with Trainer save
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint)) 
        else:
            logger.warning(f"Pretrained model path '{pretrained_path}' not found. Starting training from scratch.")

    # 6. Training Phase
    if config.get('training.epochs', 0) > 0 or config.get('training.iterations', 0) > 0:
        logger.info("Starting training phase...")
        trainer = Trainer(config, model, train_loader, val_loader, noise_scheduler, logger)
        trainer.train()
        logger.info("Training phase completed.")
    else:
        logger.info("Skipping training phase as 'training.epochs' and 'training.iterations' are 0.")

    # 7. Inference and Evaluation Phase
    logger.info("Starting inference and evaluation phase...")
    inferrer: Optional[Inferrer] = None
    if model_type == "mdm_transformer":
        inferrer = Inferrer(config, model, tokenizer, noise_scheduler)
        logger.info("Inferrer initialized for MDM model.")
    elif model_type == "arm_transformer":
        logger.info("Inferrer not needed for ARM model evaluation directly.")

    llama_model_for_ppl: Optional[AutoModelForCausalLM] = None
    llama_tokenizer_for_ppl: Optional[AutoTokenizer] = None
    if "perplexity" in config.get('evaluation.metrics', []) and config.get('evaluation.perplexity_llm_model_path'):
        ppl_model_path: str = config.get('evaluation.perplexity_llm_model_path')
        try:
            llama_tokenizer_for_ppl = AutoTokenizer.from_pretrained(ppl_model_path)
            if llama_tokenizer_for_ppl.pad_token is None:
                llama_tokenizer_for_ppl.add_special_tokens({'pad_token': '[PAD]'})
            llama_model_for_ppl = AutoModelForCausalLM.from_pretrained(ppl_model_path).to(device)
            logger.info(f"LLaMA model '{ppl_model_path}' loaded for perplexity calculation.")
        except Exception as e:
            logger.error(f"Failed to load LLaMA model for perplexity from {ppl_model_path}: {e}")
            llama_model_for_ppl = None
            llama_tokenizer_for_ppl = None

    evaluator = Evaluator(
        config,
        model,
        test_loader,
        inferrer,
        tokenizer,
        llama_model_for_ppl,
        llama_tokenizer_for_ppl
    )
    logger.info("Evaluator initialized.")

    inference_strategies: List[str] = config.get('inference.inference_strategies', ['vanilla'])
    evaluation_results: Dict[str, Any] = evaluator.run_all_evaluations(inference_strategies)
    
    logger.info("Evaluation results:")
    for strategy, metrics in evaluation_results.items():
        logger.info(f"  Strategy: {strategy}")
        for metric, value in metrics.items():
            logger.info(f"    {metric}: {value:.4f}")

    logger.info("Inference and evaluation phase completed.")


def main() -> None:
    """Main function to parse arguments and start the experiment."""
    parser = argparse.ArgumentParser(description="Reproduce 'Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions'")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the configuration YAML file.'
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = Config(args.config)
    except (FileNotFoundError, yaml.YAMLError, Exception) as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Set up global logger
    logger_instance = setup_logger(config) # Pass config directly to setup_logger
    logger_instance.info(f"Configuration loaded from: {args.config}")
    logger_instance.info(f"Experiment name: {config.get('general.experiment_name')}")

    # Set random seed
    set_seed(config.get('general.seed', 42))

    # Run the experiment
    try:
        run_experiment(config)
    except Exception as e:
        logger_instance.critical(f"An unhandled error occurred during the experiment: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Finalize Weights & Biases run if active
        if 'wandb' in sys.modules and sys.modules['wandb'].run is not None:
            sys.modules['wandb'].finish()
        logger_instance.info("Experiment finished.")


if __name__ == '__main__':
    main()

