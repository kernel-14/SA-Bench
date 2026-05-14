
import argparse
from config import Config
from training.trainer import Trainer
from utils.misc import get_logger, load_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Train SAM 2 model.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode to run: 'train' or 'eval'.")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to a custom config file (e.g., YAML or another Python file).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a model checkpoint to resume training or for evaluation.")
    # Add more arguments as needed for fine-tuning, specific ablations, etc.

    args = parser.parse_args()

    # Initialize configuration
    config = Config()
    # TODO: if config_file is provided, load and override defaults

    logger = get_logger("main")
    logger.info(f"Starting SAM 2 in {args.mode} mode.")
    logger.info(f"Using device: {config.DEVICE}")

    trainer = Trainer(config)

    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        trainer.model, trainer.optimizer, start_step, start_loss = load_checkpoint(
            trainer.model, trainer.optimizer, args.checkpoint
        )
        trainer.current_step = start_step # Resume from loaded step
        logger.info(f"Resumed from step {start_step} with loss {start_loss:.4f}")

    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval":
        logger.info("Evaluation mode selected. Running validation on the validation set.")
        val_metrics = trainer.validate()
        logger.info(f"Final Validation Metrics: {val_metrics}")
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()

