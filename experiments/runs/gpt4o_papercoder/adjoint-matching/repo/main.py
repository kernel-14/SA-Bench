## main.py
import hydra
from omegaconf import DictConfig
import torch
from utils import set_seed, save_config, log_message
from dataset_loader import DatasetLoader
from model import BaseModel
from memoryless_noise_schedule import NoiseSchedule
from trainer import Trainer
from adjoint_matching import AdjointMatching
from evaluation import Evaluation

@hydra.main(config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main function to orchestrate the workflow for pretraining, fine-tuning,
    and evaluation of Flow Matching and Diffusion models based on Adjoint Matching methodology.

    Args:
        config (DictConfig): Loaded configuration from config.yaml.
    """
    try:
        # Initialize reproducibility
        set_seed(config.general.seed)

        # Log paths and configuration
        log_dir = config.logging.log_dir
        checkpoint_path = config.checkpoint.checkpoint_path
        
        log_message(f"[INFO] Starting pipeline execution...", log_path=f"{log_dir}/execution.log")
        save_config(config, f"{log_dir}/config.yaml")

        # Initialize DatasetLoader and load datasets
        log_message("[INFO] Initializing DatasetLoader...")
        dataset_loader = DatasetLoader(config)
        train_loader, val_loader, test_loader = dataset_loader.load_data()

        # Initialize memoryless noise schedule
        log_message("[INFO] Generating memoryless noise schedule...")
        noise_schedule = NoiseSchedule(config)
        log_message(f"[INFO] Noise schedule generated: {noise_schedule.get_schedule()}")

        # Initialize model
        log_message("[INFO] Initializing BaseModel...")
        model = BaseModel(params=config)

        # Load checkpoint if specified
        resume_from_checkpoint = config.checkpoint.resume_from_checkpoint
        if resume_from_checkpoint:
            checkpoint_path = config.checkpoint.checkpoint_path
            model.load_checkpoint(checkpoint_path)

        # Training or pretraining
        log_message("[INFO] Initiating Trainer...")
        trainer = Trainer(model=model, config=config, train_loader=train_loader, val_loader=val_loader)
        trainer.train()  # Train the base model

        # Fine-tuning using Adjoint Matching
        log_message("[INFO] Initiating Adjoint Matching for Fine-tuning...")
        adjoint_matching = AdjointMatching(model=model, config=config, train_loader=train_loader)
        adjoint_matching.fine_tune()  # Fine-tune the model with memoryless schedules

        # Evaluate final model
        log_message("[INFO] Evaluating Model...")
        evaluation = Evaluation(model=model, config=config, test_loader=test_loader)
        metrics = evaluation.evaluate_metrics()
        log_message(f"[INFO] Evaluation Results: {metrics}", log_path=f"{log_dir}/evaluation.log")

        # Save final model checkpoint
        model.save_checkpoint(f"{checkpoint_path}/final_model.pth")
        log_message("[INFO] Final model checkpoint saved.")

    except Exception as e:
        log_message(f"[ERROR] Execution failed with error: {str(e)}", log_path=f"{log_dir}/execution.log")
        raise

if __name__ == "__main__":
    main()
