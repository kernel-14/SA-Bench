## main.py
import os
import logging
import torch
from dataset_loader import DatasetLoader
from model import ConsistencyModel
from trainer import Trainer
from evaluation import Evaluation
from utils import load_config, setup_logging, validate_environment, save_checkpoint

def main(config_path: str = "config.yaml"):
    """
    Main entry point of the pipeline. Orchestrates the flow of configuration parsing,
    dataset loading, model initialization, training, and evaluation.
    
    Args:
        config_path (str): Path to the configuration file. Default: "config.yaml".
    """
    # Step 1: Load Configuration
    config = load_config(config_path)
    logging.info("Configuration loaded successfully.")

    # Step 2: Validate Environment and Prepare Logging
    validate_environment()
    setup_logging(log_type="training", frequency_steps=config["logging"]["log_frequency_steps"])

    # Step 3: Load Dataset
    logging.info("Initializing DatasetLoader...")
    dataset_loader = DatasetLoader(config)
    dataloaders = dataset_loader.load_data()
    logging.info("Datasets loaded successfully.")

    # Step 4: Initialize the Consistency Model
    logging.info("Initializing ConsistencyModel...")
    model = ConsistencyModel(config)
    logging.info("Consistency model initialized.")

    # Step 5: Train the Model
    logging.info("Starting training process...")
    trainer = Trainer(model=model, dataloaders=dataloaders, config=config)
    trainer.train()
    logging.info("Training process completed.")

    # Step 6: Evaluate the Model
    logging.info("Starting evaluation process...")
    evaluator = Evaluation(model=model, dataloaders=dataloaders, config=config)
    metrics = evaluator.evaluate_metrics()
    logging.info(f"Evaluation completed. Results: {metrics}")

    # Step 7: Save Final Checkpoint
    checkpoint_dir = config["logging"]["checkpoint_path"]
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_checkpoint(model, final_checkpoint_path)
    logging.info(f"Final model checkpoint saved at: {final_checkpoint_path}")

    # Step 8: Log Results
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
