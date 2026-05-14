## main.py

import os
import logging
from config import Config
from dataset_loader import DatasetLoader
from model import DiffusionModel
from trainer import Trainer
from sampler import Sampler
from evaluation import Evaluation
from utils import Utils


def main(config_path: str = "config.yaml") -> None:
    """
    Entry point for the Diffusion Model pipeline.

    Args:
        config_path (str): Path to the configuration YAML file. Defaults to 'config.yaml'.
    """
    # Configure logging for better traceability
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pipeline.log", mode="w")
        ]
    )

    logging.info("Starting pipeline execution.")

    # Step 1: Load configuration
    logging.info("Loading configuration from file.")
    config = Config(config_path).load_config()
    logging.info("Configuration successfully loaded.")

    # Step 2: Prepare datasets
    logging.info("Preparing datasets.")
    dataset_loader = DatasetLoader(config)
    train_data, test_data = dataset_loader.load_data()
    logging.info(f"Dataset preparation complete. Training data shape: {train_data.shape}, Test data shape: {test_data.shape}.")

    # Step 3: Initialize model
    logging.info("Initializing the diffusion model.")
    diffusion_model = DiffusionModel(config)
    logging.info("Diffusion model initialized.")

    # Step 4: Train the model
    logging.info("Training the diffusion model's score function.")
    trainer = Trainer(diffusion_model, (train_data, test_data), config)
    trainer.pretrain_score_function()
    trainer.train()
    logging.info("Model training complete.")

    # Step 5: Perform sampling
    logging.info("Initiating sampling process.")
    sampler = Sampler(diffusion_model, config)
    sampled_data = sampler.sample()
    logging.info("Sampling process complete.")

    # Step 6: Evaluate metrics
    logging.info("Evaluating the performance of the sampling process.")
    evaluator = Evaluation(diffusion_model, (test_data, sampled_data), config)
    evaluation_metrics = evaluator.evaluate(y_sampled=sampled_data, x_true=test_data, num_iterations=1)
    logging.info("Evaluation metrics computed.")

    # Save evaluation results
    evaluator.save_results(evaluation_metrics)
    logging.info(f"Evaluation metrics saved in {config['output']['directory']}.")

    # Step 7: Save checkpoints and plots
    utils = Utils()
    checkpoint_dir = config["output"]["directory"]
    utils.save_checkpoint(diffusion_model, epoch="final", output_directory=checkpoint_dir)
    logging.info(f"Final model checkpoint saved in {checkpoint_dir}.")
    utils.plot_metrics(evaluation_metrics, output_directory=checkpoint_dir)
    logging.info(f"Convergence plots saved in {checkpoint_dir}.")

    logging.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
