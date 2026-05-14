## main.py

import torch
import logging
from config import Config
from dataset_loader import DatasetLoader
from vae_model import VAEModel
from flow_matching_model import FlowMatchingModel
from trainer import Trainer
from evaluation import Evaluation


def initialize_logger() -> None:
    """
    Initializes logging configuration for the application.
    Logs detailed information about training, evaluation, and errors.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(config_path: str = "config.yaml") -> None:
    """
    Entry point of the application. Initializes components and orchestrates the training and evaluation pipeline.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # Step 1: Initialize logger
    initialize_logger()
    logging.info("Starting the Pyramidal Flow Matching for Efficient Video Generative Modeling.")

    # Step 2: Load configuration
    logging.info(f"Loading configuration from: {config_path}")
    config = Config(config_path)

    training_config = config.get_training_config()
    dataset_config = config.get_dataset_config()
    model_config = config.get_model_config()
    logging.info("Configuration loaded successfully.")

    # Step 3: Initialize DatasetLoader
    logging.info("Initializing DatasetLoader...")
    dataset_loader = DatasetLoader(config.config)

    # Step 4: Load datasets
    logging.info("Loading image datasets...")
    image_loader = dataset_loader.load_images()
    logging.info("Loading video datasets for training...")
    video_loader_short = dataset_loader.load_videos(duration="short")
    video_loader_long = dataset_loader.load_videos(duration="long")

    # Step 5: Initialize Models
    logging.info("Initializing VAEModel...")
    vae_model = VAEModel(
        latent_dim=model_config["vae"]["latent_dim"],
        downsample_ratio=tuple(model_config["vae"]["downsampling_ratio"]),
    )

    logging.info("Initializing FlowMatchingModel...")
    flow_model = FlowMatchingModel(
        num_stages=model_config["flow_matching"]["num_stages"],
        params=model_config,
    )

    # Step 6: Initialize Trainer
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model_vae=vae_model,
        model_flow=flow_model,
        datasets=dataset_loader,
        config=config.config,
    )

    # Step 7: Train Models
    logging.info("Training process initiated...")
    trainer.train()
    logging.info("Training process completed.")

    # Step 8: Evaluate Models
    logging.info("Initializing Evaluation...")
    evaluation = Evaluation(model=flow_model, dataset=video_loader_long)

    logging.info("Starting evaluation process...")
    evaluation_results = evaluation.evaluate()
    logging.info(f"Evaluation results: {evaluation_results}")

    # Step 9: Conduct Optional Human Study (if enabled)
    if config.get_evaluation_config().get("human_study", False):
        logging.info("Conducting human study for qualitative evaluation...")
        # Placeholder - human study logic should be implemented separately for deployment environments
        evaluation.conduct_user_study([], [])
        logging.info("Human study completed.")

    # Step 10: Wrap up
    logging.info("Pyramidal Flow Matching finished successfully. Exiting application.")


if __name__ == "__main__":
    # Run the main procedure with default configuration file path
    main()
