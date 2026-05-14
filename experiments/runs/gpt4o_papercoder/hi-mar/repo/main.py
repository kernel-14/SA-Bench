# main.py

import torch
import argparse
import yaml
from typing import Dict, Any
from dataset_loader import DatasetLoader
from model import VAE, HiMARTransformer, DiffusionHead
from trainer import Trainer
from evaluation import Evaluation

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load the configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def run_experiment(experiment_type: str = "ImageNet", model_variant: str = "Base") -> None:
    """
    Run the complete experimentation pipeline (training and evaluation).

    Args:
        experiment_type (str): Type of experiment ("ImageNet" or "MSCOCO").
        model_variant (str): Model scale variant ("Base", "Large", "Huge").
    """
    # Load configuration
    config = load_config()

    # Load Dataset
    dataset_loader = DatasetLoader(config)
    if experiment_type == "ImageNet":
        train_dataset, val_dataset = dataset_loader.load_ImageNet()
    elif experiment_type == "MSCOCO":
        train_dataset, val_dataset = dataset_loader.load_MSCOCO()
    else:
        raise ValueError(f"Invalid experiment type: {experiment_type}. Must be 'ImageNet' or 'MSCOCO'.")

    # Configure DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    # Model Initialization
    pretrained_vae_path = config["vae"]["pretrained_path"]
    resolutions = config["vae"]["resolutions"]
    vae = VAE(pretrained_path=pretrained_vae_path, resolutions=resolutions)

    model_config = {
        "Base": {"layers": 24, "hidden_size": 768},
        "Large": {"layers": 32, "hidden_size": 1024},
        "Huge": {"layers": 40, "hidden_size": 1280}
    }
    if model_variant not in model_config:
        raise ValueError(f"Invalid model_variant: {model_variant}. Must be 'Base', 'Large', or 'Huge'.")
    transformer_params = model_config[model_variant]
    hi_mar_transformer = HiMARTransformer(
        layers=transformer_params["layers"],
        hidden_size=transformer_params["hidden_size"],
        scale_embedding=True
    )

    # Initialize Diffusion Heads for both phases
    phase1_diffusion_head = DiffusionHead(
        type="MLP",
        params={"hidden_size": transformer_params["hidden_size"], "num_layers": 6}
    )
    phase2_diffusion_head = DiffusionHead(
        type="Transformer",
        params={"hidden_size": transformer_params["hidden_size"], "num_layers": 6}
    )

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        hi_mar_transformer.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["optimizer"]["beta1"], config["training"]["optimizer"]["beta2"]),
        weight_decay=config["training"]["optimizer"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    # Trainer
    trainer = Trainer(
        model=hi_mar_transformer,
        vae=vae,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )

    # Training Loop
    print("Starting training...")
    for epoch in range(config["training"]["epochs"]):
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        trainer.train_epoch(train_loader)

        # Validation and Save Checkpoints every few epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config["training"]["epochs"]:
            print(f"Validating at epoch {epoch + 1}...")
            metrics = trainer.validate(val_loader)
            print(f"Validation Metrics at epoch {epoch + 1}: {metrics}")
            checkpoint_path = f"checkpoints/hi_mar_{experiment_type.lower()}_{model_variant.lower()}_epoch_{epoch + 1}.pth"
            trainer.save_checkpoint(checkpoint_path)

    # Final Evaluation
    evaluation = Evaluation(model=hi_mar_transformer, vae=vae, config=config)

    print("Final Evaluation Metrics:")
    if experiment_type == "ImageNet":
        # Compute FID, IS, Precision, Recall
        fid_score = evaluation.evaluate_FID(val_loader)
        is_score = evaluation.evaluate_IS(val_loader)
        print(f"FID: {fid_score}, IS: {is_score}")
    elif experiment_type == "MSCOCO":
        # Compute FID and Text-to-Image Compositional Metrics
        fid_score = evaluation.evaluate_FID(val_loader)
        composition_scores = evaluation.evaluate_composition(val_loader)
        print(f"FID: {fid_score}, Composition Scores: {composition_scores}")

if __name__ == "__main__":
    # Argument Parser for CLI interaction
    parser = argparse.ArgumentParser(description="Hi-MAR: Hierarchical Masked Autoregressive Image Generator")
    parser.add_argument(
        "--experiment_type", type=str, default="ImageNet",
        help="Type of experiment to run ('ImageNet' or 'MSCOCO')"
    )
    parser.add_argument(
        "--model_variant", type=str, default="Base",
        help="Scale variant of the model ('Base', 'Large', 'Huge')"
    )
    args = parser.parse_args()

    # Run the experiment
    run_experiment(experiment_type=args.experiment_type, model_variant=args.model_variant)
