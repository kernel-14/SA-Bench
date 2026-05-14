## main.py

import os
import torch
from dataset_loader import DatasetLoader
from p2vae import P2VAE
from fmt_model import FMTModel
from trainer import Trainer
from evaluation import Evaluation
from utils import seed_everything, load_config, save_config, create_experiment_directory

def run_experiment(config_path: str = "config.yaml") -> None:
    """Main entry point for running the experiment pipeline, including dataset preparation, model training, and evaluation.
    
    Args:
        config_path (str): Path to the configuration file (default: 'config.yaml').
    """
    
    # Load configuration
    config = load_config(config_path)
    seed = config['logging']['seed']
    seed_everything(seed)
    
    # Create experiment directory and logger
    experiment_dir = create_experiment_directory(base_dir=config['logging']['save_dir'], exp_name="experiment_run")
    save_config(config, os.path.join(experiment_dir, "config_saved.yaml"))
    print(f"Experiment directory created at {experiment_dir}")

    # Dataset preparation
    print("Initializing dataset loader...")
    dataset_loader = DatasetLoader(config=config)
    dataset_splits = dataset_loader.load_dataset()
    print("Dataset loaded and split into train/validation/test.")

    # Model initialization: P2VAE
    print("Initializing P2VAE model...")
    latent_dim = config['vae']['latent_dim']
    enc_params = config['vae']['encoder']
    dec_params = config['vae']['decoder']
    p2vae_model = P2VAE(latent_dim=latent_dim, enc_params=enc_params, dec_params=dec_params)
    print(f"P2VAE model initialized with latent dimension: {latent_dim}")
    
    # Train P2VAE
    print("Starting P2VAE training...")
    trainer = Trainer(model=p2vae_model, dataset=dataset_splits, config=config)
    trainer.train_p2vae()
    p2vae_checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], "p2vae_final.pt")
    trainer.save_checkpoint(p2vae_checkpoint_path)
    print(f"P2VAE training completed. Model checkpoint saved to {p2vae_checkpoint_path}")

    # Freeze P2VAE weights for FMT training
    print("Freezing P2VAE weights and initializing FMT model...")
    input_dim = config['fmt_model']['input_dim']
    embed_dim = config['fmt_model']['embed_dim']
    num_heads = config['fmt_model']['num_heads']
    num_layers = config['fmt_model']['num_layers']
    rnn_dim = config['fmt_model']['rnn_dim']
    fmt_model = FMTModel(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, rnn_dim=rnn_dim)
    trainer = Trainer(model=fmt_model, dataset=dataset_splits, config=config)

    print(f"FMT model initialized with embedding dimension: {embed_dim}, {num_heads} heads, and {num_layers} layers.")
    
    # Load P2VAE checkpoint
    trainer.load_checkpoint(p2vae_checkpoint_path)
    print("P2VAE checkpoint loaded and frozen.")
    
    # Train FMT
    print("Starting FMT training...")
    trainer.train_fmt()
    fmt_checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], "fmt_final.pt")
    trainer.save_checkpoint(fmt_checkpoint_path)
    print(f"FMT training completed. Model checkpoint saved to {fmt_checkpoint_path}")

    # Evaluation
    print("Evaluating model performance...")
    evaluation = Evaluation(model=fmt_model, dataset=dataset_splits, metrics=config['evaluation']['metrics'], config=config)
    evaluation_results = evaluation.evaluate()
    print(f"Evaluation results: {evaluation_results}")

    print("Generating and saving visualization outputs...")
    evaluation.visualize_results()
    print(f"Visualization saved at {config['evaluation']['visualization']['save_path']}")

    print("Experiment completed successfully.")

# Execute the main function
if __name__ == "__main__":
    run_experiment()
