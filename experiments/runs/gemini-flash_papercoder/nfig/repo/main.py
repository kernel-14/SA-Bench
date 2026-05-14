import torch
import os
import logging
from omegaconf import OmegaConf

# Import all necessary modules
from config import Config
from datasets import DatasetLoader
from models.fr_vae import FRVAE
from models.discriminator import Discriminator
from models.nfig_transformer import NFIGTransformer
from training.fr_vae_trainer import FRVAETrainer
from training.nfig_trainer import NFIGTrainer
from metrics.evaluator import Evaluator
from utils import set_seed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Main:
    """
    Orchestrates the entire NFIG reproduction pipeline:
    1. Loads configuration.
    2. Initializes data loaders.
    3. Runs FR-VAE training.
    4. Tokenizes the dataset using the trained FR-VAE.
    5. Runs NFIG Transformer training.
    6. Performs final evaluation (rFID, gFID, IS, Precision, Recall).
    """

    def __init__(self):
        """
        Initializes the Main orchestrator by loading configuration, setting up the device,
        and initializing the DatasetLoader.
        """
        logging.info("Initializing NFIG reproduction pipeline.")

        # 1. Load configuration from config.yaml
        self.config = Config("config.yaml") # Default path for config
        logging.info("Configuration loaded successfully.")
        # Print resolved config for verification
        logging.info(f"Resolved Configuration:\n{OmegaConf.to_yaml(self.config.to_omegaconf())}")

        # 2. Setup environment (device, seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Ensure seed is set for reproducibility
        # Config class should have a 'training' section with 'seed'
        set_seed(self.config.training.seed)
        logging.info(f"Random seed set to: {self.config.training.seed}")

        # 3. Initialize DatasetLoader
        self.dataset_loader = DatasetLoader(self.config)
        logging.info("DatasetLoader initialized.")

        # Paths for saving/loading checkpoints and tokenized data
        # These paths are resolved and directories created by config.py
        self.fr_vae_checkpoint_path = self.config.model_checkpoints.fr_vae_path
        self.nfig_transformer_checkpoint_path = self.config.model_checkpoints.nfig_transformer_path
        self.token_train_dir = self.config.token_data_path.train_tokens_dir

    def run_fr_vae_training(self, load_checkpoint: bool = False):
        """
        Executes the training phase for the Frequency-guided Residual-quantized VAE (FR-VAE).

        Args:
            load_checkpoint (bool): If True, attempts to load the latest checkpoint
                                    to resume training or ensure a trained model is available.
        """
        logging.info("Starting FR-VAE training phase.")

        # 1. FR-VAE Model Initialization
        # Pass the config object and DINOv2 pre-trained weights path
        fr_vae = FRVAE(
            config=self.config,
            DINOv2_base_pretrained_path=self.config.fr_vae.encoder_pretrained_weights.dino_v2_base
        ).to(self.device)
        logging.info("FRVAE model initialized.")

        # 2. Discriminator Model Initialization
        # Discriminator takes an image (B, 3, H, W)
        discriminator = Discriminator(config=self.config).to(self.device)
        logging.info("Discriminator model initialized.")

        # 3. FRVAETrainer Initialization
        fr_vae_trainer = FRVAETrainer(
            fr_vae=fr_vae,
            discriminator=discriminator,
            config=self.config,
            data_loader=self.dataset_loader
        )
        logging.info("FRVAETrainer initialized.")

        # 4. Load Checkpoint if requested
        if load_checkpoint:
            fr_vae_trainer.load_checkpoint(self.fr_vae_checkpoint_path)
            # If loaded, the trainer's internal 'start_epoch' will be updated.

        # 5. Start Training
        fr_vae_trainer.train()
        logging.info(f"FR-VAE training complete. Final model saved (likely to {self.fr_vae_checkpoint_path} if it's the best).")

    def run_nfig_transformer_training(self, load_checkpoint: bool = False):
        """
        Executes the tokenization phase and the training phase for the NFIG Transformer.

        Args:
            load_checkpoint (bool): If True, attempts to load the latest checkpoint
                                    to resume training or ensure a trained model is available.
        """
        logging.info("Starting NFIG Transformer training phase.")

        # 1. Load Best FR-VAE Checkpoint for Tokenization
        # Instantiate FRVAE and load its state dictionary
        fr_vae = FRVAE(config=self.config).to(self.device)
        if not os.path.exists(self.fr_vae_checkpoint_path):
            logging.error(f"FR-VAE checkpoint not found at {self.fr_vae_checkpoint_path}. Please train FR-VAE first or provide a valid checkpoint.")
            raise FileNotFoundError(f"FR-VAE checkpoint required for tokenization: {self.fr_vae_checkpoint_path}")
        
        # Load the state_dict and set to eval mode for inference tasks
        fr_vae.load_state_dict(torch.load(self.fr_vae_checkpoint_path, map_location=self.device)['fr_vae_state_dict'])
        fr_vae.eval()
        logging.info(f"Loaded best FR-VAE model from {self.fr_vae_checkpoint_path} for tokenization.")

        # 2. Tokenization Phase
        # Call tokenize_dataset to convert images to token sequences and save them
        # The tokenize_dataset method will handle creating the token_train_dir
        self.dataset_loader.tokenize_dataset(
            fr_vae=fr_vae,
            device=self.device,
            tokenization_batch_size=self.config.fr_vae_training.batch_size # Use FR-VAE batch size for tokenization
        )
        logging.info(f"Dataset tokenization complete. Tokens saved to {self.token_train_dir}.")

        # 3. NFIGTransformer Model Initialization
        # Derive freq_band_token_lengths from fr_vae.freq_bands.band_dims
        freq_band_token_lengths = [h * w for h, w in self.config.fr_vae.freq_bands.band_dims]
        # Instantiate NFIGTransformer
        nfig_transformer = NFIGTransformer(
            config=self.config,
            vocab_size=self.config.nfig_transformer.vocab_size,  # Codebook size from FR-VAE
            num_classes=self.config.data.num_classes,  # Number of ImageNet classes
            total_sequence_length=self.config.nfig_transformer.total_sequence_length,  # Sum of all h_i*w_i
            freq_band_token_lengths=freq_band_token_lengths  # Derived list of token counts per band
        ).to(self.device)
        logging.info("NFIGTransformer model initialized.")

        # 4. Token Data Loader Initialization (for NFIGTransformer training)
        # Create a DataLoader for the tokenized dataset
        token_dataloader_for_nfig_training = self.dataset_loader.get_token_dataloader(
            token_dir=self.token_train_dir,
            batch_size=self.config.nfig_transformer_training.batch_size,
            sequence_length=self.config.nfig_transformer.total_sequence_length
        )
        logging.info("Token data loader for NFIG training initialized.")

        # 5. NFIGTrainer Initialization
        nfig_trainer = NFIGTrainer(
            nfig_transformer=nfig_transformer,
            config=self.config,
            token_data_loader=self.dataset_loader # Pass the full DatasetLoader object
        )
        # NFIGTrainer will internally use token_dataloader_for_nfig_training via dataset_loader.get_token_dataloader
        logging.info("NFIGTrainer initialized.")

        # 6. Load Checkpoint if requested
        if load_checkpoint:
            nfig_trainer.load_checkpoint(self.nfig_transformer_checkpoint_path)

        # 7. Start Training
        nfig_trainer.train()
        logging.info(f"NFIG Transformer training complete. Final model saved (likely to {self.nfig_transformer_checkpoint_path} if it's the best).")

    def run_evaluation(self):
        """
        Executes the final evaluation phase, calculating rFID, gFID, IS, Precision, and Recall.
        """
        logging.info("Starting evaluation phase.")

        # 1. Load Best FR-VAE Checkpoint
        fr_vae = FRVAE(config=self.config).to(self.device)
        if not os.path.exists(self.fr_vae_checkpoint_path):
            logging.error(f"FR-VAE checkpoint not found at {self.fr_vae_checkpoint_path}. Cannot perform evaluation.")
            raise FileNotFoundError(f"FR-VAE checkpoint required for evaluation: {self.fr_vae_checkpoint_path}")
        fr_vae.load_state_dict(torch.load(self.fr_vae_checkpoint_path, map_location=self.device)['fr_vae_state_dict'])
        fr_vae.eval()
        logging.info(f"Loaded best FR-VAE model from {self.fr_vae_checkpoint_path} for evaluation.")

        # 2. Load Best NFIG Transformer Checkpoint
        freq_band_token_lengths = [h * w for h, w in self.config.fr_vae.freq_bands.band_dims]
        nfig_transformer = NFIGTransformer(
            config=self.config,
            vocab_size=self.config.nfig_transformer.vocab_size,
            num_classes=self.config.data.num_classes,
            total_sequence_length=self.config.nfig_transformer.total_sequence_length,
            freq_band_token_lengths=freq_band_token_lengths
        ).to(self.device)
        if not os.path.exists(self.nfig_transformer_checkpoint_path):
            logging.error(f"NFIG Transformer checkpoint not found at {self.nfig_transformer_checkpoint_path}. Cannot perform evaluation.")
            raise FileNotFoundError(f"NFIG Transformer checkpoint required for evaluation: {self.nfig_transformer_checkpoint_path}")
        nfig_transformer.load_state_dict(torch.load(self.nfig_transformer_checkpoint_path, map_location=self.device)['nfig_transformer_state_dict'])
        nfig_transformer.eval()
        logging.info(f"Loaded best NFIG Transformer model from {self.nfig_transformer_checkpoint_path} for evaluation.")

        # 3. Evaluator Initialization
        evaluator = Evaluator(config=self.config)
        logging.info("Evaluator initialized.")

        # 4. Get validation dataloader for rFID and potentially real image features for other metrics
        # The evaluator's methods will internally call dataset_loader to get appropriate dataloaders
        logging.info("Preparing validation data for evaluation.")

        # 5. Compute rFID
        rfid_score = evaluator.compute_rfid(fr_vae, self.dataset_loader)
        logging.info(f"Reconstruction FID (rFID): {rfid_score:.4f}")

        # 6. Compute gFID, IS, Precision, Recall
        # The compute_gfids method in Evaluator will handle image generation using nfig_transformer and fr_vae
        # and then calculate the metrics.
        metrics = evaluator.compute_gfids(nfig_transformer, fr_vae, self.dataset_loader)
        logging.info(f"Generation FID (gFID): {metrics['gFID']:.4f}")
        logging.info(f"Inception Score (IS): {metrics['IS']:.4f}")
        logging.info(f"Precision: {metrics['Precision']:.4f}")
        logging.info(f"Recall: {metrics['Recall']:.4f}")

        logging.info("Evaluation phase complete.")

    def run_full_pipeline(self):
        """
        Executes the entire NFIG reproduction pipeline sequentially.
        """
        logging.info("Starting full NFIG reproduction pipeline.")
        self.run_fr_vae_training()
        self.run_nfig_transformer_training()
        self.run_evaluation()
        logging.info("Full NFIG reproduction pipeline finished.")

    def to_omegaconf(self):
        """
        Helper method to access the underlying OmegaConf object of the config.
        """
        return self.config.to_omegaconf()


if __name__ == '__main__':
    # You can add command-line argument parsing here if you want to selectively run phases
    # For example: python main.py --phase fr_vae_train
    # For now, it runs the full pipeline by default.

    main_runner = Main()
    main_runner.run_full_pipeline()
