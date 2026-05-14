import os
from typing import Any

from omegaconf import OmegaConf, DictConfig


class Config:
    """
    Manages all configuration parameters for the NFIG project.
    Utilizes OmegaConf for structured configuration from YAML files.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the Config object by loading the base configuration from a YAML file.

        Args:
            config_path: The path to the primary YAML configuration file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        self._cfg: DictConfig = OmegaConf.load(config_path)
        self._resolve_paths()

    def _resolve_paths(self) -> None:
        """
        Resolves relative paths in the configuration to absolute paths
        and ensures necessary directories exist.
        """
        if "data" in self._cfg and "dataset_root" in self._cfg.data:
            self._cfg.data.dataset_root = os.path.abspath(self._cfg.data.dataset_root)

        if "fr_vae" in self._cfg and "encoder_pretrained_weights" in self._cfg.fr_vae:
            dino_path = self._cfg.fr_vae.encoder_pretrained_weights.get("dino_v2_base")
            if dino_path:
                self._cfg.fr_vae.encoder_pretrained_weights.dino_v2_base = os.path.abspath(dino_path)

        if "inference" in self._cfg and "output_dir" in self._cfg.inference:
            self._cfg.inference.output_dir = os.path.abspath(self._cfg.inference.output_dir)
            os.makedirs(self._cfg.inference.output_dir, exist_ok=True)

    def load(self, path: str) -> None:
        """
        Loads an additional configuration from a YAML file and merges it
        with the current configuration, overriding existing values.

        Args:
            path: The path to the YAML configuration file to load.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")

        loaded_cfg = OmegaConf.load(path)
        self._cfg = OmegaConf.merge(self._cfg, loaded_cfg)
        self._resolve_paths()  # Re-resolve paths after merging

    def save(self, path: str) -> None:
        """
        Saves the current configuration to a YAML file.

        Args:
            path: The path where the configuration should be saved.
        """
        OmegaConf.save(self._cfg, path)

    @property
    def data(self) -> DictConfig:
        """
        Provides access to data-related configurations.
        """
        return self._cfg.data

    @property
    def fr_vae(self) -> DictConfig:
        """
        Provides access to FR-VAE model architecture configurations.
        """
        return self._cfg.fr_vae

    @property
    def fr_vae_training(self) -> DictConfig:
        """
        Provides access to FR-VAE training configurations.
        """
        return self._cfg.fr_vae_training

    @property
    def nfig_transformer(self) -> DictConfig:
        """
        Provides access to NFIG Transformer model architecture configurations.
        """
        return self._cfg.nfig_transformer

    @property
    def nfig_transformer_training(self) -> DictConfig:
        """
        Provides access to NFIG Transformer training configurations.
        """
        return self._cfg.nfig_transformer_training

    @property
    def inference(self) -> DictConfig:
        """
        Provides access to inference-related configurations.
        """
        return self._cfg.inference

    @property
    def evaluation(self) -> DictConfig:
        """
        Provides access to evaluation-related configurations.
        """
        return self._cfg.evaluation

    def __repr__(self) -> str:
        """
        Returns a string representation of the configuration.
        """
        return self._cfg.pretty()

    def __str__(self) -> str:
        """
        Returns a string representation of the configuration.
        """
        return self._cfg.pretty()

    def __getitem__(self, key: str) -> Any:
        """
        Allows dictionary-style access to top-level configuration items.
        """
        return self._cfg[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allows dictionary-style setting of top-level configuration items.
        """
        self._cfg[key] = value


# Example usage (for testing purposes, not part of the main program flow)
if __name__ == "__main__":
    # Create a dummy config.yaml for testing
    dummy_config_content = """
data:
  dataset_name: "TestDataset"
  image_size: 128
  num_classes: 10
  dataset_root: "./data/test_dataset"

fr_vae:
  latent_dim_channels: 128
  encoder_latent_size: 8
  codebook_size: 2048
  freq_bands:
    num_bands: 5
    scaling_factors: [1, 2, 4, 8, 16]
    total_quantized_tokens: 100
  encoder_pretrained_weights:
    dino_v2_base: "./pretrained_models/dino.pth"

fr_vae_training:
  batch_size: 32
  learning_rate: 1.0e-4
  epochs: 50
  optimizer: "AdamW"
  loss_weights:
    recon_image_L2: 1.0
    recon_feature_L2: 1.0
    perceptual_loss: 0.8
    gan_loss_generator: 0.3
    codebook_loss_beta: 0.1

nfig_transformer:
  depth: 8
  embed_dim: 512
  num_heads: 8
  ffn_dim: 2048
  vocab_size: 2048
  total_sequence_length: 100
  unconditional_training_probability: 0.15

nfig_transformer_training:
  batch_size: 64
  learning_rate: 5.0e-5
  epochs: 100
  optimizer: "Adam"
  loss_type: "CrossEntropy"

inference:
  cfg_weight: 3.0
  top_k: 500
  num_samples: 1000
  output_dir: "./results/generated_images"

evaluation:
  fid_model: "inception_v3"
"""
    with open("test_config.yaml", "w") as f:
        f.write(dummy_config_content)

    print("--- Loading config from test_config.yaml ---")
    cfg = Config("test_config.yaml")

    print("\n--- Accessing configurations ---")
    print(f"Dataset root: {cfg.data.dataset_root}")
    print(f"Image size: {cfg.data.image_size}")
    print(f"FR-VAE codebook size: {cfg.fr_vae.codebook_size}")
    print(f"DINOv2 base path: {cfg.fr_vae.encoder_pretrained_weights.dino_v2_base}")
    print(f"FR-VAE LR: {cfg.fr_vae_training.learning_rate}")
    print(f"NFIG Transformer depth: {cfg.nfig_transformer.depth}")
    print(f"Inference output directory: {cfg.inference.output_dir}")
    print(f"Evaluation FID model: {cfg.evaluation.fid_model}")

    # Test merging
    additional_config_content = """
fr_vae_training:
  epochs: 75
nfig_transformer:
  depth: 10
inference:
  num_samples: 5000
"""
    with open("override_config.yaml", "w") as f:
        f.write(additional_config_content)

    print("\n--- Merging with override_config.yaml ---")
    cfg.load("override_config.yaml")
    print(f"FR-VAE training epochs after merge: {cfg.fr_vae_training.epochs}")
    print(f"NFIG Transformer depth after merge: {cfg.nfig_transformer.depth}")
    print(f"Inference num_samples after merge: {cfg.inference.num_samples}")

    # Test saving
    save_path = "saved_config.yaml"
    cfg.save(save_path)
    print(f"\n--- Current config saved to {save_path} ---")
    print("Content of saved_config.yaml:")
    with open(save_path, "r") as f:
        print(f.read())

    # Clean up dummy files
    os.remove("test_config.yaml")
    os.remove("override_config.yaml")
    os.remove("saved_config.yaml")
    os.rmdir(os.path.abspath("./results/generated_images"))
    os.rmdir(os.path.abspath("./data/test_dataset"))
    if os.path.exists(os.path.abspath("./pretrained_models")):
        os.rmdir(os.path.abspath("./pretrained_models"))

