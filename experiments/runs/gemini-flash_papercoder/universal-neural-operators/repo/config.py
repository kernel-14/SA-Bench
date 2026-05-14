import yaml
import os
from typing import Dict, Any, Optional

class Config:
    """
    Manages the loading and access of experiment configuration parameters from a YAML file.

    Attributes:
        _raw_config (Optional[Dict[str, Any]]): Stores the entire raw configuration data
                                                 loaded from the YAML file.
        model_type (Optional[str]): Type of the core operator model being used (e.g., "FNO", "MambaFNO").
                                    Set dynamically by main.py.
        scenario_type (Optional[str]): Current experimental scenario (e.g., "out_of_sample_params").
                                       Set dynamically by main.py.
        pde_configs (Dict[str, Any]): Configuration parameters for generating data for each PDE.
        pretrain_settings (Dict[str, Any]): Training parameters specific to the pre-training phase.
        finetune_settings (Dict[str, Any]): Training parameters specific to the fine-tuning phase.
        training_settings (Dict[str, Any]): General training-related parameters.
        evaluation_settings (Dict[str, Any]): Parameters related to evaluation.
        data_settings (Dict[str, Any]): General data-related parameters including base directories
                                        and data split ratios.
    """

    def __init__(self):
        """
        Initializes the Config object with default empty values.
        """
        self._raw_config: Optional[Dict[str, Any]] = None
        self.model_type: Optional[str] = None
        self.scenario_type: Optional[str] = None
        self.pde_configs: Dict[str, Any] = {}
        self.pretrain_settings: Dict[str, Any] = {}
        self.finetune_settings: Dict[str, Any] = {}
        self.training_settings: Dict[str, Any] = {}
        self.evaluation_settings: Dict[str, Any] = {}
        self.data_settings: Dict[str, Any] = {}
        self.model_settings: Dict[str, Any] = {} # Added to capture the 'model' top-level key from config.yaml

    @staticmethod
    def load_config(path: str) -> 'Config':
        """
        Loads configuration from a YAML file and populates a Config object.

        Args:
            path (str): The file path to the config.yaml file.

        Returns:
            Config: An instance of the Config class populated with parameters from the YAML file.

        Raises:
            FileNotFoundError: If the specified config file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
            KeyError: If a required top-level section is missing in the YAML file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            try:
                raw_config_data: Dict[str, Any] = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML configuration file: {e}")

        cfg = Config()
        cfg._raw_config = raw_config_data

        try:
            # Populate data-related settings
            cfg.data_settings = raw_config_data.get('data', {})
            cfg.pde_configs = cfg.data_settings.get('pde_configs', {})

            # Populate model architecture settings
            cfg.model_settings = raw_config_data.get('model', {})

            # Populate training settings
            cfg.training_settings = raw_config_data.get('training', {})
            # As per design, pretrain_settings and finetune_settings refer to the general
            # training_settings dictionary, and specific values like epochs_pretrain/finetune
            # will be accessed from within training_settings.
            cfg.pretrain_settings = cfg.training_settings
            cfg.finetune_settings = cfg.training_settings

            # Populate evaluation settings
            cfg.evaluation_settings = raw_config_data.get('evaluation', {})

            # Set global experiment settings if available
            experiment_settings = raw_config_data.get('experiment', {})
            # Dynamically setting these attributes based on the config.yaml structure
            # If not explicitly present in config.yaml, will remain None or default
            cfg.experiment_name = experiment_settings.get('name', "universal_neural_operators")
            cfg.seed = experiment_settings.get('seed', 42) # Default value as per config.yaml
            cfg.device = experiment_settings.get('device', 'cuda') # Default value as per config.yaml

        except KeyError as e:
            raise KeyError(f"Missing expected key in config file: {e}. Please check config.yaml structure.")

        return cfg

# Example usage (for local testing, not part of the main project structure)
if __name__ == "__main__":
    # Create a dummy config.yaml for testing
    dummy_config_content = """
    # Global experiment settings
    experiment:
      name: "test_experiment"
      seed: 123
      device: "cpu"

    # Data generation and dataset management settings
    data:
      base_dir: "test_data/"
      spatial_resolution: 32
      temporal_resolution: 10
      train_ratio: 0.7
      pde_configs:
        test_pde1:
          equation_type: "Test1"
          param_a: 1.0
        test_pde2:
          equation_type: "Test2"
          param_b: 2.0

    # Model Architecture Configurations
    model:
      input_dim: 10
      output_dim: 1
      hidden_dim: 64
      fno_config:
        num_fourier_modes: [8, 8]

    # Training settings
    training:
      optimizer: "Adam"
      learning_rate: 0.001
      epochs_pretrain: 100
      epochs_finetune: 50
      lr_scheduler:
        type: "CosineAnnealingLR"

    # Evaluation settings
    evaluation:
      metrics: ["mse", "nmae"]
      output_dir: "test_results/"
    """
    config_file_path = "temp_test_config.yaml"
    with open(config_file_path, "w", encoding='utf-8') as f:
        f.write(dummy_config_content)

    try:
        config = Config.load_config(config_file_path)
        print("Config loaded successfully!")
        print(f"Experiment Name: {config.experiment_name}")
        print(f"Seed: {config.seed}")
        print(f"Device: {config.device}")
        print(f"Data Base Dir: {config.data_settings.get('base_dir')}")
        print(f"PDE Configs: {config.pde_configs['test_pde1']}")
        print(f"Training Optimizer: {config.training_settings.get('optimizer')}")
        print(f"Pretrain Epochs: {config.pretrain_settings.get('epochs_pretrain')}")
        print(f"Finetune Epochs: {config.finetune_settings.get('epochs_finetune')}")
        print(f"Model Hidden Dim: {config.model_settings.get('hidden_dim')}")
        print(f"Evaluation Metrics: {config.evaluation_settings.get('metrics')}")
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        print(f"Error loading config: {e}")
    finally:
        # Clean up the dummy config file
        if os.path.exists(config_file_path):
            os.remove(config_file_path)
