import os
import yaml
import torch
from typing import List, Tuple, Dict, Any

class Config:
    """
    Centralized configuration manager for hyperparameters and settings.

    Attributes are named in uppercase as per Google-style guidelines for constants,
    but are treated as configurable parameters loaded from a YAML file.
    """

    # Static dictionary for dataset-specific overrides.
    # These values will override the defaults loaded from config.yaml if a dataset matches.
    # Learning rate and dropout for CIFAR-10, CelebA, LSUN Church, ImageNet are set
    # based on paper's findings (Appendix D, Tables 4, 5, 6) or reasonable defaults.
    _DATASET_OVERRIDES: Dict[str, Dict[str, Any]] = {
        "cifar10": {
            "RESOLUTION": 32,
            "BATCH_SIZE": 512,
            "TRAINING_STEPS": 100000,
            "LEARNING_RATE": 0.0001,  # Representative from search space for iCT-GC
            "DROPOUT_RATE": 0.0,       # Ablation studies suggest 0.0 for iCT-GC is better
            "NUM_BLOCKS": [3],         # Interpreted as [3,3,3] for channel_mult=[1,2,2]
            "CHANNEL_MULT": [1, 2, 2],
            "ATTN_RESOLUTIONS": [],
        },
        "imagenet32": { # Matches "ImageNet (32x32)"
            "RESOLUTION": 32,
            "BATCH_SIZE": 512,
            "TRAINING_STEPS": 150000,
            "LEARNING_RATE": 0.00008,
            "DROPOUT_RATE": 0.0,       # Assuming 0.0 for iCT-GC
            "NUM_BLOCKS": [3, 5, 7],
            "CHANNEL_MULT": [1, 1, 2],
            "ATTN_RESOLUTIONS": [16],
        },
        "celeba64": { # Matches "CelebA (64x64)"
            "RESOLUTION": 64,
            "BATCH_SIZE": 128,
            "TRAINING_STEPS": 150000,
            "LEARNING_RATE": 0.00008,
            "DROPOUT_RATE": 0.0,       # Assuming 0.0 for iCT-GC
            "NUM_BLOCKS": [3, 3, 4, 5],
            "CHANNEL_MULT": [1, 2, 2, 2],
            "ATTN_RESOLUTIONS": [],
        },
        "lsunchurch64": { # Matches "LSUN Church (64x64)"
            "RESOLUTION": 64,
            "BATCH_SIZE": 128,
            "TRAINING_STEPS": 150000,
            "LEARNING_RATE": 0.00008,
            "DROPOUT_RATE": 0.0,       # Assuming 0.0 for iCT-GC
            "NUM_BLOCKS": [3, 3, 4, 5],
            "CHANNEL_MULT": [1, 2, 2, 2],
            "ATTN_RESOLUTIONS": [],
        },
        "ffhq64": { # Mentioned in ECT setting in paper
            "RESOLUTION": 64,
            "BATCH_SIZE": 128, # Assuming similar to CelebA/LSUN for 64x64
            "TRAINING_STEPS": 100000, # Long training from paper (ECT-IC 4.11 @ 100k)
            "LEARNING_RATE": 0.00008, # Assuming same as CelebA/LSUN
            "DROPOUT_RATE": 0.0,       # Assuming 0.0 for iCT-GC
            "NUM_BLOCKS": [3, 3, 4, 5],
            "CHANNEL_MULT": [1, 2, 2, 2],
            "ATTN_RESOLUTIONS": [],
            "MU": 0.3, # Paper explicitly mentions 0.3 for ECT
        }
    }

    def __init__(self, config_path: str):
        """
        Initializes the Config object by loading from a YAML file,
        applying dataset-specific overrides, and post-processing attributes.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self._config_data = {}  # Store raw config data for potential saving
        self._load_from_yaml(config_path)
        self._apply_dataset_specific_overrides()
        self._post_process_attributes()
        self._validate_parameters()

    def _load_from_yaml(self, config_path: str):
        """
        Loads configuration parameters from the specified YAML file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, 'r') as f:
            self._config_data = yaml.safe_load(f)

        # Training configuration
        self.LEARNING_RATE: float = self._config_data['training']['learning_rate']
        self.BATCH_SIZE: int = self._config_data['training']['batch_size']
        self.TRAINING_STEPS: int = self._config_data['training']['training_steps']
        self.OPTIMIZER: str = self._config_data['training']['optimizer']
        self.EMA_DECAY: float = self._config_data['training']['ema_decay']
        self.SEED: int = self._config_data['training']['seed']
        self.LOG_INTERVAL_STEPS: int = self._config_data['training']['log_interval_steps']
        self.EVAL_INTERVAL_STEPS: int = self._config_data['training']['eval_interval_steps']
        self.SAVE_INTERVAL_STEPS: int = self._config_data['training']['save_interval_steps']
        self.CHECKPOINT_DIR: str = self._config_data['training']['checkpoint_dir']

        # Dataset configuration
        self.DATASET_NAME: str = self._config_data['dataset']['name']
        self.RESOLUTION: int = self._config_data['dataset']['resolution']
        self.NUM_WORKERS: int = self._config_data['dataset']['num_workers']
        self.PIN_MEMORY: bool = self._config_data['dataset']['pin_memory']

        # Model architecture configuration
        self.IMG_CHANNELS: int = self._config_data['model']['img_channels']
        self.MODEL_CHANNELS: int = self._config_data['model']['model_channels']
        self.DROPOUT_RATE: float = self._config_data['model']['dropout_rate']
        self.EMBEDDING_TYPE: str = self._config_data['model']['embedding_type']
        # These are lists and will be overridden by dataset-specific values for SongUNet
        self.NUM_BLOCKS: List[int] = self._config_data['model']['num_blocks']
        self.CHANNEL_MULT: List[int] = self._config_data['model']['channel_mult']
        self.ATTN_RESOLUTIONS: List[int] = self._config_data['model']['attn_resolutions']

        # Noise schedule configuration
        self.RHO: float = self._config_data['noise_schedule']['rho']
        self.SIGMA_0: float = self._config_data['noise_schedule']['sigma_0']
        self.SIGMA_T: float = self._config_data['noise_schedule']['sigma_T']

        # Progressive timestep scheduling
        self.S0: int = self._config_data['timestep_scheduling']['s0']
        self.S1: int = self._config_data['timestep_scheduling']['s1']

        # Timestep sampling probability configuration
        self.P_MEAN: float = self._config_data['timestep_sampling']['p_mean']
        self.P_STD: float = self._config_data['timestep_sampling']['p_std']

        # Generator-Augmented Flow (GC) specific parameters
        self.MU: float = self._config_data['gc']['mu']

        # Evaluation metrics configuration
        self.FID_NUM_SAMPLES: int = self._config_data['evaluation']['fid_num_samples']
        self.EVAL_RUNS: int = self._config_data['evaluation']['eval_runs']
        self.REAL_SAMPLES_PATH: str = self._config_data['evaluation']['real_samples_path']

        # Hardware configuration
        self.DEVICE_STR: str = self._config_data['hardware']['device'] # Store as string initially
        self.DISTRIBUTED: bool = self._config_data['hardware']['distributed']
        self.NUM_GPUS: int = self._config_data['hardware']['num_gpus']

        # Placeholder for empirically calculated data variance (sigma_d^2)
        # This value is computed once from the dataset and set externally.
        self.SIGMA_D_SQ: float = 0.0 

    def _apply_dataset_specific_overrides(self):
        """
        Applies overrides based on the specified dataset and resolution.
        Normalizes dataset names to match keys in _DATASET_OVERRIDES.
        """
        # Normalize dataset name to match keys in _DATASET_OVERRIDES
        dataset_key_base = self.DATASET_NAME.lower().replace(" ", "").replace("-", "")
        dataset_key = dataset_key_base
        
        # Append resolution for specific datasets
        if dataset_key_base in ["imagenet", "celeba", "lsunchurch", "ffhq"]:
            dataset_key = f"{dataset_key_base}{self.RESOLUTION}"

        if dataset_key in self._DATASET_OVERRIDES:
            overrides = self._DATASET_OVERRIDES[dataset_key]
            for key, value in overrides.items():
                setattr(self, key, value)
            print(f"Applied dataset-specific overrides for {self.DATASET_NAME} (resolution {self.RESOLUTION}).")
        else:
            print(f"No specific overrides found for dataset '{self.DATASET_NAME}' (resolution {self.RESOLUTION}). Using default/config.yaml values.")

        # Special handling for NUM_BLOCKS if it's a single integer for compatibility
        # with SongUNet's expected `num_blocks` structure when `channel_mult` is longer.
        # This was noted for CIFAR-10 in the paper's table interpretation.
        if len(self.NUM_BLOCKS) == 1 and len(self.CHANNEL_MULT) > 1:
            self.NUM_BLOCKS = [self.NUM_BLOCKS[0]] * len(self.CHANNEL_MULT)
            print(f"Adjusted NUM_BLOCKS to {self.NUM_BLOCKS} to match CHANNEL_MULT length.")

    def _post_process_attributes(self):
        """
        Derives and sets additional attributes, and creates necessary directories.
        """
        # Determine the torch.device
        if self.DEVICE_STR == "cuda" and torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
            if self.DISTRIBUTED and self.NUM_GPUS > 1:
                # For distributed training, accelerate handles specific GPU assignment.
                # Here, we just ensure it's set to 'cuda'.
                pass
        else:
            self.DEVICE = torch.device("cpu")
            self.DISTRIBUTED = False # Distributed training is not applicable on CPU
            self.NUM_GPUS = 1         # Only one 'device' (CPU)
        print(f"Using device: {self.DEVICE}")

        # Calculate data dimension (e.g., 32*32*3 for a 32x32 RGB image)
        self.DATA_DIM: int = self.RESOLUTION * self.RESOLUTION * self.IMG_CHANNELS

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        print(f"Checkpoint directory: {self.CHECKPOINT_DIR}")

    def _validate_parameters(self):
        """
        Performs basic sanity checks on loaded and processed parameters.
        Raises AssertionError for invalid configurations.
        """
        assert 0.0 <= self.EMA_DECAY <= 1.0, f"EMA_DECAY must be between 0 and 1, got {self.EMA_DECAY}"
        assert 0.0 <= self.MU <= 1.0, f"MU must be between 0 and 1, got {self.MU}"
        assert self.RESOLUTION > 0, f"RESOLUTION must be positive, got {self.RESOLUTION}"
        assert self.IMG_CHANNELS > 0, f"IMG_CHANNELS must be positive, got {self.IMG_CHANNELS}"
        assert self.SIGMA_0 < self.SIGMA_T, f"SIGMA_0 ({self.SIGMA_0}) must be less than SIGMA_T ({self.SIGMA_T})"
        assert self.BATCH_SIZE > 0, f"BATCH_SIZE must be positive, got {self.BATCH_SIZE}"
        assert self.TRAINING_STEPS > 0, f"TRAINING_STEPS must be positive, got {self.TRAINING_STEPS}"
        assert self.FID_NUM_SAMPLES > 0, f"FID_NUM_SAMPLES must be positive, got {self.FID_NUM_SAMPLES}"
        assert self.EVAL_RUNS > 0, f"EVAL_RUNS must be positive, got {self.EVAL_RUNS}"
        assert self.LOG_INTERVAL_STEPS > 0, f"LOG_INTERVAL_STEPS must be positive, got {self.LOG_INTERVAL_STEPS}"
        assert self.EVAL_INTERVAL_STEPS > 0, f"EVAL_INTERVAL_STEPS must be positive, got {self.EVAL_INTERVAL_STEPS}"
        assert self.SAVE_INTERVAL_STEPS > 0, f"SAVE_INTERVAL_STEPS must be positive, got {self.SAVE_INTERVAL_STEPS}"
        assert isinstance(self.NUM_BLOCKS, list), f"NUM_BLOCKS must be a list, got {type(self.NUM_BLOCKS)}"
        assert isinstance(self.CHANNEL_MULT, list), f"CHANNEL_MULT must be a list, got {type(self.CHANNEL_MULT)}"
        assert isinstance(self.ATTN_RESOLUTIONS, list), f"ATTN_RESOLUTIONS must be a list, got {type(self.ATTN_RESOLUTIONS)}"
        assert len(self.NUM_BLOCKS) == len(self.CHANNEL_MULT), \
            f"Length of NUM_BLOCKS ({len(self.NUM_BLOCKS)}) must match CHANNEL_MULT ({len(self.CHANNEL_MULT)}) for SongUNet construction."
        assert self.EMBEDDING_TYPE in ["positional", "fourier"], \
            f"EMBEDDING_TYPE must be 'positional' or 'fourier', got {self.EMBEDDING_TYPE}"
        assert self.OPTIMIZER in ["adam", "lion"], \
            f"OPTIMIZER must be 'adam' or 'lion', got {self.OPTIMIZER}"


    def save_config(self, save_path: str):
        """
        Saves the current configuration to a YAML file.
        This method reconstructs the dictionary from the current attributes to ensure
        that any dataset-specific overrides are included in the saved file.

        Args:
            save_path: The file path to save the configuration to.
        """
        config_to_save = {
            'training': {
                'learning_rate': self.LEARNING_RATE,
                'batch_size': self.BATCH_SIZE,
                'training_steps': self.TRAINING_STEPS,
                'optimizer': self.OPTIMIZER,
                'ema_decay': self.EMA_DECAY,
                'seed': self.SEED,
                'log_interval_steps': self.LOG_INTERVAL_STEPS,
                'eval_interval_steps': self.EVAL_INTERVAL_STEPS,
                'save_interval_steps': self.SAVE_INTERVAL_STEPS,
                'checkpoint_dir': self.CHECKPOINT_DIR,
            },
            'dataset': {
                'name': self.DATASET_NAME,
                'resolution': self.RESOLUTION,
                'num_workers': self.NUM_WORKERS,
                'pin_memory': self.PIN_MEMORY,
            },
            'model': {
                'img_channels': self.IMG_CHANNELS,
                'model_channels': self.MODEL_CHANNELS,
                'dropout_rate': self.DROPOUT_RATE,
                'embedding_type': self.EMBEDDING_TYPE,
                'num_blocks': self.NUM_BLOCKS,
                'channel_mult': self.CHANNEL_MULT,
                'attn_resolutions': self.ATTN_RESOLUTIONS,
            },
            'noise_schedule': {
                'rho': self.RHO,
                'sigma_0': self.SIGMA_0,
                'sigma_T': self.SIGMA_T,
            },
            'timestep_scheduling': {
                's0': self.S0,
                's1': self.S1,
            },
            'timestep_sampling': {
                'p_mean': self.P_MEAN,
                'p_std': self.P_STD,
            },
            'gc': {
                'mu': self.MU,
            },
            'evaluation': {
                'fid_num_samples': self.FID_NUM_SAMPLES,
                'eval_runs': self.EVAL_RUNS,
                'real_samples_path': self.REAL_SAMPLES_PATH,
            },
            'hardware': {
                'device': self.DEVICE_STR, # Save the string representation
                'distributed': self.DISTRIBUTED,
                'num_gpus': self.NUM_GPUS,
            }
        }
        with open(save_path, 'w') as f:
            yaml.dump(config_to_save, f, indent=2, sort_keys=False) # sort_keys=False to preserve order
        print(f"Configuration saved to {save_path}")

