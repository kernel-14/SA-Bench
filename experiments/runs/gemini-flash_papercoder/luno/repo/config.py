"""
config.py
Configuration class for managing all experiment parameters.
"""

from typing import Any, Dict


class Config:
    """
    Configuration class to hold all experiment parameters, loaded from a dictionary.

    This class's attributes strictly follow the 'Data structures and interfaces' design.
    The `__init__` method assumes the input `params` dictionary has been pre-processed
    by `main.py` (or ExperimentRunner) to flatten relevant configuration sections
    and rename keys to match the class diagram's attribute names.
    For example, if `config.yaml` has `fno_model.modes`, the `params` dictionary
    passed to this `Config` class should have a top-level key `fno_modes`.
    """

    pde_name: str
    spatial_res: int
    temporal_res: int
    num_train_traj: int
    num_val_traj: int
    num_test_traj: int
    fno_modes: int
    fno_hidden_dims: int
    fno_blocks: int
    fno_epochs: int
    lr_schedule: Dict[str, Any]
    uq_methods_config: Dict[str, Any]
    luno_rank: int
    sampling_samples: int
    ensemble_members: int
    calibration_grid_size: int
    seed: int

    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the Config object from a dictionary of parameters.

        Args:
            params: A dictionary containing all configuration parameters.
                    It's assumed that this dictionary has been pre-processed
                    to flatten specific sections and rename keys to match
                    the Config class attributes as defined in the design.
        """
        # Global experiment configuration
        # Not explicitly in diagram, but useful for runtime and logging.
        self.experiment_name: str = params.get("experiment_name", "luno_reproduction")
        self.seed: int = params.get("seed", 42)
        self.log_dir: str = params.get("log_dir", "./logs")
        self.results_dir: str = params.get("results_dir", "./results")

        # Dataset parameters (flattened for the active dataset)
        self.pde_name = params.get("pde_name", "Unknown PDE")
        self.spatial_res = params.get("spatial_res", 256)
        self.temporal_res = params.get("temporal_res", 59)
        self.num_train_traj = params.get("num_train_traj", 0)
        self.num_val_traj = params.get("num_val_traj", 0)
        # Note: 'num_test_traj' as per diagram. Main.py should map 'num_test_pairs'
        # from OOD config to this if 2D OOD experiment is active.
        self.num_test_traj = params.get("num_test_traj", 0)

        # FNO Model Architecture parameters (flattened)
        self.fno_modes = params.get("fno_modes", 12)
        self.fno_hidden_dims = params.get("fno_hidden_dims", 18)
        self.fno_blocks = params.get("fno_blocks", 4)
        # 'output_channels' is not explicitly in diagram but is a model parameter.
        self.fno_output_channels: int = params.get("fno_output_channels", 1)

        # Training configuration (partially flattened, lr_schedule kept as dict)
        # 'optimizer', 'batch_size', 'initial_time_steps', 'input_padding' are training configs.
        # Not explicitly in diagram Config class attributes, but belong to 'training.py'.
        self.training_optimizer: str = params.get("optimizer", "AdamW")
        self.training_batch_size: int = params.get("batch_size", 1)
        self.initial_time_steps: int = params.get("initial_time_steps", 10)
        self.input_padding: int = params.get("input_padding", 2)

        # 'fno_epochs' as per diagram. Main.py should map 'epochs_low_data'
        # or 'epochs_ood' based on the active experiment.
        self.fno_epochs = params.get("fno_epochs", 100)
        # 'lr_schedule' is explicitly a dict in the diagram
        self.lr_schedule = params.get("lr_schedule", {
            "name": "cosine_decay_with_warmup",
            "init_value": 1.0e-4,
            "peak_value": 1.0e-3,
            "warmup_steps": 1000,
            "decay_steps": 100000,
        })

        # Uncertainty Quantification Methods configuration (kept as nested dict)
        self.uq_methods_config = params.get("uq_methods_config", {})
        # Specific UQ parameters, flattened as per diagram, using defaults if not provided.
        # These are expected to be pulled from the 'uq_methods_config' in main.py
        # and passed as top-level keys in 'params'.
        self.luno_rank = params.get("luno_rank", 500)
        self.sampling_samples = params.get("sampling_samples", 200)
        self.ensemble_members = params.get("ensemble_members", 10)

        # Calibration configuration (flattened)
        # 'calibration_metric' is not explicitly in diagram but is useful.
        self.calibration_metric: str = params.get("calibration_metric", "NLL")
        self.calibration_grid_size = params.get("calibration_grid_size", 500)

        # Evaluation configuration (flattened)
        self.eval_autoregressive_rollout_steps: int = params.get("autoregressive_rollout_steps", 59)
        self.eval_num_rollout_trajectories: int = params.get("num_rollout_trajectories", 50)

        # Additional specific dataset configurations (for OOD datasets. Not in diagram's Config,
        # but needed for AdvectionDiffusionDatasetLoader). These are assumed to be passed
        # by main.py if relevant for the current experiment setup.
        self.ood_dt_solver: float = params.get("dt_solver", 5.0e-10)
        self.ood_diffusion_coefficient: float = params.get("diffusion_coefficient", 0.026)
        self.ood_variants: list[str] = params.get("ood_variants", ["Base"])

    def __repr__(self) -> str:
        """
        Provides a string representation of the Config object for debugging.
        """
        return f"Config({self.__dict__})"
