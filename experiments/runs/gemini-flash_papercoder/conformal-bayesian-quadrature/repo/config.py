from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


@dataclass
class Config:
    """
    Configuration dataclass for all experiment parameters and hyperparameters.

    This class centralizes settings from config.yaml, ensuring consistency
    across data generation, method execution, and evaluation.
    """

    # Global experiment parameters (from 'experiment' section in config.yaml)
    random_seed: int = 42
    num_trials: int = 10000
    num_dirichlet_samples: int = 1000
    num_dirichlet_samples_plot: int = 100000
    alpha: float = 0.4
    beta: float = 0.95
    B: float = 1.0

    # Common parameters for methods (from 'method_params' section in config.yaml)
    lambda_search_range: Tuple[float, float] = (0.0, 1.0)

    # Experiment-specific configurations (nested dictionaries mirroring config.yaml structure)
    synthetic_binomial_config: Dict[str, Any] = field(default_factory=lambda: {
        "K": 4,
        "alpha": 0.4,
        "num_calibration_samples": 10,
        "num_test_samples": 0,
    })
    synthetic_heteroskedastic_config: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": 0.1,
        "beta": 0.95,
        "num_calibration_samples": 200,
        "num_test_samples": 1000,
    })
    ms_coco_config: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": None,  # Needs to be clarified from referenced paper
        "beta": 0.95,
        "num_calibration_samples": 1000,
        "num_test_samples": 3952,
        "model_name": None,  # Needs to be clarified from referenced paper
        "model_weights_path": None,  # Needs to be clarified from referenced paper
        "dataset_path": './data/mscoco',
    })

    @classmethod
    def from_yaml(cls, yaml_data: Dict[str, Any]) -> "Config":
        """
        Creates a Config instance from a dictionary parsed from a YAML file.
        """
        # Extract global experiment parameters
        experiment_data = yaml_data.get("experiment", {})
        config_instance = cls(
            random_seed=experiment_data.get("random_seed", cls.random_seed),
            num_trials=experiment_data.get("num_trials", cls.num_trials),
            num_dirichlet_samples=experiment_data.get("num_dirichlet_samples", cls.num_dirichlet_samples),
            num_dirichlet_samples_plot=experiment_data.get("num_dirichlet_samples_plot", cls.num_dirichlet_samples_plot),
            alpha=experiment_data.get("alpha", cls.alpha),
            beta=experiment_data.get("beta", cls.beta),
            B=experiment_data.get("B", cls.B),
        )

        # Extract method-specific parameters
        method_params_data = yaml_data.get("method_params", {})
        config_instance.lambda_search_range = tuple(method_params_data.get("lambda_search_range", config_instance.lambda_search_range))

        # Extract experiment-specific configurations
        config_instance.synthetic_binomial_config.update(yaml_data.get("synthetic_binomial", {}))
        config_instance.synthetic_heteroskedastic_config.update(yaml_data.get("synthetic_heteroskedastic", {}))
        config_instance.ms_coco_config.update(yaml_data.get("ms_coco", {}))

        return config_instance

