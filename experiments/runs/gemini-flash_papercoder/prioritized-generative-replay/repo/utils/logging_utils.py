import os
import datetime
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Union, Optional

# Attempt conditional import for wandb
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    print("Warning: Weights & Biases (wandb) is not installed. WandB logging will be disabled.")

# Assuming Config class is available from config.py in the same directory or via sys.path
from config import Config


class Logger:
    """
    A Logger class for centralized experiment logging to TensorBoard and optionally Weights & Biases.
    Provides methods for logging scalars, images, histograms, saving configurations, and checkpoints.
    """

    def __init__(self, config: Config):
        """
        Initializes the logging system, setting up log directories and integrating
        with TensorBoard and optionally Weights & Biases.

        Args:
            config (Config): An instance of the Config class containing all experiment hyperparameters.
        """
        self._config: Config = config
        
        # Retrieve logging configuration
        experiment_name: str = self._config.get_hyperparam('experiment.name')
        log_base_dir: str = self._config.get_hyperparam('logging.log_dir')
        use_wandb_config: bool = self._config.get_hyperparam('logging.use_wandb')
        project_name: str = self._config.get_hyperparam('logging.project_name')

        # Create unique run directory with timestamp
        timestamp: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir: str = os.path.join(log_base_dir, experiment_name, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        print(f"Logging experiment '{experiment_name}' to: {self.run_dir}")

        # TensorBoard Initialization
        self.writer: SummaryWriter = SummaryWriter(self.run_dir)
        print(f"TensorBoard logs available at: {self.run_dir}")

        # Weights & Biases (WandB) Initialization (Conditional)
        self.use_wandb: bool = False
        if use_wandb_config:
            if _WANDB_AVAILABLE:
                try:
                    wandb.init(
                        project=project_name,
                        name=f'{experiment_name}-{timestamp}',
                        config=self._config.to_dict(), # Pass the config dictionary
                        dir=log_base_dir # Saves wandb files to this directory before syncing
                    )
                    self.use_wandb = True
                    print(f"Weights & Biases logging enabled for project '{project_name}'.")
                except Exception as e:
                    print(f"Error initializing WandB: {e}. WandB logging will be disabled.")
            else:
                print("Weights & Biases (wandb) is configured but not available. WandB logging will be disabled.")
        
        # Save initial configuration to file
        self.save_config(self._config.to_dict())

    def log_scalar(self, tag: str, value: float, step: int):
        """
        Logs a single numerical metric to TensorBoard and optionally WandB.

        Args:
            tag (str): A string identifier for the metric (e.g., "train/loss", "eval/average_reward").
            value (float): The float value of the metric.
            step (int): The current global step (e.g., environment steps, gradient steps).
        """
        self.writer.add_scalar(tag, value, global_step=step)
        if self.use_wandb:
            wandb.log({tag: value}, step=step)

    def log_image(self, tag: str, image: np.ndarray, step: int):
        """
        Logs an image to TensorBoard and optionally WandB.

        Args:
            tag (str): A string identifier for the image (e.g., "environment/observation", "diffusion/generated_sample").
            image (np.ndarray): A NumPy array representing the image. Assumes CHW format (Channels, Height, Width).
            step (int): The current global step.
        """
        # TensorBoard expects CHW
        self.writer.add_image(tag, image, global_step=step, dataformats='CHW')
        if self.use_wandb:
            # wandb.Image can often infer format, but specifying explicitly can be safer
            wandb.log({tag: wandb.Image(image, dataformats='CHW')}, step=step)

    def log_histogram(self, tag: str, values: Union[np.ndarray, torch.Tensor], step: int):
        """
        Logs a histogram of a set of numerical values to TensorBoard and optionally WandB.

        Args:
            tag (str): A string identifier for the histogram (e.g., "policy/activations", "relevance/scores").
            values (Union[np.ndarray, torch.Tensor]): A NumPy array or PyTorch tensor containing the data.
            step (int): The current global step.
        """
        # TensorBoard handles both np.ndarray and torch.Tensor
        self.writer.add_histogram(tag, values, global_step=step)
        if self.use_wandb:
            # wandb.Histogram also handles both
            wandb.log({tag: wandb.Histogram(values)}, step=step)

    def save_config(self, config_dict: Dict[str, Any]):
        """
        Saves the experiment configuration to a YAML file in the run directory.

        Args:
            config_dict (Dict[str, Any]): A dictionary containing the full experiment configuration.
        """
        config_path: str = os.path.join(self.run_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"Configuration saved to: {config_path}")

    def save_checkpoint(self, state_dict: Dict[str, Any], step: int, filename: str = 'checkpoint.pt'):
        """
        Saves the state of the models and optimizers to a checkpoint file.

        Args:
            state_dict (Dict[str, Any]): A dictionary containing all necessary states to resume training.
            step (int): The current global step at which the checkpoint is saved.
            filename (str): The base name for the checkpoint file. Default is 'checkpoint.pt'.
        """
        checkpoint_filename: str = f'step_{step}_{filename}'
        full_path: str = os.path.join(self.run_dir, checkpoint_filename)
        torch.save(state_dict, full_path)
        print(f"Checkpoint saved to: {full_path}")

        if self.use_wandb:
            # wandb.save expects a file path relative to the run directory or an absolute path
            # It copies the file to the wandb run directory and uploads it.
            wandb.save(full_path, base_path=self.run_dir) # Use base_path to maintain directory structure if saving outside run_dir
            print(f"Checkpoint uploaded to WandB: {checkpoint_filename}")

    def close(self):
        """
        Performs cleanup operations for the logging systems, ensuring all buffered
        events are written and runs are terminated gracefully.
        """
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
        print("Logger closed.")

