import os
import logging
import datetime
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Tuple, Callable

# Assuming Config class is available from config.py
# To avoid circular imports if config.py needs Logger, we use a string type hint
# and ensure Config is passed during instantiation.
try:
    from config import Config
except ImportError:
    # Fallback for testing or if config.py is not yet available
    class Config:
        def __init__(self, config_data: Dict = None):
            self._data = config_data if config_data is not None else {}
        def get(self, key: str, default: Any = None) -> Any:
            keys = key.split('.')
            current = self._data
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            return current
        def set(self, key: str, value: Any) -> None:
            keys = key.split('.')
            current = self._data
            for k in keys[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        def save(self, output_path: str) -> None:
            pass # Dummy method

class Logger:
    """
    Centralizes all logging operations, including console output, file logging,
    TensorBoard integration for metrics, images, and figures.
    Handles saving and loading model weights.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the Logger object, setting up logging directories,
        Python's logging module, and TensorBoard SummaryWriter.

        Args:
            config (Config): The configuration object for accessing settings.
        """
        self.config: Config = config

        # 1. Setup run directory and subdirectories
        base_results_dir: str = self.config.get('paths.results_dir', 'results')
        experiment_name: str = self.config.get('experiment_name', 'default_experiment')
        timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir: str = os.path.join(base_results_dir, f"{experiment_name}_{timestamp}")
        
        self.log_file_path_base: str = os.path.join(self.run_dir, self.config.get('paths.log_dir', 'logs'))
        self.checkpoints_dir: str = os.path.join(self.run_dir, self.config.get('paths.checkpoint_dir', 'checkpoints'))
        self.figures_dir: str = os.path.join(self.run_dir, self.config.get('paths.figures_dir', 'figures'))
        self.tb_logs_dir: str = os.path.join(self.run_dir, 'tb_logs') # TensorBoard specific sub-directory

        # Create all necessary directories
        os.makedirs(self.log_file_path_base, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.tb_logs_dir, exist_ok=True)

        # 2. Setup standard Python logging
        self._logger: logging.Logger = logging.getLogger(experiment_name)
        self._logger.setLevel(logging.INFO)

        # Ensure no duplicate handlers if __init__ is called multiple times
        if not self._logger.handlers:
            formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Console handler
            console_handler: logging.StreamHandler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

            # File handler
            log_file_name: str = f"{experiment_name}.log"
            file_handler: logging.FileHandler = logging.FileHandler(os.path.join(self.log_file_path_base, log_file_name))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        self.log_info(f"Logger initialized for experiment: '{experiment_name}'")
        self.log_info(f"All logs and results will be stored in: {self.run_dir}")

        # 3. Setup TensorBoard SummaryWriter
        self._writer: SummaryWriter = SummaryWriter(log_dir=self.tb_logs_dir)
        self.log_info(f"TensorBoard logs available at: {self.tb_logs_dir}")

    def log_info(self, message: str) -> None:
        """
        Logs an informational message to the console and the log file.

        Args:
            message (str): The message to log.
        """
        self._logger.info(message)

    def log_metric(self, name: str, value: float, step: int = 0, tag: str = 'train') -> None:
        """
        Logs a numerical metric to TensorBoard and the log file.

        Args:
            name (str): The name of the metric (e.g., 'loss', 'accuracy').
            value (float): The value of the metric.
            step (int, optional): The global step at which the metric is recorded. Defaults to 0.
            tag (str, optional): A tag to categorize the metric (e.g., 'train', 'eval').
                                 Used to group metrics in TensorBoard. Defaults to 'train'.
        """
        full_metric_name: str = f"{tag}/{name}"
        self._writer.add_scalar(full_metric_name, value, global_step=step)
        self._logger.info(f"Step {step} [{tag}] {name}: {value:.6f}")

    def log_figure(self, name: str, fig: plt.Figure, step: int = 0) -> None:
        """
        Logs a matplotlib figure to TensorBoard and saves it as an image file.

        Args:
            name (str): The name of the figure.
            fig (matplotlib.figure.Figure): The matplotlib Figure object to log.
            step (int, optional): The global step associated with the figure. Defaults to 0.
        """
        # Save figure to file
        fig_filename: str = f"{name}_step_{step}.png"
        fig_path: str = os.path.join(self.figures_dir, fig_filename)
        try:
            fig.savefig(fig_path, bbox_inches='tight')
            self.log_info(f"Saved figure '{name}' to {fig_path}")
        except Exception as e:
            self._logger.error(f"Failed to save figure '{name}' to file: {e}")

        # Log figure to TensorBoard
        try:
            self._writer.add_figure(name, fig, global_step=step)
            self.log_info(f"Logged figure '{name}' to TensorBoard at step {step}")
        except Exception as e:
            self._logger.error(f"Failed to log figure '{name}' to TensorBoard: {e}")

        # Close the figure to free up memory
        plt.close(fig)

    def save_model_weights(self, model: torch.nn.Module, path: str) -> None:
        """
        Saves the state dictionary of a PyTorch model to a specified path within the checkpoints directory.

        Args:
            model (torch.nn.Module): The PyTorch model whose weights are to be saved.
            path (str): The relative path/filename within the checkpoints directory
                        (e.g., 'agent_final.pth', 'probe_layer_0_epoch_10.pt').
        """
        full_save_path: str = os.path.join(self.checkpoints_dir, path)
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        try:
            torch.save(model.state_dict(), full_save_path)
            self.log_info(f"Model weights saved to: {full_save_path}")
        except Exception as e:
            self._logger.error(f"Failed to save model weights to {full_save_path}: {e}")

    def load_model_weights(self, model: torch.nn.Module, path: str) -> None:
        """
        Loads model weights from a specified path within the checkpoints directory into a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model to load weights into.
            path (str): The relative path/filename within the checkpoints directory
                        (e.g., 'agent_final.pth', 'probe_layer_0_epoch_10.pt').
        """
        full_load_path: str = os.path.join(self.checkpoints_dir, path)
        if not os.path.exists(full_load_path):
            self._logger.error(f"Checkpoint file not found at: {full_load_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {full_load_path}")
        try:
            state_dict: Dict[str, torch.Tensor] = torch.load(full_load_path)
            model.load_state_dict(state_dict)
            self.log_info(f"Model weights loaded from: {full_load_path}")
        except Exception as e:
            self._logger.error(f"Failed to load model weights from {full_load_path}: {e}")
            raise

    def close(self) -> None:
        """
        Closes the TensorBoard SummaryWriter. Should be called at the end of an experiment.
        """
        if self._writer:
            self._writer.close()
            self.log_info("TensorBoard SummaryWriter closed.")


if __name__ == '__main__':
    # --- Dummy Config for Testing ---
    dummy_config_content = """
    experiment_name: "test_logger_experiment"
    paths:
      results_dir: "test_results"
      log_dir: "run_logs"
      checkpoint_dir: "models"
      figures_dir: "visuals"
    """
    with open("dummy_config_for_logger.yaml", "w") as f:
        f.write(dummy_config_content)

    print("--- Testing Logger class ---")
    
    # Initialize Config
    config_instance = Config("dummy_config_for_logger.yaml")

    # Initialize Logger
    logger = Logger(config_instance)

    # Test log_info
    logger.log_info("This is an informational message.")
    logger.log_info("Another info message for the log file.")

    # Test log_metric
    for i in range(5):
        logger.log_metric("loss", 1.0 / (i + 1), step=i, tag="train")
        logger.log_metric("accuracy", 0.5 + i * 0.1, step=i, tag="eval")

    # Test log_figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1, 2], [1, 3, 2], label="Sample Line")
    ax.set_title("Sample Figure")
    ax.legend()
    logger.log_figure("sample_plot", fig, step=1)
    plt.close(fig) # ensure fig is closed after logging

    # Test save_model_weights and load_model_weights
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
        def forward(self, x):
            return self.linear(x)

    model_to_save = DummyModel()
    logger.save_model_weights(model_to_save, "dummy_model_epoch_0.pth")

    model_to_load = DummyModel()
    logger.load_model_weights(model_to_load, "dummy_model_epoch_0.pth")

    # Test non-existent checkpoint
    try:
        logger.load_model_weights(model_to_load, "non_existent_model.pth")
    except FileNotFoundError:
        logger.log_info("Successfully caught FileNotFoundError for non-existent model.")

    # Clean up (optional, but good for testing)
    logger.close()
    
    print("\nVerifying created directories and files:")
    print(f"Run directory: {logger.run_dir}")
    print(f"Log file directory: {logger.log_file_path_base}")
    print(f"Checkpoints directory: {logger.checkpoints_dir}")
    print(f"Figures directory: {logger.figures_dir}")
    print(f"TensorBoard logs directory: {logger.tb_logs_dir}")

    # Inspect contents
    print("\nContents of run directory:")
    for root, dirs, files in os.walk(logger.run_dir):
        level = root.replace(logger.run_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

    # Cleanup generated files and directories
    import shutil
    if os.path.exists(config_instance.get('paths.results_dir')):
        shutil.rmtree(config_instance.get('paths.results_dir'))
        print(f"\nCleaned up '{config_instance.get('paths.results_dir')}' directory.")
    os.remove("dummy_config_for_logger.yaml")
    print("Cleaned up dummy_config_for_logger.yaml.")
    print("--- Logger testing complete ---")

