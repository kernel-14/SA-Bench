import logging
import os
from datetime import datetime
import wandb
from typing import Any, Dict, Optional

# To avoid potential circular imports, we define a placeholder for the Config class
# that matches the interface required by this module. In main.py, the actual Config
# object will be passed.
class _ConfigPlaceholder:
    """
    A placeholder for the Config class to allow type hinting and access to its
    methods without creating a direct import dependency that might lead to
    circular imports in a larger project structure.
    """
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value."""
        raise NotImplementedError("This is a placeholder. Use actual Config object.")

    @property
    def config_dict(self) -> Dict[str, Any]:
        """Returns the underlying configuration dictionary."""
        raise NotImplementedError("This is a placeholder. Use actual Config object.")

# Use the placeholder for type hinting in this module
Config = _ConfigPlaceholder

def setup_logger(config: Config) -> logging.Logger:
    """
    Sets up a global logger for the project, configuring both Python's standard
    logging module for console and file output, and integrating with Weights & Biases
    (W&B) for experiment tracking.

    Args:
        config (Config): The global configuration object, containing settings
                         like log directory, experiment name, etc., obtained from
                         config.yaml.

    Returns:
        logging.Logger: The configured Python logger instance that can be used
                        throughout the application.
    """
    log_dir = config.get('general.log_dir', 'logs')
    experiment_name = config.get('general.experiment_name', 'default_experiment')
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate a unique timestamp for the current run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{experiment_name}_{timestamp}"

    # --- Python Logging Setup ---
    log_file_path = os.path.join(log_dir, f"{run_id}.log")

    # Get a named logger instance to avoid interfering with other libraries' logging
    logger = logging.getLogger("MDM_Project_Logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent messages from being passed to the root logger handlers

    # Clear existing handlers to prevent duplicate logs if setup_logger is called multiple times
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Console handler
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info("Python logging initialized.")
    logger.info(f"Log messages will be saved to: {log_file_path}")

    # --- Weights & Biases (W&B) Setup ---
    try:
        wandb.init(
            project=experiment_name,
            name=run_id,
            config=config.config_dict,  # Automatically log all hyperparameters from the Config object
            dir=log_dir,                # Store W&B-specific files (e.g., runs, metadata) in the log directory
            reinit=True                 # Allow re-initialization, useful in interactive environments
        )
        logger.info("Weights & Biases initialized successfully.")
        logger.info(f"W&B Project: {experiment_name}, Run: {run_id}")
    except wandb.errors.UsageError as e:
        logger.warning(f"Failed to initialize Weights & Biases: {e}.")
        logger.warning("Running without W&B. To enable, ensure you are logged in (wandb login) and have internet.")
        # If W&B initialization fails, clean up any partial run and potentially disable further W&B calls
        if wandb.run is not None:
            wandb.finish()
        # Optionally, set an environment variable to completely disable wandb logging for this session
        os.environ['WANDB_MODE'] = 'offline' # This will collect data but not sync immediately
        logger.warning("W&B set to 'offline' mode due to initialization failure.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during W&B initialization: {e}")
        if wandb.run is not None:
            wandb.finish()
        os.environ['WANDB_MODE'] = 'offline'
        logger.warning("W&B set to 'offline' mode due to an unexpected error.")


    return logger


if __name__ == '__main__':
    # This block is for testing the logger module in isolation.
    # It uses a mock Config class to simulate the actual Config object.

    class MockConfig:
        """A simplified mock Config class for testing purposes."""
        def __init__(self, data: Dict[str, Any]):
            self._data = data
        
        def get(self, key: str, default: Any = None) -> Any:
            keys = key.split('.')
            current = self._data
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            return current
        
        @property
        def config_dict(self) -> Dict[str, Any]:
            return self._data
            
    # Create a dummy config for testing
    dummy_config_data = {
        'general': {
            'experiment_name': 'test_logger_module',
            'seed': 42,
            'log_dir': 'temp_logs_for_logger_test'
        },
        'model': {
            'type': 'mock_model',
            'version': '1.0'
        },
        'training': {
            'epochs': 1
        }
    }
    mock_config = MockConfig(dummy_config_data)

    print("--- Initializing logger ---")
    test_logger = setup_logger(mock_config)

    test_logger.info("Logger initialized. This is an INFO message.")
    test_logger.debug("This is a DEBUG message (should not be visible by default).")
    test_logger.warning("This is a WARNING message.")
    test_logger.error("This is an ERROR message.")
    test_logger.critical("This is a CRITICAL message.")

    # Simulate W&B logging if it was successfully initialized
    if wandb.run is not None:
        test_logger.info("W&B run is active. Simulating metric logging...")
        wandb.log({"epoch": 0, "loss": 0.1, "accuracy": 0.95})
        test_logger.info("Metrics logged to W&B.")
        wandb.finish() # End the W&B run
        test_logger.info("W&B run finished.")
    else:
        test_logger.warning("W&B was not active or failed to initialize.")
        # If W&B was explicitly disabled (e.g. by 'offline' mode), it won't crash
        # but also won't log anything.

    # Clean up the temporary log directory and files
    log_dir_to_clean = mock_config.get('general.log_dir')
    if os.path.exists(log_dir_to_clean):
        print(f"\n--- Cleaning up temporary log directory: {log_dir_to_clean} ---")
        for f in os.listdir(log_dir_to_clean):
            file_path = os.path.join(log_dir_to_clean, f)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
        try:
            os.rmdir(log_dir_to_clean)
            print("Temporary log directory removed.")
        except OSError as e:
            print(f"Error removing directory {log_dir_to_clean}: {e}")
    print("--- Test complete ---")
