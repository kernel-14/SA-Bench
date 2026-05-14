import logging
import os
from datetime import datetime
from typing import Union, Tuple, Any

import torch
from torch.utils.tensorboard import SummaryWriter

# Assuming config.py is in the root directory and can be imported directly
from config import Config


def setup_logging(config: Config, stage: str) -> Tuple[logging.Logger, SummaryWriter]:
    """
    Initializes and configures the standard Python logger and sets up a
    torch.utils.tensorboard.SummaryWriter for TensorBoard logging.
    It creates necessary log directories and saves the configuration.

    Args:
        config (Config): An instance of the Config class, providing access to
                         logging configuration parameters.
        stage (str): A string indicating the current stage
                     (e.g., "p2vae_train", "fmt_train", "eval", "finetune")
                     to organize logs.

    Returns:
        Tuple[logging.Logger, SummaryWriter]: The configured Python logger
                                              and TensorBoard SummaryWriter instance.
    """
    # 1. Generate Run Name and Log Directory
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    project_name = config.get('logging.project_name', 'GenerativePDEFoundationModel')
    run_name = f"{project_name}_{current_time}_{stage}"

    base_log_dir = config.get('logging.log_dir', './logs')
    run_log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)

    # 2. Configure Python Logger
    # Use a named logger for the project to avoid interfering with other loggers
    logger = logging.getLogger(project_name)
    # Set default logging level to INFO, can be overridden by config if a setting existed
    logger.setLevel(logging.INFO) 
    
    # Check if handlers already exist to prevent adding duplicates when called multiple times
    # in different parts of an application (e.g., if main creates a logger, then a trainer also tries)
    if not logger.handlers:
        # Formatter for log messages
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler to output logs to stdout
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler to write logs to a file within the run's log directory
        file_handler = logging.FileHandler(os.path.join(run_log_dir, f'{stage}.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging initialized for run: {run_name}")
        logger.info(f"Log files will be saved to: {run_log_dir}")
    else:
        logger.info(f"Logger '{project_name}' already has handlers. Skipping handler setup.")
        # If handlers already exist, ensure the file handler points to the correct run_log_dir
        # This part assumes setup_logging is typically called once per main execution.
        # If it needs to re-configure file output for sub-stages, more complex logic is needed
        # to remove old file handlers and add new ones. For this design, we keep it simple.

    # 3. Initialize TensorBoard Writer
    tb_writer = SummaryWriter(log_dir=run_log_dir)
    logger.info(f"TensorBoard writer initialized at: {run_log_dir}")

    # 4. Log Configuration for reproducibility
    config.save_config(os.path.join(run_log_dir, 'config.yaml'))
    logger.info("Current configuration saved to run directory.")

    return logger, tb_writer


def log_message(logger_obj: logging.Logger, message: str, level: int = logging.INFO) -> None:
    """
    Provides a standardized way to log text messages through the configured Python logger.

    Args:
        logger_obj (logging.Logger): The `logging.Logger` instance returned by `setup_logging`.
        message (str): The string message to log.
        level (int, optional): The logging level (e.g., logging.INFO, logging.WARNING, logging.ERROR).
                               Defaults to logging.INFO.
    """
    logger_obj.log(level, message)


def log_scalar(writer: SummaryWriter, tag: str, scalar_value: Union[float, int, torch.Tensor], global_step: int) -> None:
    """
    Logs a single scalar value to the TensorBoard writer.

    Args:
        writer (SummaryWriter): The `SummaryWriter` instance returned by `setup_logging`.
        tag (str): A string tag (e.g., "Loss/train", "Metrics/L2RE_val") to categorize the scalar
                   in TensorBoard.
        scalar_value (Union[float, int, torch.Tensor]): The numerical value to log.
                                                       If a `torch.Tensor`, it will be converted to a scalar.
        global_step (int): The current training step or epoch number.
    """
    if isinstance(scalar_value, torch.Tensor):
        # Ensure it's a 0-dimensional tensor before calling .item()
        if scalar_value.numel() == 1:
            scalar_value = scalar_value.item()
        else:
            # Handle cases where a tensor with multiple elements is passed unintentionally
            logging.warning(f"Attempted to log a non-scalar tensor with tag '{tag}'. Logging only the first element.")
            scalar_value = scalar_value.flatten()[0].item()
            
    writer.add_scalar(tag, scalar_value, global_step)


def log_image(writer: SummaryWriter, tag: str, image_tensor: torch.Tensor, global_step: int, normalize: bool = True) -> None:
    """
    Logs an image or a batch of images to the TensorBoard writer.
    Useful for visualizing reconstructions, predictions, or intermediate feature maps.

    Args:
        writer (SummaryWriter): The `SummaryWriter` instance.
        tag (str): A string tag (e.g., "Reconstructions/P2VAE", "Predictions/FMT_Rollout").
        image_tensor (torch.Tensor): A `torch.Tensor` representing the image(s).
                                     Expected shape: (N, C, H, W) or (C, H, W).
                                     Values should ideally be in [0, 1] for float or [0, 255] for byte.
        global_step (int): The current training step or epoch number.
        normalize (bool, optional): Whether to normalize the image pixel values to [0, 1] if they
                                    are not already. This is generally recommended for float tensors.
                                    Defaults to True.
    """
    # Ensure image_tensor is on CPU and float type for SummaryWriter
    image_tensor = image_tensor.cpu().float()

    # Add a batch dimension if only a single image (C, H, W) is passed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0) # Becomes (1, C, H, W)
    elif image_tensor.dim() != 4:
        logging.warning(f"Image tensor for tag '{tag}' has unsupported dimensions: {image_tensor.dim()}. "
                        "Expected 3 (C,H,W) or 4 (N,C,H,W). Skipping image logging.")
        return

    # Normalize pixel values to [0, 1] if specified and not already in this range.
    # SummaryWriter can handle [0, 255] byte images automatically, but for float, [0, 1] is standard.
    if normalize:
        min_val = image_tensor.min()
        max_val = image_tensor.max()
        if max_val > min_val:
            image_tensor = (image_tensor - min_val) / (max_val - min_val)
        else: # Handle cases of constant images (e.g., all zeros)
            image_tensor = torch.zeros_like(image_tensor)

    # Use 'NCHW' dataformats for PyTorch tensor conventions
    writer.add_images(tag, image_tensor, global_step, dataformats='NCHW')


def close_writers(writer: SummaryWriter) -> None:
    """
    Closes the TensorBoard writer, ensuring all buffered events are written to disk.

    Args:
        writer (SummaryWriter): The `SummaryWriter` instance.
    """
    writer.close()

