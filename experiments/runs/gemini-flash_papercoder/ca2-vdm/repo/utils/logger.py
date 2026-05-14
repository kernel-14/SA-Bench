import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

# To avoid circular import if Config were to import logger,
# we use TYPE_CHECKING for type hints only.
if TYPE_CHECKING:
    from config import Config


def setup_logging(config: "Config") -> None:
    """
    Configures the Python logging system for the application.

    This function sets up a root logger with both a console handler
    and a file handler. It uses settings provided by the Config object
    to determine the logging level and file paths.

    Args:
        config (Config): An instance of the Config class containing
                         application settings, including log_level,
                         save_path, and log_dir.
    """
    # 1. Configuration Extraction
    log_level_str = getattr(config, 'log_level', 'INFO')
    save_path = getattr(config, 'save_path', './')
    log_dir_name = getattr(config, 'log_dir', 'logs')

    # Map string log level to logging module's constants
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Ensure the log directory exists
    log_directory = os.path.join(save_path, log_dir_name)
    os.makedirs(log_directory, exist_ok=True)

    # Construct a timestamped log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"run_{timestamp}.log"
    log_file_path = os.path.join(log_directory, log_file_name)

    # 2. Logger Initialization
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # 3. Handler Management: Clear existing handlers to prevent duplicate output
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 4. Formatter Definition
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # 5. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 6. File Handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 7. Post-configuration Logging
    # Use a logger from this module to avoid immediate circular dependency
    # with the root logger during setup, though it's technically fine here.
    # It's good practice to use a named logger for specific modules.
    module_logger = logging.getLogger(__name__)
    module_logger.info(
        f"Logging configured to level '{log_level_str}'."
        f" Console output enabled. Log file: '{log_file_path}'"
    )
