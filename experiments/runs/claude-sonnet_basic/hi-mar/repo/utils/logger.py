"""
Logging utilities for Hi-MAR training.
"""

import os
import logging
import sys
from datetime import datetime


def setup_logger(output_dir, is_main=True, name='hi_mar'):
    """
    Setup logger that writes to both console and file.

    Args:
        output_dir: directory to save log file
        is_main: whether this is the main process
        name: logger name
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if is_main:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(output_dir, f'train_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
