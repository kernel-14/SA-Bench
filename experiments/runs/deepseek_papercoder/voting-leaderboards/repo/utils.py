"""
utils.py – General-purpose helper functions for the reproduction pipeline.

Provides centralized seed setting for reproducibility, JSON I/O, directory creation,
and a timestamp generator. This module has no internal dependencies and can be imported
by any other part of the project.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def set_all_seeds(seed: int) -> None:
    """
    Set random seeds for Python's built-in random module and NumPy to ensure
    deterministic behaviour across the entire pipeline.

    Args:
        seed: Integer seed value.
    """
    # Ensure integer type
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: Any, path: str) -> None:
    """
    Serialise a Python object to a JSON file. The parent directory is created
    automatically if it does not exist.

    Args:
        data: JSON‑serialisable object (usually dict or list).
        path: Full file path where the JSON will be written.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Any:
    """
    Deserialise a JSON file back into a Python object.

    Args:
        path: Path to the JSON file.

    Returns:
        The deserialised Python object (dict, list, etc.).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamp() -> str:
    """
    Create a compact, filesystem‑safe timestamp string (UTC).

    Format: YYYYMMDD_HHMMSS

    Returns:
        Timestamp string, e.g., '20241201_153045'.
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    """
    Safely create a directory (and any missing parent directories). Does nothing
    if the directory already exists.

    Args:
        path: Directory path to create.
    """
    if path:  # Avoid passing an empty string to makedirs
        os.makedirs(path, exist_ok=True)
