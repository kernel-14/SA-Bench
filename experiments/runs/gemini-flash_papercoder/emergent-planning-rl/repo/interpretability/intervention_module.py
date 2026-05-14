```python
import torch
import torch.nn as nn
import numpy as np
import random
from typing import Any, Dict, List, Tuple, Optional, Union, Callable

# Attempt to import core dependencies
try:
    from config import Config
    from utils.logger import Logger
    from environments.base_environment import BaseEnvironment
    from environments.sokoban import SokobanEnv # Needed for level_info specifics
    # Import specific environment constants for clarity and type safety
    from environments.sokoban import (
        WALL_IDX, EMPTY_IDX, BOX_ON_EMPTY_IDX, AGENT_ON_EMPTY_IDX,
        BOX_ON_TARGET_IDX, AGENT_ON_TARGET_IDX, TARGET_EMPTY_IDX, ACTION_MAP as SOKOBAN_ACTION_MAP
    )
    from agents.base_agent import BaseAgentModel
    from agents.drc_agent import DRCAgent # For specific internal structure
    from agents.resnet_agent import ResNetAgent # For specific internal structure
    from interpretability.probe_model import ProbeModel
except ImportError:
    # Dummy classes for standalone testing or if dependencies are not yet available
    print("Warning: Could not import core dependencies. Using dummy classes for InterventionModule.")

    class Config:
        """Dummy Config class for self-testing."""
        def __init__(self, data: Dict = None): self._data = data if data is not None else {}
        def get(self, key: str, default: Any = None) -> Any:
            keys = key.split('.')
            current = self._data
            for k in keys:
                if isinstance(current, dict) and k in current: current = current[k]
                else: return default
            return current
        def set(self, key: str, value: Any) -> None: pass
        def save(self, output_path: str) -> None: pass

    class Logger:
        def __init__(self, config: Config): pass
        def log_info(self, message: str) -> None: print(f"INFO: {message}")
        def log_metric(self, name: str, value: float, step: int = 0, tag: str = 'train') -> None: print(f"METRIC: {tag}/{name} @ {step}: {value}")
        def log_figure(self, name: str, fig: Any, step: int = 0) -> None: pass
        def save_model_weights(self, model: Any, path: str) -> None: pass
        def load_model_weights(self, model: Any, path: str) -> None: pass
        def close(self) -> None: pass
    
    class BaseEnvironment:
        """Dummy BaseEnvironment class."""
        def __init__(self, config: Config) -> None: self.config = config
        def reset(self, level_config