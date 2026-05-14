"""Logging utilities for SCoRe training."""

import logging
import sys
import json
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """Setup standardized logging for SCoRe."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


class TrainingLogger:
    """Logger for tracking training metrics."""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = defaultdict(list)
        self.step = 0
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log a dictionary of metrics."""
        if step is not None:
            self.step = step
        else:
            self.step += 1
        
        for key, value in metrics.items():
            self.metrics[key].append((self.step, value))
    
    def save(self, filename: str = "metrics.json"):
        """Save all logged metrics to JSON."""
        path = os.path.join(self.log_dir, filename)
        # Convert to serializable format
        serializable = {}
        for key, values in self.metrics.items():
            serializable[key] = [(int(s), float(v)) for s, v in values]
        
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def get_summary(self, window: int = 100) -> Dict[str, float]:
        """Get summary statistics over last `window` steps."""
        summary = {}
        for key, values in self.metrics.items():
            if len(values) >= window:
                recent = [v for _, v in values[-window:]]
            else:
                recent = [v for _, v in values]
            summary[f"{key}_mean"] = np.mean(recent)
            summary[f"{key}_std"] = np.std(recent)
        return summary


__all__ = ["setup_logging", "TrainingLogger"]
