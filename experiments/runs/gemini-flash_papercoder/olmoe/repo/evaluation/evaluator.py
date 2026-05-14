```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from accelerate import Accelerator
from typing import Dict, Union, Any, List, Tuple
import yaml
import os
import subprocess
import json # For parsing DCLM/OLMES output if it's JSON
import math # For perplexity
from sklearn.metrics import accuracy_score, f1_score # For simpler metrics

# Local imports from the project structure
from config import Config
from utils.logger import Logger
from model.olmoe_model import OLMoEModel # Assuming OLMoEModel includes generate method


class Evaluator:
    """
    Orchestrates evaluation procedures for the OLMoE model at different stages:
    during pretraining, after pretraining, and after adaptation.
    It integrates with internal metric calculations and external evaluation scripts/standards.
    """

    def __init__(self, model: OLMoEModel, tokenizer: PreTrainedTokenizer, config: Config, logger: Logger, accelerator: Accelerator):
        """
        Initializes the Evaluator with the necessary components.

        Args:
            model (OLMoEModel): The OLMoE model instance to be evaluated.
            tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
            config (Config): The global configuration object.
            logger (Logger): The logger instance for tracking experiments.
            accelerator (Accelerator): The Hugging Face Accelerator instance for distributed operations.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.accelerator = accelerator # Store accelerator for distributed operations
        self.device = self.accelerator.device # Use accelerator's device

        # Set model to evaluation mode
        self.model.eval()
        # Model is already on self.device via accelerator.prepare in trainers/main.py

        # Load OLMES task configurations if path is provided
        self.olmes_tasks_config = self._load_olmes_config()

        # Validate DCLM repository path
        self.dclm_repo_path = self.config.evaluation.dclm_eval_repo_path
        if not os.path.isdir(self.dclm_repo_path) and self.accelerator.is_main_process:
            self.logger.log({"warning": f"DCLM evaluation path '{self.dclm_repo_path}' does not exist. DCLM evaluation will be skipped."}, step=-1)
            self.dclm_repo_path = None # Disable DCLM if path is invalid

        # Store a reference to the base model path for OLMES/DCLM if needed for comparison
        self.evaluation_base_model_path = config.evaluation.evaluation_base_model_path
        if not self.evaluation_base_model