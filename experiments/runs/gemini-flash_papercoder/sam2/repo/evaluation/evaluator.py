```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# Import custom modules
from config import Config
from model.sam2_model import SAM2Model
from data.prompt_simulator import PromptSimulator
from model.memory_modules import MemoryBank  # Import MemoryBank directly
from utils.metrics import calculate_miou, calculate_j_and_f, calculate_g


class Evaluator(object):
    """
    Evaluates the SAM2Model's performance across various segmentation tasks.
    Supports interactive video segmentation (offline/online), semi-supervised VOS,
    and image segmentation.
    """

    def __init__(self, model: SAM2Model, config: Config, device: torch.device):
        """
        Initializes the Evaluator.

        Args:
            model (SAM2Model): The main SAM 2 model instance.
            config (Config): The global configuration object.
            device (torch.device): The computational device (cuda or cpu).
        """
        self.model = model
        self.config = config
        self.device = device

        self.model.eval()
        self.model.to(self.device)

        self.prompt_simulator = PromptSimulator(self.config)

        # Interactive evaluation settings
        self.num_clicks: int = self.config.get("evaluation.interactive.num_clicks", 3)
        self.max_interacted_frames: int = self.config.get("evaluation.interactive.max_interacted_frames", 8)
        self.online_iou_threshold: float = self.config.get("evaluation.interactive.online_iou_threshold", 0.75)

        # Annotation time parameters for interactive evaluation metrics
        self.T_loc: float = self.config.get("evaluation.interactive.T_loc", 1.0)
        self.T_click: float = self.config.get("evaluation.interactive.T_click", 1.5)
        self.T_exam_300frame: float = self.config.get("evaluation.interactive.T_exam_300frame", 30.0)

        # Semi-supervised VOS prompt types
        self.vos_prompt_types: List[str] = self.config.get("evaluation.semisupervised_vos.prompt_types", [])
        
        # Image segmentation prompt types
        self.image_prompt_types: List[str] = self.config.get("evaluation.image_segmentation.prompt_types", [])

        print(f"Evaluator initialized on device: {self.device}")
        print(f"  Interactive eval num clicks per interaction: {self.num_clicks}, max interactive rounds: {self.max_interacted_frames}")

    def evaluate_interactive_video(