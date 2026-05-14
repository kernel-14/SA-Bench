## utils.py
"""
Utility functions for reproducing "Interpreting Emergent Planning in Model-Free RL".

Provides:
- Random seed setting
- Boxoban level loading and parsing
- Sokoban symbolic observation encoding
- Basic board rendering with plan arrows
- Configuration class for loading YAML settings.
"""

import random
import numpy as np
from typing import List, Optional, Tuple
import os
import glob
import yaml
from PIL import Image, ImageDraw, ImageFont
import sys


# ----------------------------------------------------------------------
# 1. Random seed
# ----------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across random, numpy, and torch
    (if available). Also configures PyTorch deterministic mode if desired.
    """
    random.seed(seed)
    np.random.seed(seed)

    # If PyTorch is imported, set its seeds
    if "torch" in sys.modules:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Optionally enforce deterministic algorithms (can slow down)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# 2. Boxoban level loading
# ----------------------------------------------------------------------
def load_boxoban_levels(
    file_path: str,
    subset: Optional[str] = None,
    num_levels: Optional[int] = None
) -> List[str]:
    """
    Load Sokoban levels from a Boxoban dataset file or directory.

    Args:
        file_path: Path to a single .txt file or a directory containing multiple
                   .txt files. If a directory, all .txt files inside are read.
        subset: Ignored in this implementation; kept for compatibility.
        num_levels: If specified, return only the first `num_levels` levels.

    Returns:
        A list of level strings, each of length 64, representing an 8x8 board.
    """
    lines = []
    if os.path.isdir(file_path):
        # Read all .txt files recursively (Boxoban structure sometimes has subdirs)
        txt_files = glob.glob(os.path.join(file_path, "**/*.txt"), recursive=True)
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in directory {file_path}")
        for txt_file in sorted(txt_files):
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        lines.append(stripped)
    else:
        # Single file
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)

    # Validate every line has exactly 64 characters
    valid_lines = []
    for line in lines:
        if len(line) == 64:
            valid_lines.append(line)
        else:
            print(f"Warning: skipping level line of length {len(line)} (expected 64)")

    if num_levels is not None and num_levels < len(valid_lines):
        valid_lines = valid_lines[:num_levels]

    return valid_lines


# ----------------------------------------------------------------------
# 3. Symbolic observation encoding
# ----------------------------------------------------------------------
# Mapping from level character to one-hot channel index
CHAR_TO_CHANNEL = {
    '#': 0,   # wall
    ' ': 1,   # empty floor
    '$': 2,   # box on floor
    '@': 3,   # agent on floor
    '*': 4,   # box on target
    '+': 5,   # agent on target
    '.': 6,   # empty target
}

NUM_CHANNELS = 7
BOARD_SIZE = 8


def one_hot_encode(level_str: str) -> np.ndarray:
    """
    Convert a 64-character Sokoban level string into an 8x8x7 symbolic
    observation (one-hot per cell).

    Args:
        level_str: A string of exactly 64 characters from the Boxoban charset.

    Returns:
        np.ndarray of shape (8, 8, 7) with float32 values.
    """
    if len(level_str) != BOARD_SIZE * BOARD_SIZE:
        raise ValueError(f"Level string must have {BOARD_SIZE*BOARD_SIZE} characters, got {len(level_str)}")

    obs = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), dtype=np.float32)
    for idx, ch in enumerate(level_str):
        row = idx // BOARD_SIZE
        col = idx % BOARD_SIZE
        if ch in CHAR_TO_CHANNEL:
            obs[row, col, CHAR_TO_CHANNEL[ch]] = 1.0
        else:
            # Unknown character → treat as empty floor (channel 1) and warn
            print(f"Warning: unknown character '{ch}' at position ({row},{col}) – treating as empty floor")
            obs[row, col, 1] = 1.0
    return obs


# ----------------------------------------------------------------------
# 4. Board drawing with plan arrows
# ----------------------------------------------------------------------
# Color definitions (R, G, B)
TILE_COLORS = {
    0: (40, 40, 40),      # wall: dark grey
    1: (200, 200, 200),   # floor: light grey
    2: (140, 70, 20),     # box on floor: brown
    3: (255, 255, 0),     # agent on floor: yellow
    4: (140, 70, 20),     # box on target (draw symbol later)
    5: (255, 255, 0),     # agent on target
    6: (255, 80, 80),     # empty target: red tint
}

# Arrow thickness, size, colors
ARROW_COLOR_AGENT = (0, 128, 128)   # teal
ARROW_COLOR_BOX = (128, 0, 128)     # purple
ARROW_LEN_FRAC = 0.4                # fraction of cell size for arrow shaft
ARROW_HEAD_LEN = 0.2
ARROW_WIDTH = 2


def _draw_arrow(
    draw: ImageDraw.Draw,
    cell_center: Tuple[int, int],
    direction: int,
    color: Tuple[int, int, int],
    cell_size: int
) -> None:
    """
    Draw an arrow inside a grid cell pointing in the given direction.

    direction: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT.
    """
    if direction == 0:   # NEVER
        return

    cx, cy = cell_center
    half = cell_size // 2
    quarter = cell_size // 4

    # Define arrow shaft endpoints and head offsets
    if direction == 1:   # UP
        start = (cx, cy + quarter)
        end = (cx, cy - quarter)
        head_left = (cx - quarter//2, cy - quarter + quarter//2)
        head_right = (cx + quarter//2, cy - quarter + quarter//2)
    elif direction == 2: # DOWN
        start = (cx, cy - quarter)
        end = (cx, cy + quarter)
        head_left = (cx - quarter//2, cy + quarter - quarter//2)
        head_right = (cx + quarter//2, cy + quarter - quarter//2)
    elif direction == 3: # LEFT
        start = (cx + quarter, cy)
        end = (cx - quarter, cy)
        head_left = (cx - quarter + quarter//2, cy - quarter//2)
        head_right = (cx - quarter + quarter//2, cy + quarter//2)
    elif direction == 4: # RIGHT
        start = (cx - quarter, cy)
        end = (cx + quarter, cy)
        head_left = (cx + quarter - quarter//2, cy - quarter//2)
        head_right = (cx + quarter - quarter//2, cy + quarter//2)
    else:
        return

    # Draw shaft
    draw.line([start, end], fill=color, width=ARROW_WIDTH)
    # Draw arrowhead (triangle)
    draw.polygon([end, head_left, head_right], fill=color)


def draw_grid(
    obs: np.ndarray,
    agent_arrows: np.ndarray,
    box_arrows: np.ndarray,
    cell_size: int = 48,
    show_targets: bool = True
) -> np.ndarray:
    """
    Render a Sokoban board as an RGB image with optional plan arrows.

    Args:
        obs: 8x8x7 symbolic observation (one-hot floats).
        agent_arrows: 8x8 int array, values 0-4 (0=NEVER, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT).
        box_arrows: 8x8 int array, same encoding.
        cell_size: pixel size of each grid cell.
        show_targets: whether to draw a target marker on empty targets or boxes on targets.

    Returns:
        RGB image as a numpy array of shape (8*cell_size, 8*cell_size, 3) uint8.
    """
    img_width = img_height = BOARD_SIZE * cell_size
    img = Image.new("RGB", (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Optional font for drawing targets
    try:
        font = ImageFont.truetype("arial.ttf", size=cell_size//4)
    except IOError:
        font = ImageFont.load_default()

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            # Determine tile type from one-hot
            channel = obs[row, col].argmax()
            tile_color = TILE_COLORS.get(channel, (128, 128, 128))
            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size - 1
            y1 = y0 + cell_size - 1

            # Draw tile background
            draw.rectangle([x0, y0, x1, y1], fill=tile_color, outline=(100, 100, 100))

            # Draw target symbol on appropriate channels (if show_targets)
            if show_targets:
                if channel == 6:  # empty target
                    # Draw a red cross
                    draw.line([(x0, y0), (x1, y1)], fill=(255, 0, 0), width=2)
                    draw.line([(x1, y0), (x0, y1)], fill=(255, 0, 0), width=2)
                elif channel in (4, 5):  # box on target or agent on target
                    # Draw a small hollow circle
                    center = (x0 + cell_size//2, y0 + cell_size//2)
                    radius = cell_size//4
                    draw.ellipse(
                        [center[0] - radius, center[1] - radius,
                         center[0] + radius, center[1] + radius],
                        outline=(255, 0, 0),
                        width=2
                    )
                # Other channels: no extra symbol

            # Overlay agent and box arrows (using plan directions)
            cell_center = (x0 + cell_size//2, y0 + cell_size//2)
            _draw_arrow(draw, cell_center, int(agent_arrows[row, col]), ARROW_COLOR_AGENT, cell_size)
            _draw_arrow(draw, cell_center, int(box_arrows[row, col]), ARROW_COLOR_BOX, cell_size)

    # Convert PIL image to numpy array
    rgb_array = np.array(img, dtype=np.uint8)
    return rgb_array


# ----------------------------------------------------------------------
# 5. Configuration class
# ----------------------------------------------------------------------
class Config:
    """
    Configuration container for all experiments, loaded from a YAML file.

    Attributes:
        seed: int, random seed
        output_dir: str, path for saving results
        checkpoint_dir: str, path for model checkpoints
        env: dict, environment parameters
        agent: dict, agent architecture parameters
        training: dict, IMPALA training parameters
        probing: dict, linear probing parameters
        interventions: dict, causal intervention settings
        dynamics: dict, training dynamics analysis settings
        dataset: dict, paths to Boxoban splits
    """
    def __init__(self, config_dict: dict):
        self.seed = config_dict.get("seed", 42)
        self.output_dir = config_dict.get("output_dir", "./outputs")
        self.checkpoint_dir = config_dict.get("checkpoint_dir", "./checkpoints")
        self.env = config_dict.get("environment", {})
        self.agent = config_dict.get("agent", {})
        self.training = config_dict.get("training", {})
        self.probing = config_dict.get("probing", {})
        self.interventions = config_dict.get("interventions", {})
        self.dynamics = config_dict.get("dynamics", {})
        self.dataset = config_dict.get("dataset", {})

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the config.yaml file.

        Returns:
            Config instance.
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def __repr__(self) -> str:
        return (f"Config(seed={self.seed}, output_dir='{self.output_dir}')")


# Optionally, default configuration creation can be added
if __name__ == "__main__":
    # Example usage:
    cfg = Config.from_yaml("config.yaml")
    print(cfg)
    # Quick one-hot test
    lvl = "  ###  ## #  ##  # #  $   . . . #  @     .#     #  #  ## #######  "
    print(lvl)
    obs = one_hot_encode(lvl)
    print(obs.shape)
