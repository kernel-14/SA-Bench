```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import os
from scipy.stats import linregress
from typing import Any, Dict, List, Tuple, Optional, Union

# Attempt to import Config class
try:
    from config import Config
except ImportError:
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
    print("Warning: Could not import 'Config' from 'config.py'. Using a dummy Config class.")

# Attempt to import environment constants for type safety and consistency
try:
    from environments.sokoban import (
        WALL_IDX, EMPTY_IDX, BOX_ON_EMPTY_IDX, AGENT_ON_EMPTY_IDX,
        BOX_ON_TARGET_IDX, AGENT_ON_TARGET_IDX, TARGET_EMPTY_IDX
    )
    from environments.mini_pacman import (
        WALL_MP_IDX, FOOD_MP_IDX, PILL_MP_IDX, AGENT_MP_IDX,
        GHOST_BASE_MP_IDX, GHOST_CHANNELS_PER_GHOST, MAX_GHOSTS,
        GHOST_STATE_NORMAL, GHOST_STATE_EDIBLE, GHOST_STATE_FLASHING, COLOR_MAP_MP
    )
except ImportError:
    print("Warning: Could not import environment constants. Defining local dummies.")
    # Dummy definitions for environment constants if import fails
    WALL_IDX, EMPTY_IDX, BOX_ON_EMPTY_IDX, AGENT_ON_EMPTY_IDX, BOX_ON_TARGET_IDX, AGENT_ON_TARGET_IDX, TARGET_EMPTY_IDX = range(7)
    WALL_MP_IDX, FOOD_MP_IDX, PILL_MP_IDX, AGENT_MP_IDX = range(4)
    GHOST_BASE_MP_IDX = 4
    GHOST_CHANNELS_PER_GHOST = 2
    MAX_GHOSTS = 5
    GHOST_STATE_NORMAL, GHOST_STATE_EDIBLE, GHOST_STATE_FLASHING = range(3)
    COLOR_MAP_MP = {
        "wall": (0, 0, 0), "empty": (200, 200, 200), "food": (255, 255, 0), "pill": (255, 100, 0),
        "agent": (0, 255, 0), "ghost_normal": (255, 0, 0), "ghost_edible": (0, 0, 255), "ghost_flashing": (100, 100, 255)
    }


# Constants for concept class labels (consistent with ConceptLabeler)
NEVER: int = 0
UP: int = 1
DOWN: int = 2
LEFT: int = 3
RIGHT: int = 4

# Maps integer class labels to human-readable strings
_DIRECTION_CLASS_MAP: Dict[int, str] = {
    NEVER: 'NEVER', UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'
}

# Mapping for arrow directions (delta_x, delta_y for matplotlib.patches.Arrow)
# Matplotlib arrows have x_start, y_start, dx, dy
# (0,0) is top-left, x increases right, y increases down. Cell centers are (c, r)
_ARROW_VECTORS: Dict[int, Tuple[float, float]] = {
    UP: (0, -0.4),      # Upward arrow (y decreases)
    DOWN: (0, 0.4),     # Downward arrow (y increases)
    LEFT: (-0.4, 0),    # Leftward arrow (x decreases)
    RIGHT: (0.4, 0)     # Rightward arrow (x increases)
}

# RGB colors for Sokoban board elements (0-255 scale)
_SOKOBAN_CELL_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    WALL_IDX: (0, 0, 0),            # Black
    EMPTY_IDX: (200, 200, 200),     # Light Grey
    BOX_ON_EMPTY_IDX: (139, 69, 19),# Brown
    AGENT_ON_EMPTY_IDX: (0, 255, 0),# Green (Agent)
    BOX_ON_TARGET_IDX: (100, 50, 0),# Darker Brown
    AGENT_ON_TARGET_IDX: (0, 150, 0),# Darker Green
    TARGET_EMPTY_IDX: (255, 0, 0)   # Red (Target)
}

class Visualization:
    """
    Provides utilities for generating all plots and figures presented in the paper.
    This includes plotting F1 scores, success rates, and correlations. Crucially,
    it will have functions to render game boards and overlay decoded concept predictions
    as colored arrows or other markers.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the Visualization object, setting up plotting aesthetics and parameters.

        Args:
            config (Config): The configuration object for accessing settings.
        """
        self.config: Config = config
        
        # General plot settings
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 9
        self.plot_dpi: int = self.config.get('plot_dpi', 100) # Default for saving
        plt.rcParams['figure.dpi'] = self.plot_dpi
        
        # Color palettes for different plot series
        self.palette_qualitative = sns.color_palette("deep", 10) # For distinct series
        self.palette_sequential = sns.color_palette("viridis", 10) # For ordered series (e.g. layers)

        # Arrow colors for concepts as specified in the paper (Figure 1, Figure 10 legend)
        self.ca_arrow_color: str = 'teal'
        self.cb_arrow_color: str = 'purple'
        self.mini_pacman_ca_arrow_color: str = 'teal'
        self.mini_pacman_cross_color: str = 'teal' # For AgentApproach_MiniPacMan_16

        # Default cell size for pixel rendering
        self.sokoban_cell_size: int = 32
        self.mini_pacman_cell_size: int = 20

    def _get_sokoban_pixel_representation(self, state: np.ndarray, cell_size: int) -> np.ndarray:
        """
        Converts a symbolic Sokoban state (H, W, 7) into a pixel-based RGB image.

        Args:
            state (np.ndarray): The symbolic state of shape (H, W, 7).
            cell_size (int): The number of pixels for each grid cell's side.

        Returns:
            np.ndarray: An RGB image representation of the board, shape (H*cell_size, W*cell_size, 3).
        """
        grid_h, grid_w, _ = state.shape
        pixel_image = np.zeros((grid_h * cell_size, grid_w * cell_size, 3), dtype=np.uint8)

        for r in range(grid_h):
            for c in range(grid_w):
                # Find the active channel (which element is present)
                active_channel: int = np.argmax(state[r, c])
                color: Tuple[int, int, int] = _SOKOBAN_CELL_COLOR_MAP.get(active_channel, (128, 128, 128)) # Grey for unknown
                
                # Fill the corresponding cell area in the pixel image
                pixel_image[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = color
        return pixel_image

    def plot_sokoban_board(self, ax: plt.axes.Axes, state: np.ndarray) -> None:
        """
        Renders a Sokoban game board from its symbolic observation into a visual pixel
        representation on a given matplotlib.axes.Axes object.

        Args:
            ax (matplotlib.axes.Axes): The Axes object to draw on.
            state (np.ndarray): A NumPy array of shape (H, W, C), representing the symbolic state.
        """
        pixel_image: np.ndarray = self._get_sokoban_pixel_representation(state, self.sokoban_cell_size)
        ax.imshow(pixel_image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        # Adjust limits to ensure cells are displayed correctly without gaps/truncation
        ax.set_xlim(-0.5, state.shape[1] - 0.5)
        ax.set_ylim(state.shape[0] - 0.5, -0.5) # Invert y-axis to match numpy array indexing (row 0 at top)

    def overlay_concept_arrows(self, ax: plt.axes.Axes, concept_preds: np.ndarray, concept_type: str, layer_idx: Optional[int] = None) -> None:
        """
        Overlays directional arrows or other markers on a rendered Sokoban board to visualize
        decoded concept predictions (C_A, C_B).

        Args:
            ax (matplotlib.axes.Axes): The Axes object to draw on (assumed to contain a Sokoban board).
            concept_preds (np.ndarray): A NumPy array of shape (H, W) containing integer predictions
                                        for each grid cell (0: NEVER, 1: UP, 2: DOWN, etc.).
            concept_type (str): A string indicating the concept (e.g., "CA", "CB").
            layer_idx (Optional[int]): Optional, for logging/debugging or adding to titles.
        """
        grid_h, grid_w = concept_preds.shape
        arrow_color: str = self.ca_arrow_color if concept_type == "CA" else self.cb_arrow_color
        
        # Arrow properties (adjusted for cell center and aesthetic scaling)
        arrow_width: float = 0.1
        head_width: float = 0.3
        head_length: float = 0.2

        for r in range(grid_h):
            for c in range(grid_w):
                predicted_class: int = concept_preds[r, c]
                if predicted_class != NEVER:
                    dx, dy = _ARROW_VECTORS.get(predicted_class, (0,0))
                    
                    # Arrow starts at center of cell (c, r) and points in direction (dx, dy)
                    ax.arrow(c, r, dx, dy,
                             width=arrow_width, head_width=head_width, head_length=head_length,
                             facecolor=arrow_color, edgecolor=arrow_color, zorder=2) # zorder to ensure visibility

    def plot_macro_f1_curve(self, data: Dict[str, Tuple[List[float], List[float]]],
                            x_label: str, y_label: str, title: str,
                            x_values: Optional[List[float]] = None) -> plt.Figure:
        """
        Generates a line plot with error bars, commonly used for Macro F1 scores or success rates.

        Args:
            data (Dict[str, Tuple[List[float], List[float]]]): A dictionary where keys are plot labels
                                                               (e.g., "Layer 1 - CA") and values are
                                                               tuples of (mean_values, std_dev_values).
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            title (str): Title of the plot.
            x_values (Optional[List[float]]): Optional list of x-axis values. If None,
                                               range(len(mean_values)) is used.

        Returns:
            matplotlib.figure.Figure: The generated matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = sns.color_palette("deep", len(data))
        for i, (label, (means, std_devs)) in enumerate(data.items()):
            x: List[float] = x_values if x_values is not None else list(range(len(means)))
            ax.plot(x, means, label=label, color=colors[i], marker='o', markersize=4)
            ax.fill_between(x, np.array(means) - np.array(std_devs), np.array(means) + np.array(std_devs),
                            color=colors[i], alpha=0.2)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        return fig

    def plot_correlation(self, x_data_dict: Dict[str, List[float]], y_data_dict: Dict[str, List[float]],
                         x_label: str, y_label: str, title: str) -> plt.Figure:
        """
        Generates a scatter plot to visualize correlations between two variables,
        potentially showing multiple series with different colors.

        Args:
            x_data_dict (Dict[str, List[float]]): Dictionary where keys are series labels and values
                                                 are lists of x-axis data points.
            y_data_dict (Dict[str, List[float]]): Dictionary where keys are series labels and values
                                                 are lists of y-axis data points.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            title (str): Title of the plot.

        Returns:
            matplotlib.figure.Figure: The generated matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = sns.color_palette("deep", len(x_data_dict))
        for i, label in enumerate(x_data_dict.keys()):
            x_series: np.ndarray = np.array(x_data_dict[label])
            y_series: np.ndarray = np.array(y_data_dict[label])
            
            ax.scatter(x_series, y_series, label=label, color=colors[i], alpha=0.7)
            
            # Optionally plot linear regression line
            if len(x_series) > 1: # Need at least 2 points for regression
                slope, intercept, r_value, p_value, std_err = linregress(x_series, y_series)
                x_reg = np.array([x_series.min(), x_series.max()])
                y_reg = slope * x_reg + intercept
                ax.plot(x_reg, y_reg, color=colors[i], linestyle='--', alpha=0.5,
                        label=f'Regress ({label}) R^2={r_value**2:.2f}')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        return fig

    def _get_mini_pacman_pixel_representation(self, symbolic_state: np.ndarray, cell_size: int) -> np.ndarray:
        """
        Converts a symbolic Mini PacMan state (H, W, 14) into a pixel-based RGB image.
        This function reconstructs the internal game elements from the symbolic observation
        to render them.

        Args:
            symbolic_state (np.ndarray): The symbolic state of shape (H, W, 14).
            cell_size (int): The number of pixels for each grid cell's side.

        Returns:
            np.ndarray: An RGB image representation of the board, shape (H*cell_size, W*cell_size, 3).
        """
        grid_h, grid_w, _ = symbolic_state.shape
        pixel_image = np.zeros((grid_h * cell_size, grid_w * cell_size, 3), dtype=np.uint8)

        # Iterate through each cell to draw base elements (walls/empty)
        for r in range(grid_h):
            for c in range(grid_w):
                if symbolic_state[r, c, WALL_MP_IDX] == 1:
                    color: Tuple[int, int, int] = COLOR_MAP_MP["wall"]
                else:
                    color = COLOR_MAP_MP["empty"]
                pixel_image[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = color
        
        # Overlay other elements based on channels
        for r in range(grid_h):
            for c in range(grid_w):
                # Food
                if symbolic_state[r, c, FOOD_MP_IDX] == 1:
                    center_x, center_y = c * cell_size + cell_size // 2, r * cell_size + cell_size // 2
                    radius = cell_size // 4
                    # Draw a small dot (approximated circle)
                    for x_offset in range(-radius, radius + 1):
                        for y_offset in range(-radius, radius + 1):
                            if x_offset**2 + y_offset**2 <= radius**2:
                                pixel_image[center_y + y_offset, center_x + x_offset] = COLOR_MAP_MP["food"]

                # Pill
                if symbolic_state[r, c, PILL_MP_IDX] == 1:
                    center_x, center_y = c * cell_size + cell_size // 2, r * cell_size + cell_size // 2
                    radius = cell_size // 2 - 2 # Slightly smaller than cell
                    # Draw a circle
                    for x_offset in range(-radius, radius + 1):
                        for y_offset in range(-radius, radius + 1):
                            if x_offset**2 + y_offset**2 <= radius**2:
                                pixel_image[center_y + y_offset, center_x + x_offset] = COLOR_MAP_MP["pill"]

                # Agent
                if symbolic_state[r, c, AGENT_MP_IDX] == 1:
                    pixel_image[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = COLOR_MAP_MP["agent"]

                # Ghosts
                for ghost_id in range(MAX_GHOSTS):
                    ghost_pos_channel: int = GHOST_BASE_MP_IDX + ghost_id * GHOST_CHANNELS_PER_GHOST
                    ghost_state_channel: int = ghost_pos_channel + 1
                    
                    if symbolic_state[r, c, ghost_pos_channel] == 1:
                        ghost_state_code: int = symbolic_state[r, c, ghost_state_channel]
                        if ghost_state_code == GHOST_STATE_FLASHING:
                            color = COLOR_MAP_MP["ghost_flashing"]
                        elif ghost_state_code == GHOST_STATE_EDIBLE:
                            color = COLOR_MAP_MP["ghost_edible"]
                        else: # GHOST_STATE_NORMAL
                            color = COLOR_MAP_MP["ghost_normal"]
                        pixel_image[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = color
        return pixel_image

    def plot_mini_pacman_board(self, ax: plt.axes.Axes, state: np.ndarray) -> None:
        """
        Renders a Mini PacMan game board from its symbolic observation.

        Args:
            ax (matplotlib.axes.Axes): The Axes object to draw on.
            state (np.ndarray): A NumPy array of shape (H, W, 14) representing the symbolic state.
        """
        pixel_image: np.ndarray = self._get_mini_pacman_pixel_representation(state, self.mini_pacman_cell_size)
        ax.imshow(pixel_image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, state.shape[1] - 0.5)
        ax.set_ylim(state.shape[0] - 0.5, -0.5)

    def overlay_mini_pacman_concept_arrows(self, ax: plt.axes.Axes, concept_preds: np.ndarray, concept_type: str) -> None:
        """
        Overlays concept visualizations for Mini PacMan (arrows for direction, crosses for presence).

        Args:
            ax (matplotlib.axes.Axes): The Axes object to draw on.
            concept_preds (np.ndarray): A NumPy array of shape (H, W) containing integer predictions.
            concept_type (str): A string indicating the concept (e.g., "AgentApproachDirection_MiniPacMan_16",
                                "AgentApproach_MiniPacMan_16").
        """
        grid_h, grid_w = concept_preds.shape

        if concept_type == "AgentApproach_MiniPacMan_16":
            # Draw teal crosses
            for r in range(grid_h):
                for c in range(grid_w):
                    if concept_preds[r, c] == 1: # Assuming 1 means "AGAIN" for binary concepts
                        ax.scatter(c, r, marker='x', color=self.mini_pacman_cross_color, s=100, linewidth=2, zorder=2)
        elif concept_type == "AgentApproachDirection_MiniPacMan_16":
            # Draw teal arrows
            arrow_width: float = 0.1
            head_width: float = 0.3
            head_length: float = 0.2
            for r in range(grid_h):
                for c in range(grid_w):
                    predicted_class: int = concept_preds[r, c]
                    if predicted_class != NEVER:
                        dx, dy = _ARROW_VECTORS.get(predicted_class, (0,0))
                        ax.arrow(c, r, dx, dy,
                                 width=arrow_width, head_width=head_width, head_length=head_