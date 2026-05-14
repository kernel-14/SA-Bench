## visualization.py
"""
Visualization of the DRC agent's internal plans using trained linear probes.

The `PlanVisualizer` class renders Sokoban boards overlaid with coloured
arrows that represent the agent’s current internal plan according to the
concepts "Agent Approach Direction" (teal) and "Box Push Direction" (purple).
It supports single‑frame plots, layer‑by‑layer comparisons, and tick‑by‑tick
plan evolution during "thinking steps".

All drawing is performed with OpenCV (`cv2`); multi‑panel figures use
Matplotlib.  Probes are instances of `probes.LinearProbe` (kernel size 1).
The base board image is obtained from `utils.draw_grid` without any arrows,
so that we can overlay the arrows with our own colour scheme.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Internal imports (avoid circular dependencies)
# -----------------------------------------------------------------------------
from utils import draw_grid, set_seed
from probes import LinearProbe

# -----------------------------------------------------------------------------
# Arrow conventions: class index → direction description
# -----------------------------------------------------------------------------
# The classes used throughout the project are:
#   0 = UP
#   1 = DOWN
#   2 = LEFT
#   3 = RIGHT
#   4 = NEVER
#
# Direction vectors for drawing (x, y offsets relative to image coordinate system)
# where x is horizontal (positive right) and y is vertical (positive down).
DIR_VECTORS: Dict[int, Optional[Tuple[int, int]]] = {
    0: (0, -1),   # UP
    1: (0, 1),    # DOWN
    2: (-1, 0),   # LEFT
    3: (1, 0),    # RIGHT
    4: None,      # NEVER
}

# -----------------------------------------------------------------------------
#  PlanVisualizer class
# -----------------------------------------------------------------------------
class PlanVisualizer:
    """
    Creates images of Sokoban boards with overlaid plan arrows decoded from
    internal cell states using trained linear probes.

    Parameters
    ----------
    cell_size : int
        Pixel size of each grid cell (default 20).
    arrow_length_ratio : float
        Length of the arrow shaft as a fraction of `cell_size`.
    agent_arrow_color_bgr : Tuple[int, int, int]
        BGR colour for agent arrows (teal by default: (128,128,0)).
    box_arrow_color_bgr : Tuple[int, int, int]
        BGR colour for box arrows (purple by default: (128,0,128)).
    arrow_thickness : int
        Thickness of the drawn arrows (default 2).
    """

    def __init__(
        self,
        cell_size: int = 20,
        arrow_length_ratio: float = 0.4,
        agent_arrow_color_bgr: Tuple[int, int, int] = (128, 128, 0),
        box_arrow_color_bgr: Tuple[int, int, int] = (128, 0, 128),
        arrow_thickness: int = 2,
    ):
        self.cell_size = cell_size
        self.arrow_length_ratio = arrow_length_ratio
        self.agent_color = agent_arrow_color_bgr
        self.box_color = box_arrow_color_bgr
        self.arrow_thickness = arrow_thickness

    # -------------------------------------------------------------------------
    #  Internal helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _probe_to_arrows(
        cell_state: torch.Tensor,
        probe: LinearProbe,
    ) -> np.ndarray:
        """
        Run a 1×1 linear probe on a single cell state and return the predicted
        class indices for each grid cell.

        Args:
            cell_state: tensor of shape ``(32, 8, 8)`` (single sample, no batch).
            probe: a ``LinearProbe`` instance (must be in eval mode).

        Returns:
            ``np.ndarray`` of shape ``(8, 8)`` with integer class labels
            (0‑4, see DIR_VECTORS).
        """
        # Ensure input has batch dimension (1, C, H, W)
        if cell_state.dim() == 3:
            x = cell_state.unsqueeze(0)
        else:
            x = cell_state

        # Move probe to the same device as the input
        original_device = next(probe.parameters()).device
        if x.device != original_device:
            probe.to(x.device)
        probe.eval()

        with torch.no_grad():
            logits = probe(x)                     # (1, 5, 8, 8)
            preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (8, 8)

        # Restore probe’s device if it was moved
        if x.device != original_device:
            probe.to(original_device)

        return preds.astype(np.int32)

    def _draw_arrow(
        self,
        img_bgr: np.ndarray,
        cx: int,
        cy: int,
        dx: int,
        dy: int,
        colour: Tuple[int, int, int],
    ) -> None:
        """
        Draw a single arrow on the BGR image using OpenCV.

        cx, cy  – centre of the grid cell (pixel coordinates).
        dx, dy  – normalised direction vector; f.e. (0, -1) is up.
        colour  – BGR colour tuple.
        """
        length = int(self.cell_size * self.arrow_length_ratio)
        end_x = int(cx + dx * length)
        end_y = int(cy + dy * length)

        cv2.arrowedLine(
            img_bgr,
            (cx, cy),
            (end_x, end_y),
            colour,
            thickness=self.arrow_thickness,
            tipLength=0.3,
            line_type=cv2.LINE_AA,
        )

    def _overlay_arrows(
        self,
        img_bgr: np.ndarray,
        agent_pred: np.ndarray,
        box_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Draw agent and box arrows on a base BGR image.

        agent_pred, box_pred – (8, 8) arrays of class indices.
        """
        for r in range(8):
            for c in range(8):
                cx = int(c * self.cell_size + self.cell_size / 2)
                cy = int(r * self.cell_size + self.cell_size / 2)

                # Agent arrows
                adir = DIR_VECTORS.get(int(agent_pred[r, c]))
                if adir is not None:
                    self._draw_arrow(img_bgr, cx, cy, *adir, self.agent_color)

                # Box arrows
                bdir = DIR_VECTORS.get(int(box_pred[r, c]))
                if bdir is not None:
                    self._draw_arrow(img_bgr, cx, cy, *bdir, self.box_color)

        return img_bgr

    # -------------------------------------------------------------------------
    #  Main public API
    # -------------------------------------------------------------------------
    def plot_internal_plan(
        self,
        obs: np.ndarray,
        cell_state: torch.Tensor,
        probe_agent: LinearProbe,
        probe_box: LinearProbe,
        title: str = "",
    ) -> np.ndarray:
        """
        Create a single RGB image of the Sokoban board with the agent’s current
        internal plan decoded from one layer/tick.

        Args:
            obs: symbolic observation of shape ``(8, 8, 7)``.
            cell_state: ConvLSTM cell state for one layer, shape ``(32, 8, 8)``.
            probe_agent: 1×1 linear probe trained on ``C_A``.
            probe_box: 1×1 linear probe trained on ``C_B``.
            title: optional title (currently not used in output; can be added externally).

        Returns:
            RGB image as a uint8 numpy array of shape ``(H, W, 3)``.
        """
        # 1. Obtain a clean board image (no arrows) from utils.draw_grid.
        #    We pass empty arrow arrays (all zeros → no arrows drawn).
        empty = np.zeros((8, 8), dtype=np.int32)
        base_rgb = draw_grid(obs, agent_arrows=empty, box_arrows=empty, cell_size=self.cell_size)

        # 2. Convert to BGR for OpenCV drawing
        base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

        # 3. Predict class indices using probes
        agent_pred = self._probe_to_arrows(cell_state, probe_agent)
        box_pred   = self._probe_to_arrows(cell_state, probe_box)

        # 4. Overlay arrows
        plan_bgr = self._overlay_arrows(base_bgr, agent_pred, box_pred)

        # 5. Convert back to RGB and return
        plan_rgb = cv2.cvtColor(plan_bgr, cv2.COLOR_BGR2RGB)
        return plan_rgb

    def plot_across_layers(
        self,
        obs: np.ndarray,
        layer_cell_states: List[torch.Tensor],
        probes_agent_list: List[LinearProbe],
        probes_box_list: List[LinearProbe],
        layer_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a horizontal panel comparing internal plans decoded from different
        ConvLSTM layers.

        Args:
            obs: single observation (8,8,7).
            layer_cell_states: list of cell state tensors, one per layer (each (32,8,8)).
            probes_agent_list: list of probes (C_A) for each layer.
            probes_box_list: list of probes (C_B) for each layer.
            layer_names: optional subplot titles (e.g., ``["Layer 1","Layer 2","Layer 3"]``).
            save_path: if given, save the figure to disk.

        Returns:
            The Matplotlib figure.
        """
        n = len(layer_cell_states)
        if n == 0:
            raise ValueError("layer_cell_states must not be empty")

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
        axes = axes[0]  # 1D array after squeeze

        for i in range(n):
            img = self.plot_internal_plan(
                obs,
                layer_cell_states[i],
                probes_agent_list[i],
                probes_box_list[i],
            )
            axes[i].imshow(img)
            axes[i].axis("off")
            if layer_names and i < len(layer_names):
                axes[i].set_title(layer_names[i])

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_over_ticks(
        self,
        obs_per_tick: List[np.ndarray],
        tick_cell_states_list: List[torch.Tensor],
        probe_agent: LinearProbe,
        probe_box: LinearProbe,
        tick_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize how the internal plan evolves over internal ticks (e.g., during
        "thinking steps").  Creates a horizontal panel with one subplot per tick.

        Args:
            obs_per_tick: list of observations (each (8,8,7)) – usually identical
                          if the agent is stationary, but can vary.
            tick_cell_states_list: list of cell state tensors (each (32,8,8)), one
                                   per tick.
            probe_agent: linear probe for C_A (same layer as the cell states).
            probe_box: linear probe for C_B.
            tick_labels: optional subplot titles (e.g., ``["Tick 1", "Tick 2", ...]``).
            save_path: if given, save the figure to disk.

        Returns:
            The Matplotlib figure.
        """
        n = len(tick_cell_states_list)
        if n == 0:
            raise ValueError("tick_cell_states_list must not be empty")
        if len(obs_per_tick) != n:
            raise ValueError("obs_per_tick and tick_cell_states_list must have same length")

        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
        axes = axes[0]

        for i in range(n):
            img = self.plot_internal_plan(
                obs_per_tick[i],
                tick_cell_states_list[i],
                probe_agent,
                probe_box,
            )
            axes[i].imshow(img)
            axes[i].axis("off")
            if tick_labels and i < len(tick_labels):
                axes[i].set_title(tick_labels[i])

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # -------------------------------------------------------------------------
    #  Standalone arrow extraction (for external use)
    # -------------------------------------------------------------------------
    def get_arrows(
        self,
        cell_state: torch.Tensor,
        probe_agent: LinearProbe,
        probe_box: LinearProbe,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the raw (`C_A`, `C_B`) prediction arrays without rendering an image.

        Returns:
            agent_arrows : (8, 8) int array
            box_arrows   : (8, 8) int array
        """
        agent = self._probe_to_arrows(cell_state, probe_agent)
        box   = self._probe_to_arrows(cell_state, probe_box)
        return agent, box

