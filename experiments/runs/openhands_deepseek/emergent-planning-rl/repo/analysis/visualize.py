"""
Visualization utilities for internal plans decoded from DRC agent cell states.

Creates visualizations similar to Figures 1, 3, 5, 7, 8, etc. in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional

from ..environment.sokoban import (
    E_WALL, E_EMPTY, E_BOX, E_AGENT, E_TARGET,
    E_BOX_ON_TARGET, E_AGENT_ON_TARGET,
)

# Colors for Sokoban elements
COLORS = {
    E_WALL: "#333333",
    E_EMPTY: "#F5F5DC",
    E_BOX: "#CD853F",
    E_AGENT: "#4169E1",
    E_TARGET: "#FF6B6B",
    E_BOX_ON_TARGET: "#2ECC40",
    E_AGENT_ON_TARGET: "#4169E1",
}

# Arrow colors for concepts
TEAL = "#20B2AA"  # Agent approach
PURPLE = "#9370DB"  # Box push

# Direction vectors for arrows
DIR_VECTORS = {
    "UP": (0, -0.35),
    "DOWN": (0, 0.35),
    "LEFT": (-0.35, 0),
    "RIGHT": (0.35, 0),
}


def draw_grid(ax, grid: np.ndarray, cmap=None):
    """Draw Sokoban grid."""
    H, W = grid.shape
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")

    for r in range(H):
        for c in range(W):
            e = grid[r, c]
            color = COLORS.get(e, "#FFFFFF")
            rect = patches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                linewidth=1, edgecolor="black",
                facecolor=color,
            )
            ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])


def draw_arrows(ax, concept_labels: np.ndarray, color: str = TEAL):
    """
    Draw arrows for concept predictions.

    Args:
        concept_labels: (H, W) array with class indices 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=NEVER
        color: arrow color
    """
    H, W = concept_labels.shape
    for r in range(H):
        for c in range(W):
            cls = concept_labels[r, c]
            if cls == 4:  # NEVER
                continue
            direction = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}[cls]
            dx, dy = DIR_VECTORS[direction]
            ax.arrow(
                c, r, dx, dy,
                head_width=0.2, head_length=0.15,
                fc=color, ec=color, linewidth=2,
                length_includes_head=True,
            )


def visualize_plan(
    grid: np.ndarray,
    agent_approach: Optional[np.ndarray] = None,
    box_push: Optional[np.ndarray] = None,
    title: str = "Internal Plan",
    save_path: Optional[str] = None,
):
    """
    Visualize the agent's internal plan as decoded from cell states.

    Args:
        grid: Sokoban grid (H, W)
        agent_approach: (H, W) C_A concept predictions (teal arrows)
        box_push: (H, W) C_B concept predictions (purple arrows)
        title: plot title
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    draw_grid(ax, grid)

    if box_push is not None:
        draw_arrows(ax, box_push, color=PURPLE)
    if agent_approach is not None:
        draw_arrows(ax, agent_approach, color=TEAL)

    ax.set_title(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def visualize_probe_predictions(
    grid: np.ndarray,
    predictions: np.ndarray,
    concept_type: str = "agent_approach",
    title: str = "Probe Predictions",
    save_path: Optional[str] = None,
):
    """
    Visualize probe predictions over a grid.

    Args:
        grid: Sokoban grid
        predictions: (H, W) array of predicted classes
        concept_type: "agent_approach" or "box_push"
        title: plot title
        save_path: optional save path
    """
    color = TEAL if concept_type == "agent_approach" else PURPLE
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    draw_grid(ax, grid)
    draw_arrows(ax, predictions, color=color)
    ax.set_title(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def visualize_plan_evolution(
    grids: List[np.ndarray],
    predictions_list: List[np.ndarray],
    concept_type: str = "box_push",
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize plan evolution over time steps or internal ticks.

    Args:
        grids: list of Sokoban grids at each step
        predictions_list: list of (H, W) prediction arrays
        concept_type: "agent_approach" or "box_push"
        titles: optional titles for each subplot
        save_path: optional save path
    """
    n = len(grids)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    axes = axes.flatten() if hasattr(axes, "flatten") else axes

    color = TEAL if concept_type == "agent_approach" else PURPLE

    for i in range(n):
        ax = axes[i]
        draw_grid(ax, grids[i])
        draw_arrows(ax, predictions_list[i], color=color)
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
