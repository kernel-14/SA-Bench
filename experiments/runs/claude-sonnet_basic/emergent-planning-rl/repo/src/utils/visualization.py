"""
Visualization utilities for internal plans and interventions.

Used to create figures like those in the paper showing:
- Internal plans decoded from cell states
- Plan formation over time
- Effect of interventions on plans
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional, Dict
import torch

from probing.concepts import ConceptClass


# Color scheme for Sokoban visualization
SQUARE_COLORS = {
    0: '#2c2c2c',  # wall - dark gray
    1: '#f5f5dc',  # empty - beige
    2: '#8B4513',  # box - brown
    3: '#4169E1',  # agent - blue
    4: '#228B22',  # box on target - green
    5: '#9370DB',  # agent on target - purple
    6: '#FFD700',  # target - gold
}

# Arrow colors for concepts
CA_COLOR = '#00CED1'   # Teal for agent approach direction
CB_COLOR = '#9370DB'   # Purple for box push direction

# Direction to arrow offset
DIRECTION_ARROWS = {
    ConceptClass.UP: (0, 0.3),
    ConceptClass.DOWN: (0, -0.3),
    ConceptClass.LEFT: (-0.3, 0),
    ConceptClass.RIGHT: (0.3, 0),
}

DIRECTION_ARROW_DIRS = {
    ConceptClass.UP: (0, 1),
    ConceptClass.DOWN: (0, -1),
    ConceptClass.LEFT: (-1, 0),
    ConceptClass.RIGHT: (1, 0),
}


def visualize_sokoban_grid(
    grid: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = '',
) -> plt.Axes:
    """
    Visualize a Sokoban grid.
    
    Args:
        grid: 8x8 integer array
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    H, W = grid.shape
    
    for i in range(H):
        for j in range(W):
            color = SQUARE_COLORS.get(grid[i, j], '#ffffff')
            rect = mpatches.Rectangle(
                (j, H - 1 - i), 1, 1,
                linewidth=0.5, edgecolor='gray', facecolor=color
            )
            ax.add_patch(rect)
    
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')
    
    return ax


def visualize_internal_plan(
    grid: np.ndarray,
    ca_labels: np.ndarray,
    cb_labels: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = '',
    show_ca: bool = True,
    show_cb: bool = True,
) -> plt.Axes:
    """
    Visualize the agent's internal plan overlaid on the Sokoban grid.
    
    Args:
        grid: 8x8 integer array
        ca_labels: C_A concept labels (8x8)
        cb_labels: C_B concept labels (8x8)
        ax: Matplotlib axes
        title: Plot title
        show_ca: Whether to show C_A arrows (teal)
        show_cb: Whether to show C_B arrows (purple)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    # Draw grid
    visualize_sokoban_grid(grid, ax, title)
    
    H, W = grid.shape
    
    # Draw arrows for C_A (agent approach direction)
    if show_ca:
        for i in range(H):
            for j in range(W):
                label = ca_labels[i, j]
                if label != ConceptClass.NEVER:
                    dx, dy = DIRECTION_ARROW_DIRS[label]
                    # Arrow center
                    cx = j + 0.5
                    cy = H - 1 - i + 0.5
                    ax.annotate(
                        '', 
                        xy=(cx + dx * 0.3, cy + dy * 0.3),
                        xytext=(cx - dx * 0.3, cy - dy * 0.3),
                        arrowprops=dict(
                            arrowstyle='->', 
                            color=CA_COLOR, 
                            lw=2.0,
                        )
                    )
    
    # Draw arrows for C_B (box push direction)
    if show_cb:
        for i in range(H):
            for j in range(W):
                label = cb_labels[i, j]
                if label != ConceptClass.NEVER:
                    dx, dy = DIRECTION_ARROW_DIRS[label]
                    cx = j + 0.5
                    cy = H - 1 - i + 0.5
                    ax.annotate(
                        '',
                        xy=(cx + dx * 0.3, cy + dy * 0.3),
                        xytext=(cx - dx * 0.3, cy - dy * 0.3),
                        arrowprops=dict(
                            arrowstyle='->',
                            color=CB_COLOR,
                            lw=2.0,
                        )
                    )
    
    return ax


def visualize_plan_formation(
    grids: List[np.ndarray],
    ca_labels_list: List[np.ndarray],
    cb_labels_list: List[np.ndarray],
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = None,
    show_ca: bool = True,
    show_cb: bool = True,
) -> plt.Figure:
    """
    Visualize plan formation over multiple steps/ticks.
    
    Args:
        grids: List of grids at each step
        ca_labels_list: List of C_A labels at each step
        cb_labels_list: List of C_B labels at each step
        titles: Optional list of titles
        figsize: Figure size
        show_ca: Whether to show C_A arrows
        show_cb: Whether to show C_B arrows
        
    Returns:
        Matplotlib figure
    """
    n = len(grids)
    if figsize is None:
        figsize = (4 * n, 4)
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for i, (grid, ca, cb, ax) in enumerate(zip(grids, ca_labels_list, cb_labels_list, axes)):
        title = titles[i] if titles else f'Step {i+1}'
        visualize_internal_plan(grid, ca, cb, ax, title, show_ca, show_cb)
    
    plt.tight_layout()
    return fig


def decode_plan_from_probe(
    cell_state: torch.Tensor,
    probe_ca: 'LinearProbe',
    probe_cb: 'LinearProbe',
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode internal plan from cell state using probes.
    
    Args:
        cell_state: Cell state tensor (hidden_channels, H, W)
        probe_ca: Trained probe for C_A
        probe_cb: Trained probe for C_B
        device: Device
        
    Returns:
        (ca_labels, cb_labels): Decoded concept labels (H, W)
    """
    cell_tensor = cell_state.unsqueeze(0).to(device)
    
    with torch.no_grad():
        ca_pred = probe_ca.predict(cell_tensor).squeeze(0).cpu().numpy()
        cb_pred = probe_cb.predict(cell_tensor).squeeze(0).cpu().numpy()
    
    return ca_pred, cb_pred


def plot_macro_f1_by_layer(
    results: Dict,
    title: str = 'Macro F1 by Layer',
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot macro F1 scores by layer (like Figure 4 in paper).
    
    Args:
        results: Dict with probe results
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for concept_idx, concept in enumerate(['ca', 'cb']):
        ax = axes[concept_idx]
        
        layers = []
        f1_1x1 = []
        f1_3x3 = []
        f1_1x1_std = []
        f1_3x3_std = []
        
        for layer_key in sorted(results.keys()):
            if layer_key.startswith('layer_'):
                layer_num = int(layer_key.split('_')[1])
                layers.append(layer_num)
                
                layer_results = results[layer_key].get(concept, {})
                f1_1x1.append(layer_results.get('1x1', {}).get('mean_f1', 0))
                f1_3x3.append(layer_results.get('3x3', {}).get('mean_f1', 0))
                f1_1x1_std.append(layer_results.get('1x1', {}).get('std_f1', 0))
                f1_3x3_std.append(layer_results.get('3x3', {}).get('std_f1', 0))
        
        # Baseline
        baseline_1x1 = results.get('baseline', {}).get(concept, {}).get('1x1', {}).get('mean_f1', 0)
        baseline_3x3 = results.get('baseline', {}).get(concept, {}).get('3x3', {}).get('mean_f1', 0)
        
        x = np.array(layers)
        
        ax.errorbar(x - 0.1, f1_1x1, yerr=f1_1x1_std, fmt='o-', label='1x1 probe', capsize=3)
        ax.errorbar(x + 0.1, f1_3x3, yerr=f1_3x3_std, fmt='s-', label='3x3 probe', capsize=3)
        ax.axhline(baseline_1x1, linestyle='--', color='gray', alpha=0.7, label='Baseline 1x1')
        ax.axhline(baseline_3x3, linestyle=':', color='gray', alpha=0.7, label='Baseline 3x3')
        
        concept_name = 'C_A (Agent Approach)' if concept == 'ca' else 'C_B (Box Push)'
        ax.set_title(f'{concept_name}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Macro F1')
        ax.set_xticks(layers)
        ax.legend()
        ax.set_ylim(0, 1)
    
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_plan_refinement(
    ca_f1_per_tick: Dict[int, float],
    cb_f1_per_tick: Dict[int, float],
    title: str = 'Plan Refinement During Thinking Steps',
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Plot plan refinement during thinking steps (like Figure 6 in paper).
    
    Args:
        ca_f1_per_tick: C_A F1 at each tick
        cb_f1_per_tick: C_B F1 at each tick
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ticks = sorted(ca_f1_per_tick.keys())
    ca_f1s = [ca_f1_per_tick[t] for t in ticks]
    cb_f1s = [cb_f1_per_tick[t] for t in ticks]
    
    ax.plot(ticks, ca_f1s, 'b-o', label='C_A (Agent Approach)', markersize=4)
    ax.plot(ticks, cb_f1s, 'r-s', label='C_B (Box Push)', markersize=4)
    
    ax.set_xlabel('Internal Tick')
    ax.set_ylabel('Macro F1')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
