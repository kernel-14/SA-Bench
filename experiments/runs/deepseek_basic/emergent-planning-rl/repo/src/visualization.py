"""
Visualization Utilities for Plan Analysis.

Provides functions to visualize:
- Internal plans (C_A and C_B decoded from src.probes)
- Plan formation over time
- Intervention effects
- Probe performance metrics (Figure 4 style)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
from src.probes import ConceptClasses


# Color scheme matching paper
COLOR_CA = '#008080'      # Teal for agent approach direction
COLOR_CB = '#800080'      # Purple for box push direction
COLOR_NEVER = '#cccccc'   # Gray for NEVER
COLOR_WALL = '#333333'    # Dark gray for walls
COLOR_EMPTY = '#ffffff'   # White for empty
COLOR_BOX = '#8B4513'     # Brown for boxes
COLOR_AGENT = '#FFD700'   # Gold for agent
COLOR_TARGET = '#FF6347'  # Tomato for targets

# Direction arrow angles
DIRECTION_ANGLES = {
    0: None,      # NEVER - no arrow
    1: 90,        # UP
    2: -90,       # DOWN
    3: 180,       # LEFT
    4: 0,         # RIGHT
}


def visualize_internal_plan(
    level: np.ndarray,
    ca_predictions: Optional[np.ndarray] = None,
    cb_predictions: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = 'Internal Plan',
    show_agent: bool = True,
    show_boxes: bool = True,
    show_targets: bool = True,
    arrow_scale: float = 0.3,
) -> plt.Axes:
    """
    Visualize an internal plan as decoded from src.probes.
    
    This replicates Figure 5 / Figures 10-12 from the paper.
    Teal arrows = C_A (agent approach direction)
    Purple arrows = C_B (box push direction)
    
    Args:
        level: (8, 8) integer array of the Sokoban board
        ca_predictions: (8, 8) integer array of C_A predictions (optional)
        cb_predictions: (8, 8) integer array of C_B predictions (optional)
        ax: Matplotlib axes to draw on (creates new if None)
        title: Plot title
        show_agent: Whether to show agent
        show_boxes: Whether to show boxes
        show_targets: Whether to show targets
        arrow_scale: Scale factor for arrow size
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Draw background grid
    for y in range(8):
        for x in range(8):
            val = level[y, x]
            
            # Determine background color
            if val in (0,):  # WALL
                color = COLOR_WALL
            else:
                color = COLOR_EMPTY
            
            rect = plt.Rectangle((x, 7 - y), 1, 1, facecolor=color, edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
            
            # Draw entities
            if val in (4, 5) and show_targets:  # BOX_ON_TARGET, AGENT_ON_TARGET
                # Target marker
                circle = plt.Circle((x + 0.5, 7 - y + 0.5), 0.15, facecolor=COLOR_TARGET, edgecolor='none')
                ax.add_patch(circle)
            elif val == 6 and show_targets:  # TARGET
                circle = plt.Circle((x + 0.5, 7 - y + 0.5), 0.15, facecolor=COLOR_TARGET, edgecolor='none')
                ax.add_patch(circle)
            
            if val in (2, 4) and show_boxes:  # BOX, BOX_ON_TARGET
                rect = plt.Rectangle((x + 0.15, 7 - y + 0.15), 0.7, 0.7, 
                                     facecolor=COLOR_BOX, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
            
            if val in (3, 5) and show_agent:  # AGENT, AGENT_ON_TARGET
                circle = plt.Circle((x + 0.5, 7 - y + 0.5), 0.35, 
                                   facecolor=COLOR_AGENT, edgecolor='black', linewidth=1)
                ax.add_patch(circle)
    
    # Draw C_A arrows (teal)
    if ca_predictions is not None:
        for y in range(8):
            for x in range(8):
                direction = ca_predictions[y, x]
                if direction != 0:  # Not NEVER
                    angle = DIRECTION_ANGLES.get(direction)
                    if angle is not None:
                        dx = np.cos(np.radians(angle)) * arrow_scale
                        dy = np.sin(np.radians(angle)) * arrow_scale
                        ax.arrow(x + 0.5 - dx/2, 7 - y + 0.5 - dy/2, dx, dy,
                                head_width=0.2, head_length=0.2, fc=COLOR_CA, ec=COLOR_CA,
                                alpha=0.8, linewidth=1.5)
    
    # Draw C_B arrows (purple)
    if cb_predictions is not None:
        for y in range(8):
            for x in range(8):
                direction = cb_predictions[y, x]
                if direction != 0:  # Not NEVER
                    angle = DIRECTION_ANGLES.get(direction)
                    if angle is not None:
                        dx = np.cos(np.radians(angle)) * arrow_scale
                        dy = np.sin(np.radians(angle)) * arrow_scale
                        ax.arrow(x + 0.5 - dx/2, 7 - y + 0.5 - dy/2, dx, dy,
                                head_width=0.2, head_length=0.2, fc=COLOR_CB, ec=COLOR_CB,
                                alpha=0.8, linewidth=1.5)
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    
    # Legend
    legend_patches = []
    if ca_predictions is not None:
        legend_patches.append(mpatches.Patch(color=COLOR_CA, label='Agent Approach (C_A)'))
    if cb_predictions is not None:
        legend_patches.append(mpatches.Patch(color=COLOR_CB, label='Box Push (C_B)'))
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    return ax


def visualize_plan_formation(
    plan_sequence: List[Dict],
    level: np.ndarray,
    ncols: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Visualize how an internal plan develops over time/ticks.
    
    Replicates plan formation figures (e.g., Figures 13-17).
    
    Args:
        plan_sequence: List of dicts with 'ca_pred' and/or 'cb_pred' and 'time_label'
        level: The Sokoban level
        ncols: Number of columns in the grid
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    nrows = (len(plan_sequence) + ncols - 1) // ncols
    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, plan in enumerate(plan_sequence):
        if i < len(axes):
            visualize_internal_plan(
                level,
                ca_predictions=plan.get('ca_pred'),
                cb_predictions=plan.get('cb_pred'),
                ax=axes[i],
                title=plan.get('time_label', f'Step {i}'),
            )
    
    # Hide unused axes
    for i in range(len(plan_sequence), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_probe_performance_bar(
    results: Dict[str, Dict[int, float]],
    title: str = 'Probe Macro F1 Scores',
    baseline_key: str = 'baseline',
) -> plt.Figure:
    """
    Plot probe performance as a bar chart (Figure 4 style).
    
    Args:
        results: Dict mapping '1x1'/'3x3'/'baseline' -> {layer: macro_f1}
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    probe_types = list(results.keys())
    layers = sorted(results[probe_types[0]].keys())
    
    x = np.arange(len(layers))
    width = 0.8 / len(probe_types)
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    
    for i, pt in enumerate(probe_types):
        values = [results[pt][l] for l in layers]
        ax.bar(x + i * width, values, width, label=pt, color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Macro F1')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(probe_types) - 1) / 2)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def plot_test_time_refinement(
    tick_f1s: Dict[int, float],
    title: str = 'Test-Time Plan Refinement',
    color: str = '#2196F3',
) -> plt.Figure:
    """
    Plot how macro F1 improves over internal ticks during thinking steps.
    
    Replicates Figure 6 / Appendix A.3.1.
    
    Args:
        tick_f1s: Dict mapping tick_number -> macro_f1
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    ticks = sorted(tick_f1s.keys())
    values = [tick_f1s[t] for t in ticks]
    
    ax.plot(ticks, values, 'o-', color=color, linewidth=2, markersize=6)
    ax.fill_between(ticks, values, alpha=0.3, color=color)
    
    ax.set_xlabel('Internal Tick')
    ax.set_ylabel('Macro F1')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_emergence_correlation(
    ca_f1s: List[float],
    cb_f1s: List[float],
    extra_solved: List[float],
    checkpoints: List[int],
    title: str = 'Emergence of Planning',
) -> plt.Figure:
    """
    Plot the correlation between probe F1 and planning-like behavior.
    
    Replicates Figure 9 / Appendix C.3.
    
    Args:
        ca_f1s: C_A macro F1 values
        cb_f1s: C_B macro F1 values
        extra_solved: Additional levels solved with thinking steps
        checkpoints: Training checkpoints
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # C_A correlation
    ax1.scatter(ca_f1s, extra_solved, alpha=0.6, color=COLOR_CA)
    ax1.set_xlabel('Macro F1 (C_A)')
    ax1.set_ylabel('Extra Levels Solved (%)')
    ax1.set_title('C_A vs Planning Behavior')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation line
    if len(ca_f1s) > 1:
        z = np.polyfit(ca_f1s, extra_solved, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(ca_f1s), max(ca_f1s), 100)
        ax1.plot(x_line, p(x_line), '--', color='red', alpha=0.7)
    
    # C_B correlation
    ax2.scatter(cb_f1s, extra_solved, alpha=0.6, color=COLOR_CB)
    ax2.set_xlabel('Macro F1 (C_B)')
    ax2.set_ylabel('Extra Levels Solved (%)')
    ax2.set_title('C_B vs Planning Behavior')
    ax2.grid(True, alpha=0.3)
    
    if len(cb_f1s) > 1:
        z = np.polyfit(cb_f1s, extra_solved, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(cb_f1s), max(cb_f1s), 100)
        ax2.plot(x_line, p(x_line), '--', color='red', alpha=0.7)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def visualize_intervention_effect(
    level: np.ndarray,
    plan_before: np.ndarray,
    plan_after: np.ndarray,
    intervention_positions: List[Tuple[int, int]],
    intervention_type: str = 'CA',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the effect of an intervention on the agent's plan.
    
    Replicates Figures 7, 8 and Appendix B.1 figures.
    
    Args:
        level: Sokoban level
        plan_before: (8, 8) predictions before intervention
        plan_after: (8, 8) predictions after intervention
        intervention_positions: List of (y, x) positions intervened on
        intervention_type: 'CA' or 'CB'
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Before
    if intervention_type == 'CA':
        visualize_internal_plan(level, ca_predictions=plan_before, ax=axes[0],
                               title='Plan Without Intervention')
        visualize_internal_plan(level, ca_predictions=plan_after, ax=axes[2],
                               title='Plan With Intervention')
    else:
        visualize_internal_plan(level, cb_predictions=plan_before, ax=axes[0],
                               title='Plan Without Intervention')
        visualize_internal_plan(level, cb_predictions=plan_after, ax=axes[2],
                               title='Plan With Intervention')
    
    # Intervention map
    ax = axes[1]
    visualize_internal_plan(level, ax=ax, title='Intervention')
    for y, x in intervention_positions:
        rect = plt.Rectangle((x, 7 - y), 1, 1, facecolor='none', 
                            edgecolor='white', linewidth=3, linestyle='-')
        ax.add_patch(rect)
        ax.plot(x + 0.5, 7 - y + 0.5, 'x', color='white', markersize=12, markeredgewidth=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_intervention_success_rates(
    success_rates: Dict[int, Dict[str, float]],
    title: str = 'Intervention Success Rates',
) -> plt.Figure:
    """
    Plot intervention success rates across layers.
    
    Replicates Table 1 / Figures 28-31 style.
    
    Args:
        success_rates: Dict mapping layer -> {'trained': rate, 'random': rate}
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    layers = sorted(success_rates.keys())
    trained = [success_rates[l]['trained'] for l in layers]
    random = [success_rates[l]['random'] for l in layers]
    
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, trained, width, label='Trained Probe', color='#4CAF50')
    ax.bar(x + width/2, random, width, label='Random Probe', color='#FF5722')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.legend()
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    return fig
