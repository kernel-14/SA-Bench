## plotter.py
"""Visualization module for reproducing Figures 1(a), 1(b), and 2 from:
    Murthy et al., "Global Convergence of Policy Gradient in Average Reward
    MDPs", ICLR 2024.

This module is a pure visualization layer with no dependency on MDP logic.
It receives pre-computed results from experiments.py (via main.py) and
renders them into publication-quality figures matching the paper's plots.

Figures produced:
    - Figure 1(a): Average reward vs iteration for three MDP sizes.
      Validates: larger (|S|, |A|) → slower convergence.
    - Figure 1(b): Average reward vs iteration for four reward variance levels.
      Validates: higher reward variance (C_r) → slower convergence.
    - Figure 2: Change in average reward vs iteration for three kernel types.
      Validates: higher kernel diameter (C_p) → slower convergence.

All figures are saved to output_dir with dpi=150 and bbox_inches='tight'.

Configuration values used (from config.yaml):
    plotting.figure_dpi: 150
    plotting.figure_format: "png"
    plotting.figure1a.filename: "figure1a.png"
    plotting.figure1b.filename: "figure1b.png"
    plotting.figure2.filename: "figure2.png"
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
from numpy import ndarray


class Plotter:
    """Renders PPG experiment results into figures matching the paper.

    This class is the sole visualization component of the project. It is
    intentionally decoupled from all MDP logic — it only knows about
    reward histories (lists of floats), complexity metrics (dicts of floats),
    and matplotlib.

    All three public plot methods follow the same pattern:
        1. Create a new figure and axes
        2. Iterate over results in a fixed, explicit order
        3. Plot reward curves with descriptive labels
        4. Set axis labels, title, legend, and grid
        5. Save via _save_figure and close the figure

    Attributes:
        output_dir: Directory path where figures are saved. Must exist
            before calling any plot method (created by Config.__post_init__).
    """

    def __init__(self, output_dir: str = "results/") -> None:
        """Initialize the Plotter with an output directory and matplotlib style.

        Sets the matplotlib style for consistent, publication-quality figures.
        Tries 'seaborn-v0_8' first (matplotlib >= 3.6), falls back to 'ggplot'
        for older versions. Both styles provide clean grids and readable fonts.

        Also ensures the output directory exists as a safety measure, even
        though Config.__post_init__ should have already created it.

        Args:
            output_dir: Directory path for saving figures. Default 'results/'
                matches config.yaml experiment.output_dir. The directory is
                created if it does not exist (idempotent via exist_ok=True).
        """
        self.output_dir: str = output_dir

        # Ensure output directory exists (safety measure; Config also creates it)
        os.makedirs(self.output_dir, exist_ok=True)

        # Set matplotlib style for publication-quality figures
        # 'seaborn-v0_8' is the correct name for matplotlib >= 3.6
        # Fall back to 'ggplot' for older matplotlib versions
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('ggplot')

    def plot_figure1a(self, results: Dict[Tuple[int, int], Dict]) -> None:
        """Plot Figure 1(a): Average reward vs iteration for three MDP sizes.

        Reproduces Figure 1(a) from Section 4 of the paper. Shows that
        larger (|S|, |A|) leads to slower convergence of PPG, validating
        the theoretical bound in Theorem 1 where L_2^Π scales with MDP size.

        The three curves correspond to (S,A) ∈ {(3,3), (9,9), (81,81)},
        all using the same non-uniform kernel and max-variance reward
        construction (Appendix C.1). The y-axis shows absolute average
        reward ρ^{π_k}, which should be monotonically non-decreasing
        (Lemma 5 guarantee).

        Args:
            results: Dictionary keyed by (S, A) tuples. Each value must
                contain at minimum:
                    'reward_history': List[float] of length exp1_iterations.
                        reward_history[k] = ρ^{π_k} (average reward at
                        iteration k, before the (k+1)-th update).
                Optional keys (used if present):
                    'complexity': Dict with 'L2', 'C_m', etc. (not plotted
                        but could be used for annotations).
                    'eta': float — step size used.
                Missing keys are skipped gracefully (no crash).

        Output:
            Saves 'figure1a.png' to self.output_dir with dpi=150.
            The figure shows three curves, one per MDP size, with:
                - X-axis: Iteration number (0 to len(reward_history)-1)
                - Y-axis: Average reward ρ^{π_k}
                - Legend: '|S|=3, |A|=3', '|S|=9, |A|=9', '|S|=81, |A|=81'
                - Title: 'Figure 1(a): Convergence vs State/Action Space Size'
        """
        # Create figure and axes
        fig: matplotlib.figure.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=(8, 5))

        # Fixed iteration order for consistent plotting (smallest to largest)
        # Do not rely on dict ordering — use explicit order matching the paper.
        plot_order: List[Tuple[int, int]] = [(3, 3), (9, 9), (81, 81)]

        for size_pair in plot_order:
            # Skip if this MDP size was not computed (e.g., experiment skipped)
            if size_pair not in results:
                continue

            entry: Dict = results[size_pair]

            # Extract reward history — required field
            if 'reward_history' not in entry:
                continue

            reward_history: List[float] = entry['reward_history']
            if len(reward_history) == 0:
                continue

            S: int = size_pair[0]
            A: int = size_pair[1]

            # Build x-axis (iteration indices) and y-axis (reward values)
            x: ndarray = np.arange(len(reward_history))
            y: ndarray = np.array(reward_history, dtype=np.float64)

            # Label matching the paper's Figure 1(a) description
            label: str = f'|S|={S}, |A|={A}'

            ax.plot(x, y, label=label, linewidth=1.5)

        # Axis labels from config.yaml plotting.figure1a
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title(
            'Figure 1(a): Convergence vs State/Action Space Size',
            fontsize=13,
            fontweight='bold',
        )

        # Legend and grid for readability
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.4)

        # Tight layout to prevent label clipping
        fig.tight_layout()

        # Save and close
        self._save_figure(fig, 'figure1a.png')
        plt.close(fig)

    def plot_figure1b(self, results: Dict[str, Dict]) -> None:
        """Plot Figure 1(b): Average reward vs iteration for four reward variances.

        Reproduces Figure 1(b) from Section 4 of the paper. Shows that
        higher reward variance (larger C_r) leads to slower convergence,
        validating Theorem 1's dependence on C_r through L_2^Π.

        All four curves use the same (16,16) MDP with a shared random
        Dirichlet transition kernel (Appendix C.2). The only difference
        is the reward function's variance level. The y-axis shows absolute
        average reward ρ^{π_k}.

        Args:
            results: Dictionary keyed by variance level strings. Each value
                must contain at minimum:
                    'reward_history': List[float] of length exp2_iterations.
                        reward_history[k] = ρ^{π_k}.
                Valid keys: 'no_variance', 'low_variance', 'high_variance',
                    'max_variance'. Missing keys are skipped gracefully.
                Optional keys:
                    'label': str — human-readable label (used if present,
                        otherwise falls back to the built-in label map).
                    'complexity': Dict with 'C_r', etc.

        Output:
            Saves 'figure1b.png' to self.output_dir with dpi=150.
            The figure shows four curves with:
                - X-axis: Iteration number
                - Y-axis: Average reward ρ^{π_k}
                - Legend: 'No Variance', 'Low Variance', 'High Variance',
                          'Max Variance' (ordered from fastest to slowest)
                - Title: 'Figure 1(b): Convergence vs Reward Variance (C_r)'
        """
        # Create figure and axes
        fig: matplotlib.figure.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=(8, 5))

        # Fixed iteration order: fastest to slowest convergence
        # This ordering matches the paper's visual presentation.
        plot_order: List[str] = [
            'no_variance',
            'low_variance',
            'high_variance',
            'max_variance',
        ]

        # Built-in label map (used when 'label' key is absent from results entry)
        label_map: Dict[str, str] = {
            'no_variance':   'No Variance',
            'low_variance':  'Low Variance',
            'high_variance': 'High Variance',
            'max_variance':  'Max Variance',
        }

        for key in plot_order:
            # Skip if this variance level was not computed
            if key not in results:
                continue

            entry: Dict = results[key]

            # Extract reward history — required field
            if 'reward_history' not in entry:
                continue

            reward_history: List[float] = entry['reward_history']
            if len(reward_history) == 0:
                continue

            # Build x-axis and y-axis arrays
            x: ndarray = np.arange(len(reward_history))
            y: ndarray = np.array(reward_history, dtype=np.float64)

            # Use label from results entry if available, else use built-in map
            label: str = entry.get('label', label_map.get(key, key))

            ax.plot(x, y, label=label, linewidth=1.5)

        # Axis labels from config.yaml plotting.figure1b
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title(
            'Figure 1(b): Convergence vs Reward Variance (C_r)',
            fontsize=13,
            fontweight='bold',
        )

        # Legend and grid
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.4)

        fig.tight_layout()

        # Save and close
        self._save_figure(fig, 'figure1b.png')
        plt.close(fig)

    def plot_figure2(self, results: Dict[str, Dict]) -> None:
        """Plot Figure 2: Change in average reward vs iteration for three kernels.

        Reproduces Figure 2 from Section 4 of the paper. Shows that higher
        kernel diameter (larger C_p) leads to slower convergence, validating
        Theorem 1's dependence on C_p through L_2^Π.

        All three curves use the same (16,16) MDP with a shared high-variance
        reward function (Appendix C.3). The y-axis shows the *change* in
        average reward: ρ^{π_k} - ρ^{π_0}, i.e., improvement over the
        initial uniform policy. This matches the paper's caption: "overall
        change in average reward as a function of iterations."

        Args:
            results: Dictionary keyed by kernel type strings. Each value
                must contain at minimum:
                    'reward_history': List[float] of length exp3_iterations.
                        reward_history[k] = ρ^{π_k}.
                Valid keys: 'uniform', 'nonuniform', 'deterministic'.
                    Missing keys are skipped gracefully.
                Optional keys:
                    'C_p': float — estimated C_p constant. If present,
                        included in the legend label for quantitative
                        validation of the theoretical claim.

        Output:
            Saves 'figure2.png' to self.output_dir with dpi=150.
            The figure shows three curves with:
                - X-axis: Iteration number (0 to exp3_iterations-1)
                - Y-axis: Change in average reward ρ^{π_k} - ρ^{π_0}
                - Legend: 'Uniform (C_p=X.XXX)', 'Non-uniform (C_p=X.XXX)',
                          'Deterministic (C_p=X.XXX)' (C_p shown if available)
                - Title: 'Figure 2: Convergence vs Transition Kernel (C_p)'
        """
        # Create figure and axes
        fig: matplotlib.figure.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=(8, 5))

        # Fixed iteration order: expected fastest to slowest convergence
        # Uniform (C_p ≈ 0) → Non-uniform → Deterministic (highest C_p)
        plot_order: List[str] = ['uniform', 'nonuniform', 'deterministic']

        # Built-in base label map (C_p appended dynamically if available)
        base_label_map: Dict[str, str] = {
            'uniform':       'Uniform',
            'nonuniform':    'Non-uniform',
            'deterministic': 'Deterministic',
        }

        for key in plot_order:
            # Skip if this kernel type was not computed
            if key not in results:
                continue

            entry: Dict = results[key]

            # Extract reward history — required field
            if 'reward_history' not in entry:
                continue

            reward_history: List[float] = entry['reward_history']
            if len(reward_history) == 0:
                continue

            # Compute change in average reward: ρ^{π_k} - ρ^{π_0}
            # reward_history[0] is the initial reward under the uniform policy.
            rewards_array: ndarray = np.array(reward_history, dtype=np.float64)
            initial_reward: float = float(rewards_array[0])
            delta_y: ndarray = rewards_array - initial_reward

            # Build x-axis (iteration indices)
            x: ndarray = np.arange(len(reward_history))

            # Build label: include C_p value if available in results entry
            base_label: str = base_label_map.get(key, key)
            c_p_value: Optional[float] = entry.get('C_p', None)

            if c_p_value is not None:
                label: str = f'{base_label} (C_p={c_p_value:.3f})'
            else:
                label = base_label

            ax.plot(x, delta_y, label=label, linewidth=1.5)

        # Axis labels from config.yaml plotting.figure2
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Change in Average Reward', fontsize=12)
        ax.set_title(
            'Figure 2: Convergence vs Transition Kernel (C_p)',
            fontsize=13,
            fontweight='bold',
        )

        # Legend and grid
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.4)

        fig.tight_layout()

        # Save and close
        self._save_figure(fig, 'figure2.png')
        plt.close(fig)

    def plot_complexity_table(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Print a formatted table of MDP complexity constants to stdout.

        Displays the complexity constants C_m, C_p, C_r, κ_r, L_2^Π, and
        η_max for each experiment configuration. This is diagnostic output
        for validating that the computed constants match theoretical
        expectations from Table 1/2 of the paper.

        This method prints to stdout rather than saving a figure because
        the complexity table is diagnostic information, not a paper figure.

        Expected theoretical bounds (Lemma 18):
            ||Φ||_∞ ≤ 2,  κ_r ≤ 2,  C_p ≤ √|A|,  C_r ≤ √|A|

        Args:
            metrics: Dictionary keyed by configuration labels (e.g.,
                '|S|=3,|A|=3', 'no_variance', 'uniform'). Each value is
                a dict with keys:
                    'C_m':     float — max_π ||(I - Φ P^π)^{-1}||_∞
                    'C_p':     float — max_{π,π'} ||P^{π'}-P^π||_∞/||π'-π||_2
                    'C_r':     float — max_{π,π'} ||r^{π'}-r^π||_∞/||π'-π||_2
                    'kappa_r': float — max_π ||Φ r^π||_∞
                    'L2':      float — L_2^Π restricted smoothness constant
                    'eta_max': float — 1/L_2^Π (maximum valid step size)
                Missing keys are displayed as 'N/A'.

        Output:
            Prints a formatted table to stdout. Example:
            ┌─────────────────┬────────┬────────┬────────┬─────────┬────────┬─────────┐
            │ Config          │  C_m   │  C_p   │  C_r   │ kappa_r │   L2   │ eta_max │
            ├─────────────────┼────────┼────────┼────────┼─────────┼────────┼─────────┤
            │ |S|=3,|A|=3     │ 2.1234 │ 0.5678 │ 0.9012 │  1.2345 │ 3.4567 │  0.2893 │
            └─────────────────┴────────┴────────┴────────┴─────────┴────────┴─────────┘
        """
        if not metrics:
            print("  [Complexity Table] No metrics to display.")
            return

        # Column headers
        headers: List[str] = ['Config', 'C_m', 'C_p', 'C_r', 'kappa_r', 'L2', 'eta_max']

        # Column widths (characters)
        col_widths: Dict[str, int] = {
            'Config':  max(16, max(len(k) for k in metrics.keys()) + 2),
            'C_m':     8,
            'C_p':     8,
            'C_r':     8,
            'kappa_r': 9,
            'L2':      10,
            'eta_max': 10,
        }

        # Helper to format a float value with 4 decimal places
        def fmt_val(val: Optional[float]) -> str:
            """Format a float value for table display."""
            if val is None:
                return 'N/A'
            if val == float('inf'):
                return 'inf'
            return f'{val:.4f}'

        # Build separator line
        sep_parts: List[str] = ['-' * col_widths[h] for h in headers]
        separator: str = '-+-'.join(sep_parts)
        separator = '-' + separator + '-'

        # Build header line
        header_parts: List[str] = [
            h.center(col_widths[h]) for h in headers
        ]
        header_line: str = ' | '.join(header_parts)

        # Print table
        print()
        print('  MDP Complexity Constants (Table 1/2 of the paper):')
        print('  ' + separator)
        print('  ' + header_line)
        print('  ' + separator)

        for config_label, metric_dict in metrics.items():
            # Extract values with fallback to None for missing keys
            c_m: Optional[float] = metric_dict.get('C_m', None)
            c_p: Optional[float] = metric_dict.get('C_p', None)
            c_r: Optional[float] = metric_dict.get('C_r', None)
            kappa_r: Optional[float] = metric_dict.get('kappa_r', None)
            l2: Optional[float] = metric_dict.get('L2', None)
            eta_max: Optional[float] = metric_dict.get('eta_max', None)

            # Format each cell
            row_values: List[str] = [
                config_label.ljust(col_widths['Config']),
                fmt_val(c_m).center(col_widths['C_m']),
                fmt_val(c_p).center(col_widths['C_p']),
                fmt_val(c_r).center(col_widths['C_r']),
                fmt_val(kappa_r).center(col_widths['kappa_r']),
                fmt_val(l2).center(col_widths['L2']),
                fmt_val(eta_max).center(col_widths['eta_max']),
            ]

            row_line: str = ' | '.join(row_values)
            print('  ' + row_line)

        print('  ' + separator)
        print()

    def _save_figure(
        self,
        fig: matplotlib.figure.Figure,
        filename: str,
    ) -> None:
        """Save a matplotlib figure to the output directory.

        Centralizes figure saving logic so that dpi, bbox_inches, and
        path construction are defined in one place. If the output format
        needs to change (e.g., to PDF), only this method needs updating.

        The method does NOT close the figure — the caller is responsible
        for calling plt.close(fig) after _save_figure returns. This
        separation allows the caller to perform additional operations on
        the figure (e.g., display it interactively) before closing.

        Args:
            fig: The matplotlib Figure object to save. Must be a valid,
                non-closed figure. The figure is saved as-is without any
                additional modifications.
            filename: Filename for the saved figure (e.g., 'figure1a.png').
                The full path is constructed as os.path.join(output_dir,
                filename). The file extension determines the format
                (matplotlib infers format from extension).

        Output:
            Saves the figure to os.path.join(self.output_dir, filename)
            with dpi=150 (from config.yaml plotting.figure_dpi) and
            bbox_inches='tight' (prevents legend/label clipping).

        Note:
            Calls os.makedirs(self.output_dir, exist_ok=True) as a safety
            measure before saving, in case the directory was deleted after
            Config.__post_init__ created it.
        """
        # Safety: ensure output directory exists before saving
        os.makedirs(self.output_dir, exist_ok=True)

        # Construct full file path
        filepath: str = os.path.join(self.output_dir, filename)

        # Save with settings from config.yaml plotting section:
        #   figure_dpi: 150
        #   bbox_inches: 'tight' (prevents label/legend clipping)
        fig.savefig(
            filepath,
            dpi=150,
            bbox_inches='tight',
            facecolor='white',  # Ensure white background for all styles
        )

        print(f"  [Plotter] Saved figure: {filepath}")
