## visualization.py

"""
Visualisation module for the numerical convergence experiment of the
randomized midpoint diffusion sampler.

Generates log‑log plots of the KL divergence versus the total number of
iterations T for each (d,k) configuration, matching the style of Figure 2
in the paper.  A reference line corresponding to the theoretical
O(log^4 T / T^3) rate – i.e., slope –3 on a log‑log scale – is added for
visual comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any


class Visualizer:
    """
    Provides a static method to create and optionally save convergence plots.
    """

    @staticmethod
    def plot_kl_vs_T(
        results: Dict[Tuple[int, int], Dict[str, Any]],
        save_path: Optional[str] = None,
        reference_slope: float = -3.0,          # matches config.yaml default
        dpi: int = 150,                         # matches config.yaml default
        fmt: str = "png",                       # matches config.yaml default
        figure_size_per_subplot: Tuple[float, float] = (5.0, 4.0),
    ) -> plt.Figure:
        """
        Generate a tiled figure of log‑log plots of KL divergence versus T.

        Args:
            results: dictionary mapping (d, k) tuples to experiment results
                containing at least {'T': list[int], 'KL': list[float]}.
            save_path: if not None, the figure is saved to this path using
                the specified format and dpi.
            reference_slope: slope of the theoretical rate line (default –3).
            dpi: resolution for saved figure (default 150).
            fmt: file format for saving (default 'png').
            figure_size_per_subplot: (width, height) in inches for each subplot;
                total figure size scales with the number of subplots.

        Returns:
            The matplotlib Figure object.
        """
        # --- Prepare data: sort configurations for consistent layout ---
        sorted_keys = sorted(results.keys(), key=lambda dk: (dk[0], dk[1]))
        if not sorted_keys:
            print("No data to plot.")   # pragma: no cover
            return None

        # --- Create subplots ---
        ncols = len(sorted_keys)
        fig, axes = plt.subplots(
            1, ncols,
            figsize=(figure_size_per_subplot[0] * ncols, figure_size_per_subplot[1]),
            squeeze=False,
        )

        for idx, (d, k) in enumerate(sorted_keys):
            ax = axes[0, idx]
            data = results[(d, k)]

            # Extract and sanitise data
            T_vals = np.asarray(data['T'], dtype=np.float64)
            KL_vals = np.asarray(data['KL'], dtype=np.float64)

            # Remove any non‑positive KL values to avoid log10 issues
            mask = KL_vals > 0.0
            T_vals = T_vals[mask]
            KL_vals = KL_vals[mask]

            if len(T_vals) == 0:
                ax.set_title(f"d={d}, k={k}  (no valid data)")
                continue

            # --- Empirical curve ---
            ax.loglog(T_vals, KL_vals, "o-", color="blue", label="Empirical")

            # --- Reference slope line (slope = reference_slope) ---
            # Anchor the line at the first data point
            T0 = T_vals[0]
            KL0 = KL_vals[0]
            # Generate a smooth log‑spaced array from T0 to max T
            T_ref = np.logspace(
                np.log10(T0), np.log10(T_vals[-1]), num=200
            )
            KL_ref = KL0 * (T_ref / T0) ** reference_slope
            ax.loglog(
                T_ref, KL_ref,
                "--", color="black",
                label=f"Slope {reference_slope}",
            )

            # --- Labels, title, grid, legend ---
            ax.set_xlabel("Number of iterations T")
            ax.set_ylabel("KL divergence")
            ax.set_title(f"d = {d}, k = {k}")
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            ax.legend()

        plt.tight_layout()

        # --- Save or display ---
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, format=fmt, bbox_inches="tight")
        else:
            plt.show()   # pragma: no cover (interactive only)

        return fig


# ---------------------------------------------------------------------------
# Example usage (if run as a standalone script for testing purposes)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy results to verify the plotting code
    dummy_results = {
        (10, 10): {
            "T": [100, 200, 500, 1000, 2000, 5000],
            "KL": [0.5, 0.12, 0.03, 0.008, 0.002, 0.0005],
        },
        (100, 10): {
            "T": [100, 200, 500, 1000, 2000, 5000],
            "KL": [2.0, 0.8, 0.2, 0.05, 0.015, 0.004],
        },
        (500, 100): {
            "T": [100, 200, 500, 1000, 2000],
            "KL": [5.0, 1.5, 0.4, 0.1, 0.03],
        },
    }
    Visualizer.plot_kl_vs_T(dummy_results, save_path=None)  # will display
