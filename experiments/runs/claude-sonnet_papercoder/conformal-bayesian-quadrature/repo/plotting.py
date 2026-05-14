## plotting.py
"""Visualization functions for reproducing Figures 3 and 4 from the paper.

This module is a pure rendering layer. All numerical computation happens
upstream in the experiment runners and cbq_core. Functions here accept
pre-computed arrays and produce publication-quality figures.

Figures reproduced:
  - Figure 3: Side-by-side histograms of chosen λ values for CRC and CBQ-HPD
    across M=10,000 trials, with the risk-exceeding region highlighted in red.
  - Figure 4: Probability density of L⁺ for λ ∈ {0.7, 0.8, 0.9} estimated
    from 100,000 Dirichlet samples via KDE.
  - Posterior illustration: Multiple sampled worst-case quantile functions
    with 95% credible interval band (Figures 1/2 style).

Config values used (passed as arguments from main.py, sourced from config.yaml):
  - plotting.dpi = 150
  - plotting.hist_bins = 50
  - plotting.hist_alpha = 0.7
  - plotting.figure3_path = "results/fig3_lambda_histograms.png"
  - plotting.figure4_path = "results/fig4_L_plus_density.png"
  - exp1_synthetic_binomial.alpha = 0.4
  - exp1_synthetic_binomial.risk_threshold = 0.6
  - cbq.n_mc_figure = 100000

References:
    Paper Figure 3: Histogram of λ_crc and λ_hpd^0.95 across 10,000 trials.
    Paper Figure 4: Density of L⁺ for λ ∈ {0.7, 0.8, 0.9} with 100,000 samples.
    Paper Figures 1/2: Illustration of posterior quantile function distribution.
    config.yaml: plotting.*, exp1_synthetic_binomial.*, cbq.n_mc_figure.
"""

from typing import Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import cbq_core


# ---------------------------------------------------------------------------
# Figure 3: Lambda histograms
# ---------------------------------------------------------------------------


def plot_lambda_histograms(
    lambda_crc: np.ndarray,
    lambda_hpd: np.ndarray,
    alpha_threshold: float,
    alpha: float,
    save_path: str,
    hist_bins: int = 50,
    hist_alpha: float = 0.7,
    dpi: int = 150,
) -> None:
    """Reproduce Figure 3: histograms of chosen λ values for CRC and CBQ-HPD.

    Creates a 1×2 subplot comparing the distribution of λ values chosen by
    CRC (left) and CBQ-HPD at β=0.95 (right) across M=10,000 trials. The
    region where per-trial risk exceeds α (λ < alpha_threshold) is shaded
    red in both panels, matching the paper's Figure 3 caption.

    For Experiment 1 (synthetic binomial), the true risk is 1 − λ, so risk
    exceeds α=0.4 iff λ < 0.6. The paper reports that CRC has 21.20% of
    trials in this region while CBQ-HPD has only 0.03%.

    Config values used (from config.yaml):
      - plotting.hist_bins = 50 (default)
      - plotting.hist_alpha = 0.7 (default)
      - plotting.dpi = 150 (default)
      - exp1_synthetic_binomial.risk_threshold = 0.6 → alpha_threshold
      - exp1_synthetic_binomial.alpha = 0.4 → alpha

    Args:
        lambda_crc: Array of shape (M,) containing the λ chosen by CRC in
            each of the M=10,000 trials. May contain np.inf values (excluded
            from histogram via finite masking).
        lambda_hpd: Array of shape (M,) containing the λ chosen by CBQ-HPD
            (β=0.95) in each trial. May contain np.inf values.
        alpha_threshold: Critical λ value below which true risk exceeds α.
            For Experiment 1: 0.6 (config.yaml: exp1_synthetic_binomial.
            risk_threshold = 0.6). Trials with λ < alpha_threshold are
            "failures" (risk > α).
        alpha: Target risk level, used for axis labels and legend text.
            For Experiment 1: 0.4 (config.yaml: exp1_synthetic_binomial.
            alpha = 0.4).
        save_path: Full path to save the figure, e.g.
            "results/fig3_lambda_histograms.png" (config.yaml:
            plotting.figure3_path).
        hist_bins: Number of histogram bins. Default 50 per config.yaml
            (plotting.hist_bins = 50).
        hist_alpha: Histogram bar transparency. Default 0.7 per config.yaml
            (plotting.hist_alpha = 0.7).
        dpi: Figure resolution in dots per inch. Default 150 per config.yaml
            (plotting.dpi = 150).

    Returns:
        None. Saves the figure to save_path and closes the matplotlib figure
        to prevent memory leaks.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> lam_crc = rng.uniform(0.3, 0.9, size=10000)
        >>> lam_hpd = rng.uniform(0.6, 1.0, size=10000)
        >>> plot_lambda_histograms(lam_crc, lam_hpd, 0.6, 0.4, "/tmp/fig3.png")
    """
    # Filter out np.inf values for histogram plotting.
    # np.inf arises when no λ in the grid satisfies the decision rule criterion.
    lambda_crc_finite: np.ndarray = lambda_crc[np.isfinite(lambda_crc)]
    lambda_hpd_finite: np.ndarray = lambda_hpd[np.isfinite(lambda_hpd)]

    # Create figure with two side-by-side panels.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ------------------------------------------------------------------
    # Left panel: CRC histogram
    # ------------------------------------------------------------------
    ax_crc = axes[0]

    ax_crc.hist(
        lambda_crc_finite,
        bins=hist_bins,
        alpha=hist_alpha,
        color="steelblue",
        edgecolor="white",
        label="CRC",
    )

    # Determine x-axis range for the red shading region.
    # Use the minimum of the data or 0 as the left boundary.
    x_min_crc: float = float(np.min(lambda_crc_finite)) if len(lambda_crc_finite) > 0 else 0.0
    x_left_crc: float = min(x_min_crc, 0.0)

    # Red shading for the risk-exceeding region: λ < alpha_threshold.
    ax_crc.axvspan(
        xmin=x_left_crc,
        xmax=alpha_threshold,
        alpha=0.2,
        color="red",
        label=f"Risk > α={alpha:.1f}",
    )

    # Vertical dashed line at the risk threshold.
    ax_crc.axvline(
        x=alpha_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"λ = {alpha_threshold}",
    )

    ax_crc.set_title("CRC: Distribution of $\\lambda_{\\mathrm{crc}}$", fontsize=13)
    ax_crc.set_xlabel("λ chosen", fontsize=11)
    ax_crc.set_ylabel("Count", fontsize=11)
    ax_crc.legend(fontsize=9)

    # ------------------------------------------------------------------
    # Right panel: CBQ-HPD histogram
    # ------------------------------------------------------------------
    ax_hpd = axes[1]

    ax_hpd.hist(
        lambda_hpd_finite,
        bins=hist_bins,
        alpha=hist_alpha,
        color="darkorange",
        edgecolor="white",
        label="CBQ-HPD (β=0.95)",
    )

    # Determine x-axis range for the red shading region.
    x_min_hpd: float = float(np.min(lambda_hpd_finite)) if len(lambda_hpd_finite) > 0 else 0.0
    x_left_hpd: float = min(x_min_hpd, 0.0)

    # Red shading for the risk-exceeding region: λ < alpha_threshold.
    ax_hpd.axvspan(
        xmin=x_left_hpd,
        xmax=alpha_threshold,
        alpha=0.2,
        color="red",
        label=f"Risk > α={alpha:.1f}",
    )

    # Vertical dashed line at the risk threshold.
    ax_hpd.axvline(
        x=alpha_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"λ = {alpha_threshold}",
    )

    ax_hpd.set_title(
        "CBQ-HPD (β=0.95): Distribution of $\\lambda_{\\mathrm{hpd}}^{0.95}$",
        fontsize=13,
    )
    ax_hpd.set_xlabel("λ chosen", fontsize=11)
    ax_hpd.set_ylabel("Count", fontsize=11)
    ax_hpd.legend(fontsize=9)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: L+ density plots
# ---------------------------------------------------------------------------


def plot_L_plus_density(
    losses_dict: Dict[float, np.ndarray],
    save_path: str,
    alpha_ref: float = 0.4,
    dpi: int = 150,
) -> None:
    """Reproduce Figure 4: probability density of L⁺ for λ ∈ {0.7, 0.8, 0.9}.

    Plots smooth KDE density curves for the L⁺ distribution at each λ value,
    estimated from 100,000 Dirichlet samples (config.yaml: cbq.n_mc_figure =
    100000). The vertical dashed line at α=0.4 shows the target risk level,
    making it visually clear what fraction of the L⁺ distribution falls below
    α for each λ — this is exactly Pr(L⁺ ≤ α) used in the CBQ-HPD rule.

    The paper's Figure 4 caption: "Probability density for L⁺ with
    λ ∈ {0.7, 0.8, 0.9} estimated using 100,000 Dirichlet samples."
    Higher λ → lower losses → L⁺ distribution shifted left → higher
    Pr(L⁺ ≤ α).

    Config values used (from config.yaml):
      - cbq.n_mc_figure = 100000 (upstream, determines sample count in losses_dict)
      - exp1_synthetic_binomial.alpha = 0.4 → alpha_ref
      - plotting.dpi = 150 (default)
      - plotting.figure4_path = "results/fig4_L_plus_density.png" → save_path

    Args:
        losses_dict: Dictionary mapping each λ value to an array of L⁺ samples
            of shape (n_mc_figure,) = (100,000,). Keys are typically
            {0.7, 0.8, 0.9} per config.yaml (exp1_synthetic_binomial.
            figure4_lambdas = [0.7, 0.8, 0.9]). Pre-computed by
            compute_L_plus_for_figure4 in experiments/exp1_synthetic_binomial.py
            using cbq_core.compute_L_plus_samples.
        save_path: Full path to save the figure, e.g.
            "results/fig4_L_plus_density.png" (config.yaml:
            plotting.figure4_path).
        alpha_ref: Target risk level for the reference vertical line. Default
            0.4 per config.yaml (exp1_synthetic_binomial.alpha = 0.4).
        dpi: Figure resolution. Default 150 per config.yaml (plotting.dpi).

    Returns:
        None. Saves the figure to save_path and closes the matplotlib figure.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> losses_07 = np.random.default_rng(0).uniform(0.0, 0.5, 100000)
        >>> losses_08 = np.random.default_rng(1).uniform(0.0, 0.4, 100000)
        >>> losses_09 = np.random.default_rng(2).uniform(0.0, 0.3, 100000)
        >>> d = {0.7: losses_07, 0.8: losses_08, 0.9: losses_09}
        >>> plot_L_plus_density(d, "/tmp/fig4.png")
    """
    # Color palette for the three λ values (matplotlib default cycle).
    # Higher λ → lower expected loss → density shifted left.
    _COLORS: List[str] = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Create single-panel figure.
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Sort lambda values for consistent ordering and color assignment.
    sorted_lambdas: List[float] = sorted(losses_dict.keys())

    for idx, lam in enumerate(sorted_lambdas):
        L_plus_samples: np.ndarray = losses_dict[lam]
        color: str = _COLORS[idx % len(_COLORS)]

        # Construct KDE from the 100,000 L⁺ samples.
        # scipy.stats.gaussian_kde uses Scott's rule for bandwidth selection,
        # which is appropriate for 100,000 samples and produces smooth curves.
        kde: gaussian_kde = gaussian_kde(L_plus_samples)

        # Evaluation grid: L⁺ is bounded in [0, B=1] since it is a convex
        # combination of losses in [0, B]. Use 500 points for smooth curves.
        x_grid: np.ndarray = np.linspace(0.0, 1.0, 500)

        # Evaluate KDE on the grid.
        density: np.ndarray = kde(x_grid)

        # Plot density curve.
        ax.plot(
            x_grid,
            density,
            label=f"λ = {lam}",
            linewidth=2.0,
            color=color,
        )

        # Fill under the curve with low alpha for visual clarity.
        ax.fill_between(
            x_grid,
            density,
            alpha=0.1,
            color=color,
        )

    # Vertical dashed reference line at α (target risk level).
    # This line shows what fraction of L⁺ falls below α for each λ,
    # which is exactly Pr(L⁺ ≤ α) used in the CBQ-HPD decision rule.
    ax.axvline(
        x=alpha_ref,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"α = {alpha_ref}",
    )

    # Axis labels and title.
    ax.set_xlabel("$L^+$ (upper bound on expected loss)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(
        "Distribution of $L^+$ for different λ values",
        fontsize=13,
    )

    # Fix x-axis to [0, 1] since L⁺ ∈ [0, B=1].
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim(bottom=0.0)

    ax.legend(fontsize=10)

    # Finalization.
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Posterior illustration (Figures 1/2 style)
# ---------------------------------------------------------------------------


def plot_posterior_illustration(
    losses: np.ndarray,
    B: float,
    n_samples: int,
    rng: np.random.Generator,
    save_path: str,
    dpi: int = 150,
) -> None:
    """Reproduce the Figure 1/2 style illustration of posterior quantile functions.

    Shows multiple sampled worst-case quantile functions K*(t) (step functions
    in blue) consistent with the observed calibration losses, along with a 95%
    credible interval band in black. This illustrates how the Bayesian
    quadrature approach characterizes uncertainty over the quantile function.

    The worst-case quantile function K*(t) from Proposition B.2 is a right-
    continuous step function:
        K*(t) = ℓ₍ᵢ₎  for t ∈ (t₍ᵢ₋₁₎, t₍ᵢ₎]

    where t₍₀₎=0, t₍₁₎,...,t₍ₙ₎ are the random quantile levels drawn from
    Dir(1,...,1) and ℓ₍₁₎ ≤ ... ≤ ℓ₍ₙ₎ are the order statistics of the
    observed losses.

    Args:
        losses: Observed calibration losses of shape (n,). These are the
            per-sample losses ℓ(zᵢ, λ) for a fixed λ in a single trial.
            Values should be in [0, B].
        B: Upper bound on losses. Set to 1.0 for all experiments per
            config.yaml (exp*.B = 1.0). Appended as ℓ₍ₙ₊₁₎ = B.
        n_samples: Number of Dirichlet spacing samples to draw for the
            illustration. Use a small value (e.g., 50) for visual clarity —
            too many overlapping step functions obscure the pattern.
        rng: NumPy Generator for reproducible sampling. Created via
            utils.get_rng(seed) in main.py.
        save_path: Full path to save the figure.
        dpi: Figure resolution. Default 150 per config.yaml (plotting.dpi).

    Returns:
        None. Saves the figure to save_path and closes the matplotlib figure.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> losses = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        >>> plot_posterior_illustration(losses, B=1.0, n_samples=30, rng=rng,
        ...                            save_path="/tmp/fig_illustration.png")
    """
    n: int = len(losses)

    # Step 1: Sort losses to get order statistics ℓ₍₁₎ ≤ ... ≤ ℓ₍ₙ₎.
    # Append B as ℓ₍ₙ₊₁₎ = B (the worst-case unobserved future loss).
    losses_sorted: np.ndarray = np.sort(losses)                    # shape (n,)
    losses_extended: np.ndarray = np.append(losses_sorted, B)      # shape (n+1,)

    # Step 2: Sample Dirichlet spacings U ~ Dir(1,...,1) with n+1 components.
    # Shape: (n_samples, n+1)
    spacings: np.ndarray = cbq_core.sample_dirichlet_spacings(
        n=n,
        num_samples=n_samples,
        rng=rng,
    )

    # Step 3: Compute cumulative sums to get quantile levels t₍₁₎,...,t₍ₙ₊₁₎.
    # t_samples[s, i] = t₍ᵢ₊₁₎ for sample s (0-indexed: i=0 → t₍₁₎, ..., i=n → t₍ₙ₊₁₎=1).
    # Shape: (n_samples, n+1)
    t_samples: np.ndarray = np.cumsum(spacings, axis=1)

    # Create figure.
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # ------------------------------------------------------------------
    # Step 4: Plot individual sampled step functions (worst-case quantile
    # functions K*(t) from Proposition B.2).
    # ------------------------------------------------------------------
    # Each step function is plotted as a piecewise constant (right-continuous)
    # function. We construct explicit (x, y) coordinates for the step plot.
    for s in range(n_samples):
        # Quantile levels for this sample: t₍₁₎, ..., t₍ₙ₎, t₍ₙ₊₁₎=1.
        # Shape: (n+1,)
        t_s: np.ndarray = t_samples[s]  # cumulative sum, last entry = 1.0

        # Build x-coordinates for the step function.
        # The step function starts at t=0 with value ℓ₍₁₎, then jumps at
        # each t₍ᵢ₎ to ℓ₍ᵢ₊₁₎. We represent each jump with two x-points
        # at the same t value (left and right limits).
        #
        # Pattern: [0, t₍₁₎, t₍₁₎, t₍₂₎, t₍₂₎, ..., t₍ₙ₎, t₍ₙ₎, 1]
        # Pattern: [ℓ₍₁₎, ℓ₍₁₎, ℓ₍₂₎, ℓ₍₂₎, ..., ℓ₍ₙ₎, ℓ₍ₙ₊₁₎, ℓ₍ₙ₊₁₎]
        #
        # For n+1 intervals, we have n+1 jump points (t₍₁₎,...,t₍ₙ₊₁₎).
        # The step function has n+1 constant segments.

        # x-coordinates: interleave jump points to create vertical transitions.
        # Start at 0, then for each jump point t₍ᵢ₎ add two entries.
        x_coords: List[float] = [0.0]
        for i in range(n + 1):
            x_coords.append(float(t_s[i]))
            x_coords.append(float(t_s[i]))

        # y-coordinates: constant value on each segment, then jump.
        # Segment 0: [0, t₍₁₎) → ℓ₍₁₎ = losses_extended[0]
        # Segment i: [t₍ᵢ₎, t₍ᵢ₊₁₎) → ℓ₍ᵢ₊₁₎ = losses_extended[i]
        # Last segment: [t₍ₙ₎, 1] → ℓ₍ₙ₊₁₎ = B = losses_extended[n]
        y_coords: List[float] = []
        for i in range(n + 1):
            # Value on segment i (before the jump at t₍ᵢ₊₁₎).
            y_coords.append(float(losses_extended[i]))
            # Value after the jump at t₍ᵢ₊₁₎ (right-continuous).
            if i < n:
                y_coords.append(float(losses_extended[i + 1]))
            else:
                # Last point: stays at B.
                y_coords.append(float(losses_extended[n]))

        ax.plot(
            x_coords,
            y_coords,
            color="#4a90d9",
            alpha=0.15,
            linewidth=0.8,
        )

    # ------------------------------------------------------------------
    # Step 5: Compute and plot 95% credible interval band.
    # For each t in a fine grid, evaluate K*(t) for all n_samples and
    # compute the 2.5th and 97.5th percentiles.
    # ------------------------------------------------------------------
    # Fine grid of t values for the credible interval.
    t_grid: np.ndarray = np.linspace(0.0, 1.0, 200)

    # For each sample s and each t in t_grid, find which segment t falls in.
    # K*(t) = losses_extended[i] where i = searchsorted(t_samples[s], t).
    # t_samples[s] has shape (n+1,) with values in (0, 1].
    # np.searchsorted returns the index i such that t_samples[s][i-1] < t <= t_samples[s][i].
    # This gives the correct segment index for the right-continuous step function.
    #
    # Shape of K_star_matrix: (n_samples, len(t_grid))
    K_star_matrix: np.ndarray = np.zeros((n_samples, len(t_grid)), dtype=float)

    for s in range(n_samples):
        # For each t in t_grid, find the segment index.
        # searchsorted with side='left' gives the first index where t_samples[s][idx] >= t.
        # This corresponds to the segment [t₍ᵢ₋₁₎, t₍ᵢ₎] containing t.
        indices: np.ndarray = np.searchsorted(t_samples[s], t_grid, side="left")
        # Clip to valid range [0, n] to handle t=0 and t=1 edge cases.
        indices = np.clip(indices, 0, n)
        K_star_matrix[s] = losses_extended[indices]

    # Compute 2.5th and 97.5th percentiles across samples for each t.
    ci_low: np.ndarray = np.percentile(K_star_matrix, 2.5, axis=0)
    ci_high: np.ndarray = np.percentile(K_star_matrix, 97.5, axis=0)

    # Plot the credible interval band.
    ax.fill_between(
        t_grid,
        ci_low,
        ci_high,
        alpha=0.3,
        color="black",
        label="95% Credible Interval",
    )

    # ------------------------------------------------------------------
    # Step 6: Plot observed loss points at their mean quantile levels.
    # The mean quantile level for the i-th order statistic under Dir(1,...,1)
    # is i/(n+1) (the expected value of the i-th order statistic of n
    # uniform samples). We use the mean of t_samples across all draws.
    # ------------------------------------------------------------------
    # Mean quantile levels for the n order statistics (not including t₍ₙ₊₁₎=1).
    # t_samples[:, :-1] has shape (n_samples, n); mean over axis=0 gives (n,).
    mean_t_levels: np.ndarray = np.mean(t_samples[:, :-1], axis=0)  # shape (n,)

    ax.scatter(
        mean_t_levels,
        losses_sorted,
        color="red",
        zorder=5,
        s=40,
        label="Observed losses $\\ell_{(i)}$",
    )

    # ------------------------------------------------------------------
    # Labels, limits, and finalization
    # ------------------------------------------------------------------
    ax.set_xlabel("Quantile level $t$", fontsize=12)
    ax.set_ylabel("Loss value $K(t)$", fontsize=12)
    ax.set_title(
        "Posterior distribution over quantile functions",
        fontsize=13,
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, B * 1.05])

    # Add a custom legend entry for the sampled step functions.
    step_patch: mpatches.Patch = mpatches.Patch(
        color="#4a90d9",
        alpha=0.5,
        label="Sampled $K^*(t)$ functions",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=[step_patch] + handles,
        labels=["Sampled $K^*(t)$ functions"] + labels,
        fontsize=9,
        loc="upper left",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
