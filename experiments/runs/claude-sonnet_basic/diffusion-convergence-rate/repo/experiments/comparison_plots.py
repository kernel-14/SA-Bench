"""
Comparison plots for convergence rates.

Reproduces Figures 1 and 3 from the paper:
"Instance-dependent Convergence Theory for Diffusion Models"
by Yuchen Jiao and Gen Li (2025).

Figure 1: Comparison of iteration complexity as a function of L (left)
          and as a function of epsilon when L=infinity (right).

Figure 3: TV distance achieved by various theories for fixed T.
"""

import numpy as np
import sys
import os


def tv_distance_our_result(T, L, d, eps=None):
    """
    TV distance from Theorem 1: O(min{d^{3/2}, d*L^{1/2}, d^{1/2}*L^{3/2}} * log^4(T) / T^{3/2})

    For iteration complexity: T ~ min{d, d^{2/3}*L^{1/3}, d^{1/3}*L} * eps^{-2/3} * log^{8/3}(T)
    """
    if eps is not None:
        # Return iteration complexity
        return min(d, d**(2/3) * L**(1/3), d**(1/3) * L) * eps**(-2/3)
    # Return TV distance for given T
    log_T = np.log(max(T, 2))
    return min(d**(3/2), d * L**(1/2), d**(1/2) * L**(3/2)) * log_T**4 / T**(3/2)


def tv_distance_li_jiao_2024(T, L, d, eps=None):
    """
    Li and Jiao (2024): O(d^{1/3} * L * eps^{-2/3}) iteration complexity.
    TV distance: O(d^{1/2} * L^{3/2} * log^4(T) / T^{3/2})
    """
    if eps is not None:
        return d**(1/3) * L * eps**(-2/3)
    log_T = np.log(max(T, 2))
    return d**(1/2) * L**(3/2) * log_T**4 / T**(3/2)


def tv_distance_li_yan_2024(T, d, eps=None):
    """
    Li and Yan (2024a): O(d * eps^{-1}) iteration complexity.
    TV distance: O(d * log^2(T) / T)
    """
    if eps is not None:
        return d * eps**(-1)
    log_T = np.log(max(T, 2))
    return d * log_T**2 / T


def tv_distance_benton_2023(T, d, eps=None):
    """
    Benton et al. (2023): O(d * eps^{-2}) iteration complexity.
    TV distance: O(d^{1/2} * log(T) / T^{1/2})
    """
    if eps is not None:
        return d * eps**(-2)
    log_T = np.log(max(T, 2))
    return d**(1/2) * log_T / T**(1/2)


def tv_distance_li_cai_2024(T, d, eps=None):
    """
    Li and Cai (2024): O(d^{5/4} * eps^{-1/2}) iteration complexity.
    TV distance: O(d^{5/8} * log^{1/4}(T) / T^{1/4})
    """
    if eps is not None:
        return d**(5/4) * eps**(-1/2)
    log_T = np.log(max(T, 2))
    return d**(5/8) * log_T**(1/4) / T**(1/4)


def tv_distance_gupta_2024(T, L, d, eps=None):
    """
    Gupta et al. (2024): O(d^{1/3} * L * eps^{-2/3}) iteration complexity (similar to Li & Jiao).
    """
    if eps is not None:
        return d**(1/3) * L * eps**(-2/3)
    log_T = np.log(max(T, 2))
    return d**(1/2) * L**(3/2) * log_T**4 / T**(3/2)


def generate_figure1(output_dir):
    """
    Generate Figure 1: Comparison of iteration complexity.

    Left: Iteration complexity as a function of L when eps = O(1).
    Right: Iteration complexity as a function of eps when L = infinity.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping Figure 1")
        return

    d = 100  # Fixed dimension
    eps = 1.0  # Fixed accuracy for left plot

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Iteration complexity vs L
    ax = axes[0]
    L_values = np.logspace(0, 4, 100)  # L from 1 to 10^4

    # Our result: min{d, d^{2/3}*L^{1/3}, d^{1/3}*L} * eps^{-2/3}
    our_complexity = np.array([min(d, d**(2/3) * L**(1/3), d**(1/3) * L) * eps**(-2/3)
                                for L in L_values])

    # Li and Jiao (2024): d^{1/3} * L * eps^{-2/3}
    li_jiao_complexity = d**(1/3) * L_values * eps**(-2/3)

    # Li and Yan (2024a): d * eps^{-1} (no L dependence)
    li_yan_complexity = d * eps**(-1) * np.ones_like(L_values)

    # Benton et al. (2023): d * eps^{-2} (no L dependence)
    benton_complexity = d * eps**(-2) * np.ones_like(L_values)

    # Li and Cai (2024): d^{5/4} * eps^{-1/2} (no L dependence)
    li_cai_complexity = d**(5/4) * eps**(-1/2) * np.ones_like(L_values)

    ax.loglog(L_values, our_complexity, "r-", linewidth=2.5, label="Ours (Theorem 1)", zorder=5)
    ax.loglog(L_values, li_jiao_complexity, "b--", linewidth=2, label="Li & Jiao (2024)")
    ax.loglog(L_values, li_yan_complexity, "g-.", linewidth=2, label="Li & Yan (2024a)")
    ax.loglog(L_values, benton_complexity, "m:", linewidth=2, label="Benton et al. (2023)")
    ax.loglog(L_values, li_cai_complexity, "c--", linewidth=2, label="Li & Cai (2024)")

    ax.set_xlabel(r"$ (Lipschitz constant)", fontsize=12)
    ax.set_ylabel(r"Iteration complexity", fontsize=12)
    ax.set_title(r"Iteration complexity vs $ ($arepsilon = O(1)$)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right plot: Iteration complexity vs eps when L = infinity
    ax = axes[1]
    eps_values = np.logspace(-3, 0, 100)  # eps from 10^{-3} to 1

    # Our result with L = infinity: d * eps^{-2/3}
    our_complexity_linf = d * eps_values**(-2/3)

    # Li and Yan (2024a): d * eps^{-1}
    li_yan_complexity_eps = d * eps_values**(-1)

    # Benton et al. (2023): d * eps^{-2}
    benton_complexity_eps = d * eps_values**(-2)

    # Li and Cai (2024): d^{5/4} * eps^{-1/2}
    li_cai_complexity_eps = d**(5/4) * eps_values**(-1/2)

    ax.loglog(eps_values, our_complexity_linf, "r-", linewidth=2.5, label="Ours (Theorem 1)", zorder=5)
    ax.loglog(eps_values, li_yan_complexity_eps, "g-.", linewidth=2, label="Li & Yan (2024a)")
    ax.loglog(eps_values, benton_complexity_eps, "m:", linewidth=2, label="Benton et al. (2023)")
    ax.loglog(eps_values, li_cai_complexity_eps, "c--", linewidth=2, label="Li & Cai (2024)")

    ax.set_xlabel(r"$arepsilon$ (accuracy)", fontsize=12)
    ax.set_ylabel(r"Iteration complexity", fontsize=12)
    ax.set_title(r"Iteration complexity vs $arepsilon$ ( = \infty$)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Figure 1: Comparison of Theorem 1 with Prior Results", fontsize=13)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"figure1_comparison.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Figure 1 saved to {path}")
    plt.close()


def generate_figure3(output_dir):
    """
    Generate Figure 3: TV distance achieved by various theories for fixed T.

    Three subplots:
    - Left: T = O(d)
    - Middle: T = O(d^{3/2})
    - Right: T = O(d^2)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping Figure 3")
        return

    d = 100  # Fixed dimension
    L_values = np.logspace(0, 4, 200)  # L from 1 to 10^4

    T_settings = [
        {"T": d, "label": r" = O(d)$"},
        {"T": int(d**(3/2)), "label": r" = O(d^{3/2})$"},
        {"T": d**2, "label": r" = O(d^2)$"},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, setting in zip(axes, T_settings):
        T = setting["T"]

        # Our result
        our_tv = np.array([tv_distance_our_result(T, L, d) for L in L_values])

        # Li and Jiao (2024)
        li_jiao_tv = np.array([tv_distance_li_jiao_2024(T, L, d) for L in L_values])

        # Li and Yan (2024a) - no L dependence
        li_yan_tv = tv_distance_li_yan_2024(T, d) * np.ones_like(L_values)

        # Benton et al. (2023) - no L dependence
        benton_tv = tv_distance_benton_2023(T, d) * np.ones_like(L_values)

        # Li and Cai (2024) - no L dependence
        li_cai_tv = tv_distance_li_cai_2024(T, d) * np.ones_like(L_values)

        # Clip to [0, 1] (TV distance is at most 1)
        our_tv = np.minimum(our_tv, 1.0)
        li_jiao_tv = np.minimum(li_jiao_tv, 1.0)
        li_yan_tv = min(li_yan_tv[0], 1.0) * np.ones_like(L_values)
        benton_tv = min(benton_tv[0], 1.0) * np.ones_like(L_values)
        li_cai_tv = min(li_cai_tv[0], 1.0) * np.ones_like(L_values)

        ax.semilogx(L_values, our_tv, "r-", linewidth=2.5, label="Ours (Theorem 1)", zorder=5)
        ax.semilogx(L_values, li_jiao_tv, "b--", linewidth=2, label="Li & Jiao (2024)")
        ax.semilogx(L_values, li_yan_tv, "g-.", linewidth=2, label="Li & Yan (2024a)")
        ax.semilogx(L_values, benton_tv, "m:", linewidth=2, label="Benton et al. (2023)")
        ax.semilogx(L_values, li_cai_tv, "c--", linewidth=2, label="Li & Cai (2024)")

        ax.set_xlabel(r"$ (Lipschitz constant)", fontsize=12)
        ax.set_ylabel(r"TV distance $arepsilon$", fontsize=12)
        ax.set_title(setting["label"], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])

    plt.suptitle("Figure 3: TV Distance Achieved by Various Theories", fontsize=13)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"figure3_tv_comparison.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Figure 3 saved to {path}")
    plt.close()


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating Figure 1...")
    generate_figure1(output_dir)

    print("Generating Figure 3...")
    generate_figure3(output_dir)


if __name__ == "__main__":
    main()
