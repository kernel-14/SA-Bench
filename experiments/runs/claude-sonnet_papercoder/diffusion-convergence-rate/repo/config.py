## config.py
"""Configuration dataclass for reproducing Instance-dependent Convergence Theory
for Diffusion Models.

This module defines the Config dataclass that bundles all hyperparameters for
a single experiment configuration. It has zero external dependencies beyond
the Python standard library.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Experiment configuration for the randomized midpoint diffusion sampler.

    Attributes:
        d: Data dimension. Takes values 10, 100, 500 in the three paper configs.
        k_active: Number of non-degenerate dimensions. First k_active diagonal
            entries of the target covariance are drawn from Unif(0, 10);
            remaining d - k_active entries are zero.
        K: Number of sampler rounds. Fixed at 10 per paper (Appendix A).
        T_values: List of total iteration counts to sweep. Each T gives one
            data point on the convergence plot. All values must satisfy
            2*T % K == 0 so that N = 2T/K is an integer.
        c0: Schedule constant controlling alpha_hat[T+1] = 1/T^c0. Must be
            positive. Default 2.0 (paper: "sufficiently large").
        c1: Schedule constant controlling step size. Must be positive and
            c1/c0 sufficiently large. Default 10.0 (ratio = 5 with c0=2.0).
        seed: Random seed for constructing the covariance diagonal (the
            k_active values drawn from Unif(0, 10)). Default 42.
        device: Computation device, either "cpu" or "cuda". Default "cpu".
        figure_dir: Output directory for saved figures. Default "figures".
        label: Human-readable identifier for this configuration, e.g.
            "fig2a", "fig2b", "fig2c". Used as dict keys and plot labels.
    """

    d: int
    k_active: int
    K: int = 10
    T_values: List[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000, 2000, 5000])
    c0: float = 2.0
    c1: float = 10.0
    seed: int = 42
    device: str = "cpu"
    figure_dir: str = "figures"
    label: str = ""

    def __post_init__(self) -> None:
        """Validate all fields after initialization.

        Raises:
            ValueError: If any field violates its constraints.
            TypeError: If any field has the wrong type.
        """
        # Type checks
        if not isinstance(self.d, int):
            raise TypeError(f"d must be int, got {type(self.d).__name__}")
        if not isinstance(self.k_active, int):
            raise TypeError(f"k_active must be int, got {type(self.k_active).__name__}")
        if not isinstance(self.K, int):
            raise TypeError(f"K must be int, got {type(self.K).__name__}")
        if not isinstance(self.T_values, list):
            raise TypeError(f"T_values must be list, got {type(self.T_values).__name__}")
        if not isinstance(self.c0, (int, float)):
            raise TypeError(f"c0 must be numeric, got {type(self.c0).__name__}")
        if not isinstance(self.c1, (int, float)):
            raise TypeError(f"c1 must be numeric, got {type(self.c1).__name__}")
        if not isinstance(self.seed, int):
            raise TypeError(f"seed must be int, got {type(self.seed).__name__}")
        if not isinstance(self.device, str):
            raise TypeError(f"device must be str, got {type(self.device).__name__}")
        if not isinstance(self.figure_dir, str):
            raise TypeError(f"figure_dir must be str, got {type(self.figure_dir).__name__}")
        if not isinstance(self.label, str):
            raise TypeError(f"label must be str, got {type(self.label).__name__}")

        # Ensure float types for schedule constants
        self.c0 = float(self.c0)
        self.c1 = float(self.c1)

        # Value constraints
        if self.d < 1:
            raise ValueError(f"d must be >= 1, got {self.d}")
        if self.k_active < 1:
            raise ValueError(f"k_active must be >= 1, got {self.k_active}")
        if self.k_active > self.d:
            raise ValueError(
                f"k_active ({self.k_active}) must be <= d ({self.d})"
            )
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if len(self.T_values) < 1:
            raise ValueError("T_values must contain at least one element")
        for T in self.T_values:
            if not isinstance(T, int):
                raise TypeError(f"All T_values must be int, got {type(T).__name__} for value {T}")
            if T <= 0:
                raise ValueError(f"All T_values must be positive, got {T}")
            if (2 * T) % self.K != 0:
                raise ValueError(
                    f"T={T} is invalid: 2*T must be divisible by K={self.K} "
                    f"so that N = 2T/K is an integer. "
                    f"Got 2*{T} % {self.K} = {(2 * T) % self.K}"
                )
        if self.c0 <= 0.0:
            raise ValueError(f"c0 must be positive, got {self.c0}")
        if self.c1 <= 0.0:
            raise ValueError(f"c1 must be positive, got {self.c1}")
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got '{self.device}'")

        # Warn if c1/c0 ratio is small (paper requires it to be "sufficiently large")
        ratio = self.c1 / self.c0
        if ratio < 3.0:
            import warnings
            warnings.warn(
                f"c1/c0 ratio = {ratio:.2f} may be too small. "
                f"The paper requires c1/c0 to be sufficiently large. "
                f"Consider increasing c1 or decreasing c0.",
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def from_dict(cls, cfg: dict) -> "Config":
        """Construct a Config from a plain Python dictionary.

        This enables loading from a parsed YAML config file or from an
        argparse namespace converted to dict. Keys must match field names.
        Missing keys fall back to field defaults.

        Args:
            cfg: Dictionary with configuration values. Supported keys:
                d, k_active, K, T_values, c0, c1, seed, device,
                figure_dir, label.

        Returns:
            A validated Config instance.

        Raises:
            ValueError: If required fields (d, k_active) are missing.
            ValueError: If any field violates its constraints.
        """
        if "d" not in cfg:
            raise ValueError("Required field 'd' missing from config dict")
        if "k_active" not in cfg:
            raise ValueError("Required field 'k_active' missing from config dict")

        # Extract with defaults matching field defaults
        d: int = int(cfg["d"])
        k_active: int = int(cfg["k_active"])
        K: int = int(cfg.get("K", 10))
        raw_T_values = cfg.get("T_values", [50, 100, 200, 500, 1000, 2000, 5000])
        T_values: List[int] = [int(t) for t in raw_T_values]
        c0: float = float(cfg.get("c0", 2.0))
        c1: float = float(cfg.get("c1", 10.0))
        seed: int = int(cfg.get("seed", 42))
        device: str = str(cfg.get("device", "cpu"))
        figure_dir: str = str(cfg.get("figure_dir", "figures"))
        label: str = str(cfg.get("label", ""))

        return cls(
            d=d,
            k_active=k_active,
            K=K,
            T_values=T_values,
            c0=c0,
            c1=c1,
            seed=seed,
            device=device,
            figure_dir=figure_dir,
            label=label,
        )

    @classmethod
    def default_configs(
        cls,
        K: int = 10,
        T_values: Optional[List[int]] = None,
        c0: float = 2.0,
        c1: float = 10.0,
        seed: int = 42,
        device: str = "cpu",
        figure_dir: str = "figures",
    ) -> List["Config"]:
        """Return the three paper configurations from Figure 2 (Appendix A).

        The three configurations are:
            (a) d=10,  k_active=10,  label="fig2a"
            (b) d=100, k_active=10,  label="fig2b"
            (c) d=500, k_active=100, label="fig2c"

        All three share the same K, T_values, c0, c1, seed, device, figure_dir.

        Args:
            K: Number of sampler rounds. Default 10.
            T_values: List of T values to sweep. If None, uses the default
                [50, 100, 200, 500, 1000, 2000, 5000].
            c0: Schedule constant. Default 2.0.
            c1: Schedule constant. Default 10.0.
            seed: Random seed. Default 42.
            device: Computation device. Default "cpu".
            figure_dir: Output directory for figures. Default "figures".

        Returns:
            List of three Config objects corresponding to the three paper
            configurations.
        """
        if T_values is None:
            T_values = [50, 100, 200, 500, 1000, 2000, 5000]

        shared_kwargs = dict(
            K=K,
            T_values=list(T_values),
            c0=c0,
            c1=c1,
            seed=seed,
            device=device,
            figure_dir=figure_dir,
        )

        configs: List[Config] = [
            cls(d=10,  k_active=10,  label="fig2a", **shared_kwargs),
            cls(d=100, k_active=10,  label="fig2b", **shared_kwargs),
            cls(d=500, k_active=100, label="fig2c", **shared_kwargs),
        ]
        return configs

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"Config(label={self.label!r}, d={self.d}, k_active={self.k_active}, "
            f"K={self.K}, T_values={self.T_values}, c0={self.c0}, c1={self.c1}, "
            f"seed={self.seed}, device={self.device!r}, figure_dir={self.figure_dir!r})"
        )

    def summary(self) -> str:
        """Return a multi-line human-readable summary of the configuration.

        Returns:
            Formatted string with all configuration fields and their values.
        """
        lines = [
            f"Config Summary [{self.label}]",
            f"  Data dimension d       : {self.d}",
            f"  Active dimensions k    : {self.k_active}",
            f"  Sampler rounds K       : {self.K}",
            f"  T values               : {self.T_values}",
            f"  Schedule c0            : {self.c0}",
            f"  Schedule c1            : {self.c1}",
            f"  c1/c0 ratio            : {self.c1 / self.c0:.2f}",
            f"  Random seed            : {self.seed}",
            f"  Device                 : {self.device}",
            f"  Figure output dir      : {self.figure_dir}",
        ]
        return "\n".join(lines)
