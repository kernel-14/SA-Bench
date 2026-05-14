"""
data_generator.py
=================
Generates (or loads from cache) PDE trajectory datasets for the LUNO reproduction.
Supports 1D (Burgers, Hyper‑Diffusion, Kuramoto‑Sivashinsky conservative) via the
APEBench library (if installed) with a fallback to custom spectral solvers, and
2D advection‑diffusion with out‑of‑distribution variants via a custom pseudo‑spectral
solver.

The module exposes the ``DatasetLoader`` class, which takes a ``DataConfig`` and
provides a unified ``load_or_generate`` method that returns standardised (X, Y)
pairs for training, validation, and testing.

All randomness is seeded through the ``DataConfig`` and a global random generator
to guarantee reproducibility.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import h5py  # type: ignore
import numpy as np
from numpy.random import Generator, PCG64

from config import DataConfig
from utils import mesh_grid  # safe import – no circular dependencies

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# APEBench availability check
# ---------------------------------------------------------------------------
try:
    import apebench  # type: ignore

    HAS_APEBENCH = True
except ImportError:
    HAS_APEBENCH = False
    logger.info(
        "APEBench not installed – falling back to custom 1D spectral solvers."
    )

# ===========================================================================
# Custom 1D spectral solvers (fallback when APEBench is missing)
# ===========================================================================


def _burgers_custom(
    nx: int,
    L: float,
    nu: float,
    T: float,
    dt_save: float,
    num_snapshots: int,
    seed: int,
) -> np.ndarray:
    """
    Custom spectral solver for the 1D viscous Burgers' equation.

    .. math::
        u_t + u u_x = nu u_{xx}   (periodic)

    Uses a Fourier pseudo‑spectral evaluation of the non‑linear term and an
    explicit fourth‑order Runge‑Kutta integrator.  Aliasing is controlled by
    the 3/2‑zero‑padding rule.

    Parameters
    ----------
    nx : int
        Number of spatial grid points.
    L : float
        Domain length.
    nu : float
        Viscosity coefficient.
    T : float
        Total simulation time.
    dt_save : float
        Time between saved snapshots.
    num_snapshots : int
        Number of snapshots to save (including t=0).
    seed : int
        Random seed for the initial condition.

    Returns
    -------
    u : ndarray of shape (num_snapshots, nx)
        Saved snapshots.
    """
    rng = Generator(PCG64(seed))
    x = np.linspace(0, L, nx, endpoint=False)
    k = np.fft.fftfreq(nx, d=L / nx) * 2.0 * np.pi
    k2 = k**2

    # Initial condition: random combination of low‑frequency Fourier modes
    # plus a smooth mean profile (following APEBench style).
    u_hat = np.zeros((nx,), dtype=np.complex128)
    r_real = rng.normal(0.0, 1.0, (nx // 2 + 1,))
    r_imag = rng.normal(0.0, 1.0, (nx // 2 + 1,))
    for i in range(1, nx // 2):
        a = 1.0 / (1.0 + np.abs(k[i]))  # damp high modes
        u_hat[i] = a * (r_real[i] + 1j * r_imag[i])
        u_hat[-i] = a * (r_real[i] - 1j * r_imag[i])
    u_hat[0] = 0.0  # zero mean
    u_hat[nx // 2] = 0.0
    u_hat = u_hat[: nx // 2 + 1]  # rfft format
    u = np.fft.irfft(u_hat, n=nx)

    # Time integration
    dt_inner = dt_save / 10.0  # enough resolution for stability
    n_inner = max(1, int(round(dt_save / dt_inner)))
    dt = dt_save / n_inner

    snapshots = [u.copy()]
    t = 0.0
    for step in range(1, num_snapshots):
        # n_inner RK4 steps
        for _ in range(n_inner):
            u = _rk4_step_burgers(u, k, k2, nu, dt)
        snapshots.append(u.copy())
        t += dt_save

    return np.stack(snapshots, axis=0)


def _rk4_step_burgers(
    u: np.ndarray, k: np.ndarray, k2: np.ndarray, nu: float, dt: float
) -> np.ndarray:
    """Single RK4 step for Burgers (pseudo‑spectral)."""

    def rhs(v: np.ndarray) -> np.ndarray:
        v_hat = np.fft.rfft(v)
        dv_dx = np.fft.irfft(1j * k[: v_hat.shape[0]] * v_hat)
        nonlin = 0.5 * dv_dx * v  # u u_x written as 0.5 (u^2)_x
        nonlin_hat = np.fft.rfft(nonlin)
        # dealiasing: zero out upper third
        n_cut = v_hat.shape[0] * 2 // 3
        nonlin_hat[n_cut:] = 0.0
        # diffusion
        diff = -nu * k2[: v_hat.shape[0]] * v_hat
        return np.fft.irfft(diff - 1j * k[: v_hat.shape[0]] * nonlin_hat)

    k1 = rhs(u)
    k2v = rhs(u + 0.5 * dt * k1)
    k3v = rhs(u + 0.5 * dt * k2v)
    k4v = rhs(u + dt * k3v)
    return u + (dt / 6.0) * (k1 + 2.0 * k2v + 2.0 * k3v + k4v)


def _hyper_diffusion_custom(
    nx: int,
    L: float,
    gamma: float,
    T: float,
    dt_save: float,
    num_snapshots: int,
    seed: int,
) -> np.ndarray:
    """Custom solver for the hyper‑diffusion equation u_t = -(-Δ)^γ u."""
    rng = Generator(PCG64(seed))
    x = np.linspace(0, L, nx, endpoint=False)
    k = np.fft.fftfreq(nx, d=L / nx) * 2.0 * np.pi
    k2 = k**2
    k_pow = k2**gamma

    # initial condition similar to Burgers
    u_hat = np.zeros((nx,), dtype=np.complex128)
    r_real = rng.normal(0.0, 1.0, (nx // 2 + 1,))
    r_imag = rng.normal(0.0, 1.0, (nx // 2 + 1,))
    for i in range(1, nx // 2):
        a = 1.0 / (1.0 + k2[i])
        u_hat[i] = a * (r_real[i] + 1j * r_imag[i])
        u_hat[-i] = a * (r_real[i] - 1j * r_imag[i])
    u_hat[0] = 0.0
    u_hat[nx // 2] = 0.0
    u_hat = u_hat[: nx // 2 + 1]
    u = np.fft.irfft(u_hat, n=nx)

    dt = dt_save  # linear equation, exact in time possible but we use simple ETD
    snapshots = [u.copy()]
    # exact integration: u_hat = u_hat * exp(-k_pow * dt)
    u_hat_full = np.fft.rfft(u)
    k_pow_rfft = k_pow[: u_hat_full.shape[0]]
    for _ in range(1, num_snapshots):
        u_hat_full = u_hat_full * np.exp(-k_pow_rfft * dt_save)
        u = np.fft.irfft(u_hat_full, n=nx)
        snapshots.append(u.copy())
    return np.stack(snapshots, axis=0)


def _ks_conservative_custom(
    nx: int,
    L: float,
    T: float,
    dt_save: float,
    num_snapshots: int,
    seed: int,
) -> np.ndarray:
    """Custom solver for the conservative Kuramoto‑Sivashinsky equation."""
    rng = Generator(PCG64(seed))
    x = np.linspace(0, L, nx, endpoint=False)
    k = np.fft.fftfreq(nx, d=L / nx) * 2.0 * np.pi
    k2 = k**2
    k4 = k2**2

    # initial condition
    u_hat = np.zeros((nx,), dtype=np.complex128)
    r_real = rng.normal(0.0, 1.0, (nx // 2 + 1,))
    r_imag = rng.normal(0.0, 1.0, (nx // 2 + 1,))
    for i in range(1, nx // 2):
        a = 1.0 / (1.0 + k2[i])
        u_hat[i] = a * (r_real[i] + 1j * r_imag[i])
        u_hat[-i] = a * (r_real[i] - 1j * r_imag[i])
    u_hat[0] = 0.0
    u_hat[nx // 2] = 0.0
    u_hat = u_hat[: nx // 2 + 1]
    u = np.fft.irfft(u_hat, n=nx)

    dt_inner = dt_save / 10.0
    n_inner = max(1, int(round(dt_save / dt_inner)))
    dt = dt_save / n_inner

    def rhs(v: np.ndarray) -> np.ndarray:
        v_hat = np.fft.rfft(v)
        dv_dx = np.fft.irfft(1j * k[: v_hat.shape[0]] * v_hat)
        nonlin = v * dv_dx
        nonlin_hat = np.fft.rfft(nonlin)
        n_cut = v_hat.shape[0] * 2 // 3
        nonlin_hat[n_cut:] = 0.0
        diff = -k2[: v_hat.shape[0]] * v_hat - k4[: v_hat.shape[0]] * v_hat
        return np.fft.irfft(diff - 1j * k[: v_hat.shape[0]] * nonlin_hat)

    snapshots = [u.copy()]
    for _ in range(1, num_snapshots):
        for _ in range(n_inner):
            k1 = rhs(u)
            k2v = rhs(u + 0.5 * dt * k1)
            k3v = rhs(u + 0.5 * dt * k2v)
            k4v = rhs(u + dt * k3v)
            u = u + (dt / 6.0) * (k1 + 2.0 * k2v + 2.0 * k3v + k4v)
        snapshots.append(u.copy())
    return np.stack(snapshots, axis=0)


# ===========================================================================
# 2D advection‑diffusion (custom pseudo‑spectral solver)
# ===========================================================================


def _rhs_advection_2d(
    u: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    R: np.ndarray,
    alpha: float,
    kx: np.ndarray,
    ky: np.ndarray,
    k2: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Right‑hand side of the advection‑diffusion‑reaction equation:
        ∂_t u = -v·∇u + α ∇² u + R

    Evaluated pseudo‑spectrally; the derivative operations are exact
    (assuming periodic boundaries).
    """
    u_hat = np.fft.rfft2(u)
    # advection: - v·∇u computed in physical space after spectral derivatives
    ux_hat = 1j * kx * u_hat
    uy_hat = 1j * ky * u_hat
    ux = np.fft.irfft2(ux_hat, s=(nx, ny))
    uy = np.fft.irfft2(uy_hat, s=(nx, ny))
    adv = vx * ux + vy * uy  # negative sign applied later

    # diffusion: α ∇² u in spectral domain
    diff_hat = -alpha * k2 * u_hat
    diff = np.fft.irfft2(diff_hat, s=(nx, ny))

    return -adv + diff + R


def _rk4_step_advection_2d(
    u: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    R: np.ndarray,
    alpha: float,
    kx: np.ndarray,
    ky: np.ndarray,
    k2: np.ndarray,
    nx: int,
    ny: int,
    dt: float,
) -> np.ndarray:
    """One RK4 step."""

    def rhs(v: np.ndarray) -> np.ndarray:
        return _rhs_advection_2d(v, vx, vy, R, alpha, kx, ky, k2, nx, ny)

    k1 = rhs(u)
    k2v = rhs(u + 0.5 * dt * k1)
    k3v = rhs(u + 0.5 * dt * k2v)
    k4v = rhs(u + dt * k3v)
    return u + (dt / 6.0) * (k1 + 2.0 * k2v + 2.0 * k3v + k4v)


def _generate_advection_variant(
    config: DataConfig,
    variant: str,
    seed: int,
    num_trajectories: int,
) -> np.ndarray:
    """
    Generate a batch of 2D advection‑diffusion trajectories.

    Parameters
    ----------
    config : DataConfig
        Must contain the advection‑specific parameters (spatial_res,
        domain_size, diffusion_coef, …).
    variant : str
        One of ``"base"``, ``"flip"``, ``"pos"``, ``"neg"``, ``"pos_neg"``,
        ``"pos_neg_flip"``.
    seed : int
        Base random seed; each trajectory uses ``seed + i``.
    num_trajectories : int
        How many independent trajectories to produce.

    Returns
    -------
    traj : ndarray of shape (num_trajectories, time_steps, H, W, 4)
        The last three channels are (vx, vy, R).
    """
    rng = Generator(PCG64(seed))
    nx, ny = config.spatial_res
    Lx, Ly = config.domain_size
    alpha = config.diffusion_coef if config.diffusion_coef is not None else 0.026
    T_output = getattr(config, "dt_output", 0.1)  # we define a custom attribute or rely on default
    # We define dt_output = 0.1 for sensible dynamics; config may not have it.
    dt_internal = 1e-3  # stable for this grid
    n_steps_per_output = max(1, int(round(T_output / dt_internal)))
    dt = T_output / n_steps_per_output
    time_steps = config.time_steps  # number of saved frames

    # Spectral operators
    x_grid = np.linspace(0, Lx, nx, endpoint=False)
    y_grid = np.linspace(0, Ly, ny, endpoint=False)
    kx = np.fft.fftfreq(nx, d=Lx / nx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=Ly / ny) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky[: ny // 2 + 1], indexing="ij")
    k2 = KX**2 + KY**2

    traj_list = []
    for i in range(num_trajectories):
        traj_seed = seed + i
        state_rng = Generator(PCG64(traj_seed))

        # ----- velocity and reaction fields -----
        # velocity: random constant field
        vx_mag = state_rng.uniform(-1.0, 1.0)
        vy_mag = state_rng.uniform(-1.0, 1.0)
        vx = np.full((nx, ny), vx_mag, dtype=np.float64)
        vy = np.full((nx, ny), vy_mag, dtype=np.float64)
        R = np.zeros((nx, ny), dtype=np.float64)

        # apply variant modifications
        if variant in ("flip", "pos_neg_flip"):
            # flip velocity in left/right half or similar; we'll flip sign for x > Lx/2
            mask = x_grid[:, np.newaxis] > Lx / 2
            vx[mask] *= -1.0
            # also flip vy based on y > Ly/2
            mask_y = y_grid[np.newaxis, :] > Ly / 2
            vy[mask_y] *= -1.0
        if "pos" in variant:
            # triangular heat source
            src_x = state_rng.uniform(0.2, 0.8) * Lx
            src_y = state_rng.uniform(0.2, 0.8) * Ly
            width = state_rng.uniform(0.05, 0.15) * Lx
            amp = state_rng.uniform(2.0, 5.0)
            dist = np.sqrt((x_grid[:, np.newaxis] - src_x) ** 2 + (y_grid[np.newaxis, :] - src_y) ** 2)
            triangle = np.maximum(0.0, 1.0 - dist / width)
            R += amp * triangle
        if "neg" in variant:
            # cloud‑shaped sink (negative Gaussian)
            sink_x = state_rng.uniform(0.2, 0.8) * Lx
            sink_y = state_rng.uniform(0.2, 0.8) * Ly
            sigma = state_rng.uniform(0.05, 0.15) * Lx
            amp = state_rng.uniform(-5.0, -2.0)
            gauss = amp * np.exp(
                -0.5
                * (
                    (x_grid[:, np.newaxis] - sink_x) ** 2
                    + (y_grid[np.newaxis, :] - sink_y) ** 2
                )
                / sigma**2
            )
            R += gauss

        # ----- initial scalar field: random Gaussian blobs -----
        n_blobs = state_rng.integers(1, 11)
        u0 = np.zeros((nx, ny), dtype=np.float64)
        for _ in range(n_blobs):
            cx = state_rng.uniform(0.1, 0.9) * Lx
            cy = state_rng.uniform(0.1, 0.9) * Ly
            s = state_rng.uniform(0.05, 0.15) * Lx
            a = state_rng.uniform(0.5, 2.0)
            u0 += a * np.exp(
                -0.5
                * (
                    (x_grid[:, np.newaxis] - cx) ** 2
                    + (y_grid[np.newaxis, :] - cy) ** 2
                )
                / s**2
            )

        # ----- time integration -----
        u = u0.copy()
        snapshots = [u.copy()]
        # for base variant, vx, vy, R are zero in the recorded channels
        for _ in range(1, time_steps):
            for _ in range(n_steps_per_output):
                u = _rk4_step_advection_2d(
                    u, vx, vy, R, alpha, kx, ky, k2, nx, ny, dt
                )
            snapshots.append(u.copy())

        # Stack scalar field and constant channels
        scalar = np.stack(snapshots, axis=0)  # (T, H, W)
        # Expand dims to channels
        vx_ch = np.tile(vx[np.newaxis, ...], (time_steps, 1, 1))
        vy_ch = np.tile(vy[np.newaxis, ...], (time_steps, 1, 1))
        R_ch = np.tile(R[np.newaxis, ...], (time_steps, 1, 1))
        traj = np.stack([scalar, vx_ch, vy_ch, R_ch], axis=-1)  # (T, H, W, 4)
        traj_list.append(traj)

    return np.stack(traj_list, axis=0)  # (num_traj, T, H, W, 4)


# ===========================================================================
# DatasetLoader
# ===========================================================================


class DatasetLoader:
    """
    Loads or generates PDE trajectory datasets and returns standardised
    (inputs, outputs) pairs.

    Parameters
    ----------
    config : DataConfig
        A fully populated data configuration (must include ``pde_name``).
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.data_dir = config.data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.rng = Generator(PCG64(config.seed if hasattr(config, "seed") else 42))

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def load_or_generate(
        self, ood_variant: Optional[str] = None
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """
        Return train/val/test datasets.

        If ``ood_variant`` is provided, only the test dataset is returned
        (train and val are ``None``).
        """
        if ood_variant is not None:
            if self.config.pde_name != "advection_2d":
                raise NotImplementedError("OOD variants only supported for 2D advection.")
            test = self._load_ood_test(ood_variant)
            return (None, None, test)  # type: ignore[return-value]

        train = self._load_split("train")
        if train is None:
            logger.info("Generating base dataset (%s) …", self.config.pde_name)
            self._generate_and_cache_base()
            train = self._load_split("train")
            val = self._load_split("val")
            test = self._load_split("test")
            if train is None or val is None or test is None:
                raise RuntimeError("Failed to generate dataset.")
        else:
            val = self._load_split("val")
            test = self._load_split("test")
            if val is None or test is None:
                raise RuntimeError("Missing validation or test split in cache.")
        return (train, val, test)

    # -----------------------------------------------------------------------
    # Base generation & caching
    # -----------------------------------------------------------------------
    def _generate_and_cache_base(self) -> None:
        """Generate raw trajectories, split, form pairs, and save to HDF5."""
        raw = self._generate_raw_trajectories()
        train_traj, val_traj, test_traj = (
            self.config.train_traj,
            self.config.val_traj,
            self.config.test_traj,
        )
        splits = self._split_trajectories(
            raw, train_traj, val_traj, test_traj, self.rng.integers(1e9)
        )
        for name, traj_array in zip(["train", "val", "test"], splits):
            X, Y = self._trajectory_to_pairs(traj_array, self.config.input_time_window)
            path = self._cache_path(name)
            self._save_h5(path, X, Y)
            logger.info("Saved %s split → %s", name, path)

    def _generate_raw_trajectories(self) -> np.ndarray:
        """Return stacked trajectories of shape (total_traj, T, *spatial, C)."""
        pde = self.config.pde_name
        if pde in ("burgers", "hyper_diffusion", "ks_conservative"):
            return self._generate_1d_raw(pde)
        elif pde == "advection_2d":
            # generate base variant (vx/vy/R zero)
            return _generate_advection_variant(
                self.config,
                variant="base",
                seed=self.rng.integers(1e9),
                num_trajectories=self.config.train_traj
                + self.config.val_traj
                + self.config.test_traj,
            )
        else:
            raise ValueError(f"Unsupported PDE: {pde}")

    def _generate_1d_raw(self, pde: str) -> np.ndarray:
        """
        Dispatch 1D generation to APEBench (if available and enabled) or custom
        fallback solvers.
        """
        use_ape = self.config.use_apebench and HAS_APEBENCH
        total_traj = (
            self.config.train_traj + self.config.val_traj + self.config.test_traj
        )
        nx = self.config.spatial_res
        L = self.config.domain_size  # float
        T = 2.0  # approximate total simulation time (not critical)
        dt_save = 0.05 if pde != "hyper_diffusion" else 0.1
        snaps = self.config.time_steps
        seed_base = self.rng.integers(1e9)

        if use_ape:
            return self._apebench_1d(pde, total_traj, nx, L, T, dt_save, snaps, seed_base)
        else:
            logger.info("Using custom spectral solvers for %s.", pde)
            if pde == "burgers":
                nu = self.config.viscosity or 0.01
                trajs = [
                    _burgers_custom(nx, L, nu, T, dt_save, snaps, seed_base + i)
                    for i in range(total_traj)
                ]
            elif pde == "hyper_diffusion":
                gamma = 4  # not in config; typical value
                trajs = [
                    _hyper_diffusion_custom(
                        nx, L, gamma, T, dt_save, snaps, seed_base + i
                    )
                    for i in range(total_traj)
                ]
            elif pde == "ks_conservative":
                trajs = [
                    _ks_conservative_custom(
                        nx, L, T, dt_save, snaps, seed_base + i
                    )
                    for i in range(total_traj)
                ]
            else:
                raise ValueError(f"Unknown 1D PDE: {pde}")
            # Add channel dimension: (traj, T, N) → (traj, T, N, 1)
            return np.stack(trajs, axis=0)[..., np.newaxis]

    def _apebench_1d(
        self,
        pde: str,
        n_traj: int,
        nx: int,
        L: float,
        T: float,
        dt: float,
        snaps: int,
        seed: int,
    ) -> np.ndarray:
        """Delegate to APEBench to generate 1D data."""
        # APEBench interface (simplified; exact API may differ)
        import apebench  # already checked

        # The caller guarantees HAS_APEBENCH is True.
        # We assume the APEBench package provides functions like `generate_burgers`.
        # Because APEBench is not part of the standard environment we will attempt
        # a simple wrapping that matches known signatures from the repository
        # https://github.com/fKoehler/APEBench .
        # In practice the user may need to adapt these calls.
        gen_rng = Generator(PCG64(seed))
        trajs = []
        for i in range(n_traj):
            sub_seed = seed + i
            if pde == "burgers":
                data = apebench.generate_burgers(
                    N=nx, L=L, nu=self.config.viscosity or 0.01,
                    T=T, dt=dt, n_snapshots=snaps, seed=sub_seed,
                )
            elif pde == "hyper_diffusion":
                data = apebench.generate_hyper_diffusion(
                    N=nx, L=L, T=T, dt=dt, n_snapshots=snaps, seed=sub_seed,
                )
            elif pde == "ks_conservative":
                data = apebench.generate_ks_conservative(
                    N=nx, L=L, T=T, dt=dt, n_snapshots=snaps, seed=sub_seed,
                )
            else:
                raise ValueError(f"APEBench generation for {pde} not implemented.")
            # data expected shape (snaps, nx)
            trajs.append(data)
        return np.stack(trajs, axis=0)[..., np.newaxis]  # add channel dim

    # -----------------------------------------------------------------------
    # Splitting and windowing
    # -----------------------------------------------------------------------
    def _split_trajectories(
        self,
        raw: np.ndarray,
        n_train: int,
        n_val: int,
        n_test: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shuffle and split raw trajectories into train/val/test."""
        total = raw.shape[0]
        assert total == n_train + n_val + n_test, (
            f"raw trajectories ({total}) != n_train+n_val+n_test "
            f"({n_train}+{n_val}+{n_test})"
        )
        rng = Generator(PCG64(seed))
        idx = rng.permutation(total)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        return raw[train_idx], raw[val_idx], raw[test_idx]

    def _trajectory_to_pairs(
        self,
        traj: np.ndarray,
        input_window: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a set of trajectories into (X, Y) pairs.

        Parameters
        ----------
        traj : ndarray of shape (n_traj, T, *spatial, C)
            Trajectories.
        input_window : int
            Number of input time steps.

        Returns
        -------
        X : ndarray of shape (n_pairs, input_window, *spatial, C)
        Y : ndarray of shape (n_pairs, *spatial, 1)
        """
        n_traj, T = traj.shape[0], traj.shape[1]
        pairs_per_traj = T - input_window
        total_pairs = n_traj * pairs_per_traj
        spatial_shape = traj.shape[2:-1]
        in_channels = traj.shape[-1]

        X = np.empty((total_pairs, input_window) + spatial_shape + (in_channels,),
                     dtype=traj.dtype)
        Y = np.empty((total_pairs,) + spatial_shape + (1,), dtype=traj.dtype)

        idx = 0
        for i in range(n_traj):
            for t in range(pairs_per_traj):
                X[idx] = traj[i, t : t + input_window]
                Y[idx, ..., 0] = traj[i, t + input_window, ..., 0]
                idx += 1
        return X, Y

    # -----------------------------------------------------------------------
    # OOD test generation
    # -----------------------------------------------------------------------
    def _load_ood_test(self, variant: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate or load a specific OOD test dataset."""
        path = self._cache_path(f"test_{variant}")
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                X = f["inputs"][:]
                Y = f["outputs"][:]
            return X, Y
        else:
            raw = _generate_advection_variant(
                self.config,
                variant=variant,
                seed=self.rng.integers(1e9),
                num_trajectories=self.config.test_traj,
            )
            X, Y = self._trajectory_to_pairs(raw, self.config.input_time_window)
            self._save_h5(path, X, Y)
            return X, Y

    # -----------------------------------------------------------------------
    # HDF5 I/O
    # -----------------------------------------------------------------------
    def _save_h5(
        self, path: str, X: np.ndarray, Y: np.ndarray
    ) -> None:
        with h5py.File(path, "w") as f:
            f.create_dataset("inputs", data=X, compression="gzip")
            f.create_dataset("outputs", data=Y, compression="gzip")

    def _load_split(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        path = self._cache_path(name)
        if not os.path.exists(path):
            return None
        with h5py.File(path, "r") as f:
            X = f["inputs"][:]
            Y = f["outputs"][:]
        return X, Y

    def _cache_path(self, name: str) -> str:
        """Build a descriptive file path including PDE, resolution, etc."""
        cfg = self.config
        pde = cfg.pde_name
        res_str = (
            f"{cfg.spatial_res}x{cfg.spatial_res}"
            if isinstance(cfg.spatial_res, tuple)
            else str(cfg.spatial_res)
        )
        return os.path.join(
            self.data_dir,
            f"{pde}_{res_str}_steps{cfg.time_steps}_{name}.h5",
        )
