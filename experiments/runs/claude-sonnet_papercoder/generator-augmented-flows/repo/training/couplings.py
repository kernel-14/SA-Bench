## training/couplings.py
"""Data-noise coupling strategies for iCT-GC consistency model training.

This module implements the ``Coupling`` class, which handles the initial
data-noise pairing strategy ``(x★, z)`` before any model-based augmentation.

Three coupling modes are supported, corresponding to the three model variants
compared in Table 1 of the paper:

1. **IC (Independent Coupling)**: ``q_I = p★(x★) · p_z(z)`` — the standard
   baseline (iCT-IC). Data and noise are paired independently.

2. **OT (Minibatch Optimal Transport)**: The iCT-OT baseline from Dou et al.
   (2024) / Pooladian et al. (2023). Finds a permutation of the noise batch
   that minimizes total L2 transport cost ``Σ_i ||x★_i - z_{π(i)}||²``.

3. **GC (Generator-Augmented)**: Delegates to IC here — the actual GC
   augmentation (endpoint prediction + re-coupling) is performed inside
   ``trainer.py`` using the consistency model. The ``Coupling`` class only
   handles the initial ``(x★, z)`` pairing.

The key insight from the paper (Section 4.1): GC's coupling is constructed
dynamically using the model's prediction ``x̂_ti = sg(f_θ(x_ti, σ_ti))``.
This model-dependent step belongs in the trainer, not here. So ``Coupling``
with mode ``'gc'`` simply returns the IC pair; the trainer applies the
generator-augmented transformation on top via the Bernoulli mixing mask ``m``.

Config values used (from config.yaml):
    coupling:      gc          (default for iCT-GC experiments)
    ot_batch_size: 512         (CIFAR-10, ImageNet-32) or 128 (CelebA, LSUN)

Typical usage in trainer.py::

    coupling = Coupling(mode=config.coupling, ot_batch_size=config.ot_batch_size)

    # Inside training step:
    x_star, z = coupling.get_coupled_pairs(x_star, z)
    # For 'ic' and 'gc': x_star, z unchanged
    # For 'ot': z is permuted to reduce transport cost

References:
    - Pooladian et al. (2023): Multisample flow matching with minibatch OT
    - Dou et al. (2024): OT coupling in consistency models
    - POT library: https://pythonot.github.io/
    - scipy.optimize.linear_sum_assignment: Hungarian algorithm
"""

import warnings
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------

try:
    import ot as pot  # Python Optimal Transport (POT) library
    _POT_AVAILABLE: bool = True
except ImportError:
    _POT_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE: bool = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Coupling class
# ---------------------------------------------------------------------------


class Coupling:
    """Data-noise coupling strategy for consistency model training.

    Handles the initial ``(x★, z)`` pairing before any model-based
    augmentation. For the GC mode, this class returns the IC pair unchanged;
    the generator-augmented transformation is applied in ``trainer.py``.

    The class is stateless after ``__init__`` — calling ``get_coupled_pairs``
    multiple times with the same inputs returns identical results (for IC/GC)
    or deterministically optimal results (for OT, given the same batch).

    Attributes:
        mode: Coupling strategy identifier. One of ``'ic'``, ``'ot'``,
            ``'gc'``. Default ``'gc'`` (used for iCT-GC experiments).
        ot_batch_size: Reference batch size for OT solver selection.
            Batches with ``B <= ot_batch_size`` use Hungarian (exact);
            larger batches use Sinkhorn (approximate). Only relevant when
            ``mode='ot'``. Default 512.
    """

    # Threshold for switching from Hungarian to Sinkhorn.
    # Hungarian is O(B³); for B > 256 it becomes slow on CPU.
    _HUNGARIAN_MAX_BATCH: int = 256

    # Sinkhorn regularization parameter.
    # Smaller values → closer to exact OT but slower convergence.
    # 0.1 is a standard choice for image data.
    _SINKHORN_REG: float = 0.1

    # Number of Sinkhorn iterations.
    _SINKHORN_NUMITER: int = 100

    def __init__(
        self,
        mode: str = "gc",
        ot_batch_size: int = 512,
    ) -> None:
        """Initialise the coupling strategy.

        Args:
            mode: Coupling strategy to use. Must be one of:
                - ``'ic'``: Independent coupling ``q_I = p★ × p_z``.
                  Returns ``(x_star, z)`` unchanged. Used for iCT-IC baseline.
                - ``'gc'``: Generator-augmented coupling. Returns ``(x_star, z)``
                  unchanged here; the GC augmentation is applied in trainer.py.
                  Used for iCT-GC experiments (default).
                - ``'ot'``: Minibatch optimal transport coupling. Permutes ``z``
                  to minimise total L2 transport cost. Used for iCT-OT baseline.
            ot_batch_size: Reference batch size for OT solver selection.
                Batches with ``B <= _HUNGARIAN_MAX_BATCH`` use Hungarian;
                larger batches use Sinkhorn. Only used when ``mode='ot'``.
                Config values: 512 (CIFAR-10, ImageNet-32), 128 (CelebA, LSUN).

        Raises:
            ValueError: If ``mode`` is not one of ``'ic'``, ``'ot'``, ``'gc'``.
            ValueError: If ``ot_batch_size < 1``.
        """
        _VALID_MODES: Tuple[str, ...] = ("ic", "ot", "gc")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown coupling mode '{mode}'. "
                f"Must be one of: {_VALID_MODES}. "
                "Config default: coupling = gc."
            )

        if ot_batch_size < 1:
            raise ValueError(
                f"ot_batch_size must be at least 1, got {ot_batch_size}. "
                "Config values: 512 (CIFAR-10/ImageNet-32), 128 (CelebA/LSUN)."
            )

        self.mode: str = mode
        self.ot_batch_size: int = int(ot_batch_size)

        # Warn at init time if OT mode is requested but dependencies are missing
        if mode == "ot" and not _SCIPY_AVAILABLE and not _POT_AVAILABLE:
            warnings.warn(
                "Coupling mode='ot' requires either scipy or the POT library. "
                "Neither is available. Install with: "
                "pip install scipy POT. "
                "OT coupling will fall back to independent coupling.",
                ImportWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_coupled_pairs(
        self,
        x_star: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a coupled ``(x★, z)`` pair according to the coupling strategy.

        Dispatches to the appropriate coupling implementation based on
        ``self.mode``. For ``'ic'`` and ``'gc'`` modes, the inputs are
        returned unchanged. For ``'ot'`` mode, ``z`` is permuted to reduce
        the total L2 transport cost between ``x_star`` and ``z``.

        The returned tensors are on the same device as the inputs. No
        gradient computation is performed here — coupling is a preprocessing
        step, not part of the computational graph.

        Args:
            x_star: Batch of real data samples of shape ``(B, C, H, W)``
                in ``[-1, 1]``. Sampled from the training dataset.
            z: Batch of noise samples of shape ``(B, C, H, W)``.
                Sampled from ``N(0, I)`` by the trainer.

        Returns:
            Tuple ``(x_star_coupled, z_coupled)`` where:
            - For ``'ic'`` and ``'gc'``: both tensors are the unchanged inputs.
            - For ``'ot'``: ``x_star_coupled`` is unchanged, ``z_coupled`` is
              a permutation of ``z`` that minimises ``Σ_i ||x★_i - z_{π(i)}||²``.

        Raises:
            ValueError: If ``x_star`` and ``z`` have different batch sizes.
            RuntimeError: If ``x_star`` and ``z`` are on different devices.
        """
        if x_star.shape[0] != z.shape[0]:
            raise ValueError(
                f"x_star and z must have the same batch size. "
                f"Got x_star.shape[0]={x_star.shape[0]} and "
                f"z.shape[0]={z.shape[0]}."
            )

        if x_star.device != z.device:
            raise RuntimeError(
                f"x_star and z must be on the same device. "
                f"Got x_star.device={x_star.device} and "
                f"z.device={z.device}."
            )

        if self.mode in ("ic", "gc"):
            return self._independent_coupling(x_star, z)
        elif self.mode == "ot":
            return self._minibatch_ot_coupling(x_star, z)
        else:
            # Unreachable due to __init__ validation; defensive fallback.
            raise ValueError(
                f"Unknown coupling mode '{self.mode}'. "
                "This should not happen — check __init__ validation."
            )

    # ------------------------------------------------------------------
    # Private coupling implementations
    # ------------------------------------------------------------------

    def _independent_coupling(
        self,
        x_star: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the independent coupling ``q_I = p★(x★) · p_z(z)``.

        This is the standard baseline coupling where data and noise are
        paired independently. No computation is performed — the inputs are
        returned as-is.

        This is also the initial pairing used by GC: the generator-augmented
        transformation in ``trainer.py`` builds on top of this IC pair by
        predicting a cleaner endpoint ``x̂_ti = sg(f_θ(x_ti, σ_ti))`` and
        re-coupling it with the same noise vector ``z``.

        Args:
            x_star: Batch of real data samples of shape ``(B, C, H, W)``.
            z: Batch of noise samples of shape ``(B, C, H, W)``.

        Returns:
            Tuple ``(x_star, z)`` — the unchanged inputs.
        """
        return x_star, z

    def _minibatch_ot_coupling(
        self,
        x_star: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the minibatch OT coupling that minimises L2 transport cost.

        Finds a permutation ``π`` of the noise batch that minimises the total
        squared L2 transport cost:
            min_π Σ_i ||x★_i - z_{π(i)}||²

        For small batches (B ≤ ``_HUNGARIAN_MAX_BATCH``), uses the exact
        Hungarian algorithm via ``scipy.optimize.linear_sum_assignment``.
        For larger batches, uses the Sinkhorn algorithm via the POT library
        (approximate but faster).

        If neither scipy nor POT is available, falls back to independent
        coupling with a warning.

        Implementation details:
        1. Flatten ``x_star`` and ``z`` to 2D for cost computation.
        2. Compute pairwise L2 cost matrix on GPU via ``torch.cdist``.
        3. Move cost matrix to CPU numpy for scipy/POT.
        4. Solve assignment problem to get permutation ``col_ind``.
        5. Reorder ``z`` by the permutation.
        6. Return ``(x_star, z_reordered)``.

        Args:
            x_star: Batch of real data samples of shape ``(B, C, H, W)``
                in ``[-1, 1]``.
            z: Batch of noise samples of shape ``(B, C, H, W)`` from
                ``N(0, I)``.

        Returns:
            Tuple ``(x_star, z_reordered)`` where ``z_reordered`` is a
            permutation of ``z`` that minimises the total L2 transport cost.
            ``x_star`` is returned unchanged.
        """
        batch_size: int = x_star.shape[0]

        # Step 1: Flatten to 2D for cost computation
        # x_flat: (B, C*H*W), z_flat: (B, C*H*W)
        x_flat: torch.Tensor = x_star.view(batch_size, -1).float()
        z_flat: torch.Tensor = z.view(batch_size, -1).float()

        # Step 2: Compute pairwise squared L2 cost matrix on GPU
        # C[i, j] = ||x★_i - z_j||²
        # torch.cdist computes L2 distances; squaring gives L2² cost
        # Shape: (B, B)
        cost_matrix: torch.Tensor = torch.cdist(
            x_flat, z_flat, p=2.0
        ).pow(2)

        # Step 3: Move to CPU numpy for scipy/POT solvers
        # Use float64 for numerical stability in OT solvers
        cost_np: np.ndarray = cost_matrix.detach().cpu().numpy().astype(
            np.float64
        )

        # Step 4: Solve assignment problem
        # Choose solver based on batch size and available libraries
        col_ind: np.ndarray = self._solve_assignment(
            cost_np=cost_np,
            batch_size=batch_size,
        )

        # Step 5: Reorder z by the optimal permutation
        # col_ind[i] = j means sample i in x_star is matched to sample j in z
        col_ind_tensor: torch.Tensor = torch.from_numpy(col_ind).long().to(
            z.device
        )
        z_reordered: torch.Tensor = z[col_ind_tensor]

        # Step 6: Return (x_star unchanged, z permuted)
        return x_star, z_reordered

    def _sinkhorn_coupling(
        self,
        x_star: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the Sinkhorn-based OT coupling (Sinkhorn-only path).

        A cleaner Sinkhorn-only implementation without the Hungarian fallback.
        Uses the POT library's ``ot.sinkhorn`` to compute a regularised
        optimal transport plan, then extracts a hard permutation via argmax.

        This method is provided as a standalone alternative to
        ``_minibatch_ot_coupling`` for cases where the user explicitly wants
        Sinkhorn without the automatic solver selection logic.

        Args:
            x_star: Batch of real data samples of shape ``(B, C, H, W)``.
            z: Batch of noise samples of shape ``(B, C, H, W)``.

        Returns:
            Tuple ``(x_star, z_reordered)`` where ``z_reordered`` is a
            permutation of ``z`` derived from the Sinkhorn transport plan.

        Raises:
            ImportError: If the POT library is not installed.
        """
        if not _POT_AVAILABLE:
            raise ImportError(
                "The POT library is required for _sinkhorn_coupling. "
                "Install it with: pip install POT==0.9.1"
            )

        batch_size: int = x_star.shape[0]

        # Flatten to 2D and compute cost matrix
        x_flat: torch.Tensor = x_star.view(batch_size, -1).float()
        z_flat: torch.Tensor = z.view(batch_size, -1).float()

        cost_matrix: torch.Tensor = torch.cdist(
            x_flat, z_flat, p=2.0
        ).pow(2)

        cost_np: np.ndarray = cost_matrix.detach().cpu().numpy().astype(
            np.float64
        )

        # Normalise cost for Sinkhorn numerical stability
        cost_max: float = float(cost_np.max())
        if cost_max > 0.0:
            cost_normalized: np.ndarray = cost_np / cost_max
        else:
            cost_normalized = cost_np

        # Uniform marginal distributions
        a: np.ndarray = np.ones(batch_size, dtype=np.float64) / batch_size
        b: np.ndarray = np.ones(batch_size, dtype=np.float64) / batch_size

        # Compute Sinkhorn transport plan: shape (B, B)
        # Each row sums to 1/B; entry T[i,j] is the mass transported
        # from x★_i to z_j.
        transport_plan: np.ndarray = pot.sinkhorn(
            a=a,
            b=b,
            M=cost_normalized,
            reg=self._SINKHORN_REG,
            numItermax=self._SINKHORN_NUMITER,
            warn=False,
        )

        # Extract hard permutation via argmax over columns
        # col_ind[i] = argmax_j T[i, j] = best match for x★_i
        col_ind: np.ndarray = np.argmax(transport_plan, axis=1)

        # Reorder z by permutation
        col_ind_tensor: torch.Tensor = torch.from_numpy(col_ind).long().to(
            z.device
        )
        z_reordered: torch.Tensor = z[col_ind_tensor]

        return x_star, z_reordered

    # ------------------------------------------------------------------
    # Private solver selection helper
    # ------------------------------------------------------------------

    def _solve_assignment(
        self,
        cost_np: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """Select and run the appropriate OT assignment solver.

        Solver selection logic:
        - If ``batch_size <= _HUNGARIAN_MAX_BATCH`` and scipy is available:
          use Hungarian algorithm (exact, O(B³)).
        - Otherwise, if POT is available: use Sinkhorn (approximate, faster).
        - If neither is available: fall back to identity permutation (IC)
          with a warning.

        The cost matrix is normalised before passing to Sinkhorn for
        numerical stability. Hungarian does not require normalisation.

        Args:
            cost_np: Pairwise squared L2 cost matrix of shape ``(B, B)``
                as a float64 numpy array. Entry ``[i, j]`` is
                ``||x★_i - z_j||²``.
            batch_size: Number of samples ``B``. Used for solver selection
                and for constructing the fallback identity permutation.

        Returns:
            Integer numpy array of shape ``(B,)`` representing the optimal
            permutation ``π``. Entry ``col_ind[i]`` is the index of the
            noise sample matched to data sample ``i``.
        """
        # --- Hungarian algorithm (exact, scipy) ---
        if batch_size <= self._HUNGARIAN_MAX_BATCH and _SCIPY_AVAILABLE:
            row_ind: np.ndarray
            col_ind: np.ndarray
            row_ind, col_ind = linear_sum_assignment(cost_np)
            # For a square matrix, row_ind is always [0, 1, ..., B-1].
            # col_ind is the optimal permutation.
            return col_ind

        # --- Sinkhorn algorithm (approximate, POT) ---
        if _POT_AVAILABLE:
            # Normalise cost for numerical stability
            cost_max: float = float(cost_np.max())
            if cost_max > 0.0:
                cost_normalized: np.ndarray = cost_np / cost_max
            else:
                cost_normalized = cost_np.copy()

            # Uniform marginal distributions
            a: np.ndarray = np.ones(batch_size, dtype=np.float64) / batch_size
            b: np.ndarray = np.ones(batch_size, dtype=np.float64) / batch_size

            # Compute Sinkhorn transport plan
            try:
                transport_plan: np.ndarray = pot.sinkhorn(
                    a=a,
                    b=b,
                    M=cost_normalized,
                    reg=self._SINKHORN_REG,
                    numItermax=self._SINKHORN_NUMITER,
                    warn=False,
                )
                # Hard assignment via argmax
                col_ind_sinkhorn: np.ndarray = np.argmax(
                    transport_plan, axis=1
                )
                return col_ind_sinkhorn

            except Exception as exc:
                warnings.warn(
                    f"Sinkhorn OT failed with error: {exc}. "
                    "Falling back to Hungarian algorithm if scipy is available, "
                    "otherwise using identity permutation.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                # Try Hungarian as fallback even for large batches
                if _SCIPY_AVAILABLE:
                    _, col_ind_fallback = linear_sum_assignment(cost_np)
                    return col_ind_fallback

        # --- Hungarian fallback for large batches (scipy only, no POT) ---
        if _SCIPY_AVAILABLE:
            warnings.warn(
                f"POT library not available for batch_size={batch_size} > "
                f"{self._HUNGARIAN_MAX_BATCH}. "
                "Using Hungarian algorithm (may be slow for large batches). "
                "Install POT for faster Sinkhorn: pip install POT==0.9.1",
                RuntimeWarning,
                stacklevel=3,
            )
            _, col_ind_hungarian: np.ndarray = linear_sum_assignment(cost_np)
            return col_ind_hungarian

        # --- Last resort: identity permutation (IC fallback) ---
        warnings.warn(
            "Neither scipy nor POT is available for OT coupling. "
            "Falling back to identity permutation (equivalent to IC). "
            "Install dependencies: pip install scipy POT",
            ImportWarning,
            stacklevel=3,
        )
        return np.arange(batch_size, dtype=np.int64)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a human-readable summary of the coupling configuration."""
        if self.mode == "ic":
            return (
                "Coupling(mode='ic')\n"
                "  Strategy: Independent coupling q_I = p★(x★) · p_z(z)\n"
                "  Used for: iCT-IC baseline"
            )
        elif self.mode == "gc":
            return (
                "Coupling(mode='gc')\n"
                "  Strategy: Independent coupling (GC augmentation in trainer)\n"
                "  Used for: iCT-GC experiments (default)"
            )
        elif self.mode == "ot":
            solver: str = (
                "Hungarian (exact)"
                if _SCIPY_AVAILABLE and self.ot_batch_size <= self._HUNGARIAN_MAX_BATCH
                else "Sinkhorn (approximate, POT)"
                if _POT_AVAILABLE
                else "identity fallback (no OT library)"
            )
            return (
                f"Coupling(mode='ot', ot_batch_size={self.ot_batch_size})\n"
                f"  Strategy: Minibatch OT — minimise Σ_i ||x★_i - z_{{π(i)}}||²\n"
                f"  Solver: {solver}\n"
                f"  Used for: iCT-OT baseline"
            )
        else:
            return f"Coupling(mode='{self.mode}')"
