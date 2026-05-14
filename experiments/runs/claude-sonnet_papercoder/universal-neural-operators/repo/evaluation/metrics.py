## evaluation/metrics.py
"""
Standalone evaluation metrics for the multi-physics neural operator
pretraining framework described in:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

Implements three evaluation metrics as static methods on the ``Metrics``
class:
  - ``nmae``:       Range-normalized mean absolute error, equation (3) in
                    the paper. Returned as a percentage.
  - ``mse``:        Mean squared error over all elements.
  - ``relative_l2``: Per-sample relative L2 norm, averaged over the batch.
  - ``compute_all``: Convenience wrapper that returns all three metrics in
                    a single dict.

Design contract (Data structures and interfaces):
  Metrics:
    nmae(pred, target, eps=1e-8) -> float
    mse(pred, target) -> float
    relative_l2(pred, target, eps=1e-8) -> float
    compute_all(pred, target, eps=1e-8) -> Dict[str, float]

Tensor layout convention (Shared Knowledge):
  All spatial fields are channel-first: [B, C, H, W] for 2D or [B, C, L]
  for 1D. Methods handle arbitrary spatial shape [B, C, *spatial] via
  reshape(B, -1).

Config alignment:
  evaluation.nmae_eps: 1.0e-8  -> default eps for nmae() and compute_all().
  evaluation.metrics: ["nmae", "mse"]  -> both are implemented here.

Separation from training/losses.py:
  training/losses.py contains nn.Module subclasses returning tensors for
  backpropagation. This module returns Python floats for reporting. The NMAE
  formula is identical in both; the difference is tensor vs. float output.

NO imports from other project files. torch and standard library only.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


class Metrics:
    """Evaluation metrics for neural operator predictions.

    All methods are static — no instance state is required. The class acts
    as a namespace grouping related metric computations. Usage::

        # As static methods:
        nmae_pct = Metrics.nmae(pred, target, eps=1e-8)
        mse_val  = Metrics.mse(pred, target)
        rel_l2   = Metrics.relative_l2(pred, target)
        all_metrics = Metrics.compute_all(pred, target)

        # Or via an instance (equivalent):
        m = Metrics()
        all_metrics = m.compute_all(pred, target)

    Tensor layout:
        All methods accept tensors of shape ``[B, C, *spatial]`` where:
          - ``B`` = batch size
          - ``C`` = number of output channels
          - ``*spatial`` = one or more spatial dimensions (e.g. ``(L,)``
            for 1D, ``(H, W)`` for 2D, ``(D, H, W)`` for 3D).

        The spatial and channel dimensions are jointly flattened to a single
        dimension ``N = C * prod(spatial)`` for per-sample computations.

    Aggregation note:
        All methods operate on a single tensor containing the **full** set of
        samples (or a batch). The ``Evaluator`` is responsible for
        concatenating per-batch predictions before calling these methods to
        ensure NMAE is computed over the full test set ``|D_test|`` as
        specified in equation (3) of the paper.
    """

    # -----------------------------------------------------------------------
    # NMAE — equation (3) from the paper
    # -----------------------------------------------------------------------

    @staticmethod
    def nmae(
        pred: Tensor,
        target: Tensor,
        eps: float = 1e-8,
    ) -> float:
        """Range-normalized mean absolute error (equation 3 in the paper).

        Implements the NMAE formula exactly as defined in the paper:

            NMAE(θ) = (1/|D_test|) * Σ_{(a,u)∈D_test}
                      ||G_θ(a) - u||_{1,G} / (max_G(u) - min_G(u) + ε)

        where ``||·||_{1,G}`` is the **mean** L1 norm over the spatial grid G
        (sum of absolute errors divided by the number of grid points), and
        ``max_G(u) - min_G(u)`` is the value range of the **target** field
        per sample.

        Per-sample computation:
            1. Flatten ``[C, *spatial]`` to a 1-D vector of length
               ``N = C * prod(spatial)``.
            2. Numerator: ``mean(|pred_i - target_i|)`` over N elements.
            3. Denominator: ``max(target_i) - min(target_i) + eps`` over N
               elements.
            4. Divide numerator by denominator.
        Then average over the batch and multiply by 100 to get a percentage.

        The ``eps`` value of ``1e-8`` matches ``evaluation.nmae_eps`` in
        ``config.yaml``. It prevents division by zero when the target field
        is spatially constant (value range = 0).

        Args:
            pred: Model predictions with shape ``[B, C, *spatial]``.
            target: Ground-truth values with the same shape as ``pred``.
            eps: Small positive constant added to the value-range denominator
                to prevent division by zero. Default ``1e-8`` matches
                ``evaluation.nmae_eps`` in ``config.yaml``.

        Returns:
            NMAE as a Python ``float`` in **percentage** units. For example,
            a return value of ``0.0120`` means 0.0120 %, matching the
            notation in Table 1 of the paper.

        Raises:
            ValueError: If ``pred`` and ``target`` have different shapes.
            ValueError: If the input tensors have fewer than 2 dimensions
                (batch dimension plus at least one other dimension required).

        Example::

            pred   = torch.randn(16, 1, 64, 64)   # [B, C, H, W]
            target = torch.randn(16, 1, 64, 64)
            nmae_pct = Metrics.nmae(pred, target)  # e.g. 0.0120
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape; "
                f"got pred={tuple(pred.shape)} and target={tuple(target.shape)}."
            )
        if pred.dim() < 2:
            raise ValueError(
                f"Input tensors must have at least 2 dimensions [B, ...]; "
                f"got {pred.dim()} dimension(s)."
            )

        batch_size: int = pred.shape[0]

        # ── Step 1: Flatten all non-batch dims to [B, N] ──────────────────
        # N = C * prod(spatial_dims).  This handles 1D, 2D, 3D uniformly.
        pred_flat: Tensor = pred.reshape(batch_size, -1)      # [B, N]
        target_flat: Tensor = target.reshape(batch_size, -1)  # [B, N]

        # ── Step 2: Per-sample mean absolute error over the grid ──────────
        # ||G_θ(a) - u||_{1,G} = mean_{grid} |pred - target|
        # Shape: [B]
        abs_error: Tensor = torch.abs(pred_flat - target_flat)  # [B, N]
        l1_per_sample: Tensor = abs_error.mean(dim=1)           # [B]

        # ── Step 3: Per-sample value range of the target ──────────────────
        # max_G(u) - min_G(u) computed over all grid points for each sample.
        # Uses TARGET only (not prediction), consistent with equation (3).
        # Shape: [B]
        target_max: Tensor = target_flat.max(dim=1).values   # [B]
        target_min: Tensor = target_flat.min(dim=1).values   # [B]
        value_range: Tensor = target_max - target_min        # [B]

        # ── Step 4: Per-sample normalized error ───────────────────────────
        # Divide mean L1 error by value range (+ eps for numerical safety).
        # Shape: [B]
        nmae_per_sample: Tensor = l1_per_sample / (value_range + eps)  # [B]

        # ── Step 5: Mean over batch, convert to percentage ─────────────────
        # Paper reports NMAE as a percentage (e.g. 0.0120 in Table 1 means
        # 0.0120 %, corresponding to a raw ratio of 0.000120).
        nmae_pct: float = (nmae_per_sample.mean() * 100.0).item()

        return nmae_pct

    # -----------------------------------------------------------------------
    # MSE — primary metric reported in Tables 1 and 2
    # -----------------------------------------------------------------------

    @staticmethod
    def mse(
        pred: Tensor,
        target: Tensor,
    ) -> float:
        """Mean squared error over all elements.

        Computes:

            MSE = mean_{B, C, *spatial}( (pred - target)^2 )

        This is the standard MSE averaged over all elements in the tensor
        (batch × channels × spatial dimensions). It is the primary metric
        reported in Tables 1 and 2 of the paper (values on the order of
        ``1e-7`` to ``1e-5``).

        Note: The paper reports MSE values on the (potentially normalized)
        prediction space. The ``Evaluator`` is responsible for denormalizing
        predictions before calling this method if raw-space metrics are
        required.

        Args:
            pred: Model predictions with shape ``[B, C, *spatial]``.
            target: Ground-truth values with the same shape as ``pred``.

        Returns:
            MSE as a Python ``float``.

        Raises:
            ValueError: If ``pred`` and ``target`` have different shapes.
            ValueError: If the input tensors have fewer than 2 dimensions.

        Example::

            pred   = torch.randn(16, 1, 64, 64)
            target = torch.randn(16, 1, 64, 64)
            mse_val = Metrics.mse(pred, target)  # e.g. 1.009e-7
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape; "
                f"got pred={tuple(pred.shape)} and target={tuple(target.shape)}."
            )
        if pred.dim() < 2:
            raise ValueError(
                f"Input tensors must have at least 2 dimensions [B, ...]; "
                f"got {pred.dim()} dimension(s)."
            )

        # Compute MSE over all elements (batch × channels × spatial).
        # Using torch.nn.functional.mse_loss with reduction='mean' is
        # equivalent to ((pred - target) ** 2).mean() but avoids creating
        # an intermediate squared-difference tensor of the same size.
        mse_val: float = torch.nn.functional.mse_loss(
            pred, target, reduction="mean"
        ).item()

        return mse_val

    # -----------------------------------------------------------------------
    # Relative L2 — standard FNO literature alternative
    # -----------------------------------------------------------------------

    @staticmethod
    def relative_l2(
        pred: Tensor,
        target: Tensor,
        eps: float = 1e-8,
    ) -> float:
        """Per-sample relative L2 norm, averaged over the batch.

        Computes:

            rel_l2 = mean_{B}( ||pred_i - target_i||_2 / (||target_i||_2 + ε) )

        where the L2 norms are computed over all non-batch dimensions
        (channels × spatial) for each sample ``i``.

        This is the standard training and evaluation metric used in the
        original FNO paper (Li et al., 2021). It is scale-invariant with
        respect to the target magnitude, making it useful when target values
        vary significantly across samples or physics problems. It is not
        explicitly defined in the paper being reproduced but is included as
        a standard FNO-literature alternative.

        Args:
            pred: Model predictions with shape ``[B, C, *spatial]``.
            target: Ground-truth values with the same shape as ``pred``.
            eps: Small positive constant added to the target L2 norm
                denominator to prevent division by zero when the target is
                identically zero. Default ``1e-8``.

        Returns:
            Mean relative L2 error as a Python ``float`` (dimensionless
            ratio, not a percentage).

        Raises:
            ValueError: If ``pred`` and ``target`` have different shapes.
            ValueError: If the input tensors have fewer than 2 dimensions.

        Example::

            pred   = torch.randn(16, 2, 64, 64)
            target = torch.randn(16, 2, 64, 64)
            rel_l2 = Metrics.relative_l2(pred, target)  # e.g. 0.042
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape; "
                f"got pred={tuple(pred.shape)} and target={tuple(target.shape)}."
            )
        if pred.dim() < 2:
            raise ValueError(
                f"Input tensors must have at least 2 dimensions [B, ...]; "
                f"got {pred.dim()} dimension(s)."
            )

        batch_size: int = pred.shape[0]

        # ── Flatten all non-batch dims to [B, N] ──────────────────────────
        pred_flat: Tensor = pred.reshape(batch_size, -1)      # [B, N]
        target_flat: Tensor = target.reshape(batch_size, -1)  # [B, N]

        # ── Per-sample L2 norm of the error ───────────────────────────────
        # ||pred_i - target_i||_2 for each sample in the batch.
        # Shape: [B]
        diff: Tensor = pred_flat - target_flat                          # [B, N]
        error_norm: Tensor = torch.norm(diff, p=2, dim=1)              # [B]

        # ── Per-sample L2 norm of the target ──────────────────────────────
        # ||target_i||_2 + eps to avoid division by zero.
        # Shape: [B]
        target_norm: Tensor = (
            torch.norm(target_flat, p=2, dim=1) + eps
        )  # [B]

        # ── Per-sample relative L2 error ──────────────────────────────────
        rel_l2_per_sample: Tensor = error_norm / target_norm  # [B]

        # ── Mean over batch ────────────────────────────────────────────────
        rel_l2_val: float = rel_l2_per_sample.mean().item()

        return rel_l2_val

    # -----------------------------------------------------------------------
    # compute_all — convenience wrapper
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_all(
        pred: Tensor,
        target: Tensor,
        eps: float = 1e-8,
    ) -> Dict[str, float]:
        """Compute all three metrics and return them in a single dict.

        Calls :meth:`nmae`, :meth:`mse`, and :meth:`relative_l2`
        independently (no shared intermediate computations) so that each
        method remains independently testable and the results are identical
        to calling each method separately.

        The returned dict structure is used by:
          - ``Evaluator.evaluate()`` to aggregate results across batches.
          - ``ResultsTable.add_row()`` in ``utils/logging_utils.py`` to
            format Tables 1 and 2 (expects keys ``'mse'`` and ``'nmae'``).

        Dict keys and units:
          - ``'nmae'``:       float, **percentage** (e.g. ``0.0120`` means
                              0.0120 %, matching Table 1 notation).
          - ``'mse'``:        float, raw MSE (e.g. ``1.009e-7``).
          - ``'relative_l2'``: float, dimensionless ratio (e.g. ``0.042``).

        The ``eps`` parameter is forwarded to both :meth:`nmae` and
        :meth:`relative_l2`. It defaults to ``1e-8``, matching
        ``evaluation.nmae_eps`` in ``config.yaml``.

        Args:
            pred: Model predictions with shape ``[B, C, *spatial]``.
            target: Ground-truth values with the same shape as ``pred``.
            eps: Small positive constant for numerical stability in NMAE
                and relative L2 denominators. Default ``1e-8``.

        Returns:
            Dict with keys ``'nmae'`` (float, percentage), ``'mse'``
            (float), and ``'relative_l2'`` (float).

        Raises:
            ValueError: If ``pred`` and ``target`` have different shapes.
            ValueError: If the input tensors have fewer than 2 dimensions.

        Example::

            pred   = torch.randn(100, 2, 64, 64)  # full test set
            target = torch.randn(100, 2, 64, 64)
            results = Metrics.compute_all(pred, target)
            # results = {
            #     'nmae': 0.0120,       # percentage
            #     'mse': 1.009e-7,      # raw MSE
            #     'relative_l2': 0.042  # dimensionless
            # }
        """
        # Validate shapes once here; individual methods also validate, but
        # doing it here avoids redundant error messages when all three are
        # called in sequence.
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape; "
                f"got pred={tuple(pred.shape)} and target={tuple(target.shape)}."
            )
        if pred.dim() < 2:
            raise ValueError(
                f"Input tensors must have at least 2 dimensions [B, ...]; "
                f"got {pred.dim()} dimension(s)."
            )

        nmae_val: float = Metrics.nmae(pred, target, eps=eps)
        mse_val: float = Metrics.mse(pred, target)
        rel_l2_val: float = Metrics.relative_l2(pred, target, eps=eps)

        return {
            "nmae": nmae_val,
            "mse": mse_val,
            "relative_l2": rel_l2_val,
        }
