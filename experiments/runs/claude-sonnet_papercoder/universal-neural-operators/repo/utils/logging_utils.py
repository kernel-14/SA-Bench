## utils/logging_utils.py
"""
Logging utilities for the multi-physics neural operator pretraining framework.

Provides:
  - get_logger: factory for consistently formatted Python loggers with optional
    file output. Used by all training, evaluation, and experiment modules.
  - ResultsTable: accumulates per-model metrics and formats them as a table
    matching Tables 1 and 2 in the paper (Model | MSE | NMAE (%) |
    Avg. epoch (s) | Param.).

Design contract (Data structures and interfaces):
  get_logger(name: str, log_file: Optional[str]) -> logging.Logger
  ResultsTable:
    add_row(model_name: str, metrics: Dict[str, float],
            epoch_time: float, n_params: Optional[int]) -> None
    print_table() -> None
    save_csv(path: str) -> None

NO imports from other project files. Standard library only.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Log format string — used for both console and file handlers.
# Format: [2024-01-15 12:34:56][module_name][INFO] message
LOG_FORMAT: str = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# Column widths for print_table() — chosen to accommodate the longest
# expected values from Tables 1 and 2 in the paper.
_COL_MODEL: int = 28
_COL_MSE: int = 16
_COL_NMAE: int = 12
_COL_EPOCH: int = 16
_COL_PARAMS: int = 14


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------


def get_logger(
    name: str,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create or retrieve a named logger with consistent formatting.

    The logger writes to the console (stderr) always, and optionally to a
    file when ``log_file`` is provided. Calling this function multiple times
    with the same ``name`` returns the same logger without adding duplicate
    handlers (idempotent).

    The logger level is set to ``DEBUG`` internally so that all messages are
    passed through; individual handlers can be reconfigured by callers if
    finer control is needed. ``config.yaml`` specifies ``logging.level:
    "INFO"`` — callers that want INFO-only output should call
    ``logger.setLevel(logging.INFO)`` after construction.

    ``logger.propagate`` is set to ``False`` to prevent double-logging when
    the root logger also has handlers (common in Jupyter environments and
    when multiple modules call ``get_logger``).

    Args:
        name: Logger name, typically the module or class name
            (e.g. ``'pretrainer'``, ``'evaluator'``, ``'main'``).
        log_file: Optional path to a log file. Parent directories are
            created automatically if they do not exist. The file is opened
            in append mode so logs accumulate across runs.

    Returns:
        A configured ``logging.Logger`` instance.

    Example::

        logger = get_logger("pretrainer", log_file="logs/pretrain.log")
        logger.info("Epoch 1 started")
    """
    logger: logging.Logger = logging.getLogger(name)

    # Idempotency guard: if handlers are already attached, return as-is.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Console handler (always present) ──────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler (optional) ────────────────────────────────────────────
    if log_file is not None:
        log_dir: str = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


class ResultsTable:
    """Accumulates per-model experiment results and formats them as a table.

    The output format mirrors Tables 1 and 2 in the paper:

    Table 1 (with parameter count):
        Model                        | MSE              | NMAE (%)     | Avg. epoch (s)  | Param.
        Mamba FNO (pretr.)           | 1.009e-07        | 0.0120       | 21.91           | ≈ 1e+07

    Table 2 (without parameter count):
        Model                        | MSE              | NMAE (%)     | Avg. epoch (s)
        Mamba FNO (pretr.)           | 3.910e-06        | 0.0041       | 131.20

    The ``n_params`` column is included in ``print_table()`` and
    ``save_csv()`` only when at least one row has a non-``None`` value.

    Attributes:
        _table_name: Display name printed as the table header.
        _rows: List of row dicts, each with keys ``model_name``, ``mse``,
            ``nmae_pct``, ``epoch_time``, ``n_params``.

    Example::

        table = ResultsTable("Experiment 1 — Out-of-Sample Parameters")
        table.add_row(
            model_name="Mamba FNO (pretr.)",
            metrics={"mse": 1.009e-7, "nmae": 0.000120},
            epoch_time=21.91,
            n_params=10_000_000,
        )
        table.print_table()
        table.save_csv("results/exp1.csv")
    """

    def __init__(self, table_name: str = "Results") -> None:
        """Initialise an empty results table.

        Args:
            table_name: Human-readable name displayed as the table header
                when ``print_table()`` is called.
        """
        self._table_name: str = table_name
        self._rows: List[Dict] = []

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def add_row(
        self,
        model_name: str,
        metrics: Dict[str, float],
        epoch_time: float,
        n_params: Optional[int] = None,
    ) -> None:
        """Append a result row to the table.

        The ``metrics`` dict must contain at minimum the keys ``'mse'`` and
        ``'nmae'``. The ``'nmae'`` value is expected as a raw fraction (e.g.
        ``0.000120`` for 0.0120 %) and is converted to percentage internally
        for display. This matches the NMAE formula in equation (3) of the
        paper, where the result is a dimensionless ratio; the tables report
        it multiplied by 100.

        Args:
            model_name: Display name for the model, e.g.
                ``"Mamba FNO (pretr.)"`` or ``"FNO (scratch)"``.
            metrics: Dict with at least ``'mse'`` (float) and ``'nmae'``
                (float, raw fraction). Additional keys are ignored.
            epoch_time: Average wall-clock time per training epoch in
                seconds, as reported in the ``Avg. epoch (s)`` column of
                Tables 1 and 2.
            n_params: Total parameter count for the model. Optional — when
                ``None``, the Param. column shows ``"N/A"`` in
                ``print_table()`` and an empty string in the CSV. Table 2
                in the paper omits this column; pass ``None`` for those rows.

        Raises:
            KeyError: If ``metrics`` does not contain ``'mse'`` or
                ``'nmae'``.
        """
        mse: float = float(metrics["mse"])
        nmae_raw: float = float(metrics["nmae"])
        nmae_pct: float = nmae_raw * 100.0

        self._rows.append(
            {
                "model_name": str(model_name),
                "mse": mse,
                "nmae_pct": nmae_pct,
                "epoch_time": float(epoch_time),
                "n_params": n_params,  # int or None
            }
        )

    def print_table(self) -> None:
        """Print the results table to stdout.

        The table header, separator, and data rows are formatted with fixed
        column widths. The ``Param.`` column is included only when at least
        one row has a non-``None`` ``n_params`` value, matching the
        difference between Table 1 (has Param.) and Table 2 (omits it) in
        the paper.

        If no rows have been added, a placeholder message is printed instead
        of an empty table.
        """
        has_params: bool = any(row["n_params"] is not None for row in self._rows)

        # ── Header ────────────────────────────────────────────────────────
        title_line: str = f"  {self._table_name}  "
        print()
        print("=" * (len(title_line) + 4))
        print(f"  {title_line}")
        print("=" * (len(title_line) + 4))

        if not self._rows:
            print("  (No results yet)")
            print()
            return

        # ── Column header row ─────────────────────────────────────────────
        header: str = (
            f"{'Model':<{_COL_MODEL}}"
            f"{'MSE':>{_COL_MSE}}"
            f"{'NMAE (%)':>{_COL_NMAE}}"
            f"{'Avg. epoch (s)':>{_COL_EPOCH}}"
        )
        if has_params:
            header += f"{'Param.':>{_COL_PARAMS}}"

        separator: str = "-" * len(header)
        print(header)
        print(separator)

        # ── Data rows ─────────────────────────────────────────────────────
        for row in self._rows:
            mse_str: str = f"{row['mse']:.3e}"
            nmae_str: str = f"{row['nmae_pct']:.4f}"
            epoch_str: str = f"{row['epoch_time']:.2f}"

            line: str = (
                f"{row['model_name']:<{_COL_MODEL}}"
                f"{mse_str:>{_COL_MSE}}"
                f"{nmae_str:>{_COL_NMAE}}"
                f"{epoch_str:>{_COL_EPOCH}}"
            )

            if has_params:
                if row["n_params"] is not None:
                    # Format as "≈ 1e+07" to match paper notation "≈ 10^7"
                    params_str: str = f"\u2248 {row['n_params']:.0e}"
                else:
                    params_str = "N/A"
                line += f"{params_str:>{_COL_PARAMS}}"

            print(line)

        print(separator)
        print()

    def save_csv(self, path: str) -> None:
        """Write the results table to a CSV file.

        Columns written: ``model_name``, ``mse``, ``nmae_pct``,
        ``epoch_time``, ``n_params``. Values are stored as raw floats (not
        formatted strings) for downstream analysis. When ``n_params`` is
        ``None``, an empty string is written in that cell.

        Parent directories are created automatically if they do not exist.

        Args:
            path: Destination file path, e.g. ``"results/exp1.csv"``.
        """
        parent_dir: str = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        fieldnames: List[str] = [
            "model_name",
            "mse",
            "nmae_pct",
            "epoch_time",
            "n_params",
        ]

        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(
                    {
                        "model_name": row["model_name"],
                        "mse": row["mse"],
                        "nmae_pct": row["nmae_pct"],
                        "epoch_time": row["epoch_time"],
                        # Empty string for None so the CSV is valid
                        "n_params": row["n_params"]
                        if row["n_params"] is not None
                        else "",
                    }
                )
