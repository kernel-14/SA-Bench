## utils/logger.py
"""Logger utility for Prioritized Generative Replay (PGR).

Provides a unified logging interface over W&B and TensorBoard backends,
with a CSV fallback for offline multi-seed aggregation.
"""

import csv
import io
import os
import pathlib
from typing import Any, Dict, Optional

import numpy as np

# Optional backend imports — guarded so the code runs without either installed.
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

try:
    from omegaconf import OmegaConf, DictConfig
    _OMEGACONF_AVAILABLE = True
except ImportError:
    _OMEGACONF_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless servers.
import matplotlib.pyplot as plt


class Logger:
    """Unified logging wrapper over W&B and TensorBoard with CSV fallback.

    Abstracts over two backends so callers never need to know which is active.
    A plain CSV file is always written alongside the primary backend for
    lightweight offline inspection and multi-seed aggregation.

    Attributes:
        use_wandb: Whether the W&B backend is active.
        run_name: Human-readable identifier for this run.
        log_dir: Root directory for all log artefacts.
    """

    def __init__(
        self,
        config: Any,
        use_wandb: bool = True,
        log_dir: str = "logs",
    ) -> None:
        """Initialises the logger and the chosen backend.

        Args:
            config: Hydra/OmegaConf DictConfig (or plain dict) carrying all
                hyperparameters.  Used to populate the W&B run config and to
                derive the run name.
            use_wandb: If True and wandb is installed, use W&B as the primary
                backend.  Falls back to TensorBoard when wandb is unavailable.
            log_dir: Root directory under which per-run subdirectories are
                created.  Overrides ``config.logging.log_dir`` when provided
                explicitly.
        """
        self._config = config

        # ── Resolve log directory ────────────────────────────────────────────
        # Prefer the explicit argument; fall back to config value.
        resolved_log_dir: str = log_dir
        if hasattr(config, "logging") and hasattr(config.logging, "log_dir"):
            resolved_log_dir = log_dir if log_dir != "logs" else config.logging.log_dir

        self.log_dir: pathlib.Path = pathlib.Path(resolved_log_dir)

        # ── Build a human-readable run name ──────────────────────────────────
        env_name: str = "unknown_env"
        relevance_type: str = "unknown_relevance"
        seed: int = 0

        if hasattr(config, "env") and hasattr(config.env, "name"):
            env_name = str(config.env.name)
        if hasattr(config, "relevance") and hasattr(config.relevance, "type"):
            relevance_type = str(config.relevance.type)
        if hasattr(config, "training") and hasattr(config.training, "seed"):
            seed = int(config.training.seed)
        elif hasattr(config, "env") and hasattr(config.env, "seed"):
            seed = int(config.env.seed)

        self.run_name: str = f"{env_name}_{relevance_type}_seed{seed}"

        # ── Create run-specific directory ────────────────────────────────────
        self._run_dir: pathlib.Path = self.log_dir / self.run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # ── Determine effective backend ──────────────────────────────────────
        self.use_wandb: bool = use_wandb and _WANDB_AVAILABLE

        # ── Initialise primary backend ───────────────────────────────────────
        self._writer: Optional[Any] = None  # TensorBoard SummaryWriter or None.
        self._wandb_run: Optional[Any] = None

        if self.use_wandb:
            project_name: str = "prioritized-generative-replay"
            if hasattr(config, "logging") and hasattr(config.logging, "project_name"):
                project_name = str(config.logging.project_name)

            # Convert OmegaConf config to a plain dict for W&B.
            config_dict: Dict[str, Any] = {}
            if _OMEGACONF_AVAILABLE and isinstance(config, DictConfig):
                config_dict = dict(OmegaConf.to_container(config, resolve=True))
            elif isinstance(config, dict):
                config_dict = config

            self._wandb_run = wandb.init(
                project=project_name,
                name=self.run_name,
                config=config_dict,
                dir=str(self.log_dir),
                reinit=True,
            )
        elif _TB_AVAILABLE:
            tb_log_dir: str = str(self._run_dir / "tensorboard")
            self._writer = SummaryWriter(log_dir=tb_log_dir)
        else:
            # Neither backend available — CSV-only mode.
            pass

        # ── Initialise CSV fallback ──────────────────────────────────────────
        self._csv_path: pathlib.Path = self._run_dir / "metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer: Optional[csv.DictWriter] = None
        # DictWriter is initialised lazily on the first log() call once we
        # know the full set of metric keys.
        self._csv_fieldnames: Optional[list] = None

        # ── Numpy artefact directory ─────────────────────────────────────────
        self._npy_dir: pathlib.Path = self._run_dir / "histograms"
        self._npy_dir.mkdir(parents=True, exist_ok=True)

        # ── Figure artefact directory ────────────────────────────────────────
        self._fig_dir: pathlib.Path = self._run_dir / "figures"
        self._fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """Logs scalar metrics to the active backend and the CSV file.

        Args:
            metrics: Mapping from metric name to scalar value.  Keys should
                follow the ``"namespace/metric_name"`` convention, e.g.
                ``"eval/episode_return"`` or ``"train/critic_loss"``.
            step: Global environment step counter used as the x-axis.
        """
        if not metrics:
            return

        # ── W&B ──────────────────────────────────────────────────────────────
        if self.use_wandb and self._wandb_run is not None:
            wandb.log(metrics, step=step)

        # ── TensorBoard ──────────────────────────────────────────────────────
        if self._writer is not None:
            for key, value in metrics.items():
                self._writer.add_scalar(key, float(value), global_step=step)

        # ── CSV fallback ─────────────────────────────────────────────────────
        self._write_csv_row(metrics, step)

    def log_histogram(
        self,
        name: str,
        values: np.ndarray,
        step: int,
    ) -> None:
        """Logs a histogram of values (e.g. relevance score distributions).

        Used for Fig. 6b analysis — called every ``relevance_eval_freq`` steps
        from ``PGRTrainer`` after ``Evaluator.compute_relevance_distribution()``.

        Args:
            name: Histogram identifier, e.g.
                ``"relevance/pgr_curiosity_scores"``.
            values: 1-D numpy array of scalar values to histogram.
            step: Global environment step counter.
        """
        values = np.asarray(values, dtype=np.float32).ravel()

        # ── W&B ──────────────────────────────────────────────────────────────
        if self.use_wandb and self._wandb_run is not None:
            wandb.log({name: wandb.Histogram(values)}, step=step)

        # ── TensorBoard ──────────────────────────────────────────────────────
        if self._writer is not None:
            self._writer.add_histogram(name, values, global_step=step)

        # ── Persist raw array for offline analysis ───────────────────────────
        safe_name: str = name.replace("/", "_")
        npy_path: pathlib.Path = self._npy_dir / f"{safe_name}_step{step}.npy"
        np.save(str(npy_path), values)

    def log_figure(self, name: str, fig: Any, step: int) -> None:
        """Logs a matplotlib figure (e.g. tSNE projections for Fig. 2).

        The figure is always saved to disk as a PNG.  After logging, the
        figure is closed to prevent memory leaks during long training runs.

        Args:
            name: Figure identifier, e.g. ``"tsne/epoch_130"``.
            fig: A ``matplotlib.figure.Figure`` instance.
            step: Global environment step counter.
        """
        # ── Save to disk first (always) ───────────────────────────────────────
        safe_name: str = name.replace("/", "_")
        png_path: pathlib.Path = self._fig_dir / f"{safe_name}_step{step}.png"
        fig.savefig(str(png_path), dpi=150, bbox_inches="tight")

        # ── W&B ──────────────────────────────────────────────────────────────
        if self.use_wandb and self._wandb_run is not None:
            wandb.log({name: wandb.Image(fig)}, step=step)

        # ── TensorBoard ──────────────────────────────────────────────────────
        if self._writer is not None:
            img_array: np.ndarray = self._figure_to_numpy(fig)
            # TensorBoard expects (N, C, H, W) with values in [0, 1].
            img_tensor = (
                np.transpose(img_array, (2, 0, 1))[np.newaxis].astype(np.float32)
                / 255.0
            )
            import torch  # Local import to avoid hard dependency at module level.
            self._writer.add_image(
                name,
                torch.from_numpy(img_tensor),
                global_step=step,
            )

        # ── Close figure to free memory ───────────────────────────────────────
        plt.close(fig)

    def close(self) -> None:
        """Finalises all backends and flushes pending writes.

        Should be called at the end of ``PGRTrainer.train()`` and in
        ``main.py`` after training completes.
        """
        if self.use_wandb and self._wandb_run is not None:
            wandb.finish()

        if self._writer is not None:
            self._writer.close()

        if self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()

        print(f"Training complete. Logs saved to {self._run_dir}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _write_csv_row(self, metrics: Dict[str, float], step: int) -> None:
        """Appends a row to the CSV metrics file.

        The DictWriter is initialised lazily on the first call so that the
        column set is determined from the first metrics dict.  Subsequent
        calls with new keys will add those keys with empty values for prior
        rows (standard DictWriter ``extrasaction='ignore'`` behaviour).

        Args:
            metrics: Scalar metrics to record.
            step: Global environment step.
        """
        row: Dict[str, Any] = {"step": step, **{k: float(v) for k, v in metrics.items()}}

        if self._csv_writer is None:
            # Initialise the writer with the keys seen in the first call.
            self._csv_fieldnames = ["step"] + sorted(metrics.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._csv_fieldnames,
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()
        else:
            # Extend fieldnames if new keys appear in later calls.
            new_keys = [k for k in row if k not in self._csv_fieldnames]
            if new_keys:
                self._csv_fieldnames.extend(sorted(new_keys))
                # Re-open the file and rewrite with updated header is complex;
                # instead we silently accept that new keys are ignored in prior
                # rows (extrasaction='ignore' handles extra keys gracefully).
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=self._csv_fieldnames,
                    extrasaction="ignore",
                )

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    @staticmethod
    def _figure_to_numpy(fig: Any) -> np.ndarray:
        """Converts a matplotlib Figure to an (H, W, 3) uint8 numpy array.

        Args:
            fig: A ``matplotlib.figure.Figure`` instance.

        Returns:
            RGB image array of shape (H, W, 3) with dtype uint8.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        import PIL.Image  # type: ignore[import]
        pil_img = PIL.Image.open(buf).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
