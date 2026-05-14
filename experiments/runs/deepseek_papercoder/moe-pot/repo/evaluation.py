## evaluation.py
"""Autoregressive evaluation and inference timing for MoE‑POT."""

from __future__ import annotations

import time
from typing import Dict, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from config import Config
from model import Model


class Evaluation:
    """Compute zero‑shot / fine‑tuned L2 relative error and single‑step
    inference time for a pre‑trained MoE‑POT model.

    The class supports evaluation on a single dataset (via a plain
    ``DataLoader``) or on multiple datasets (via a dictionary mapping
    dataset names to loaders).  Autoregressive rollout is used to
    predict ``num_rollout_steps`` future frames, then the L2RE is
    averaged over all samples while respecting per‑dataset masks.

    Parameters
    ----------
    model : Model
        The MoE‑POT neural operator, already on the target device and
        in evaluation mode.
    loader : DataLoader or Dict[str, DataLoader]
        If a single ``DataLoader`` is given, the dataset name is set
        to ``"default"``.  Each loader is expected to yield batches of
        ``(context, target_seq, mask_seq)`` with shapes:
        * context : ``(B, T_in, H, W, C)``
        * target_seq : ``(B, rollout_steps, H, W, C)``
        * mask_seq : ``(B, rollout_steps, H, W, C)`` or broadcastable
    config : Config
        Global configuration (defines rollout length, warmup/repeat
        for timing, etc.).
    """

    def __init__(
        self,
        model: Model,
        loader: Union[DataLoader, Dict[str, DataLoader]],
        config: Config,
    ) -> None:
        self.model = model
        self.config = config

        # Normalise loaders to a dictionary so that the same code works
        # for single‑ and multi‑dataset evaluation.
        if isinstance(loader, DataLoader):
            self._loaders: Dict[str, DataLoader] = {"default": loader}
        else:
            self._loaders = loader

        # Determine device from model parameters (assumes model is on a single device)
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        self.model.eval()

    # ------------------------------------------------------------------
    # Core evaluation metric
    # ------------------------------------------------------------------

    def compute_l2re(self, num_rollout_steps: int = None) -> Dict[str, float]:
        """Evaluate the model by autoregressively predicting future
        frames and computing the L2 Relative Error (L2RE).

        The L2RE is computed only on valid data positions as indicated
        by the mask returned by the dataloader (padded channels and
        irregular domains are automatically ignored).

        Parameters
        ----------
        num_rollout_steps : int, optional
            Number of future time steps to predict.  If ``None``,
            ``config.rollout_steps`` is used.

        Returns
        -------
        Dict[str, float]
            Mapping from dataset name to its average L2RE (lower is better).
        """
        if num_rollout_steps is None:
            num_rollout_steps = self.config.rollout_steps

        self.model.eval()
        results: Dict[str, float] = {}

        with torch.no_grad():
            for ds_name, loader in self._loaders.items():
                total_l2re = 0.0
                num_samples = 0

                for batch in loader:
                    # Unpack batch – exact format must match the evaluation
                    # dataloader provided by DatasetLoader.
                    context, target_seq, mask_seq = batch
                    context = context.to(self.device, non_blocking=True)
                    target_seq = target_seq.to(self.device, non_blocking=True)
                    mask_seq = mask_seq.to(self.device, non_blocking=True)

                    B = context.shape[0]
                    cur = context  # (B, T_in, H, W, C)

                    pred_list = []
                    for _ in range(num_rollout_steps):
                        out, _ = self.model(cur)      # out: (B, H, W, C)
                        pred_list.append(out.unsqueeze(1))    # (B, 1, H, W, C)
                        # Shift the input window
                        cur = torch.cat(
                            [cur[:, 1:, ...], out.unsqueeze(1)], dim=1
                        )

                    # Concatenate predictions -> (B, rollout_steps, H, W, C)
                    pred_seq = torch.cat(pred_list, dim=1)

                    # Compute per‑sample L2RE (masked)
                    for i in range(B):
                        l2re_val = self._masked_l2re(
                            pred_seq[i], target_seq[i], mask_seq[i]
                        )
                        total_l2re += l2re_val.item()
                        num_samples += 1

                avg_l2re = total_l2re / num_samples if num_samples > 0 else float("inf")
                results[ds_name] = avg_l2re

        return results

    # ------------------------------------------------------------------
    # Inference speed measurement
    # ------------------------------------------------------------------

    def compute_inference_time(self) -> float:
        """Measure the average single‑step inference time (ms).

        A dummy tensor of shape ``(1, T_in, C_max, H, W)`` is used,
        and measurements are taken after a warm‑up phase to eliminate
        kernel launch overhead.

        Returns
        -------
        float
            Average wall‑clock time in milliseconds for one forward
            pass (predicting the next frame).
        """
        self.model.eval()

        dummy = torch.randn(
            1,
            self.config.input_frames,
            self.config.max_channels,
            self.config.spatial_resolution[0],
            self.config.spatial_resolution[1],
            device=self.device,
        )

        # Warm‑up
        with torch.no_grad():
            for _ in range(self.config.eval_warmup_inference):
                _ = self.model(dummy)

        # Timing
        if self.device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            timings: list[float] = []

            for _ in range(self.config.eval_repeat_inference):
                starter.record()
                with torch.no_grad():
                    _ = self.model(dummy)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))  # already in ms

            avg_ms = sum(timings) / len(timings)
        else:
            # CPU fallback using perf_counter
            timings_cpu: list[float] = []
            for _ in range(self.config.eval_repeat_inference):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = self.model(dummy)
                end = time.perf_counter()
                timings_cpu.append((end - start) * 1000.0)  # convert to ms
            avg_ms = sum(timings_cpu) / len(timings_cpu)

        return avg_ms

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_l2re(
        pred: Tensor,
        target: Tensor,
        mask: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """Compute per‑sample L2RE, ignoring positions where mask==0.

        Parameters
        ----------
        pred : Tensor
            Predicted tensor of arbitrary shape (e.g.,
            ``(rollout_steps, H, W, C)``).
        target : Tensor
            Ground‑truth tensor of identical shape.
        mask : Tensor
            Binary mask of shape identical to ``pred`` or broadcastable
            to it.

        Returns
        -------
        Tensor
            Scalar L2 relative error.  ``0.0`` is returned if the mask
            contains no valid elements.
        """
        if mask.shape != pred.shape:
            mask = mask.expand_as(pred).to(pred.dtype)

        diff = pred - target
        # Weighted sum of squares (over valid positions only)
        err_sq = (diff * mask).pow(2).sum()
        tgt_sq = (target * mask).pow(2).sum()

        # Guard against division by zero
        if tgt_sq == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        return torch.sqrt(err_sq) / (torch.sqrt(tgt_sq) + eps)

