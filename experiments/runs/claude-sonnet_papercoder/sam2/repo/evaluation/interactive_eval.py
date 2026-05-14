```python
## evaluation/interactive_eval.py
"""Interactive video segmentation evaluator for SAM 2.

Implements InteractiveEvaluator with two evaluation modes:
    - Offline: multiple passes over the full video, selecting the worst frame
      per pass for correction (up to N_frame=8 passes)
    - Online: single forward pass, pausing at frames with IoU < 0.75 for
      correction (corrections only affect future frames)

Both modes use N_click=3 clicks per interacted frame and report average J&F
over N_frame=1..8 interacted frames, plus annotation time curves.

Config references (config.yaml evaluation.interactive):
    num_clicks_per_frame: 3
    max_interacted_frames: 8
    online_iou_threshold: 0.75
    time_locate_sec: 1.0
    time_click_sec: 1.5
    time_examine_per_300frames: 30
    dense_datasets: [9 dataset names]

Paper references:
    Section 6.1: "We have two settings, offline evaluation ... and online
        evaluation ..."
    Appendix F.1.2: Full details on click placement, pass structure, and
        annotation time model.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from datasets import PromptInput, VideoSample
from evaluation.metrics import JFMetric
from models.memory_bank import MemoryBank
from models.sam2 import SAM2FrameOutput, SAM2Model
from utils.click_sampler import ClickSampler
from utils.mask_utils import MaskUtils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default evaluation constants (from config.yaml evaluation.interactive)
# ---------------------------------------------------------------------------

_DEFAULT_N_CLICKS: int = 3
_DEFAULT_MAX_FRAMES: int = 8
_DEFAULT_ONLINE_IOU_THRESHOLD: float = 0.75
_DEFAULT_T_LOC: float = 1.0
_DEFAULT_T_CLICK: float = 1.5
_DEFAULT_T_EXAM_PER_300: float = 30.0
_DEFAULT_OCCLUSION_THRESHOLD: float = 0.5


class InteractiveEvaluator:
    """Evaluates SAM 2 in interactive offline and online video segmentation modes.

    Simulates the interactive annotation experience described in Section 6.1
    and Appendix F.1.2 of the SAM 2 paper. Processes each object in a video
    independently, sharing image encoder features across objects.

    Args:
        model: Initialized SAM2Model in eval mode.
        config: Full config dict loaded from config.yaml. The evaluator reads
            from config['evaluation']['interactive'].
        device: Target device string (e.g., "cuda:0", "cpu").

    Example:
        evaluator = InteractiveEvaluator(model=sam2, config=cfg, device="cuda")
        results = evaluator.evaluate_offline(dataset=my_dataset, n_frames=8, n_clicks=3)
        print(results["overall_avg_jf"])
    """

    def __init__(
        self,
        model: SAM2Model,
        config: Dict[str, Any],
        device: str = "cuda",
    ) -> None:
        self.model: SAM2Model = model
        self.config: Dict[str, Any] = config
        self.device: str = device

        # Move model to device and set to eval mode
        self.model = self.model.to(device)
        self.model.eval()

        # Extract evaluation constants from config
        interactive_cfg: Dict[str, Any] = (
            config.get("evaluation", {}).get("interactive", {})
        )

        self.n_clicks: int = int(
            interactive_cfg.get("num_clicks_per_frame", _DEFAULT_N_CLICKS)
        )
        self.max_frames: int = int(
            interactive_cfg.get("max_interacted_frames", _DEFAULT_MAX_FRAMES)
        )
        self.online_iou_threshold: float = float(
            interactive_cfg.get("online_iou_threshold", _DEFAULT_ONLINE_IOU_THRESHOLD)
        )
        self.t_loc: float = float(
            interactive_cfg.get("time_locate_sec", _DEFAULT_T_LOC)
        )
        self.t_click: float = float(
            interactive_cfg.get("time_click_sec", _DEFAULT_T_CLICK)
        )
        self.t_exam_per_300: float = float(
            interactive_cfg.get("time_examine_per_300frames", _DEFAULT_T_EXAM_PER_300)
        )

        # Instantiate shared utilities
        self.click_sampler: ClickSampler = ClickSampler()
        self.jf_metric: JFMetric = JFMetric()
        self.mask_utils: MaskUtils = MaskUtils()

        logger.info(
            "InteractiveEvaluator initialized: n_clicks=%d, max_frames=%d, "
            "online_iou_threshold=%.2f, device=%s",
            self.n_clicks,
            self.max_frames,
            self.online_iou_threshold,
            device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_offline(
        self,
        dataset: Any,
        n_frames: int = _DEFAULT_MAX_FRAMES,
        n_clicks: int = _DEFAULT_N_CLICKS,
    ) -> Dict[str, Any]:
        """Run offline interactive evaluation across all videos in the dataset.

        In offline mode, multiple passes are made through the entire video.
        Each pass selects the frame with the lowest IoU vs GT as the next
        frame to prompt. Results are reported for each N_frame value from 1
        to n_frames.

        From Appendix F.1.2: "Offline evaluation involves multiple passes over
        the entire video. We start with click prompts on the first frame,
        segment the object throughout the entire video, and then in the next
        pass, we select the frame with the lowest segmentation IoU w.r.t. the
        ground-truth as the new frame for prompting."

        Args:
            dataset: Any iterable dataset yielding VideoSample objects. Each
                VideoSample must have frames [T, C, H, W], masks [T, N, H, W],
                video_id str, frame_indices List[int], num_objects int,
                is_occluded List[bool].
            n_frames: Maximum number of interacted frames (passes). Results
                are reported for each value 1..n_frames. Defaults to 8.
            n_clicks: Number of clicks per interacted frame. Defaults to 3.

        Returns:
            Dict with keys:
                - "avg_jf_by_nframe": Dict[int, float] — mean J&F for each
                  N_frame value (1..n_frames), averaged over all videos and
                  objects.
                - "avg_jf_by_time": List[Tuple[float, float]] — (time_sec,
                  jf_score) pairs for annotation time curve plotting.
                - "per_video_jf": Dict[str, Dict[int, float]] — per-video
                  J&F for each N_frame value.
                - "overall_avg_jf": float — single summary number (mean J&F
                  at max N_frame).
        """
        n_frames = min(n_frames, self.max_frames)
        n_clicks = n_clicks if n_clicks > 0 else self.n_clicks

        # Accumulators: nframe_idx -> list of jf scores across videos/objects
        jf_accum: Dict[int, List[float]] = {i: [] for i in range(1, n_frames + 1)}
        time_accum: Dict[int, List[float]] = {i: [] for i in range(1, n_frames + 1)}
        per_video_jf: Dict[str, Dict[int, float]] = {}

        num_videos: int = 0

        with torch.no_grad():
            for sample in dataset:
                if not isinstance(sample, VideoSample):
                    logger.warning(
                        "evaluate_offline: Expected VideoSample, got %s. Skipping.",
                        type(sample).__name__,
                    )
                    continue

                video_id: str = sample.video_id
                T: int = sample.frames.shape[0]
                N: int = sample.masks.shape[1]
                video_length: int = T

                # Per-video accumulator: nframe -> list of jf per object
                video_jf_per_nframe: Dict[int, List[float]] = {
                    i: [] for i in range(1, n_frames + 1)
                }

                # Cache image encoder outputs for all frames (shared across objects)
                frame_embeds, skip_features_list = self._encode_all_frames(sample)

                # Process each object independently
                for obj_idx in range(N):
                    gt_masks_obj: List[Optional[np.ndarray]] = (
                        self._extract_gt_masks_for_object(sample, obj_idx)
                    )

                    # Skip objects with no valid GT masks
                    if not any(
                        m is not None and m.astype(bool).any()
                        for m in gt_masks_obj
                    ):
                        logger.debug(
                            "evaluate_offline: video=%s obj=%d has no valid GT masks. "
                            "Skipping.",
                            video_id,
                            obj_idx,
                        )
                        continue

                    # Run offline multi-pass evaluation for this object
                    obj_results: List[Tuple[int, float, float]] = (
                        self._run_offline_pass(
                            frame_embeds=frame_embeds,
                            skip_features_list=skip_features_list,
                            gt_masks_obj=gt_masks_obj,
                            video_length=video_length,
                            n_frames=n_frames,
                            n_clicks=n_clicks,
                        )
                    )

                    # Accumulate per-object results
                    for nf, jf_score, ann_time in obj_results:
                        video_jf_per_nframe[nf].append(jf_score)
                        jf_accum[nf].append(jf_score)
                        time_accum[nf].append(ann_time)

                # Average over objects for this video
                video_avg_jf: Dict[int, float] = {}
                for nf in range(1, n_frames + 1):
                    if video_jf_per_nframe[nf]:
                        video_avg_jf[nf] = float(
                            np.mean(video_jf_per_nframe[nf])
                        )
                    else:
                        video_avg_jf[nf] = 0.0

                per_video_jf[video_id] = video_avg_jf
                num_videos += 1

                logger.debug(
                    "evaluate_offline: video=%s done. avg_jf_nframe8=%.4f",
                    video_id,
                    video_avg_jf.get(n_frames, 0.0),
                )

        # Compute global averages
        avg_jf_by_nframe: Dict[int, float] = {}
        for nf in range(1, n_frames + 1):
            if jf_accum[nf]:
                avg_jf_by_nframe[nf] = float(np.mean(jf_accum[nf]))
            else:
                avg_jf_by_nframe[nf] = 0.0

        # Build annotation time curve: (mean_time, mean_jf) pairs
        avg_jf_by_time: List[Tuple[float, float]] = []
        for nf in range(1, n_frames + 1):
            if time_accum[nf] and jf_accum[nf]:
                mean_time: float = float(np.mean(time_accum[nf]))
                mean_jf: float = avg_jf_by_nframe[nf]
                avg_jf_by_time.append((mean_time, mean_jf))

        overall_avg_jf: float = avg_jf_by_nframe.get(n_frames, 0.0)

        logger.info(
            "evaluate_offline: %d videos processed. "
            "overall_avg_jf (nframe=%d) = %.4f",
            num_videos,
            n_frames,
            overall_avg_jf,
        )

        return {
            "avg_jf_by_nframe": avg_jf_by_nframe,
            "avg_jf_by_time": avg_jf_by_time,
            "per_video_jf": per_video_jf,
            "overall_avg_jf": overall_avg_jf,
        }

    def evaluate_online(
        self,
        dataset: Any,
        n_frames: int = _DEFAULT_MAX_FRAMES,
        n_clicks: int = _DEFAULT_N_CLICKS,
        iou_threshold: float = _DEFAULT_ONLINE_IOU_THRESHOLD,
    ) -> Dict[str, Any]:
        """Run online interactive evaluation across all videos in the dataset.

        In online mode, a single forward pass is made through the video.
        Propagation pauses when a frame's IoU with GT drops below iou_threshold,
        at which point correction clicks are added. New prompts only affect
        frames after the current paused frame.

        From Appendix F.1.2: "Online evaluation involves only one pass over
        the entire video. We start with click prompts on the first frame and
        propagate the prompts across the video, pausing propagation when
        encountering a frame with a low-quality prediction (IoU < 0.75 with
        ground-truth)."

        Args:
            dataset: Any iterable dataset yielding VideoSample objects.
            n_frames: Maximum number of interacted frames. Defaults to 8.
            n_clicks: Number of clicks per interacted frame. Defaults to 3.
            iou_threshold: IoU threshold below which to pause and add
                corrections. Defaults to 0.75 (config:
                evaluation.interactive.online_iou_threshold).

        Returns:
            Dict with same structure as evaluate_offline:
                - "avg_jf_by_nframe": Dict[int, float]
                - "avg_jf_by_time": List[Tuple[float, float]]
                - "per_video_jf": Dict[str, Dict[int, float]]
                - "overall_avg_jf": float
        """
        n_frames = min(n_frames, self.max_frames)
        n_clicks = n_clicks if n_clicks > 0 else self.n_clicks

        jf_accum: Dict[int, List[float]] = {i: [] for i in range(1, n_frames + 1)}
        time_accum: Dict[int, List[float]] = {i: [] for i in range(1, n_frames + 1)}
        per_video_jf: Dict[str, Dict[int, float]] = {}

        num_videos: int = 0

        with torch.no_grad():
            for sample in dataset:
                if not isinstance(sample, VideoSample):
                    logger.warning(
                        "evaluate_online: Expected VideoSample, got %s. Skipping.",
                        type(sample).__name__,
                    )
                    continue

                video_id: str = sample.video_id
                T: int = sample.frames.shape[0]
                N: int = sample.masks.shape[1]
                video_length: int = T

                video_jf_per_nframe: Dict[int, List[float]] = {
                    i: [] for i in range(1, n_frames + 1)
                }

                # Cache image encoder outputs for all frames
                frame_embeds, skip_features_list = self._encode_all_frames(sample)

                for obj_idx in range(N):
                    gt_masks_obj: List[Optional[np.ndarray]] = (
                        self._extract_gt_masks_for_object(sample, obj_idx)
                    )

                    if not any(
                        m is not None and m.astype(bool).any()
                        for m in gt_masks_obj
                    ):
                        logger.debug(
                            "evaluate_online: video=%s obj=%d has no valid GT masks. "
                            "Skipping.",
                            video_id,
                            obj_idx,
                        )
                        continue

                    obj_results: List[Tuple[int, float, float]] = (
                        self._run_online_pass(
                            frame_embeds=frame_embeds,
                            skip_features_list=skip_features_list,
                            gt_masks_obj=gt_masks_obj,
                            video_length=video_length,
                            n_frames=n_frames,
                            n_clicks=n_clicks,
                            iou_threshold=iou_threshold,
                        )
                    )

                    for nf, jf_score, ann_time in obj_results:
                        video_jf_per_nframe[nf].append(jf_score)
                        jf_accum[nf].append(jf_score)
                        time_accum[nf].append(ann_time)

                video_avg_jf: Dict[int, float] = {}
                for nf in range(1, n_frames + 1):
                    if video_jf_per_nframe[nf]:
                        video_avg_jf[nf] = float(
                            np.mean(video_jf_per_nframe[nf])
                        )
                    else:
                        video_avg_jf[nf] = 0.0

                per_video_jf[video_id] = video_avg_jf
                num_videos += 1

                logger.debug(
                    "evaluate_online: video=%s done. avg_jf_nframe8=%.4f",
                    video_id,
                    video_avg_jf.get(n_frames, 0.0),
                )

        avg_jf_by_nframe: Dict[int, float] = {}
        for nf in range(1, n_frames + 1):
            if jf_accum[nf]:
                avg_jf_by_nframe[nf] = float(np.mean(jf_accum[nf]))
            else:
                avg_jf_by_nframe[nf] = 0.0

        avg_jf_by_time: List[Tuple[float, float]] = []
        for nf in range(1, n_frames + 1):
            if time_accum[nf] and jf_accum[nf]:
                mean_time = float(np.mean(time_accum[nf]))
                mean_jf = avg_jf_by_nframe[nf]
                avg_jf_by_time.append((mean_time, mean_jf))

        overall_avg_jf: float = avg_jf_by_nframe.get(n_frames, 0.0)

        logger.info(
            "evaluate_online: %d videos processed. "
            "overall_avg_jf (nframe=%d) = %.4f",
            num_videos,
            n_frames,
            overall_avg_jf,
        )

        return {
            "avg_jf_by_nframe": avg_jf_by_nframe,
            "avg_jf_by_time": avg_jf_by_time,
            "per_video_jf": per_video_jf,
            "overall_avg_jf": overall_avg_jf,
        }

    # ------------------------------------------------------------------
    # Core private methods
    # ------------------------------------------------------------------

    def _run_offline_pass(
        self,
        frame_embeds: List[Tensor],
        skip_features_list: List[List[Tensor]],
        gt_masks_obj: List[Optional[np.ndarray]],
        video_length: int,
        n_frames: int,
        n_clicks: int,
    ) -> List[Tuple[int, float, float]]:
        """Execute the full offline multi-pass evaluation for one object.

        Pass 1: Initial click at GT mask center on frame 0, then 2 correction
        clicks based on error region on frame 0. Propagate full video.

        Passes 2..n_frames: Find worst frame (lowest IoU vs GT), add 3
        correction clicks at error region center. Re-propagate full video
        from scratch with all accumulated prompts.

        Args:
            frame_embeds: List of T frame embedding tensors, each [1, C, H/16, W/16].
            skip_features_list: List of T skip feature lists, each containing
                [stride4_feat, stride8_feat].
            gt_masks_obj: List of T GT masks for this object. None or all-zero
                mask indicates the object is absent/occluded in that frame.
            video_length: Total number of frames T.
            n_frames: Maximum number of passes (interacted frames).
            n_clicks: Number of clicks per interacted frame (always 3).

        Returns:
            List of (n_frame, jf_score, annotation_time) tuples, one per pass.
            n_frame ranges from 1 to n_frames.
        """
        results: List[Tuple[int, float, float]] = []

        # Accumulate all prompted frames across passes: List[(frame_idx, PromptInput)]
        all_prompted_frames: List[Tuple[int, PromptInput]] = []

        # ------------------------------------------------------------------
        # Pass 1: Initial prompts on frame 0
        # ------------------------------------------------------------------
        # Find the first frame with a valid GT mask (object present)
        first_valid_frame: int = self._find_first_valid_frame(gt_masks_obj)

        if first_valid_frame < 0:
            # No valid frames — return zeros for all passes
            for nf in range(1, n_frames + 1):
                ann_time = self._compute_annotation_time(nf, video_length, "offline")
                results.append((nf, 0.0, ann_time))
            return results

        # Build initial prompts for the first valid frame
        # Click 1: center of GT mask (positive click)
        # Clicks 2-3: iterative correction clicks based on model response
        initial_prompts: PromptInput = self._build_initial_prompts_iterative(
            frame_embed=frame_embeds[first_valid_frame],
            skip_features=skip_features_list[first_valid_frame],
            gt_mask=gt_masks_obj[first_valid_frame],
            frame_idx=first_valid_frame,
            n_clicks=n_clicks,
        )

        all_prompted_frames = [(first_valid_frame, initial_prompts)]

        # Propagate full video with initial prompts
        outputs: List[SAM2FrameOutput] = self._propagate_full_video(
            frame_embeds=frame_embeds,
            skip_features_list=skip_features_list,
            prompted_frames=all_prompted_frames,
            video_length=video_length,
        )

        # Compute J&F after pass 1
        jf_score: float = self._compute_jf_from_outputs(outputs, gt_masks_obj)
        ann_time: float = self._compute_annotation_time(1, video_length, "offline")
        results.append((1, jf_score, ann_time))

        # ------------------------------------------------------------------
        # Passes 2..n_frames: find worst frame, add correction clicks
        # ------------------------------------------------------------------
        for pass_idx in range(2, n_frames + 1):
            # Find the frame with the lowest IoU vs GT (excluding occluded frames)
            worst_frame_idx: int = self._find_worst_frame(outputs, gt_masks_obj)

            if worst_frame_idx < 0:
                # No valid frame to correct — repeat last score
                ann_time = self._compute_annotation_time(
                    pass_idx, video_length, "offline"
                )
                results.append((pass_idx, jf_score, ann_time))
                continue

            # Get current prediction on worst frame for correction click sampling
            worst_output: SAM2FrameOutput = outputs[worst_frame_idx]
            pred_mask_worst: np.ndarray = self._output_to_binary_mask(worst_output)
            gt_mask_worst: Optional[np.ndarray] = gt_masks_obj[worst_frame_idx]

            if gt_mask_worst is None:
                gt_mask_worst = np.zeros_like(pred_mask_worst)

            # Build correction prompts: 3 clicks at error region center
            correction_prompts: PromptInput = self._build_correction_prompts(
                gt_mask=gt_mask_worst,
                pred_mask=pred_mask_worst,
                frame_idx=worst_frame_idx,
                n_clicks=n_clicks,
            )

            all_prompted_frames.append((worst_frame_idx, correction_prompts))

            # Re-propagate full video from scratch with all accumulated prompts
            outputs = self._propagate_full_video(
                frame_embeds=frame_embeds,
                skip_features_list=skip_features_list,
                prompted_frames=all_prompted_frames,
                video_length=video_length,
            )

            jf_score = self._compute_jf_from_outputs(outputs, gt_masks_obj)
            ann_time = self._compute_annotation_time(
                pass_idx, video_length, "offline"
            )
            results.append((pass_idx, jf_score, ann_time))

        return results

    def _run_online_pass(
        self,
        frame_embeds: List[Tensor],
        skip_features_list: List[List[Tensor]],
        gt_masks_obj: List[Optional[np.ndarray]],
        video_length: int,
        n_frames: int,
        n_clicks: int,
        iou_threshold: float,
    ) -> List[Tuple[int, float, float]]:
        """Execute online single-pass evaluation for one object.

        Single forward pass through the video. Pauses when a frame's IoU
        with GT drops below iou_threshold. Corrections only affect future
        frames (memory bank is NOT reset between corrections).

        From Appendix F.1.2: "Unlike the previous offline evaluation, in this
        setting, the new prompts only affect the frames after the current
        paused frame but not the frames before it."

        Args:
            frame_embeds: List of T frame embedding tensors.
            skip_features_list: List of T skip feature lists.
            gt_masks_obj: List of T GT masks for this object.
            video_length: Total number of frames T.
            n_frames: Maximum number of interacted frames.
            n_clicks: Number of clicks per interacted frame.
            iou_threshold: IoU threshold for pausing (default 0.75).

        Returns:
            List of (n_frame, jf_score, annotation_time) tuples. One entry
            per interaction point (including the initial frame 0 interaction).
        """
        results: List[Tuple[int, float, float]] = []

        # Reset memory bank for this object
        self.model.reset_memory()

        # Build a fresh per-object memory bank
        memory_bank: MemoryBank = self._create_memory_bank()

        T: int = video_length
        outputs: List[Optional[SAM2FrameOutput]] = [None] * T
        prompted_frames_count: int = 0

        # ------------------------------------------------------------------
        # Find first valid frame
        # ------------------------------------------------------------------
        first_valid_frame: int = self._find_first_valid_frame(gt_masks_obj)

        if first_valid_frame < 0:
            for nf in range(1, n_frames + 1):
                ann_time = self._compute_annotation_time(nf, video_length, "online")
                results.append((nf, 0.0, ann_time))
            return results

        # ------------------------------------------------------------------
        # Process frames before first_valid_frame (no GT — propagate without prompts)
        # ------------------------------------------------------------------
        for t in range(first_valid_frame):
            output_t: SAM2FrameOutput = self._forward_single_frame(
                frame_embed=frame_embeds[t],
                skip_features=skip_features_list[t],
                prompt=None,
                memory_bank=memory_bank,
                frame_idx=t,
                is_prompted=False,
            )
            outputs[t] = output_t

        # ------------------------------------------------------------------
        # Initial interaction on first_valid_frame
        # ------------------------------------------------------------------
        gt_mask_first: Optional[np.ndarray] = gt_masks_obj[first_valid_frame]
        if gt_mask_first is None:
            gt_mask_first = np.zeros(
                (frame_embeds[0].shape[2] * 16, frame_embeds[0].shape[3] * 16),
                dtype=np.float32,
            )

        # Build initial prompts iteratively (click 1 at center, clicks 2-3 at error)
        initial_prompts: PromptInput = self._build_initial_prompts_iterative(
            frame_embed=frame_embeds[first_valid_frame],
            skip_features=skip_features_list[first_valid_frame],
            gt_mask=gt_mask_first,
            frame_idx=first_valid_frame,
            n_clicks=n_clicks,
        )

        output_first: SAM2FrameOutput = self._forward_single_frame(
            frame_embed=frame_embeds[first_valid_frame],
            skip_features=skip_features_list[first_valid_frame],
            prompt=initial_prompts,
            memory_bank=memory_bank,
            frame_idx=first_valid_frame,
            is_prompted=True,
        )
        outputs[first_valid_frame] = output_first
        prompted_frames_count = 1

        # Record J&F after first interaction
        jf_score: float = self._compute_jf_from_outputs_partial(
            outputs, gt_masks_obj, up_to_frame=first_valid_frame
        )
        ann_time: float = self._compute_annotation_time(1, video_length, "online")
        results.append((1, jf_score, ann_time))

        # ------------------------------------------------------------------
        # Stream forward through remaining frames
        # ------------------------------------------------------------------
        for t in range(first_valid_frame + 1, T):
            # Propagate this frame without prompts
            output_t = self._forward_single_frame(
                frame_embed=frame_embeds[t],
                skip_features=skip_features_list[t],
                prompt=None,
                memory_bank=memory_bank,
                frame_idx=t,
                is_prompted=False,
            )
            outputs[t] = output_t

            # Check if we should pause for correction
            if prompted_frames_count >= n_frames:
                # Already used all allowed interactions — continue without pausing
                continue

            gt_mask_t: Optional[np.ndarray] = gt_masks_obj[t]
            if gt_mask_t is None or not gt_mask_t.astype(bool).any():
                # No GT for this frame — skip IoU check
                continue

            # Compute IoU between prediction and GT
            pred_mask_t: np.ndarray = self._output_to_binary_mask(output_t)
            frame_iou: float = self.mask_utils.compute_iou_pair(
                pred_mask_t, gt_mask_t
            )

            if frame_iou < iou_threshold:
                # Pause: add correction clicks on this frame
                correction_prompts: PromptInput = self._build_correction_prompts(
                    gt_mask=gt_mask_t,
                    pred_mask=pred_mask_t,
                    frame_idx=t,
                    n_clicks=n_clicks,
                )

                # Re-run this frame with corrections
                # In online mode, corrections affect only future frames.