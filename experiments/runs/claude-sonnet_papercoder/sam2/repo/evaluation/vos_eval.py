```python
## evaluation/vos_eval.py
"""Semi-supervised VOS evaluator for SAM 2.

Implements VOSEvaluator for the semi-supervised video object segmentation
evaluation protocol described in Section 6.2 and Appendix F.1.3 of the
SAM 2 paper. Evaluates SAM 2 on all 17 zero-shot video datasets using
prompts provided only on the first frame.

Five prompt types are supported:
    - 1-click, 3-click, 5-click: iterative click refinement on frame 0
    - bounding_box: tight bounding box from GT mask on frame 0
    - gt_mask: ground-truth mask directly as mask prompt on frame 0

Config references (config.yaml evaluation.semi_supervised_vos):
    prompt_types: ["1_click", "3_click", "5_click", "bounding_box", "gt_mask"]
    all_17_datasets: [list of 17 dataset names]
    vost_metric: "J_only"

Paper references:
    Section 6.2: "We evaluate the semi-supervised video object segmentation
        (VOS) setting with click, box, or mask prompts only on the first
        frame of the video."
    Appendix F.1.3: Full details on click placement, dataset preprocessing,
        and metric computation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from datasets import PromptInput, VideoSample
from evaluation.metrics import JFMetric
from models.memory_bank import MemoryBank
from models.sam2 import SAM2FrameOutput, SAM2Model
from utils.click_sampler import ClickSampler
from utils.mask_utils import MaskUtils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants from config.yaml evaluation.semi_supervised_vos
# ---------------------------------------------------------------------------

_VALID_PROMPT_TYPES: List[str] = [
    "1_click",
    "3_click",
    "5_click",
    "bounding_box",
    "gt_mask",
]

# Datasets that use J-only metric (VOST official protocol)
_VOST_DATASETS: frozenset = frozenset({"VOST"})

# Occlusion threshold: if occlusion_score > this, output empty mask
_DEFAULT_OCCLUSION_THRESHOLD: float = 0.5

# Default device
_DEFAULT_DEVICE: str = "cuda"


class VOSEvaluator:
    """Evaluates SAM 2 in the semi-supervised VOS setting.

    Processes each object in a video independently, sharing image encoder
    features across objects. Supports five prompt types on the first frame
    and propagates predictions through the entire video using SAM 2's
    streaming memory architecture.

    From Appendix D.1: "we perform inference on each object independently.
    More specifically, we share the visual features from the image encoder
    between all the objects in the video but run all the other model
    components (such as the memory bank and the mask decoder) separately
    for each object."

    Args:
        model: Initialized SAM2Model in eval mode. Will be moved to device.
        config: Full config dict loaded from config.yaml. The evaluator reads
            from config['evaluation']['semi_supervised_vos'].
        device: Target device string (e.g., "cuda:0", "cpu").
            Defaults to "cuda".

    Example:
        evaluator = VOSEvaluator(model=sam2, config=cfg, device="cuda")
        results = evaluator.evaluate(
            dataset=davis_dataset,
            prompt_type="gt_mask",
            n_clicks=0,
        )
        print(results["JF"])  # e.g., 90.2 for DAVIS 2017 val
    """

    def __init__(
        self,
        model: SAM2Model,
        config: Dict[str, Any],
        device: str = _DEFAULT_DEVICE,
    ) -> None:
        self.model: SAM2Model = model
        self.config: Dict[str, Any] = config
        self.device: str = device

        # Move model to device and set to eval mode
        self.model = self.model.to(device)
        self.model.eval()

        # Extract evaluation configuration
        eval_cfg: Dict[str, Any] = config.get("evaluation", {})
        vos_cfg: Dict[str, Any] = eval_cfg.get("semi_supervised_vos", {})

        self.prompt_types: List[str] = list(
            vos_cfg.get("prompt_types", _VALID_PROMPT_TYPES)
        )
        self.vost_metric: str = str(
            vos_cfg.get("vost_metric", "J_only")
        )
        self.all_17_datasets: List[str] = list(
            vos_cfg.get("all_17_datasets", [])
        )
        self.occlusion_threshold: float = float(
            eval_cfg.get("benchmarking", {}).get(
                "occlusion_threshold", _DEFAULT_OCCLUSION_THRESHOLD
            )
        )

        # Instantiate shared utilities
        self.click_sampler: ClickSampler = ClickSampler()
        self.jf_metric: JFMetric = JFMetric()
        self.mask_utils: MaskUtils = MaskUtils()

        logger.info(
            "VOSEvaluator initialized: device=%s, prompt_types=%s, "
            "vost_metric=%s, occlusion_threshold=%.2f",
            device,
            self.prompt_types,
            self.vost_metric,
            self.occlusion_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset: Any,
        prompt_type: str = "gt_mask",
        n_clicks: int = 0,
    ) -> Dict[str, float]:
        """Run semi-supervised VOS evaluation on a dataset with one prompt type.

        Iterates over all sequences in the dataset, evaluates each object
        independently, and returns aggregated J&F (or J-only for VOST) metrics.

        From Section 6.2: "We evaluate the semi-supervised video object
        segmentation (VOS) setting with click, box, or mask prompts only
        on the first frame of the video."

        Args:
            dataset: Any iterable yielding VideoSample objects. Each sample
                must have frames [T, C, H, W], masks [T, N, H, W], video_id,
                frame_indices, num_objects, and is_occluded fields.
            prompt_type: One of "1_click", "3_click", "5_click",
                "bounding_box", "gt_mask". Defaults to "gt_mask".
            n_clicks: Number of clicks for click-based prompts. Derived
                automatically from prompt_type if 0:
                    "1_click" → 1, "3_click" → 3, "5_click" → 5,
                    "bounding_box" → 0, "gt_mask" → 0.
                Pass explicitly to override. Defaults to 0.

        Returns:
            Dict with keys:
                - "J": Mean Jaccard index over all sequences and objects.
                - "F": Mean boundary F-measure (0.0 for VOST).
                - "JF": Mean J&F = (J + F) / 2 (equals J for VOST).
                - "num_sequences": int — number of sequences evaluated.
                - "num_objects": int — total number of objects evaluated.

        Raises:
            ValueError: If prompt_type is not one of the valid types.
        """
        if prompt_type not in _VALID_PROMPT_TYPES:
            raise ValueError(
                f"prompt_type must be one of {_VALID_PROMPT_TYPES}, "
                f"got '{prompt_type}'."
            )

        # Derive n_clicks from prompt_type if not explicitly provided
        effective_n_clicks: int = self._get_n_clicks(prompt_type, n_clicks)

        # Determine if this dataset uses J-only metric (VOST)
        dataset_name: str = getattr(dataset, "dataset_name", "")
        use_j_only: bool = (
            dataset_name in _VOST_DATASETS
            or self.vost_metric == "J_only"
            and dataset_name.upper() == "VOST"
        )

        # Accumulators
        all_j_scores: List[float] = []
        all_f_scores: List[float] = []
        num_sequences: int = 0
        num_objects_total: int = 0

        with torch.no_grad():
            for sample in dataset:
                if not isinstance(sample, VideoSample):
                    logger.warning(
                        "VOSEvaluator.evaluate: Expected VideoSample, got %s. "
                        "Skipping.",
                        type(sample).__name__,
                    )
                    continue

                try:
                    seq_result: Dict[str, float] = self._evaluate_sequence(
                        sequence=sample,
                        prompt_type=prompt_type,
                        n_clicks=effective_n_clicks,
                    )
                except Exception as exc:
                    logger.warning(
                        "VOSEvaluator.evaluate: Failed to evaluate sequence "
                        "'%s': %s. Skipping.",
                        sample.video_id,
                        exc,
                    )
                    continue

                # Accumulate per-sequence results
                seq_j: float = seq_result.get("J", 0.0)
                seq_f: float = seq_result.get("F", 0.0)
                seq_n_objects: int = int(seq_result.get("num_objects", 1))

                # Weight by number of objects (each object contributes equally)
                for _ in range(seq_n_objects):
                    all_j_scores.append(seq_j)
                    all_f_scores.append(seq_f)

                num_sequences += 1
                num_objects_total += seq_n_objects

                logger.debug(
                    "VOSEvaluator: sequence='%s' J=%.4f F=%.4f JF=%.4f "
                    "n_objects=%d",
                    sample.video_id,
                    seq_j,
                    seq_f,
                    (seq_j + seq_f) / 2.0,
                    seq_n_objects,
                )

        # Compute global averages
        if len(all_j_scores) == 0:
            logger.warning(
                "VOSEvaluator.evaluate: No valid sequences evaluated for "
                "prompt_type='%s'. Returning zero metrics.",
                prompt_type,
            )
            return {
                "J": 0.0,
                "F": 0.0,
                "JF": 0.0,
                "num_sequences": 0,
                "num_objects": 0,
            }

        mean_j: float = float(np.mean(all_j_scores))
        mean_f: float = float(np.mean(all_f_scores)) if not use_j_only else 0.0
        mean_jf: float = (mean_j + mean_f) / 2.0 if not use_j_only else mean_j

        logger.info(
            "VOSEvaluator.evaluate: prompt_type='%s' | "
            "J=%.4f F=%.4f JF=%.4f | "
            "%d sequences, %d objects",
            prompt_type,
            mean_j,
            mean_f,
            mean_jf,
            num_sequences,
            num_objects_total,
        )

        return {
            "J": mean_j,
            "F": mean_f,
            "JF": mean_jf,
            "num_sequences": num_sequences,
            "num_objects": num_objects_total,
        }

    def evaluate_all_prompt_types(
        self,
        dataset: Any,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all five prompt types on a dataset.

        Convenience method that calls evaluate() for each prompt type and
        returns a nested dict. Matches the evaluation protocol in Table 4
        of the paper which reports results for all five prompt types.

        Args:
            dataset: Any iterable yielding VideoSample objects.

        Returns:
            Dict mapping prompt_type -> result dict from evaluate().
            Example:
                {
                    "1_click": {"J": 64.7, "F": ..., "JF": ...},
                    "3_click": {"J": 75.3, "F": ..., "JF": ...},
                    ...
                }
        """
        results: Dict[str, Dict[str, float]] = {}

        for prompt_type in _VALID_PROMPT_TYPES:
            n_clicks: int = self._get_n_clicks(prompt_type, 0)
            logger.info(
                "VOSEvaluator.evaluate_all_prompt_types: "
                "Evaluating prompt_type='%s'...",
                prompt_type,
            )
            results[prompt_type] = self.evaluate(
                dataset=dataset,
                prompt_type=prompt_type,
                n_clicks=n_clicks,
            )

        return results

    # ------------------------------------------------------------------
    # Core private methods
    # ------------------------------------------------------------------

    def _evaluate_sequence(
        self,
        sequence: VideoSample,
        prompt_type: str,
        n_clicks: int,
    ) -> Dict[str, float]:
        """Run VOS evaluation on a single video sequence.

        Processes each object independently, sharing image encoder features.
        Handles the preprocessing rule for objects not appearing in frame 0:
        creates a sub-video starting from the object's first appearance.

        From Appendix F.1.3: "In case an object doesn't appear in the first
        frame, we create a separate video for it starting from the first frame
        where the object appears."

        Args:
            sequence: VideoSample with frames [T, C, H, W] and
                masks [T, N, H, W].
            prompt_type: One of the five valid prompt types.
            n_clicks: Number of clicks for click-based prompts.

        Returns:
            Dict with keys "J", "F", "JF", "num_objects" averaged over all
            valid objects in the sequence.
        """
        frames: Tensor = sequence.frames   # [T, C, H, W]
        masks: Tensor = sequence.masks     # [T, N, H, W]
        T: int = frames.shape[0]
        N: int = sequence.num_objects

        # ------------------------------------------------------------------
        # Pre-encode all frames once (shared across all objects)
        # From Appendix D.1: "we share the visual features from the image
        # encoder between all the objects in the video"
        # ------------------------------------------------------------------
        frame_embeds: List[Tensor] = []
        skip_features_list: List[List[Tensor]] = []

        for t in range(T):
            frame_t: Tensor = frames[t].unsqueeze(0).to(self.device)
            with torch.no_grad():
                fe, sf = self.model.forward_image(frame_t)
            frame_embeds.append(fe)
            skip_features_list.append(sf)

        # ------------------------------------------------------------------
        # Process each object independently
        # ------------------------------------------------------------------
        obj_j_scores: List[float] = []
        obj_f_scores: List[float] = []
        num_valid_objects: int = 0

        for obj_idx in range(N):
            # Extract GT masks for this object: [T, H, W]
            gt_masks_obj: Tensor = masks[:, obj_idx, :, :]  # [T, H, W]

            # ------------------------------------------------------------------
            # Preprocessing: find first frame where object appears
            # Appendix F.1.3: "we create a separate video for it starting from
            # the first frame where the object appears"
            # ------------------------------------------------------------------
            first_valid_frame: int = self._find_first_valid_frame_tensor(
                gt_masks_obj
            )

            if first_valid_frame < 0:
                # Object never appears in this video — skip
                logger.debug(
                    "_evaluate_sequence: video='%s' obj=%d never appears. "
                    "Skipping.",
                    sequence.video_id,
                    obj_idx,
                )
                continue

            # Slice to sub-video starting from first_valid_frame
            # This implements the "separate video" preprocessing rule
            sub_frame_embeds: List[Tensor] = frame_embeds[first_valid_frame:]
            sub_skip_features: List[List[Tensor]] = (
                skip_features_list[first_valid_frame:]
            )
            sub_gt_masks: Tensor = gt_masks_obj[first_valid_frame:]  # [T', H, W]
            sub_T: int = sub_gt_masks.shape[0]

            if sub_T == 0:
                continue

            # ------------------------------------------------------------------
            # Build first-frame prompt
            # ------------------------------------------------------------------
            gt_mask_first: Tensor = sub_gt_masks[0]  # [H, W]

            # For click-based prompts, we need iterative refinement on frame 0
            # This requires running the model on frame 0 between clicks
            try:
                initial_prompt: PromptInput = self._get_first_frame_prompt(
                    gt_mask=gt_mask_first,
                    prompt_type=prompt_type,
                    n_clicks=n_clicks,
                    frame_embed=sub_frame_embeds[0],
                    skip_features=sub_skip_features[0],
                )
            except Exception as exc:
                logger.warning(
                    "_evaluate_sequence: Failed to build prompt for "
                    "video='%s' obj=%d: %s. Skipping.",
                    sequence.video_id,
                    obj_idx,
                    exc,
                )
                continue

            # ------------------------------------------------------------------
            # Propagate and score
            # ------------------------------------------------------------------
            try:
                obj_result: Dict[str, float] = self._propagate_and_score(
                    frame_embeds=sub_frame_embeds,
                    skip_features_list=sub_skip_features,
                    initial_prompt=initial_prompt,
                    gt_masks=sub_gt_masks,
                    video_id=sequence.video_id,
                    obj_idx=obj_idx,
                )
            except Exception as exc:
                logger.warning(
                    "_evaluate_sequence: Propagation failed for "
                    "video='%s' obj=%d: %s. Skipping.",
                    sequence.video_id,
                    obj_idx,
                    exc,
                )
                continue

            obj_j_scores.append(obj_result.get("J", 0.0))
            obj_f_scores.append(obj_result.get("F", 0.0))
            num_valid_objects += 1

        # ------------------------------------------------------------------
        # Average over objects
        # ------------------------------------------------------------------
        if num_valid_objects == 0:
            return {"J": 0.0, "F": 0.0, "JF": 0.0, "num_objects": 0}

        avg_j: float = float(np.mean(obj_j_scores))
        avg_f: float = float(np.mean(obj_f_scores))
        avg_jf: float = (avg_j + avg_f) / 2.0

        return {
            "J": avg_j,
            "F": avg_f,
            "JF": avg_jf,
            "num_objects": num_valid_objects,
        }

    def _get_first_frame_prompt(
        self,
        gt_mask: Tensor,
        prompt_type: str,
        n_clicks: int,
        frame_embed: Optional[Tensor] = None,
        skip_features: Optional[List[Tensor]] = None,
    ) -> PromptInput:
        """Generate the appropriate prompt for the first frame.

        For click-based prompts, performs iterative refinement on frame 0:
        places the initial click, runs a forward pass to get the current
        prediction, then places correction clicks at the error region center.

        From Appendix F.1.3: "the initial click is placed on the object center
        and subsequent clicks are obtained from the center of the error region."

        Args:
            gt_mask: Ground-truth binary mask for frame 0, shape [H, W].
                Values in {0, 1} or bool.
            prompt_type: One of the five valid prompt types.
            n_clicks: Number of clicks for click-based prompts.
            frame_embed: Optional pre-computed frame embedding [1, C, H/16, W/16].
                Required for iterative click refinement (3-click, 5-click).
            skip_features: Optional pre-computed skip features.
                Required for iterative click refinement.

        Returns:
            PromptInput with the appropriate prompt fields populated.

        Raises:
            ValueError: If prompt_type is not recognized.
        """
        if prompt_type == "gt_mask":
            return self._build_mask_prompt(gt_mask)

        elif prompt_type == "bounding_box":
            return self._build_box_prompt(gt_mask)

        elif prompt_type in ("1_click", "3_click", "5_click"):
            return self._build_click_prompt_iterative(
                gt_mask=gt_mask,
                n_clicks=n_clicks,
                frame_embed=frame_embed,
                skip_features=skip_features,
            )

        else:
            raise ValueError(
                f"Unknown prompt_type: '{prompt_type}'. "
                f"Must be one of {_VALID_PROMPT_TYPES}."
            )

    def _propagate_and_score(
        self,
        frame_embeds: List[Tensor],
        skip_features_list: List[List[Tensor]],
        initial_prompt: PromptInput,
        gt_masks: Tensor,
        video_id: str = "",
        obj_idx: int = 0,
    ) -> Dict[str, float]:
        """Propagate SAM 2 from first-frame prompt through entire video.

        Runs the streaming inference pipeline:
            1. Reset memory bank for this object
            2. Process frame 0 with the initial prompt
            3. Propagate through frames 1..T-1 without additional prompts
            4. Compute J&F (or J-only for VOST) on annotated frames

        From Section 4: "SAM 2 is equipped with a memory that stores
        information about the object and previous interactions, which allows
        it to generate masklet predictions throughout the video."

        Args:
            frame_embeds: List of T pre-computed frame embeddings, each
                [1, C, H/16, W/16].
            skip_features_list: List of T skip feature lists, each containing
                [stride4_feat, stride8_feat].
            initial_prompt: PromptInput for frame 0 (first frame of sub-video).
            gt_masks: Ground-truth masks for this object, shape [T, H, W].
                All-zero masks indicate the object is absent/occluded.
            video_id: Video identifier for logging. Defaults to "".
            obj_idx: Object index for logging. Defaults to 0.

        Returns:
            Dict with keys "J", "F", "JF" — sequence-level metrics averaged
            over annotated frames.
        """
        T: int = len(frame_embeds)

        # ------------------------------------------------------------------
        # Reset memory bank for this object
        # ------------------------------------------------------------------
        self.model.reset_memory()

        # ------------------------------------------------------------------
        # Process all frames sequentially
        # ------------------------------------------------------------------
        pred_masks_np: List[Optional[np.ndarray]] = []

        for t in range(T):
            frame_embed_t: Tensor = frame_embeds[t]
            skip_features_t: List[Tensor] = skip_features_list[t]

            # Determine prompt: only frame 0 gets the initial prompt
            prompt_t: Optional[PromptInput] = initial_prompt if t == 0 else None
            is_prompted: bool = (t == 0)

            with torch.no_grad():
                frame_output: SAM2FrameOutput = self.model.forward_video_frame(
                    frame_embed=frame_embed_t,
                    skip_features=skip_features_t,
                    prompts=prompt_t,
                    memory_bank=self.model.memory_bank,
                )

                # Update memory bank with current prediction
                self.model._update_memory_bank(
                    frame_embed=frame_embed_t,
                    frame_output=frame_output,
                    memory_bank=self.model.memory_bank,
                    is_prompted=is_prompted,
                    frame_idx=t,
                )

            # ------------------------------------------------------------------
            # Extract predicted mask for this frame
            # ------------------------------------------------------------------
            pred_mask_t: Optional[np.ndarray] = self._extract_pred_mask(
                frame_output=frame_output,
                gt_mask_shape=gt_masks[t].shape,
            )
            pred_masks_np.append(pred_mask_t)

        # ------------------------------------------------------------------
        # Compute sequence-level metrics
        # ------------------------------------------------------------------
        return self._compute_sequence_metrics(
            pred_masks=pred_masks_np,
            gt_masks=gt_masks,
            video_id=video_id,
            obj_idx=obj_idx,
        )

    # ------------------------------------------------------------------
    # Prompt building helpers
    # ------------------------------------------------------------------

    def _build_mask_prompt(self, gt_mask: Tensor) -> PromptInput:
        """Build a GT mask PromptInput for the first frame.

        From Appendix F.1.3: "For mask prompts, the ground-truth object masks
        on the first frame are directly used as input."

        Args:
            gt_mask: Ground-truth binary mask of shape [H, W].

        Returns:
            PromptInput with masks=[1, H, W] float32 and frame_idx=0.
        """
        # Ensure float32 and add channel dimension: [H, W] → [1, H, W]
        mask_tensor: Tensor = gt_mask.float()
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]
        elif mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor[0:1]

        return PromptInput(
            masks=mask_tensor.to(self.device),
            frame_idx=0,
        )

    def _build_box_prompt(self, gt_mask: Tensor) -> PromptInput:
        """Build a bounding box PromptInput from the GT mask.

        Computes the tight axis-aligned bounding box of the GT mask foreground
        pixels. Falls back to a center click if the box cannot be computed.

        Args:
            gt_mask: Ground-truth binary mask of shape [H, W].

        Returns:
            PromptInput with boxes=[4] float32 (x1, y1, x2, y2) and frame_idx=0.
        """
        gt_mask_np: np.ndarray = gt_mask.detach().cpu().numpy().astype(bool)
        bbox: Optional[Tuple[int, int, int, int]] = (
            self.mask_utils.get_bounding_box(gt_mask_np)
        )

        if bbox is None:
            # Degenerate mask — fall back to center click
            logger.debug(
                "_build_box_prompt: Empty GT mask, falling back to center click."
            )
            return self._build_single_click_prompt(gt_mask)

        y_min, x_min, y_max, x_max = bbox

        # Convert to (x1, y1, x2, y2) format expected by PromptEncoder
        box_tensor: Tensor = torch.tensor(
            [float(x_min), float(y_min), float(x_max), float(y_max)],
            dtype=torch.float32,
            device=self.device,
        )

        return PromptInput(
            boxes=box_tensor,
            frame_idx=0,
        )

    def _build_single_click_prompt(self, gt_mask: Tensor) -> PromptInput:
        """Build a single positive click PromptInput at the GT mask centroid.

        Args:
            gt_mask: Ground-truth binary mask of shape [H, W].

        Returns:
            PromptInput with points=[1, 2] float32 (x, y) and
            point_labels=[1] int64 and frame_idx=0.
        """
        # get_center_click returns [row, col] = [y, x]
        center_yx: Tensor = self.click_sampler.get_center_click(gt_mask)

        # Convert to (x, y) for PromptEncoder: x=col, y=row
        coords_xy: Tensor = torch.tensor(
            [float(center_yx[1]), float(center_yx[0])],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, 2]

        point_labels: Tensor = torch.tensor(
            [1], dtype=torch.long, device=self.device
        )

        return PromptInput(
            points=coords_xy,
            point_labels=point_labels,
            frame_idx=0,
        )

    def _build_click_prompt_iterative(
        self,
        gt_mask: Tensor,
        n_clicks: int,
        frame_embed: Optional[Tensor] = None,
        skip_features: Optional[List[Tensor]] = None,
    ) -> PromptInput:
        """Build an iterative click PromptInput with n_clicks on frame 0.

        Implements the iterative click refinement strategy from Appendix F.1.3:
            - Click 1: center of GT mask (positive click)
            - Clicks 2..n: center of error region between GT and current prediction

        For n_clicks > 1, requires running the model between clicks to get
        the current prediction for error region computation.

        Args:
            gt_mask: Ground-truth binary mask of shape [H, W].
            n_clicks: Total number of clicks to generate (1, 3, or 5).
            frame_embed: Pre-computed frame embedding [1, C, H/16, W/16].
                Required for n_clicks > 1.
            skip_features: Pre-computed skip features.
                Required for n_clicks > 1.

        Returns:
            PromptInput with points=[n_clicks, 2] float32 (x, y) and
            point_labels=[n_clicks] int64 and frame_idx=0.
        """
        if n_clicks <= 0:
            n_clicks = 1

        all_coords: List[Tensor] = []
        all_labels: List[Tensor] = []

        # ------------------------------------------------------------------
        # Click 1: initial positive click at GT mask center
        # ------------------------------------------------------------------
        center_yx: Tensor = self.click_sampler.get_center_click(gt_mask)
        coords_xy_1: Tensor = torch.tensor(
            [float(center_yx[1]), float(center_yx[0])],
            dtype=torch.float32,
            device=self.device,
        )
        all_coords.append(coords_xy_1)
        all_labels.append(torch.tensor(1, dtype=torch.long, device=self.device))

        if n_clicks == 1:
            # Single click — no iterative refinement needed
            points: Tensor = torch.stack(all_coords, dim=0)    # [1, 2]
            labels: Tensor = torch.stack(all_labels, dim=0)    # [1]
            return PromptInput(
                points=points,
                point_labels=labels,
                frame_idx=0,
            )

        # ------------------------------------------------------------------
        # Clicks 2..n_clicks: iterative refinement using model predictions
        # ------------------------------------------------------------------
        if frame_