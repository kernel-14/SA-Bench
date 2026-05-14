# evaluation/evaluator.py
"""
Evaluator for SAM 2 – implements all zero‑shot evaluation protocols.

This class uses a pre‑trained :class:`SAM2Model`, a :class:`ClickSimulator`,
and dataset objects to reproduce the experimental results described in the
SAM 2 paper (Sections 6 and Appendices F.1‑F.4).  It supports:

- Interactive offline evaluation (multiple full‑video passes)
- Interactive online evaluation (single forward pass)
- Semi‑supervised VOS evaluation (prompts only on first frame)
- Image segmentation evaluation (1‑click / 5‑click)

All settings are taken from the ``evaluation`` section of the configuration
file (``config.yaml``).  The evaluator also uses the functions from
``utils.metrics`` to compute J&F and mIoU.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

# Project imports (assumed to be installed / on PYTHONPATH)
from data.video_dataset import VideoDataset
from data.image_dataset import ImageDataset
from evaluation.click_simulator import ClickSimulator
from model.sam2 import SAM2Model
from utils.metrics import compute_JF, compute_mIoU


# ---------------------------------------------------------------------------
# Helper: compute per‑mask IoU for two binary masks
# ---------------------------------------------------------------------------

def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection‑over‑Union for a single frame."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0  # both empty
    return intersection / union


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """Evaluate SAM 2 under various protocols.

    Args:
        model: Pre‑trained :class:`SAM2Model` instance.
        config: Full configuration dictionary as produced by
            :meth:`config.Config.to_dict`.
        device: Torch device to use.  Defaults to ``'cuda'`` if available.
    """

    def __init__(
        self,
        model: SAM2Model,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

        # Create click simulator (centroid‑based strategy)
        self.click_sim = ClickSimulator(strategy="centroid")

        # Shortcut to evaluation config section
        self.eval_cfg = config["evaluation"]

        # Build dataset factories from config paths
        self._dataset_paths: Dict[str, str] = {}
        data_cfg = config["data"]
        for ds_name, root_key in [
            ("sav", "sav_root"),
            ("davis", "davis_root"),
            ("mose", "mose_root"),
            ("ytvos", "ytv_root"),
            ("sa1b", "sa1b_root"),
        ]:
            if root_key in data_cfg and data_cfg[root_key] is not None:
                self._dataset_paths[ds_name] = data_cfg[root_key]

    # ------------------------------------------------------------------
    #  Internal: dataset retrieval
    # ------------------------------------------------------------------

    def _get_video_dataset(
        self, dataset_name: str, sequence_length: int = 8
    ) -> VideoDataset:
        """Build a :class:`VideoDataset` for a named zero‑shot video benchmark.

        The mapping from dataset_name to root is predetermined for the 17
        datasets used in the paper.  For convenience, many of them can be
        loaded via the same root directory layout as DAVIS / MOSE.
        This is a simplified version; in practice one must supply the correct
        paths through the config.
        """
        # Use the config mapping for the known roots; if not found, raise
        if dataset_name not in self._dataset_paths:
            raise ValueError(f"Unknown dataset '{dataset_name}' – add its root to config.yaml")

        root = self._dataset_paths[dataset_name]
        # Create a temporary config subset for this single dataset
        temp_cfg = copy.deepcopy(self.config)
        # Override mix_weights to only include this dataset
        temp_cfg["data"]["mix_weights"] = {dataset_name: 1.0}
        # The root_paths argument for VideoDataset expects a dict mapping
        # dataset key to its root.  We'll use the same key as dataset_name.
        return VideoDataset(
            root_paths={dataset_name: root},
            config=temp_cfg,
            split="val",               # always evaluation
            sequence_length=sequence_length,
            augment=False,
        )

    def _get_image_dataset(self, dataset_name: str) -> ImageDataset:
        """Build an :class:`ImageDataset` for a single image dataset.

        The dataset_name may be one of the 37 benchmarks.  Currently we assume
        that all image datasets are stored in a structure similar to SA‑1B
        (images and per‑image mask RLEs).  The root is taken from config.
        """
        if dataset_name not in self._dataset_paths:
            raise ValueError(f"Unknown image dataset '{dataset_name}'.")
        root = self._dataset_paths[dataset_name]
        return ImageDataset(root=root, config=self.config, train=False)

    # ------------------------------------------------------------------
    #  Feature pre‑computation for videos
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _precompute_video_embeddings(
        self, frames: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Pre‑compute Hiera features + FPN + skip features for all frames.

        Args:
            frames: ``(T, 3, H, W)`` float32 [0,1] tensor on the target device.

        Returns:
            Dictionary with keys:
            - ``'fpn_feat'``: ``(T, C_fpn, H16, W16)`` FPN output.
            - ``'stage1'``: ``(T, C1, H/4, W/4)`` stride 4 features.
            - ``'stage2'``: ``(T, C2, H/8, W/8)`` stride 8 features.
        """
        T, C, H, W = frames.shape
        fpn_feats = []
        stage1_feats = []
        stage2_feats = []

        for t in range(T):
            # Encode a single frame with the Hiera encoder
            feats = self.model.image_encoder(frames[t].unsqueeze(0))
            # `feats` is a dict with keys 'stage1'..'stage4'
            # FPN fusion: call model._fpn_forward on this dict
            fpn = self.model._fpn_forward(feats).squeeze(0)  # (C_fpn, H16, W16)
            fpn_feats.append(fpn)
            stage1_feats.append(feats["stage1"].squeeze(0))
            stage2_feats.append(feats["stage2"].squeeze(0))

        return {
            "fpn_feat": torch.stack(fpn_feats, dim=0),   # (T, C_fpn, H16, W16)
            "stage1": torch.stack(stage1_feats, dim=0),  # (T, C1, H/4, W/4)
            "stage2": torch.stack(stage2_feats, dim=0),  # (T, C2, H/8, W/8)
        }

    # ------------------------------------------------------------------
    #  Streaming inference for a single object
    # ------------------------------------------------------------------

    def _run_streaming(
        self,
        precomp: Dict[str, torch.Tensor],
        prompt_frames: List[Tuple[int, Dict[str, Any]]],
        memory_bank_state: Optional[Dict[str, Any]] = None,
    ) -> List[torch.Tensor]:
        """Run the full streaming inference for a video object.

        This method processes frames sequentially, using the pre‑computed
        features and applying prompts at the specified frame indices.  If
        ``prompt_frames`` contains an entry for frame ``0`` with a ``'clicks'``
        key, multi‑mask will be enabled (the first positive click is ambiguous).

        Args:
            precomp: output of :meth:`_precompute_video_embeddings`.
            prompt_frames: sorted list of ``(frame_idx, prompt_dict)``.
                The ``prompt_dict`` keys may be:
                - ``'clicks'``: list of ``{'x': int, 'y': int, 'positive': bool}``.
                - ``'boxes'``: list of ``(x1, y1, x2, y2)`` pixel coordinates.
                - ``'masks'``: tensor of shape ``(1, 1, H, W)``, dense mask prompt.
            memory_bank_state: optional pre‑filled memory snapshot (for chaining).

        Returns:
            List of predicted binary masks of length ``T``, each a tensor
            of shape ``(H, W)``.
        """
        T = precomp["fpn_feat"].shape[0]
        H, W = precomp["fpn_feat"].shape[-2] * 16, precomp["fpn_feat"].shape[-1] * 16   # original resolution

        # ---- 1. Memory bank initialisation ----
        if memory_bank_state is not None:
            self.model.memory_bank.load_state(memory_bank_state)
        else:
            self.model.memory_bank.reset()

        masks = []

        # ---- 2. Process frames ----
        prompt_iter = iter(sorted(prompt_frames, key=lambda x: x[0]))
        next_prompt_frame, next_prompt = None, None
        try:
            next_prompt_frame, next_prompt = next(prompt_iter)
        except StopIteration:
            pass

        for t in range(T):
            # Current frame features
            fpn_t = precomp["fpn_feat"][t].unsqueeze(0)   # (1, C, H16, W16)
            skip4_t = precomp["stage1"][t].unsqueeze(0)   # (1, C1, H/4, W/4)
            skip8_t = precomp["stage2"][t].unsqueeze(0)   # (1, C2, H/8, W/8)

            # Memory attention
            memory_conditioned = self.model.memory_attention(
                fpn_t, self.model.memory_bank
            )  # (1, C, 64, 64)

            # ---- Build prompt tokens for this frame ----
            prompt_tokens = None
            is_prompted = False
            if next_prompt_frame is not None and next_prompt_frame == t:
                is_prompted = True

                # Encode sparse / dense prompts using PromptEncoder
                sparse_tokens = self._encode_prompts(next_prompt)
                dense_embed = None
                if "masks" in next_prompt:
                    dense_embed = self.model.prompt_encoder.encode_masks(
                        next_prompt["masks"].to(self.device)
                    )

                prompt_tokens = {
                    "sparse": sparse_tokens,
                    "dense": dense_embed,
                }

                # Determine multi‑mask flag: first prompted frame with a single
                # positive click and no prior prompts in the memory bank
                multi_mask = False
                if self.model.mask_decoder.multi_mask:
                    if len(self.model.memory_bank.prompted_queue) == 0:
                        if "clicks" in next_prompt and len(next_prompt["clicks"]) == 1:
                            if next_prompt["clicks"][0]["positive"]:
                                multi_mask = True
                prompt_tokens["multi_mask"] = multi_mask

                # Advance to next scheduled prompt
                try:
                    next_prompt_frame, next_prompt = next(prompt_iter)
                except StopIteration:
                    next_prompt_frame, next_prompt = None, None

            # ---- Mask decoder ----
            decoder_out = self.model.mask_decoder(
                image_embed=memory_conditioned,
                prompt_embed=prompt_tokens or {"sparse": None, "dense": None},
                skip_features=[skip8_t, skip4_t],  # stride8 first, then stride4
                multi_mask=prompt_tokens.get("multi_mask", False) if prompt_tokens else False,
            )

            # Compute binary mask (best candidate when multi‑mask)
            masks_logits = decoder_out["masks"].squeeze(0)   # (num_masks, H, W)
            iou_pred = decoder_out["iou_pred"].squeeze(0)     # (num_masks,)

            if prompt_tokens and prompt_tokens.get("multi_mask", False) and masks_logits.shape[0] > 1:
                best_idx = torch.argmax(iou_pred).item()
                mask_logits = masks_logits[best_idx]
            else:
                mask_logits = masks_logits[0] if masks_logits.dim() == 3 else masks_logits
            best_mask = (mask_logits.sigmoid() > 0.5).float()

            masks.append(best_mask)

            # ---- Update memory bank ----
            memory_feat = self.model.memory_encoder(
                mask=best_mask.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                image_embed=fpn_t,
            )
            self.model.memory_bank.add_memory(
                memory=memory_feat,
                is_prompted=is_prompted,
                object_pointer=decoder_out["object_pointer_token"],
            )

        return masks

    def _encode_prompts(self, prompt: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Transform the high‑level prompt dict into the sparse tokens required
        by the mask decoder."""
        sparse_parts = []

        # Clicks
        if "clicks" in prompt and prompt["clicks"]:
            click_coords = []
            click_labels = []
            for c in prompt["clicks"]:
                click_coords.append([c["x"], c["y"]])
                click_labels.append(1 if c["positive"] else 0)
            if click_coords:
                coords = torch.tensor(click_coords, dtype=torch.float32, device=self.device).unsqueeze(0)
                labels = torch.tensor(click_labels, dtype=torch.int64, device=self.device).unsqueeze(0)
                sparse_parts.append(self.model.prompt_encoder.encode_clicks(coords, labels))

        # Boxes
        if "boxes" in prompt and prompt["boxes"]:
            boxes = torch.tensor(prompt["boxes"], dtype=torch.float32, device=self.device).unsqueeze(0)
            sparse_parts.append(self.model.prompt_encoder.encode_boxes(boxes))

        if sparse_parts:
            return torch.cat(sparse_parts, dim=1)
        return None

    # ------------------------------------------------------------------
    #  Interactive offline evaluation
    # ------------------------------------------------------------------

    def evaluate_interactive_offline(self, dataset_name: str) -> Dict[str, Any]:
        """
        Run interactive offline evaluation on a video dataset.

        Simulates up to ``max_frames`` rounds of interaction, each round
        adding ``clicks_per_frame`` clicks on the frame with lowest IoU.

        Returns a dictionary containing per‑object results (list of J&F per
        pass) and aggregated statistics.
        """
        off_cfg = self.eval_cfg.interactive_offline
        video_ds = self._get_video_dataset(dataset_name)
        all_per_obj_jf = []
        total_objects = 0

        for batch_idx in range(len(video_ds)):
            batch = video_ds[batch_idx]
            frames = batch["frames"].to(self.device)                     # (T, C, H, W)
            masklets = batch.get("masklets", [])                         # list of dicts with 'mask' (T,1,H,W)
            if not masklets:
                continue

            precomp = self._precompute_video_embeddings(frames)

            for mlet in masklets:
                gt_masks = mlet["mask"].squeeze(1).cpu().numpy()         # (T, H, W)
                T = gt_masks.shape[0]

                # ---- interactive loop ----
                all_prompts: List[Tuple[int, Dict[str, Any]]] = []
                # Initial click set on frame 0
                init_clicks = self.click_sim.generate_initial_clicks(gt_masks[0])
                if not init_clicks:
                    continue  # no object on first frame → skip

                # For offline evaluation, we want to get J&F after each pass.
                per_pass_jf = []
                for pass_idx in range(off_cfg.max_frames):
                    # Schedule current prompts
                    if pass_idx == 0:
                        # first frame only, with initial clicks
                        current_prompts = [(0, {"clicks": init_clicks})]
                    else:
                        # all accumulated prompts
                        current_prompts = sorted(all_prompts, key=lambda x: x[0])

                    # Run streaming inference and get masks
                    masks_pred = self._run_streaming(precomp, current_prompts)
                    # Convert to numpy array for metric
                    masks_pred_np = torch.stack(masks_pred, dim=0).cpu().numpy()  # (T, H, W)
                    jf = compute_JF(masks_pred_np, gt_masks)["J_and_F"]
                    per_pass_jf.append(jf)

                    # If last pass, stop.
                    if pass_idx == off_cfg.max_frames - 1:
                        break

                    # Find frame with lowest IoU (only consider frames where object exists)
                    ioU_values = []
                    for t in range(T):
                        if gt_masks[t].sum() == 0:
                            ioU_values.append(-1.0)  # skip absent frames
                        else:
                            iou = _iou(masks_pred_np[t], gt_masks[t])
                            ioU_values.append(iou)
                    worst_frame = max(
                        range(T),
                        key=lambda t: ioU_values[t] if ioU_values[t] >= 0 else -1.0,
                    )
                    # Generate correction clicks on worst_frame
                    corr_clicks = self.click_sim.generate_correction_clicks(
                        pred_mask=masks_pred_np[worst_frame],
                        gt_mask=gt_masks[worst_frame],
                        random_gt_prob=0.0,  # evaluation: no random sampling
                    )
                    if corr_clicks:
                        all_prompts.append((worst_frame, {"clicks": corr_clicks}))

                all_per_obj_jf.append(per_pass_jf)
                total_objects += 1

        # Aggregate over objects
        avg_jf_per_pass = np.mean(np.array(all_per_obj_jf), axis=0).tolist() if all_per_obj_jf else []
        return {
            "dataset": dataset_name,
            "avg_J&F_per_pass": avg_jf_per_pass,
            "num_objects": total_objects,
        }

    # ------------------------------------------------------------------
    #  Interactive online evaluation
    # ------------------------------------------------------------------

    def evaluate_interactive_online(self, dataset_name: str) -> Dict[str, Any]:
        """
        Run interactive online evaluation on a video dataset.

        The evaluation performs a single forward pass and adds correction
        clicks whenever the IoU of a frame falls below ``ioU_threshold``.
        The number of prompted frames is limited by ``max_frames``.
        We report the average J&F after encountering a certain number of
        prompted frames (by counting the actual number of frames that
        received prompts).
        """
        on_cfg = self.eval_cfg.interactive_online
        video_ds = self._get_video_dataset(dataset_name)
        results_by_max_frames = {f: [] for f in range(1, on_cfg.max_frames + 1)}
        total_objects = 0

        for batch_idx in range(len(video_ds)):
            batch = video_ds[batch_idx]
            frames = batch["frames"].to(self.device)
            masklets = batch.get("masklets", [])
            if not masklets:
                continue

            precomp = self._precompute_video_embeddings(frames)

            for mlet in masklets:
                gt_masks = mlet["mask"].squeeze(1).cpu().numpy()
                T = gt_masks.shape[0]

                # ---- Online streaming ----
                # Start with first frame prompts
                init_clicks = self.click_sim.generate_initial_clicks(gt_masks[0])
                if not init_clicks:
                    continue

                # This scenario is complex: we need to allow re‑prompting on
                # frames that have already been processed.  The cleanest way is
                # to re‑run the streaming whenever a re‑prompt is needed, but
                # for online evaluation the paper specifies that new prompts
                # only affect later frames; they cannot correct backward.
                # Therefore we can process frame by frame, and if a frame needs
                # correction, we add the correction clicks to the **same** frame
                # and re‑predict it, then continue forward.  We'll simulate that
                # by managing the memory bank ourselves.

                memory_bank = copy.deepcopy(self.model.memory_bank)  # fresh state
                self.model.memory_bank.reset()

                masks_pred = []
                prompts_schedule = [(0, {"clicks": init_clicks})]
                frame_idx = 0
                n_prompted = 1  # first frame is already prompted

                while frame_idx < T:
                    # Check if there is a pending prompt for this frame
                    pending_prompts = []
                    while prompts_schedule and prompts_schedule[0][0] == frame_idx:
                        pending_prompts.append(prompts_schedule.pop(0))

                    if pending_prompts:
                        # Combine all prompts for this frame into one dict
                        combined_prompt = self._merge_prompts(pending_prompts)
                        # Run the forward for this single frame
                        # We'll implement a small helper that processes one frame
                        mask = self._process_single_frame(
                            precomp, frame_idx, combined_prompt, self.model.memory_bank
                        )
                    else:
                        # No prompts – just propagate
                        mask = self._process_single_frame(
                            precomp, frame_idx, None, self.model.memory_bank
                        )

                    masks_pred.append(mask)

                    # Check if we need correction
                    if frame_idx > 0 and n_prompted < on_cfg.max_frames:
                        iou = _iou(mask.cpu().numpy(), gt_masks[frame_idx])
                        if iou < on_cfg.ioU_threshold:
                            # Generate correction clicks on this frame
                            corr = self.click_sim.generate_correction_clicks(
                                pred_mask=mask.cpu().numpy(),
                                gt_mask=gt_masks[frame_idx],
                                random_gt_prob=0.0,
                            )
                            if corr:
                                # Add to prompts schedule to reprocess this frame
                                prompts_schedule.insert(0, (frame_idx, {"clicks": corr}))
                                n_prompted += 1
                                # Remove the memory we just added for this frame
                                # because we will re‑run it
                                self.model.memory_bank.pop_last()
                                continue  # reprocess the same frame_idx

                    frame_idx += 1

                # After streaming, compute J&F for each possible max_frames limit
                # but we only have one trajectory.  We'll compute J&F assuming
                # that the number of interactions is ``n_prompted``.
                # For reporting per‑max‑frames, we record the result at the
                # corresponding value (if n_prompted <= max_frames).
                if masks_pred:
                    masks_pred_np = torch.stack(masks_pred, dim=0).cpu().numpy()
                    jf = compute_JF(masks_pred_np, gt_masks)["J_and_F"]
                    for f in range(1, on_cfg.max_frames + 1):
                        if n_prompted <= f:
                            results_by_max_frames[f].append(jf)
                total_objects += 1

        avg_jf_per_frames = {
            f: np.mean(lst).item() if lst else 0.0
            for f, lst in results_by_max_frames.items()
        }
        return {
            "dataset": dataset_name,
            "avg_J&F_per_interacted_frames": avg_jf_per_frames,
            "num_objects": total_objects,
        }

    def _process_single_frame(
        self,
        precomp: Dict[str, torch.Tensor],
        frame_idx: int,
        prompt: Optional[Dict[str, Any]],
        memory_bank: Any,       # MemoryBank instance
    ) -> torch.Tensor:
        """Run a single frame of streaming inference and return the binary mask."""
        fpn_t = precomp["fpn_feat"][frame_idx].unsqueeze(0)
        skip4_t = precomp["stage1"][frame_idx].unsqueeze(0)
        skip8_t = precomp["stage2"][frame_idx].unsqueeze(0)

        conditioned = self.model.memory_attention(fpn_t, memory_bank)

        # Encode prompt if any
        prompt_embed = {"sparse": None, "dense": None, "multi_mask": False}
        is_prompted = False
        if prompt is not None:
            is_prompted = True
            sparse = self._encode_prompts(prompt)
            prompt_embed["sparse"] = sparse
            if "masks" in prompt:
                prompt_embed["dense"] = self.model.prompt_encoder.encode_masks(
                    prompt["masks"].to(self.device)
                )
            # multi_mask special case handled before calling _process_single_frame

        output = self.model.mask_decoder(
            image_embed=conditioned,
            prompt_embed=prompt_embed,
            skip_features=[skip8_t, skip4_t],
            multi_mask=prompt.get("multi_mask", False) if prompt else False,
        )

        masks_logits = output["masks"].squeeze(0)
        iou_pred = output["iou_pred"].squeeze(0)
        if output.get("multi_mask", False) and masks_logits.shape[0] > 1:
            best_idx = torch.argmax(iou_pred).item()
            mask_logits = masks_logits[best_idx]
        else:
            mask_logits = masks_logits[0] if masks_logits.dim() == 3 else masks_logits
        mask = (mask_logits.sigmoid() > 0.5).float()

        # Update memory bank
        memory_feat = self.model.memory_encoder(
            mask=mask.unsqueeze(0).unsqueeze(0),
            image_embed=fpn_t,
        )
        memory_bank.add_memory(
            memory=memory_feat,
            is_prompted=is_prompted,
            object_pointer=output["object_pointer_token"],
        )
        return mask

    @staticmethod
    def _merge_prompts(prompt_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple prompt dicts (same frame) into one."""
        combined: Dict[str, Any] = {}
        clicks = []
        boxes = []
        mask_prompt = None
        for p in prompt_list:
            if "clicks" in p:
                clicks.extend(p["clicks"])
            if "boxes" in p:
                boxes.extend(p["boxes"])
            if "masks" in p:
                mask_prompt = p["masks"]  # last one wins
        if clicks:
            combined["clicks"] = clicks
        if boxes:
            combined["boxes"] = boxes
        if mask_prompt is not None:
            combined["masks"] = mask_prompt
        return combined

    # ------------------------------------------------------------------
    #  Semi‑supervised VOS evaluation
    # ------------------------------------------------------------------

    def evaluate_semi_supervised(self, dataset_name: str) -> Dict[str, Any]:
        """
        Evaluate semi‑supervised VOS: prompts only on the first frame.

        Supported prompt types: ``1-click``, ``3-click``, ``5-click``, ``box``,
        ``mask`` (ground‑truth mask).  For click prompts, interactively refine
        the mask on the first frame, then propagate.
        """
        semi_cfg = self.eval_cfg.semi_supervised
        video_ds = self._get_video_dataset(dataset_name)
        results_by_prompt: Dict[str, List[float]] = {pt: [] for pt in semi_cfg.prompts}

        for batch_idx in range(len(video_ds)):
            batch = video_ds[batch_idx]
            frames = batch["frames"].to(self.device)
            masklets = batch.get("masklets", [])
            if not masklets:
                continue

            precomp = self._precompute_video_embeddings(frames)

            for mlet in masklets:
                gt_masks = mlet["mask"].squeeze(1).cpu().numpy()
                gt_mask0 = gt_masks[0]

                for prompt_type in semi_cfg.prompts:
                    # Prepare the first‑frame prompt(s) and the resulting memory state
                    memory_bank = copy.deepcopy(self.model.memory_bank)
                    self.model.memory_bank.reset()

                    if prompt_type.endswith("-click"):
                        n_clicks = int(prompt_type.split("-")[0])
                        # Interactive click refinement on first frame
                        clicks: List[Dict[str, Any]] = []
                        curr_prompt = {"clicks": self.click_sim.generate_initial_clicks(gt_mask0)}
                        mask0 = None
                        for _ in range(n_clicks):
                            # Run the first frame with the current set of clicks
                            mask0 = self._process_single_frame(
                                precomp, 0, curr_prompt, self.model.memory_bank
                            )
                            # If not the last iteration, generate correction
                            if len(clicks) < n_clicks:
                                corr = self.click_sim.generate_correction_clicks(
                                    pred_mask=mask0.cpu().numpy(),
                                    gt_mask=gt_mask0,
                                    random_gt_prob=0.0,
                                )
                                if corr:
                                    # Remove last memory and reprocess with added clicks
                                    self.model.memory_bank.pop_last()
                                    clicks.extend(corr)
                                    curr_prompt = {"clicks": clicks}
                                else:
                                    break
                        # After the loop, memory_bank holds the final state for frame 0.
                        # Now propagate to remaining frames (no further prompts)
                        final_prompts = []  # no prompts for frames > 0
                        masks = self._run_streaming(precomp, final_prompts, memory_bank_state=None)
                        # Recompute masks with the correct memory state by chaining
                        # We'll re‑run from scratch but with the accumulated memory.
                        # Instead, we pass the existing memory_bank to _run_streaming
                        masks = self._run_streaming(
                            precomp,
                            [(0, curr_prompt)],
                            memory_bank_state=memory_bank.get_state(),
                        )
                        # But the above will still run frame 0 with curr_prompt and add
                        # memory again, which is correct because it will overwrite.
                        # To be efficient, we can just run the rest of the frames with
                        # the memory_bank as it is. We'll add a helper to propagate
                        # without reprocessing frame 0.

                        # Cleanest: save memory_bank state after first‑frame refinement,
                        # then use that as the initial memory for the rest.
                        # We'll implement a small propagation helper.
                        masks_rest = self._propagate_rest(
                            precomp, start_frame=1, memory_bank_state=memory_bank.get_state()
                        )
                        masks = [mask0] + masks_rest

                    elif prompt_type == "box":
                        # bounding box from mask
                        rows = np.any(gt_mask0, axis=1)
                        cols = np.any(gt_mask0, axis=0)
                        if rows.any() and cols.any():
                            y1, y2 = np.where(rows)[0][[0, -1]]
                            x1, x2 = np.where(cols)[0][[0, -1]]
                            boxes = [[x1, y1, x2, y2]]
                        else:
                            boxes = []
                        prompt0 = {"boxes": boxes}
                        masks = self._run_streaming(precomp, [(0, prompt0)])
                    elif prompt_type == "mask":
                        # ground‑truth mask as prompt
                        masks = self._run_streaming(
                            precomp, [(0, {"masks": torch.from_numpy(gt_mask0).float().to(self.device).unsqueeze(0).unsqueeze(0)})]
                        )
                    else:
                        raise ValueError(f"Unknown prompt type: {prompt_type}")

                    # Compute J&F
                    if masks:
                        masks_np = torch.stack(masks, dim=0).cpu().numpy() if isinstance(masks, list) else masks.cpu().numpy()
                        jf = compute_JF(masks_np, gt_masks)["J_and_F"]
                    else:
                        jf = 0.0
                    results_by_prompt[prompt_type].append(jf)

        # Average over objects
        avg_results = {
            pt: float(np.mean(lst)) if lst else 0.0
            for pt, lst in results_by_prompt.items()
        }
        return {
            "dataset": dataset_name,
            "metrics": avg_results,
        }

    def _propagate_rest(
        self,
        precomp: Dict[str, torch.Tensor],
        start_frame: int,
        memory_bank_state: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """Propagate from start_frame to end using given memory bank state, no prompts."""
        self.model.memory_bank.load_state(memory_bank_state)
        T = precomp["fpn_feat"].shape[0]
        masks = []
        for t in range(start_frame, T):
            mask = self._process_single_frame(precomp, t, None, self.model.memory_bank)
            masks.append(mask)
        return masks

    # ------------------------------------------------------------------
    #  Image segmentation evaluation
    # ------------------------------------------------------------------

    def evaluate_image_segmentation(self, dataset_name: str) -> Dict[str, Any]:
        """
        Evaluate SAM 2 on a static image segmentation dataset.

        For each image and each ground‑truth mask, simulate 1‑click and 5‑click
        interactive segmentation (no video memory) and compute mIoU.
        """
        img_cfg = self.eval_cfg.image_segmentation
        img_ds = self._get_image_dataset(dataset_name)
        # For images, we treat each image as a single‑frame video.
        # We can reuse the model's forward method with T=1.
        results: Dict[str, List[float]] = {f"{c}-click": [] for c in img_cfg.clicks}
        batch_size = 1  # process one image at a time for simplicity

        dataloader = DataLoader(img_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        for batch in dataloader:
            image = batch["image"].to(self.device)       # (1, C, H, W) if batch_size=1 else (B, ...)
            masks_gt = batch["masks"]                     # (1, K, H, W) or (K, H, W)
            if masks_gt.dim() == 3:
                masks_gt = masks_gt.unsqueeze(0)          # (1, K, H, W)

            B = image.shape[0]
            for b in range(B):
                img = image[b]                             # (C, H, W)
                K = masks_gt.shape[1]
                for k in range(K):
                    gt_mask = masks_gt[b, k].cpu().numpy()
                    if gt_mask.sum() == 0:
                        continue

                    for n_clicks in img_cfg.clicks:
                        # Simulate interactive clicks on a single image
                        clicked_mask = self._simulate_image_interaction(img, gt_mask, n_clicks)
                        iou = _iou(clicked_mask.cpu().numpy(), gt_mask)
                        results[f"{n_clicks}-click"].append(iou)

        avg_mIoU = {k: float(np.mean(v)) if v else 0.0 for k, v in results.items()}
        return {
            "dataset": dataset_name,
            "mIoU": avg_mIoU,
        }

    def _simulate_image_interaction(
        self, image: torch.Tensor, gt_mask: np.ndarray, n_clicks: int
    ) -> torch.Tensor:
        """Run interactive segmentation on a single image (treated as 1‑frame video)."""
        init_clicks = self.click_sim.generate_initial_clicks(gt_mask)
        if not init_clicks:
            return torch.zeros_like(image[0])  # no object

        # Prepare first prompt
        prompt_list = [{"clicks": init_clicks}]
        masks_logits = None
        for _ in range(n_clicks - 1):
            # Run the model with current prompts
            masks_pred, _ = self._predict_single_image(image, prompt_list[-1])
            # Generate correction
            corr = self.click_sim.generate_correction_clicks(
                pred_mask=masks_pred.cpu().numpy(),
                gt_mask=gt_mask,
                random_gt_prob=0.0,
            )
            if not corr:
                break
            # Merge clicks
            new_clicks = prompt_list[-1]["clicks"] + corr
            prompt_list.append({"clicks": new_clicks})

        # Final prediction with all accumulated clicks
        final_mask, _ = self._predict_single_image(image, prompt_list[-1])
        return final_mask

    @torch.no_grad()
    def _predict_single_image(
        self, image: torch.Tensor, prompt: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass on a single frame with given prompt.

        Returns:
            binary mask (H, W) and the full model output dict.
        """
        # Ensure we have a batch dim of 1
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)

        # Build the per‑frame prompt list for SAM2Model.forward
        prompts = [prompt]   # only one frame

        # The model expects a list of prompts per frame. Use the forward method.
        output = self.model.forward(image, prompts, memory_bank_state=None)

        # Extract the predicted mask (already binarized after best selection)
        mask = output["masks"][0]  # (H, W)
        return mask, output

    # ------------------------------------------------------------------
    #  Master evaluation method
    # ------------------------------------------------------------------

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Run all evaluation protocols on all configured datasets.

        This is a convenience method that calls the individual evaluators
        and collects the results in a dictionary suitable for reporting.

        Returns:
            A nested dictionary summarising all metrics.
        """
        all_results = {}

        # ---- 1. Interactive offline (9 dense datasets) ----
        for ds in self.eval_cfg.interactive_offline.datasets:
            print(f"Evaluating interactive offline on {ds}")
            res = self.evaluate_interactive_offline(ds)
            all_results.setdefault("interactive_offline", {})[ds] = res

        # ---- 2. Interactive online (same 9 datasets) ----
        for ds in self.eval_cfg.interactive_online.datasets:
            print(f"Evaluating interactive online on {ds}")
            res = self.evaluate_interactive_online(ds)
            all_results.setdefault("interactive_online", {})[ds] = res

        # ---- 3. Semi‑supervised VOS (17 datasets) ----
        for ds in self.eval_cfg.semi_supervised.datasets:
            print(f"Evaluating semi‑supervised VOS on {ds}")
            res = self.evaluate_semi_supervised(ds)
            all_results.setdefault("semi_supervised", {})[ds] = res

        # ---- 4. Image segmentation (37 datasets) ----
        # The config lists "all 37 ...", but actual names need to be supplied.
        # Here we assume a predefined list or a way to iterate.
        # For demonstration, we expect the config to contain an explicit list.
        # We'll read the list from a hardcoded variable if not in config.
        img_datasets = self.eval_cfg.image_segmentation.datasets
        if isinstance(img_datasets, str) and img_datasets == "all 37":
            # Placeholder: use a subset for code illustration.
            img_datasets = ["ADE20K", "Cityscapes"]  # replace with full list
        for ds in img_datasets:
            print(f"Evaluating image segmentation on {ds}")
            res = self.evaluate_image_segmentation(ds)
            all_results.setdefault("image_segmentation", {})[ds] = res

        return all_results
