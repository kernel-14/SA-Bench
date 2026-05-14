# model/sam2.py

"""
SAM2Model – top‑level ensemble of all SAM 2 components.

This module implements the SAM2Model class, which ties together:
- Hiera image encoder
- FPN to fuse multi‑scale features
- Memory bank (FIFO queues for recent / prompted frames and object pointers)
- Memory attention (transformer blocks with self/cross‑attention and RoPE)
- Prompt encoder (clicks, boxes, masks)
- Mask decoder (multi‑mask, IoU prediction, occlusion head, skip connections)
- Memory encoder (mask + image embedding fusion)

The streaming forward method processes video frames one at a time,
conditioning the current frame on past memories and user prompts.
A single model instance can be reused for multiple objects by calling
:meth:`reset_memory` and then running the forward pass sequentially.

All hyper‑parameters are drawn from the configuration dictionary
produced by :class:`config.Config`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sub‑modules (imported from the designated files)
from model.hiera import Hiera, HIERA_VARIANTS
from model.memory import MemoryBank, MemoryAttention
from model.prompt_encoder import PromptEncoder
from model.mask_decoder import MaskDecoder
from model.memory_encoder import MemoryEncoder


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class SAM2Model(nn.Module):
    """
    SAM 2 – promptable visual segmentation in images and videos.

    Args:
        config: The **entire** configuration dictionary as returned by
            :class:`config.Config` (or a compatible ``AttrDict``).  The
            model extracts its parameters from the ``"model"`` section.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        # Short‑hand access to the model configuration block
        model_cfg: Dict[str, Any] = config["model"]

        # ------------------------------------------------------------------
        # 1.  Image encoder (Hiera) + FPN
        # ------------------------------------------------------------------
        self.resolution: int = model_cfg["resolution"]  # e.g. 1024
        self.image_encoder: Hiera = Hiera(
            variant=model_cfg["image_encoder"]["variant"],
            pretrained=model_cfg["image_encoder"]["pretrained"],
        )
        self._build_fpn(
            fpn_channels=model_cfg["fpn_channels"],
            variant=model_cfg["image_encoder"]["variant"],
        )

        # ------------------------------------------------------------------
        # 2.  Memory components
        # ------------------------------------------------------------------
        spatial_size: int = self.resolution // 16  # 64 for resolution=1024
        self.memory_bank: MemoryBank = MemoryBank(
            max_recent=model_cfg["memory"]["max_recent_frames"],
            max_prompted=model_cfg["memory"]["max_prompted_frames"],
            feat_dim=model_cfg["memory"]["feature_channels"],  # 64
            spatial_size=spatial_size,
        )
        self.memory_attention: MemoryAttention = MemoryAttention(
            d_model=model_cfg["fpn_channels"],
            nhead=model_cfg["memory_attention"]["num_heads"],
            num_layers=model_cfg["memory_attention"]["num_layers"],
        )

        # ------------------------------------------------------------------
        # 3.  Prompt encoder
        # ------------------------------------------------------------------
        self.prompt_encoder: PromptEncoder = PromptEncoder(
            embed_dim=model_cfg["fpn_channels"]
        )

        # ------------------------------------------------------------------
        # 4.  Mask decoder
        # ------------------------------------------------------------------
        self.mask_decoder: MaskDecoder = MaskDecoder(config)

        # ------------------------------------------------------------------
        # 5.  Memory encoder
        # ------------------------------------------------------------------
        self.memory_encoder: MemoryEncoder = MemoryEncoder(
            in_ch=model_cfg["fpn_channels"],       # 256
            out_ch=model_cfg["memory"]["feature_channels"],  # 64
        )

        # ------------------------------------------------------------------
        # Misc flags
        # ------------------------------------------------------------------
        self.use_object_pointers: bool = model_cfg["memory"]["object_pointers"]

        # Per‑frame unconditioned image embedding – will be filled by
        # ``_encode_frame`` and used by the memory encoder within the same
        # time step.
        self.unconditioned_embed: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    #  FPN (Feature Pyramid Network) helpers
    # ------------------------------------------------------------------

    def _build_fpn(self, fpn_channels: int, variant: str) -> None:
        """
        Create the two 1 × 1 projection conv layers that map Hiera stage 3
        (stride 16) and stage 4 (stride 32) features to a common channel
        dimension, and a final 3 × 3 conv to smooth the fused result.

        The output channels of each Hiera stage are read from
        :data:`HIERA_VARIANTS`.
        """
        var_cfg = HIERA_VARIANTS[variant]
        stage_channels: List[int] = var_cfg["embed_dims"]  # [C1, C2, C3, C4]

        # Stride 16  →  fpn_channels
        self.fpn_proj16: nn.Conv2d = nn.Conv2d(stage_channels[2], fpn_channels, 1)
        # Stride 32  →  fpn_channels
        self.fpn_proj32: nn.Conv2d = nn.Conv2d(stage_channels[3], fpn_channels, 1)
        # Final fusion conv
        self.fpn_final: nn.Conv2d = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)

    def _fpn_forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse stride 16 and stride 32 features into a single ``(B, C, H, W)`` map.

        Args:
            features: Dict with at least ``"stage3"`` and ``"stage4"`` tensors.

        Returns:
            Fused feature map of shape ``(B, fpn_channels, H16, W16)``.
        """
        feat16: torch.Tensor = features["stage3"]   # (B, C3, H, W)
        feat32: torch.Tensor = features["stage4"]   # (B, C4, H/2, W/2)

        proj16 = self.fpn_proj16(feat16)            # (B, fpn_channels, H, W)
        proj32 = self.fpn_proj32(feat32)            # (B, fpn_channels, H/2, W/2)

        # Upsample to match spatial size of proj16
        proj32_up = F.interpolate(
            proj32,
            size=proj16.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = proj16 + proj32_up
        return self.fpn_final(fused)                # (B, fpn_channels, H, W)

    # ------------------------------------------------------------------
    #  Memory management
    # ------------------------------------------------------------------

    def reset_memory(self) -> None:
        """Clear the memory bank for a new segmentation target."""
        self.memory_bank.reset()

    # ------------------------------------------------------------------
    #  Per‑frame encoding (image → features)
    # ------------------------------------------------------------------

    def _encode_frame(self, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run the image encoder on a single video frame.

        Args:
            frame: ``(C, H, W)`` tensor (RGB, normalised to [0, 1]).

        Returns:
            dict with keys:
            - ``"features"`` – raw Hiera output dict (``stage1``…``stage4``)
            - ``"fpn_feat"`` – fused FPN feature map ``(1, fpn_channels, H16, W16)``
        """
        # Add batch dimension
        feats: Dict[str, torch.Tensor] = self.image_encoder(frame.unsqueeze(0))
        fpn_feat = self._fpn_forward(feats)   # (1, fpn_channels, H16, W16)

        # Store the unconditioned embedding for later memory encoding
        self.unconditioned_embed = fpn_feat

        return {"features": feats, "fpn_feat": fpn_feat}

    # ------------------------------------------------------------------
    #  Core per‑frame processing (shared by forward variants)
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        fpn_feat: torch.Tensor,
        features: Dict[str, torch.Tensor],
        prompt: Optional[Dict[str, Any]],
        memory: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single frame in the streaming pipeline.

        Args:
            fpn_feat: FPN feature map for the current frame
                ``(1, fpn_channels, H16, W16)``.
            features: Raw Hiera output dict (``stage1``…``stage4``).
            prompt: Prompt dictionary for this frame.  May be ``None`` if the
                frame receives no new prompts.  If present, expected keys:
                - ``"is_prompted"`` (bool) – ``True`` if new prompts are supplied.
                - ``"coords"``, ``"labels"`` – click coordinates & labels.
                - ``"boxes"`` – bounding boxes.
                - ``"masks"`` – dense mask prompts.
                - ``"multi_mask"`` (bool, optional) – force multi‑mask mode.
            memory: Current memory bank content (returned by
                :meth:`MemoryBank.get_memories`).

        Returns:
            dict with keys:
            - ``"mask"`` – binary mask ``(H, W)``
            - ``"masks_logits"`` – all logits ``(num_masks, H, W)``
            - ``"iou_pred"`` – predicted IoU per mask ``(num_masks,)``
            - ``"occlusion_logit"`` – occlusion logit ``(1,)`` or ``None``
            - ``"object_pointer"`` – memory pointer token ``(embed_dim,)``
        """
        # ---- 1. Memory attention on FPN features ----
        conditioned_embed = self.memory_attention(fpn_feat, memory)
        # conditioned_embed: (1, fpn_channels, H16, W16)

        # ---- 2. Integrate prompts ----
        sparse_prompt: Optional[torch.Tensor] = None
        dense_prompt: Optional[torch.Tensor] = None
        is_prompted: bool = False
        if prompt is not None:
            is_prompted = prompt.get("is_prompted", False)

        if is_prompted:
            # 2a. Sparse prompts (clicks / boxes)
            if "coords" in prompt and prompt["coords"] is not None:
                # encode_clicks expects (B, N, 2) coords and (B, N) labels
                sparse_prompt = self.prompt_encoder.encode_clicks(
                    prompt["coords"], prompt["labels"]
                )
            if "boxes" in prompt and prompt["boxes"] is not None:
                box_embed = self.prompt_encoder.encode_boxes(prompt["boxes"])
                if sparse_prompt is not None:
                    sparse_prompt = torch.cat([sparse_prompt, box_embed], dim=1)
                else:
                    sparse_prompt = box_embed

            # 2b. Dense prompt (mask)
            if "masks" in prompt and prompt["masks"] is not None:
                dense_prompt = self.prompt_encoder.encode_masks(prompt["masks"])
                if dense_prompt is not None:
                    # element‑wise addition (the decoder also expects a dense
                    # key, but we already added it here; we will pass None as
                    # dense key to avoid double addition)
                    conditioned_embed = conditioned_embed + dense_prompt

        # ---- 3. Determine multi‑mask mode ----
        multi_mask: bool = False
        if prompt is not None and "multi_mask" in prompt:
            multi_mask = bool(prompt["multi_mask"])
        elif is_prompted:
            # Heuristic: first prompt with a single positive click → ambiguous
            no_prior_prompts = len(self.memory_bank.prompted_queue) == 0
            single_positive_click = False
            if "coords" in prompt and prompt["coords"] is not None:
                if (
                    prompt["coords"].shape[1] == 1
                    and prompt["labels"].shape[1] == 1
                    and prompt["labels"][0, 0] == 1
                ):
                    single_positive_click = True
            multi_mask = no_prior_prompts and single_positive_click

        # ---- 4. Mask decoder ----
        skip_features = [features["stage2"], features["stage1"]]  # stride 8, stride 4
        prompt_embed = {
            "sparse": sparse_prompt,
            "dense": None,   # already added to conditioned_embed
        }
        decoder_out = self.mask_decoder(
            image_embed=conditioned_embed,
            prompt_embed=prompt_embed,
            skip_features=skip_features,
            multi_mask=multi_mask,
        )
        # decoder_out keys: 'masks' (1,num_masks,1024,1024), 'iou_pred' (1,num_masks),
        # 'occlusion_logit' (1,1), 'object_pointer_token' (1,embed_dim)

        # ---- 5. Post‑process: select best mask for propagation ----
        masks_logits: torch.Tensor = decoder_out["masks"].squeeze(0)  # (num_masks, H, W)
        iou_pred: torch.Tensor = decoder_out["iou_pred"].squeeze(0)   # (num_masks,)
        if multi_mask and masks_logits.shape[0] > 1:
            best_idx = torch.argmax(iou_pred).item()
            best_mask_logits = masks_logits[best_idx]  # (H, W)
        else:
            best_mask_logits = masks_logits[0]           # (H, W)
        best_mask: torch.Tensor = (
            (best_mask_logits.sigmoid() > 0.5).float()
        )  # binary mask (H, W)

        # ---- 6. Encode memory for this frame ----
        memory_feat = self.memory_encoder(
            mask=best_mask.unsqueeze(0).unsqueeze(0),   # (1, 1, H, W)
            image_embed=self.unconditioned_embed,       # (1, C, H16, W16)
        )  # (1, feat_channels, H16, W16)

        # ---- 7. Update memory bank ----
        self.memory_bank.add_memory(
            memory=memory_feat,
            is_prompted=is_prompted,
            object_pointer=decoder_out["object_pointer_token"],
        )

        # ---- 8. Return all interesting tensors ----
        occlusion_logit: Optional[torch.Tensor] = decoder_out.get("occlusion_logit", None)
        obj_ptr: torch.Tensor = decoder_out["object_pointer_token"].squeeze(0)  # (embed_dim,)

        return {
            "mask": best_mask,                      # (H, W)
            "masks_logits": masks_logits,           # (num_masks, H, W)
            "iou_pred": iou_pred,                   # (num_masks,)
            "occlusion_logit": occlusion_logit,     # (1,) or None
            "object_pointer": obj_ptr,              # (embed_dim,)
        }

    # ------------------------------------------------------------------
    #  Full video forward (streaming, with on‑the‑fly encoding)
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: torch.Tensor,
        prompts: List[Dict[str, Any]],
        memory_bank_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Streaming forward pass for a video clip.

        Args:
            frames: ``(T, C, H, W)`` tensor of resized video frames.
            prompts: list of length ``T``.  Each element is either ``None``
                (if no prompt is issued for that frame) or a dict containing
                the prompt specification (see :meth:`_process_frame`).
            memory_bank_state: optional pre‑filled memory snapshot (rarely
                used during inference).

        Returns:
            dict with keys:
            - ``"masks"`` – binary masks ``(T, H, W)``
            - ``"masks_logits"`` – all logits ``(T, num_masks, H, W)``
            - ``"iou_pred"`` – predicted IoU per mask ``(T, num_masks)``
            - ``"occlusion_logit"`` – occlusion logits ``(T, 1)`` if available
            - ``"object_pointers"`` – memory pointer tokens ``(T, embed_dim)``
        """
        if memory_bank_state is not None:
            self.memory_bank.load_state(memory_bank_state)

        T = frames.size(0)
        all_masks = []
        all_masks_logits = []
        all_iou_pred = []
        all_occlusion = []
        all_obj_ptrs = []

        for t in range(T):
            # 1. Encode frame
            frame_info = self._encode_frame(frames[t])
            fpn_feat = frame_info["fpn_feat"]
            features = frame_info["features"]

            # 2. Get memory
            memory = self.memory_bank.get_memories()

            # 3. Process
            out = self._process_frame(fpn_feat, features, prompts[t], memory)
            all_masks.append(out["mask"])                        # (H, W)
            all_masks_logits.append(out["masks_logits"])         # (num_masks, H, W)
            all_iou_pred.append(out["iou_pred"])                 # (num_masks,)
            all_occlusion.append(out["occlusion_logit"])
            all_obj_ptrs.append(out["object_pointer"])           # (embed_dim,)

        # Stack along time
        masks_tensor = torch.stack(all_masks, dim=0)             # (T, H, W)
        masks_logits_tensor = torch.stack(all_masks_logits, dim=0)  # (T, num_masks, H, W)
        iou_pred_tensor = torch.stack(all_iou_pred, dim=0)       # (T, num_masks)
        obj_ptrs_tensor = torch.stack(all_obj_ptrs, dim=0)       # (T, embed_dim)

        output_dict: Dict[str, torch.Tensor] = {
            "masks": masks_tensor,
            "masks_logits": masks_logits_tensor,
            "iou_pred": iou_pred_tensor,
            "object_pointers": obj_ptrs_tensor,
        }
        if all_occlusion[0] is not None:
            output_dict["occlusion_logit"] = torch.stack(
                [o.squeeze(0) for o in all_occlusion], dim=0
            ).unsqueeze(-1)  # (T, 1)

        return output_dict

    # ------------------------------------------------------------------
    #  Pre‑computed features variant (for multi‑object tracking)
    # ------------------------------------------------------------------

    def encode_video_features(
        self, frames: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Pre‑compute image features for all video frames.

        The resulting list can be fed to :meth:`forward_with_precomputed_features`
        to avoid redundant image encoder passes when tracking many objects.

        Args:
            frames: ``(T, C, H, W)`` video tensor.

        Returns:
            list of length ``T``, each element a dict with:
            - ``"fpn_feat"`` – ``(1, C, H16, W16)``
            - ``"features"`` – raw Hiera output dict
            - ``"unconditioned_embed"`` – same as ``fpn_feat``
        """
        self.eval()
        features_list: List[Dict[str, torch.Tensor]] = []
        with torch.no_grad():
            for t in range(frames.shape[0]):
                frame_info = self._encode_frame(frames[t])
                features_list.append({
                    "fpn_feat": frame_info["fpn_feat"],
                    "features": frame_info["features"],
                    "unconditioned_embed": frame_info["fpn_feat"],
                })
        return features_list

    def forward_with_precomputed_features(
        self,
        features_list: List[Dict[str, torch.Tensor]],
        prompts: List[Dict[str, Any]],
        memory_bank_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Streaming forward pass using pre‑cached image features.

        Args:
            features_list: output of :meth:`encode_video_features`.
            prompts: same as in :meth:`forward`.
            memory_bank_state: optional pre‑filled memory snapshot.

        Returns:
            same structure as :meth:`forward`.
        """
        if memory_bank_state is not None:
            self.memory_bank.load_state(memory_bank_state)

        T = len(features_list)
        all_masks = []
        all_masks_logits = []
        all_iou_pred = []
        all_occlusion = []
        all_obj_ptrs = []

        for t in range(T):
            # Retrieve pre‑computed data
            fpn_feat = features_list[t]["fpn_feat"]
            features = features_list[t]["features"]
            self.unconditioned_embed = features_list[t]["unconditioned_embed"]

            memory = self.memory_bank.get_memories()
            out = self._process_frame(fpn_feat, features, prompts[t], memory)

            all_masks.append(out["mask"])
            all_masks_logits.append(out["masks_logits"])
            all_iou_pred.append(out["iou_pred"])
            all_occlusion.append(out["occlusion_logit"])
            all_obj_ptrs.append(out["object_pointer"])

        masks_tensor = torch.stack(all_masks, dim=0)
        masks_logits_tensor = torch.stack(all_masks_logits, dim=0)
        iou_pred_tensor = torch.stack(all_iou_pred, dim=0)
        obj_ptrs_tensor = torch.stack(all_obj_ptrs, dim=0)

        output_dict: Dict[str, torch.Tensor] = {
            "masks": masks_tensor,
            "masks_logits": masks_logits_tensor,
            "iou_pred": iou_pred_tensor,
            "object_pointers": obj_ptrs_tensor,
        }
        if all_occlusion[0] is not None:
            output_dict["occlusion_logit"] = torch.stack(
                [o.squeeze(0) for o in all_occlusion], dim=0
            ).unsqueeze(-1)

        return output_dict

