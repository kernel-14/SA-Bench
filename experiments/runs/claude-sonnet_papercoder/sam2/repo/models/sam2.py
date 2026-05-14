## Code: models/sam2.py

```python
## models/sam2.py
"""SAM 2: Segment Anything Model 2 — top-level model orchestration.

This module wires together all SAM 2 sub-modules into a coherent streaming
inference and training pipeline. It implements the core data flow described
in Section 4 and Appendix D.1 of the paper:

    Input frame → Image Encoder → Memory Attention → Mask Decoder
                                        ↑
                              Memory Bank (FIFO queues)
                                        ↑
                              Memory Encoder ← Previous predictions

Key design decisions:
    - forward_image() runs the encoder once per frame (streaming)
    - forward_video_frame() applies memory conditioning + mask prediction
    - _update_memory_bank() encodes predictions into compact 64-dim memories
    - Memory bank is stateful and must be reset between videos
    - Multi-object inference: shared encoder features, separate memory banks
    - Image mode: empty memory bank → model behaves like SAM

Config references:
    model.num_recent_memories: 6
    model.memory_feature_dim: 64
    model.object_pointer_dim: 256
    model.num_object_pointer_tokens: 4
    model.memory_attention_layers: 4
    model.num_multimask_outputs: 3
    model.mask_threshold: 0.0
    model.use_rope_2d: true
    model.use_flash_attention: true
    model.input_resolution: 1024
    model.fpn_out_channels: 256

Paper references:
    Section 4: "SAM 2 is equipped with a memory that stores information about
        the object and previous interactions."
    Section 4: "When applied to images, the memory is empty and the model
        behaves like SAM."
    Appendix D.1: "we perform inference on each object independently. More
        specifically, we share the visual features from the image encoder
        between all the objects in the video but run all the other model
        components separately for each object."
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.image_encoder import HieraImageEncoder
from models.mask_decoder import MaskDecoder, TwoWayTransformer
from models.memory_attention import MemoryAttention
from models.memory_bank import MemoryBank, MemoryBankOutput
from models.memory_encoder import MemoryEncoder
from models.prompt_encoder import PromptEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM2Config dataclass — single source of truth for all architectural params
# ---------------------------------------------------------------------------


@dataclass
class SAM2Config:
    """Configuration dataclass for SAM 2 architecture.

    All fields correspond directly to entries in config.yaml under the
    'model' section. This is the single source of truth for architectural
    hyperparameters, imported by every sub-module.

    Config references (config.yaml model section):
        image_encoder_type: "hiera_b_plus"
        fpn_out_channels: 256
        memory_attention_layers: 4
        memory_attention_self_attn: 4
        memory_attention_cross_attn: 4
        num_recent_memories: 6
        memory_feature_dim: 64
        object_pointer_dim: 256
        num_object_pointer_tokens: 4
        num_multimask_outputs: 3
        mask_threshold: 0.0
        use_rope_2d: true
        use_rpb: false
        use_flash_attention: true
        input_resolution: 1024
    """

    # Image encoder
    image_encoder_type: str = "hiera_b_plus"

    # FPN output channels (frame embedding dimension)
    fpn_out_channels: int = 256

    # Memory attention
    memory_attention_layers: int = 4
    memory_attention_self_attn: int = 4
    memory_attention_cross_attn: int = 4

    # Memory bank
    num_recent_memories: int = 6
    memory_feature_dim: int = 64
    object_pointer_dim: int = 256
    num_object_pointer_tokens: int = 4

    # Mask decoder
    num_multimask_outputs: int = 3
    mask_threshold: float = 0.0

    # Positional encoding
    use_rope_2d: bool = True
    use_rpb: bool = False
    use_flash_attention: bool = True

    # Input resolution
    input_resolution: int = 1024

    # Encoder-specific settings (populated from config.yaml nested dicts)
    encoder_drop_path_rate: float = 0.2  # default for hiera_b_plus
    global_attn_blocks: List[int] = field(
        default_factory=lambda: [12, 16, 20]  # hiera_b_plus default
    )

    # Skip connection channel dimensions (encoder-size dependent)
    # Hiera-B+ Stage 1 (stride-4): 112 channels
    # Hiera-B+ Stage 2 (stride-8): 224 channels
    skip_s4_channels: int = 112
    skip_s8_channels: int = 224

    # Mask decoder transformer settings
    transformer_depth: int = 2
    transformer_num_heads: int = 8
    transformer_mlp_dim: int = 2048

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> "SAM2Config":
        """Build SAM2Config from a plain dict (e.g., from OmegaConf.to_container).

        Handles nested config.yaml structure by extracting the 'model' sub-dict
        and mapping encoder-specific settings based on image_encoder_type.

        Args:
            cfg_dict: Plain dict from config.yaml, either the full config or
                just the 'model' sub-dict.

        Returns:
            SAM2Config instance with all fields populated.
        """
        # Support both full config dict and model-only sub-dict
        model_cfg: dict = cfg_dict.get("model", cfg_dict)

        encoder_type: str = model_cfg.get("image_encoder_type", "hiera_b_plus")

        # Extract encoder-specific drop path rate
        drop_path_rates: dict = model_cfg.get("encoder_drop_path_rates", {})
        drop_path_rate: float = drop_path_rates.get(encoder_type, 0.2)

        # Extract global attention block indices
        global_attn_blocks_map: dict = model_cfg.get("global_attn_blocks", {})
        global_attn_blocks: List[int] = global_attn_blocks_map.get(
            encoder_type, [12, 16, 20]
        )

        # Encoder channel dimensions per variant
        # (stage1_dim, stage2_dim) for skip connections
        skip_channels_map: Dict[str, Tuple[int, int]] = {
            "hiera_t":      (96,  192),
            "hiera_s":      (96,  192),
            "hiera_b_plus": (112, 224),
            "hiera_l":      (144, 288),
        }
        skip_s4, skip_s8 = skip_channels_map.get(encoder_type, (112, 224))

        return cls(
            image_encoder_type=encoder_type,
            fpn_out_channels=model_cfg.get("fpn_out_channels", 256),
            memory_attention_layers=model_cfg.get("memory_attention_layers", 4),
            memory_attention_self_attn=model_cfg.get("memory_attention_self_attn", 4),
            memory_attention_cross_attn=model_cfg.get("memory_attention_cross_attn", 4),
            num_recent_memories=model_cfg.get("num_recent_memories", 6),
            memory_feature_dim=model_cfg.get("memory_feature_dim", 64),
            object_pointer_dim=model_cfg.get("object_pointer_dim", 256),
            num_object_pointer_tokens=model_cfg.get("num_object_pointer_tokens", 4),
            num_multimask_outputs=model_cfg.get("num_multimask_outputs", 3),
            mask_threshold=model_cfg.get("mask_threshold", 0.0),
            use_rope_2d=model_cfg.get("use_rope_2d", True),
            use_rpb=model_cfg.get("use_rpb", False),
            use_flash_attention=model_cfg.get("use_flash_attention", True),
            input_resolution=model_cfg.get("input_resolution", 1024),
            encoder_drop_path_rate=drop_path_rate,
            global_attn_blocks=global_attn_blocks,
            skip_s4_channels=skip_s4,
            skip_s8_channels=skip_s8,
        )


# ---------------------------------------------------------------------------
# Shared dataclasses — data contracts between model, trainer, and evaluators
# ---------------------------------------------------------------------------


@dataclass
class PromptInput:
    """Input prompt container for a single frame.

    Shared data contract between PromptSampler (training), all evaluators
    (evaluation), and SAM2Model (inference). Defined here to avoid circular
    imports — datasets and evaluation depend on models, not the reverse.

    Coordinate convention: (x, y) pixel coordinates for points and boxes,
    consistent with PromptEncoder's expected input format.

    Attributes:
        points: Optional click coordinates of shape [N, 2] in (x, y) pixel
            space. None if no click prompts for this frame.
        point_labels: Optional click labels of shape [N] with values:
            1 = positive click, 0 = negative click, -1 = padding.
            Must be provided when points is not None.
        boxes: Optional bounding box of shape [4] or [B, 4] in (x1, y1, x2, y2)
            pixel coordinates. None if no box prompt for this frame.
        masks: Optional mask prompt of shape [1, H, W] or [B, 1, H, W].
            Values can be binary {0, 1}, probabilities [0, 1], or logits.
            None if no mask prompt for this frame.
        frame_idx: Integer index of the video frame this prompt belongs to.
            Used by propagate_video() to determine which frames are prompted.
    """

    points: Optional[torch.Tensor] = None
    point_labels: Optional[torch.Tensor] = None
    boxes: Optional[torch.Tensor] = None
    masks: Optional[torch.Tensor] = None
    frame_idx: int = 0


@dataclass
class SAM2FrameOutput:
    """Output container for a single processed video frame.

    Shared data contract between SAM2Model, Trainer, InteractiveEvaluator,
    and VOSEvaluator. Contains all outputs needed for loss computation,
    memory bank update, and evaluation metric computation.

    Attributes:
        masks: Predicted mask logits of shape [B, num_masks, H, W].
            Raw logits before sigmoid/threshold. num_masks = num_multimask_outputs+1
            when multimask_output=True, else 1.
        iou_scores: Predicted IoU scores of shape [B, num_masks].
            Sigmoid-activated (in [0, 1]) per Appendix D.2.1.
        occlusion_score: Predicted occlusion probability of shape [B, 1].
            Sigmoid-activated. High value (> 0.5) means object likely not visible.
        object_pointer: Mask decoder output token of shape [B, object_pointer_dim=256].
            Used as the object pointer stored in the memory bank.
        selected_mask_idx: Integer index of the mask selected for propagation.
            = argmax(iou_scores) when multimask_output=True and no follow-up
            prompts resolve ambiguity. = 0 when multimask_output=False.
    """

    masks: torch.Tensor
    iou_scores: torch.Tensor
    occlusion_score: torch.Tensor
    object_pointer: torch.Tensor
    selected_mask_idx: int = 0


# ---------------------------------------------------------------------------
# SAM2Model
# ---------------------------------------------------------------------------


class SAM2Model(nn.Module):
    """SAM 2: Segment Anything Model 2 — unified image and video segmentation.

    Orchestrates all sub-modules into a streaming inference pipeline:
        Image Encoder → Memory Attention → Mask Decoder
                              ↑
                    Memory Bank (FIFO queues)
                              ↑
                    Memory Encoder ← Previous predictions

    The model processes video frames one at a time (streaming), maintaining
    a memory bank of past predictions. For images, the memory bank is empty
    and the model behaves identically to SAM.

    Multi-object inference: the image encoder is run once per frame and its
    features are shared across all objects. Each object has its own memory
    bank and mask decoder state. The caller (evaluator or trainer) manages
    per-object memory banks and passes them explicitly to forward_video_frame().

    Args:
        config: SAM2Config instance containing all architectural hyperparameters.
            Use SAM2Config.from_dict(cfg_dict) to build from config.yaml.

    Example:
        config = SAM2Config()  # uses defaults for Hiera-B+
        model = SAM2Model(config)
        model.eval()

        # Image mode (SAM-like)
        image = torch.randn(1, 3, 1024, 1024)
        prompt = PromptInput(
            points=torch.tensor([[512, 512]], dtype=torch.float32),
            point_labels=torch.tensor([1]),
            frame_idx=0,
        )
        output = model.forward_image_only(image, prompt)

        # Video mode
        frames = torch.randn(1, 10, 3, 1024, 1024)  # [B, T, C, H, W]
        outputs = model.propagate_video(frames, initial_prompts=prompt)
    """

    def __init__(self, config: SAM2Config) -> None:
        super().__init__()

        self.config: SAM2Config = config

        # ------------------------------------------------------------------
        # Image encoder: Hiera backbone + FPN
        # Runs once per frame; produces unconditioned frame embedding and
        # skip features for the mask decoder.
        # Config: model.image_encoder_type, model.fpn_out_channels
        # ------------------------------------------------------------------
        self.image_encoder: HieraImageEncoder = HieraImageEncoder(config)

        # ------------------------------------------------------------------
        # Memory attention: L=4 transformer blocks conditioning frame features
        # on past predictions stored in the memory bank.
        # Config: model.memory_attention_layers=4, model.use_rope_2d=true
        # ------------------------------------------------------------------
        self.memory_attention: MemoryAttention = MemoryAttention(config)

        # ------------------------------------------------------------------
        # Prompt encoder: encodes clicks, boxes, and masks into embeddings.
        # Identical to SAM's PromptEncoder (Section 4).
        # Config: model.fpn_out_channels=256, model.input_resolution=1024
        # ------------------------------------------------------------------
        embed_size: int = config.input_resolution // 16  # stride-16 → 64 for 1024 input
        self.prompt_encoder: PromptEncoder = PromptEncoder(
            embed_dim=config.fpn_out_channels,
            image_embedding_size=(embed_size, embed_size),
            input_image_size=(config.input_resolution, config.input_resolution),
        )

        # ------------------------------------------------------------------
        # Mask decoder: two-way transformer + upsampling + output heads.
        # Extended from SAM with occlusion head and skip connections.
        # Config: model.num_multimask_outputs=3, model.fpn_out_channels=256
        # ------------------------------------------------------------------
        transformer: TwoWayTransformer = TwoWayTransformer(
            depth=config.transformer_depth,
            embedding_dim=config.fpn_out_channels,
            num_heads=config.transformer_num_heads,
            mlp_dim=config.transformer_mlp_dim,
        )
        self.mask_decoder: MaskDecoder = MaskDecoder(
            transformer_dim=config.fpn_out_channels,
            transformer=transformer,
            num_multimask_outputs=config.num_multimask_outputs,
            skip_s4_channels=config.skip_s4_channels,
            skip_s8_channels=config.skip_s8_channels,
        )

        # ------------------------------------------------------------------
        # Memory encoder: converts frame predictions into compact 64-dim
        # spatial memory features for storage in the memory bank.
        # Config: model.fpn_out_channels=256, model.memory_feature_dim=64
        # ------------------------------------------------------------------
        self.memory_encoder: MemoryEncoder = MemoryEncoder(
            in_dim=config.fpn_out_channels,
            out_dim=config.memory_feature_dim,
        )

        # ------------------------------------------------------------------
        # Memory bank: FIFO queues of spatial memories and object pointers.
        # Stateful — must be reset between videos via reset_memory().
        # Config: model.num_recent_memories=6, model.memory_feature_dim=64
        # ------------------------------------------------------------------
        self.memory_bank: MemoryBank = MemoryBank(
            max_recent_frames=config.num_recent_memories,
            memory_dim=config.memory_feature_dim,
            max_prompted_frames=2,  # M=2 per training setup (max_prompted_frames)
            object_pointer_dim=config.object_pointer_dim,
            num_object_pointer_tokens=config.num_object_pointer_tokens,
        )

        # Mask threshold for binary output (config.model.mask_threshold: 0.0)
        self.mask_threshold: float = config.mask_threshold

        logger.info(
            "SAM2Model initialized: encoder=%s, resolution=%d, "
            "memory_dim=%d, num_recent=%d",
            config.image_encoder_type,
            config.input_resolution,
            config.memory_feature_dim,
            config.num_recent_memories,
        )

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------

    def forward_image(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the image encoder to produce unconditioned frame features.

        This is the first step in the streaming pipeline. The encoder runs
        once per frame and its output is reused for both memory attention
        (conditioned embedding) and memory encoding (unconditioned embedding).

        Paper: "The image encoder is only run once for the entire interaction
        and its role is to provide unconditioned tokens (feature embeddings)
        representing each frame." (Section 4)

        Args:
            image: Input image tensor of shape [B, 3, H, W] where H=W=1024
                (or 512 for ablations). Values should be normalized to [0, 1]
                or ImageNet-normalized depending on the encoder's preprocessing.

        Returns:
            Tuple of:
                - frame_embed: FPN-fused frame embedding of shape
                  [B, fpn_out_channels, H/16, W/16] = [B, 256, 64, 64] for 1024 input.
                  This is the UNCONDITIONED embedding — memory attention has not
                  been applied yet.
                - skip_features: List of two tensors for mask decoder skip connections:
                  [0]: stride-4 features from Stage 1, shape [B, C_s4, H/4, W/4]
                  [1]: stride-8 features from Stage 2, shape [B, C_s8, H/8, W/8]
                  For Hiera-B+: C_s4=112, C_s8=224.
        """
        frame_embed, skip_features = self.image_encoder.forward(image)
        return frame_embed, skip_features

    def forward_video_frame(
        self,
        frame_embed: torch.Tensor,
        skip_features: List[torch.Tensor],
        prompts: Optional[PromptInput],
        memory_bank: MemoryBank,
    ) -> "SAM2FrameOutput":
        """Process a single video frame with memory conditioning and mask prediction.

        Implements the core per-frame processing pipeline:
            1. Retrieve memory bank contents for cross-attention
            2. Apply memory attention to condition frame features on past predictions
            3. Encode prompts (if any) into sparse and dense embeddings
            4. Run mask decoder to predict masks, IoU scores, and occlusion score
            5. Select the best mask for propagation

        Paper: "The memory attention operation takes the per-frame embedding from
        the image encoder and conditions it on the memory bank, before the mask
        decoder ingests it to form a prediction." (Section 4)

        Args:
            frame_embed: Unconditioned frame embedding from forward_image(),
                shape [B, fpn_out_channels, H/16, W/16].
            skip_features: Skip connection features from forward_image(),
                list of [stride4_feat, stride8_feat].
            prompts: Optional PromptInput for this frame. If None, the frame
                is processed as an unprompted propagation frame.
            memory_bank: MemoryBank instance containing past predictions.
                Pass an empty MemoryBank for image-mode inference.
                For multi-object inference, pass the per-object memory bank.

        Returns:
            SAM2FrameOutput containing:
                - masks: [B, num_masks, H, W] logits
                - iou_scores: [B, num_masks] sigmoid-activated scores
                - occlusion_score: [B, 1] sigmoid-activated occlusion probability
                - object_pointer: [B, 256] mask decoder output token
                - selected_mask_idx: index of mask selected for propagation
        """
        B: int = frame_embed.shape[0]
        device: torch.device = frame_embed.device

        # ------------------------------------------------------------------
        # Step 1: Retrieve memory bank contents for cross-attention
        # Returns empty tensors when bank is empty (image mode / first frame)
        # ------------------------------------------------------------------
        memory_bank_output: MemoryBankOutput = memory_bank.get_memory_for_attention()

        # ------------------------------------------------------------------
        # Step 2: Apply memory attention to condition frame features
        # When memory bank is empty, this degenerates to self-attention only
        # → model behaves like SAM (Section 4)
        # ------------------------------------------------------------------
        conditioned_embed: torch.Tensor = self.memory_attention.forward(
            curr_frame_embed=frame_embed,
            memory_bank_output=memory_bank_output,
        )
        # conditioned_embed: [B, fpn_out_channels, H/16, W/16]

        # ------------------------------------------------------------------
        # Step 3: Encode prompts into sparse and dense embeddings
        # When no prompts are provided (propagation frame), use empty sparse
        # embeddings and the no-mask dense embedding from PromptEncoder.
        # ------------------------------------------------------------------
        multimask_output: bool = False

        if prompts is not None:
            # Prepare point prompts
            points_tuple: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            if prompts.points is not None and prompts.point_labels is not None:
                # Ensure batch dimension
                pts: torch.Tensor = prompts.points
                lbls: torch.Tensor = prompts.point_labels
                if pts.ndim == 2:
                    pts = pts.unsqueeze(0).expand(B, -1, -1)
                if lbls.ndim == 1:
                    lbls = lbls.unsqueeze(0).expand(B, -1)
                points_tuple = (pts.to(device), lbls.to(device))

            # Prepare box prompts
            boxes_input: Optional[torch.Tensor] = None
            if prompts.boxes is not None:
                boxes_input = prompts.boxes.to(device)
                if boxes_input.ndim == 1:
                    boxes_input = boxes_input.unsqueeze(0).expand(B, -1)

            # Prepare mask prompts
            masks_input: Optional[torch.Tensor] = None
            if prompts.masks is not None:
                masks_input = prompts.masks.to(device)
                if masks_input.ndim == 3:
                    masks_input = masks_input.unsqueeze(0).expand(B, -1, -1, -1)

            sparse_embeddings, dense_embeddings = self.prompt_encoder.forward(
                points=points_tuple,
                boxes=boxes_input,
                masks=masks_input,
            )

            # Multi-mask output for ambiguous single-click prompts
            # Paper: "for ambiguous prompts (i.e., a single click) where there
            # may be multiple compatible target masks, we predict multiple masks"
            is_single_click: bool = (
                points_tuple is not None
                and boxes_input is None
                and masks_input is None
                and points_tuple[0].shape[1] == 1  # exactly one click
            )
            multimask_output = is_single_click

        else:
            # No prompts — propagation frame
            # PromptEncoder returns empty sparse embeddings and no-mask dense embedding
            sparse_embeddings, dense_embeddings = self.prompt_encoder.forward(
                points=None,
                boxes=None,
                masks=None,
            )
            multimask_output = False

        # ------------------------------------------------------------------
        # Step 4: Run mask decoder
        # Returns masks, IoU scores, occlusion score, and object pointer
        # ------------------------------------------------------------------
        image_pe: torch.Tensor = self.prompt_encoder.get_dense_pe()
        # image_pe: [1, fpn_out_channels, H/16, W/16] — broadcast over batch

        masks, iou_scores, occlusion_score, object_pointer = self.mask_decoder.forward(
            image_embeddings=conditioned_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            skip_features=skip_features,
            multimask_output=multimask_output,
        )
        # masks: [B, num_masks, H, W]
        # iou_scores: [B, num_masks] — sigmoid-activated
        # occlusion_score: [B, 1] — sigmoid-activated
        # object_pointer: [B, 256]

        # ------------------------------------------------------------------
        # Step 5: Select mask for propagation
        # Paper: "If no follow-up prompts resolve the ambiguity, the model
        # only propagates the mask with the highest predicted IoU for the
        # current frame." (Section 4)
        # ------------------------------------------------------------------
        selected_mask_idx: int = self._select_mask_idx(
            iou_scores=iou_scores,
            multimask_output=multimask_output,
        )

        return SAM2FrameOutput(
            masks=masks,
            iou_scores=iou_scores,
            occlusion_score=occlusion_score,
            object_pointer=object_pointer,
            selected_mask_idx=selected_mask_idx,
        )

    def _select_mask_idx(
        self,
        iou_scores: torch.Tensor,
        multimask_output: bool,
    ) -> int:
        """Select the mask index for propagation based on IoU scores.

        For multi-mask output (ambiguous single click), selects the mask with
        the highest predicted IoU score. For single-mask output (propagation
        or unambiguous prompt), always returns index 0.

        Paper: "If no follow-up prompts resolve the ambiguity, the model only
        propagates the mask with the highest predicted IoU for the current
        frame." (Section 4)

        Args:
            iou_scores: Predicted IoU scores of shape [B, num_masks].
                Sigmoid-activated, values in [0, 1].
            multimask_output: Whether multi-mask prediction was used.

        Returns:
            Integer index of the selected mask (0-based).
            For multi-mask: argmax over the multi-mask outputs (indices 1..N).
            For single-mask: always 0.
        """
        if not multimask_output or iou_scores.shape[1] <= 1:
            return 0

        # For multi-mask output, index 0 is the single-mask output and
        # indices 1..num_multimask_outputs are the multi-mask outputs.
        # Select the best among the multi-mask outputs (indices 1..N).
        # Use the first batch element's scores for index selection
        # (consistent across batch during inference; training uses all).
        multi_mask_scores: torch.Tensor = iou_scores[0, 1:]  # [num_multimask_outputs]
        best_multi_idx: int = int(torch.argmax(multi_mask_scores).item()) + 1

        return best_multi_idx

    # ------------------------------------------------------------------
    # Memory bank management
    # ------------------------------------------------------------------

    def _update_memory_bank(
        self,
        frame_embed: torch.Tensor,
        frame_output: "SAM2FrameOutput",
        memory_bank: MemoryBank,
        is_prompted: bool,
        frame_idx: int,
    ) -> None:
        """Encode current frame prediction into memory and update the memory bank.

        Called after each processed frame to store the prediction in the memory
        bank for future frame conditioning.

        IMPORTANT: Uses the UNCONDITIONED frame embedding (from forward_image),
        not the memory-attention-conditioned embedding. This is explicit in the
        paper: "summing it element-wise with the unconditioned frame embedding
        from the image-encoder" (Section 4).

        Args:
            frame_embed: Unconditioned frame embedding from forward_image(),
                shape [B, fpn_out_channels, H/16, W/16].
            frame_output: SAM2FrameOutput from forward_video_frame().
            memory_bank: MemoryBank to update. For multi-object inference,
                this is the per-object memory bank.
            is_prompted: If True, store in prompted_memories queue (no temporal PE).
                If False, store in recent_memories queue (temporal PE applied on read).
            frame_idx: Absolute frame index in the video (0-based).

        Returns:
            None. Modifies memory_bank in-place.
        """
        # Determine occlusion from sigmoid-activated occlusion score
        # occlusion_score is already sigmoid-activated from MaskDecoder
        occlusion_prob: float = float(frame_output.occlusion_score[0, 0].item())
        is_occluded: bool = occlusion_prob > 0.5

        # Get the selected mask for memory encoding
        # Shape: [B, H, W] — logits for the selected prediction
        selected_idx: int = frame_output.selected_mask_idx
        selected_mask: torch.Tensor = frame_output.masks[:, selected_idx, :, :]
        # Expand to [B, 1, H, W] for MemoryEncoder
        selected_mask = selected_mask.unsqueeze(1)

        # Encode prediction into compact 64-dim spatial memory feature
        # MemoryEncoder handles occlusion embedding internally
        memory_feature: torch.Tensor = self.memory_encoder.forward(
            frame_embedding=frame_embed,
            mask=selected_mask,
            is_occluded=is_occluded,
        )
        # memory_feature: [B, memory_feature_dim=64, H/16, W/16]

        # Add to memory bank FIFO queue
        memory_bank.add_memory(
            memory=memory_feature,
            is_prompted=is_prompted,
            object_pointer=frame_output.object_pointer,
            is_occluded=is_occluded,
            frame_idx=frame_idx,
        )

    def reset_memory(self) -> None:
        """Clear the memory bank between videos or objects.

        Must be called at the start of each new video to prevent cross-video
        contamination. Also called when switching to a different object in
        multi-object inference.