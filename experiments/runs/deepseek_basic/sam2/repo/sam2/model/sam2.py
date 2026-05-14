"""
SAM 2: Segment Anything Model 2

Full model combining:
- Image encoder (Hiera-based)
- Memory attention
- Prompt encoder
- Mask decoder
- Memory encoder
- Memory bank

The model operates in streaming fashion:
1. Image encoder runs once per frame
2. Memory attention conditions frame features on past memories
3. Prompt encoder encodes user prompts
4. Mask decoder produces segmentation mask, IoU, occlusion score, and object pointer
5. Memory encoder transforms mask + image embedding into memory
6. Memory bank stores memories for future frames

For images (single-frame), the memory is empty and the model behaves like SAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

from .image_encoder import HieraImageEncoder
from .memory_attention import MemoryAttention
from .prompt_encoder import PromptEncoder
from .mask_decoder import MaskDecoder
from .memory_encoder import MemoryEncoder
from .memory_bank import MemoryBank


class SAM2(nn.Module):
    """
    SAM 2: Segment Anything Model 2.

    Supports both image and video segmentation with interactive prompting.
    """

    def __init__(
        self,
        # Image encoder
        encoder_size: str = "base_plus",
        img_size: int = 1024,
        # Embedding dimensions
        embed_dim: int = 256,
        memory_dim: int = 64,
        # Memory
        num_memory_attn_layers: int = 4,
        num_memory_attn_heads: int = 8,
        max_recent_frames: int = 6,
        max_prompted_frames: int = 4,
        # Mask decoder
        num_multimask_outputs: int = 3,
        num_mask_decoder_blocks: int = 2,
        # Speed
        use_flash_attention: bool = False,
    ):
        super().__init__()

        self.encoder_size = encoder_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.num_multimask_outputs = num_multimask_outputs

        # Image encoder
        self.image_encoder = HieraImageEncoder(
            encoder_size=encoder_size,
            img_size=img_size,
            out_channels=embed_dim,
            use_abs_pos=True,
        )

        # Memory attention (stacks L transformer blocks with self+ cross-attention)
        self.memory_attention = MemoryAttention(
            embed_dim=embed_dim,
            num_heads=num_memory_attn_heads,
            num_layers=num_memory_attn_layers,
        )

        # Prompt encoder (identical to SAM)
        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=(img_size // 16, img_size // 16),
            input_image_size=(img_size, img_size),
        )

        # Mask decoder (largely follows SAM with occlusion head and skip connections)
        self.mask_decoder = MaskDecoder(
            embed_dim=embed_dim,
            num_multimask_outputs=num_multimask_outputs,
            num_transformer_blocks=num_mask_decoder_blocks,
        )

        # Memory encoder
        self.memory_encoder = MemoryEncoder(
            embed_dim=embed_dim,
            memory_dim=memory_dim,
        )

        # Memory bank
        self.memory_bank = MemoryBank(
            memory_dim=memory_dim,
            max_recent_frames=max_recent_frames,
            max_prompted_frames=max_prompted_frames,
            feature_map_size=(img_size // 16, img_size // 16),
        )

        # No-mask embedding for frames where object is absent
        self.no_mask_embed = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.normal_(self.no_mask_embed, std=0.02)

    def _initialize_state(self, video_id: Optional[str] = None):
        """Initialize/reinitialize memory state for a new video or object."""
        self.memory_bank.reset_state()

    def forward(
        self,
        frame: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        is_first_frame: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single frame with optional prompts.

        Args:
            frame: [B, 3, H, W] input image frame
            points: tuple of (point_coords [B, N, 2], point_labels [B, N])
            boxes: [B, N, 4] box prompts
            masks: [B, H, W] mask prompts
            multimask_output: whether to output multiple masks for ambiguity
            is_first_frame: whether this is the first frame (reset memory)

        Returns:
            dict with:
                'masks': [B, C, H, W] predicted segmentation masks
                'iou_predictions': [B, C] predicted IoU scores
                'occlusion_prediction': [B] occlusion prediction logits
                'object_pointer': [B, 256] object pointer for memory
        """
        if is_first_frame:
            self._initialize_state()

        B, _, H_img, W_img = frame.shape

        # 1. Image encoder (run once per frame, produces unconditioned embeddings)
        image_embedding, high_res_features = self.image_encoder(frame)
        # image_embedding: [B, C, H/16, W/16]
        # high_res_features: [skip4, skip8] each [B, C, stride*H, stride*W]

        # 2. Memory attention: condition current frame features on past memories
        memory_output = self.memory_bank.get_memory(frame.device)
        if memory_output.spatial_memory.shape[1] > 0:  # Has memories
            conditioned_embedding = self.memory_attention(
                image_embedding,
                memory_output,
            )
        else:
            # No memories yet: skip memory attention (like SAM)
            conditioned_embedding = image_embedding

        # 3. Prompt encoder: encode user prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        # 4. Generate dense positional encodings for the image embeddings
        image_pe = self.prompt_encoder.pe_layer(
            (image_embedding.shape[2], image_embedding.shape[3])
        ).unsqueeze(0).expand(B, -1, -1, -1)

        # 5. Mask decoder: predict masks
        masks, iou_pred, occlusion_pred, object_pointer = self.mask_decoder(
            image_embeddings=conditioned_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            high_res_features=high_res_features,
            multimask_output=multimask_output,
        )

        # 6. Memory encoder: generate memory for this frame
        # Use the best mask (highest IoU) for memory
        best_mask_idx = iou_pred.argmax(dim=1)
        best_mask = masks[torch.arange(B), best_mask_idx].unsqueeze(1)  # [B, 1, H, W]

        memory_features = self.memory_encoder(
            mask_pred=best_mask,
            image_embedding=image_embedding,  # unconditioned embedding
        )

        # 7. Store in memory bank
        self.memory_bank.add_memory(
            memory_features=memory_features,
            object_pointer=object_pointer,
            is_prompted=(points is not None or boxes is not None or masks is not None),
            is_occluded=(torch.sigmoid(occlusion_pred) < 0.5).any(),
        )

        return {
            "masks": masks,
            "iou_predictions": iou_pred,
            "occlusion_prediction": occlusion_pred,
            "object_pointer": object_pointer,
            "image_embedding": image_embedding,
            "high_res_features": high_res_features,
        }

    @torch.no_grad()
    def process_video(
        self,
        frames: torch.Tensor,
        prompts: Optional[List[Optional[Dict]]] = None,
        multimask_output: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Process a full video in streaming fashion.

        Args:
            frames: [T, 3, H, W] video frames
            prompts: list of length T with optional prompts per frame
            multimask_output: whether to output multiple masks

        Returns:
            list of dicts with predictions per frame
        """
        self._initialize_state()
        results = []

        for t in range(frames.shape[0]):
            frame = frames[t:t+1]
            frame_prompts = prompts[t] if prompts is not None and t < len(prompts) else None

            points = frame_prompts.get("points") if frame_prompts else None
            boxes = frame_prompts.get("boxes") if frame_prompts else None
            mask_prompts = frame_prompts.get("masks") if frame_prompts else None

            output = self.forward(
                frame=frame,
                points=points,
                boxes=boxes,
                masks=mask_prompts,
                multimask_output=multimask_output,
                is_first_frame=(t == 0),
            )
            results.append(output)

        return results


def build_sam2(
    encoder_size: str = "base_plus",
    img_size: int = 1024,
    checkpoint_path: Optional[str] = None,
    **kwargs,
) -> SAM2:
    """
    Build a SAM 2 model with specified configuration.

    Args:
        encoder_size: one of "tiny", "small", "base_plus", "large"
        img_size: input image size (square)
        checkpoint_path: optional path to load pretrained weights

    Returns:
        SAM2 model
    """
    model = SAM2(
        encoder_size=encoder_size,
        img_size=img_size,
        **kwargs,
    )

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model
