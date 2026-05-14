
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional

from .image_encoder import ImageEncoder
from .prompt_encoder import PromptEncoder
from .mask_decoder import MaskDecoder
from .memory_attention import MemoryAttention
from .memory_encoder import MemoryEncoder

class SAM2(nn.Module):
    def __init__(
        self,
        image_encoder_type: str,
        image_encoder_out_chans: int,
        prompt_encoder_embed_dim: int,
        mask_decoder_num_heads: int,
        mask_decoder_num_layers: int,
        mask_decoder_iou_head_depth: int,
        mask_decoder_iou_head_hidden_dim: int,
        memory_attention_num_heads: int,
        memory_attention_num_layers: int,
        memory_channels: int,
        num_mask_tokens: int,
        num_point_embeddings: int,
        image_size: int = 1024,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            encoder_type=image_encoder_type,
            out_chans=image_encoder_out_chans,
            image_size=image_size,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_encoder_embed_dim,
            image_size=image_size,
            input_image_size=image_size,
            mask_in_chans=1, # assuming masks are single channel
            num_point_embeddings=num_point_embeddings,
        )
        self.mask_decoder = MaskDecoder(
            transformer_dim=image_encoder_out_chans, # Transformer uses image encoder output dim
            transformer_num_heads=mask_decoder_num_heads,
            transformer_num_layers=mask_decoder_num_layers,
            iou_head_depth=mask_decoder_iou_head_depth,
            iou_head_hidden_dim=mask_decoder_iou_head_hidden_dim,
            num_mask_tokens=num_mask_tokens,
        )
        self.memory_attention = MemoryAttention(
            embedding_dim=image_encoder_out_chans,
            num_heads=memory_attention_num_heads,
            num_layers=memory_attention_num_layers,
            rope_dim=64, # As per paper's memory attention details (Section D.1)
        )
        self.memory_encoder = MemoryEncoder(
            embed_dim=image_encoder_out_chans,
            output_channels=memory_channels,
        )

        self.memory_bank = {} # Stores memory features and object pointers for past frames
        self.num_recent_frames_memory_bank = -1 # Config.NUM_RECENT_FRAMES_MEMORY_BANK
        self.num_prompted_frames_memory_bank = -1 # Config.NUM_PROMPTED_FRAMES_MEMORY_BANK

    def set_memory_bank_configs(self, num_recent_frames: int, num_prompted_frames: int):
        self.num_recent_frames_memory_bank = num_recent_frames
        self.num_prompted_frames_memory_bank = num_prompted_frames
    
    def reset_memory_bank(self):
        self.memory_bank = {
            "spatial_memory": [], # FIFO queue of spatial feature maps
            "object_pointers": [], # FIFO queue of object pointers
            "prompted_frames_info": [], # List of (frame_idx, spatial_feat, obj_ptr) for prompted frames
        }

    def add_to_memory_bank(self, frame_idx: int, spatial_memory_features: torch.Tensor, object_pointer: torch.Tensor, is_prompted: bool = False):
        # Handle recent frames memory (FIFO)
        if len(self.memory_bank["spatial_memory"]) >= self.num_recent_frames_memory_bank:
            self.memory_bank["spatial_memory"].pop(0)
            self.memory_bank["object_pointers"].pop(0)
        self.memory_bank["spatial_memory"].append(spatial_memory_features)
        self.memory_bank["object_pointers"].append(object_pointer)

        # Handle prompted frames memory
        if is_prompted:
            # We don't limit prompted frames by FIFO explicitly as paper says "up to M prompted frames"
            # and that memory from prompted frames is not positionally encoded.
            self.memory_bank["prompted_frames_info"].append(
                (frame_idx, spatial_memory_features, object_pointer)
            )
            # If we need to limit the size of prompted_frames_info, implement FIFO here.
            # For now, assuming it grows with prompted frames.

    def forward(
        self,
        video_frames: torch.Tensor, # (B, T, C_in, H, W) for video, (B, 1, C_in, H, W) for image
        current_frame_idx: int, # Index of the frame being processed (0 for image)
        points: Optional[torch.Tensor] = None, # (B, N_points, 2)
        labels: Optional[torch.Tensor] = None, # (B, N_points)
        boxes: Optional[torch.Tensor] = None, # (B, 4)
        masks: Optional[torch.Tensor] = None, # (B, 1, H_orig, W_orig)
        is_prompted_frame: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size, num_frames_in_input = video_frames.shape[:2]
        
        # For simplicity, process one frame at a time from video_frames input
        # In actual streaming, only one frame is passed.
        current_frame = video_frames[:, current_frame_idx, :, :, :] # (B, C_in, H, W)

        # 1. Image Encoder
        encoded_features = self.image_encoder(current_frame)
        image_embedding = encoded_features["image_embedding"] # (B, C, H_embed, W_embed)
        multiscale_features = [encoded_features["stride4"], encoded_features["stride8"]]
        H_embed, W_embed = image_embedding.shape[2:]

        # 2. Memory Attention (if memory bank is not empty)
        if self.memory_bank["spatial_memory"] and self.num_recent_frames_memory_bank > 0:
            # Concatenate spatial memory features from recent frames
            spatial_mem_feats = torch.stack(self.memory_bank["spatial_memory"], dim=1) # (B, N_recent, C_mem, H_mem, W_mem)
            spatial_mem_feats = spatial_mem_feats.view(batch_size, -1, spatial_mem_feats.shape[-3]) # (B, N_recent * H_mem * W_mem, C_mem)

            # Concatenate object pointers from recent frames
            obj_ptrs = torch.stack(self.memory_bank["object_pointers"], dim=1).view(batch_size, -1, self.memory_attention.embedding_dim) # (B, N_recent, C)

            # Also include prompted frames' memories and object pointers
            for _, spatial_feat, obj_ptr in self.memory_bank["prompted_frames_info"]:
                spatial_mem_feats = torch.cat([spatial_mem_feats, spatial_feat.view(batch_size, -1, spatial_feat.shape[-3])], dim=1)
                obj_ptrs = torch.cat([obj_ptrs, obj_ptr.view(batch_size, 1, -1)], dim=1)

            # Apply Memory Attention
            # Flatten image_embedding for transformer input (B, H*W, C)
            image_embedding_flat = image_embedding.flatten(2).permute(0, 2, 1) # (B, H_embed*W_embed, C)
            conditioned_image_embedding_flat = self.memory_attention(
                image_embedding_flat,
                spatial_mem_feats,
                obj_ptrs,
            )
            conditioned_image_embedding = conditioned_image_embedding_flat.permute(0, 2, 1).reshape(batch_size, -1, H_embed, W_embed)
        else:
            conditioned_image_embedding = image_embedding

        # 3. Prompt Encoder
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points, labels, boxes, masks, (H_embed, W_embed)
        )

        # 4. Mask Decoder
        masks_output, iou_predictions, occlusion_predictions = self.mask_decoder(
            conditioned_image_embedding,
            sparse_embeddings,
            dense_embeddings,
            multiscale_features,
        )

        # Select the best mask and its object pointer for memory encoding
        # For simplicity, we assume the first mask token corresponds to the primary object.
        # In a real scenario, highest IoU mask would be chosen.
        best_mask = masks_output[:, 0:1, :, :] # (B, 1, H_orig, W_orig)

        # 5. Memory Encoder
        spatial_memory_features = self.memory_encoder(
            image_embedding, # Use unconditioned image_embedding for memory encoding as per Fig 3
            best_mask,
            (H_embed, W_embed),
        )
        # Extract object pointer from mask decoder (mask_tokens_out corresponding to best_mask)
        # Assuming mask_tokens_out (B, N_mask_tokens, C) from mask_decoder represents object pointers
        # For simplicity, take the first mask token as the object pointer.
        object_pointer = self.mask_decoder.mask_tokens.weight[0].unsqueeze(0).repeat(batch_size, 1) # (B, C)
        # In a full impl, we'd take the mask token that generated `best_mask`

        # Update memory bank
        if self.num_recent_frames_memory_bank > 0: # Only add to memory if it's a video task
            self.add_to_memory_bank(current_frame_idx, spatial_memory_features, object_pointer, is_prompted_frame)

        return masks_output, iou_predictions, occlusion_predictions

    def process_image(
        self,
        image: torch.Tensor, # (B, C_in, H, W)
        points: Optional[torch.Tensor] = None, # (B, N_points, 2)
        labels: Optional[torch.Tensor] = None, # (B, N_points)
        boxes: Optional[torch.Tensor] = None, # (B, 4)
        masks: Optional[torch.Tensor] = None, # (B, 1, H_orig, W_orig)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a single image (single frame, no memory).
        """
        self.reset_memory_bank() # Ensure memory is empty for image processing
        
        # Add a dummy temporal dimension for compatibility with forward
        image = image.unsqueeze(1) # (B, 1, C_in, H, W)
        
        masks_output, iou_predictions, occlusion_predictions = self.forward(
            video_frames=image,
            current_frame_idx=0,
            points=points,
            labels=labels,
            boxes=boxes,
            masks=masks,
            is_prompted_frame=True, # An image is always 'prompted' in SAM-like tasks
        )
        return masks_output, iou_predictions, occlusion_predictions

