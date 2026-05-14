"""
model.py

Implements the main SAM 2 model architecture, including the Image Encoder, Memory Attention,
Prompt Encoder, Memory Bank management, and Mask Decoder. The Model class integrates
all these components to perform tasks like interactive segmentation and promptable visual segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Any, Dict, Tuple
from torchvision.models.vision_transformer import vit_b_16, vit_large_16

from utils import positional_encoding


class Model(nn.Module):
    """
    Implements the main SAM 2 architecture for promptable segmentation tasks in both images and videos.
    """

    def __init__(self, 
                 backbone: str = "Hiera", 
                 memory_size: int = 6, 
                 prompt_types: List[str] = ["clicks", "boxes", "masks"]):
        """
        Initializes the SAM 2 model.

        Args:
            backbone (str): Specifies the backbone architecture, default is "Hiera".
            memory_size (int): Number of frames the memory bank can store (FIFO queue length).
            prompt_types (list): List of supported prompt types (e.g., clicks, boxes, masks).
        """
        super(Model, self).__init__()
        
        # Load pre-trained image encoder (Hiera Transformer, Defaults to ImageNet initialization)
        self.backbone = self._initialize_backbone(backbone)
        
        # Memory modules
        self.memory_size = memory_size  # Max size for the FIFO memory queue
        self.memory_bank = []  # FIFO memory queue - stores embeddings (up to `self.memory_size` frames)
        
        # Channel dimensions for image encoder and memory
        self.embedding_dim = 768 if backbone == "Hiera" else 1024  # Adapt to backbone

        # Prompt Encoder
        self.prompt_types = prompt_types
        self.prompt_encoder = PromptEncoder(self.embedding_dim)

        # Memory Attention
        self.memory_attention = MemoryAttentionLayer(
            embedding_dim=self.embedding_dim,
            num_layers=4  # Defined based on paper
        )

        # Mask Decoder
        self.mask_decoder = MaskDecoder(
            embedding_dim=self.embedding_dim,
            prompt_dim=self.embedding_dim,
            memory_dim=self.embedding_dim
        )

        # Occlusion Prediction
        self.occlusion_head = nn.Linear(self.embedding_dim, 1)

    def _initialize_backbone(self, backbone: str) -> nn.Module:
        """
        Initializes the backbone (Hiera transformer or alternatives like ViT).

        Args:
            backbone (str): Type of transformer backbone to use.

        Returns:
            nn.Module: Initialized backbone model (e.g., pre-trained Hiera).
        """
        if backbone == "Hiera":
            # Placeholder: Use a torch equivalent for Hiera/MAE-pretrained vision transformer
            return vit_b_16(pretrained=True)  # Replace with actual pre-trained Hiera implementation
        # Extendable for other transformer backbones
        raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, 
                x: torch.Tensor, 
                prompts: Optional[torch.Tensor] = None, 
                memory: Optional[Any] = None) -> torch.Tensor:
        """
        Forward propagation through the SAM 2 architecture.

        Args:
            x (torch.Tensor): Input tensor representing the video/image frame.
            prompts (torch.Tensor, optional): Encoded user inputs like clicks or bounding boxes.
            memory (Any, optional): Previously stored memory embeddings (defaults to contents of memory bank).

        Returns:
            torch.Tensor: Segmentation mask logits for the current frame.
        """
        # Step 1: Frame Embedding via Image Encoder
        frame_embedding = self.backbone(x)

        # Step 2: Memory Conditioning via Memory Attention
        conditioned_frame = self.memory_attention(
            current_frame_embedding=frame_embedding, 
            previous_memory=memory or self.memory_bank  # Use the externally provided memory or self.memory_bank
        )

        # Step 3: Fuse with Prompt Embedding
        if prompts is not None:
            prompt_embedding = self.prompt_encoder(prompts)
        else:
            prompt_embedding = None

        # Step 4: Decode Masks from Memory-conditioned Frame
        mask_logits, iou_logits = self.mask_decoder(
            conditioned_frame, 
            prompts=prompt_embedding
        )

        # Step 5: Predict Occlusion States
        occlusion_logits = self.occlusion_head(conditioned_frame.mean(dim=(2, 3)))  # Global pooling

        return mask_logits, iou_logits, occlusion_logits

    def update_memory(self, 
                      predictions: torch.Tensor, 
                      frame_embedding: torch.Tensor) -> None:
        """
        Updates the memory bank with predictions and frame embeddings.

        Args:
            predictions (torch.Tensor): Predicted mask logits for the frame.
            frame_embedding (torch.Tensor): Frame-level embeddings from the image encoder.
        """
        # Downsample mask predictions for memory storage
        memory_entry = F.avg_pool2d(predictions, kernel_size=4) + frame_embedding

        # Add temporal position encoding for better sequence modeling
        memory_entry += positional_encoding(memory_entry.shape)

        # Update the FIFO memory queue
        if len(self.memory_bank) >= self.memory_size:
            self.memory_bank.pop(0)  # FIFO principle
        self.memory_bank.append(memory_entry)


class PromptEncoder(nn.Module):
    """
    Encodes user-provided prompts (e.g., clicks, bounding boxes, masks) into feature embeddings.
    """

    def __init__(self, embedding_dim: int):
        super(PromptEncoder, self).__init__()
        self.click_embed = nn.Embedding(2, embedding_dim)  # Pos/Neg clicks
        self.box_embed = nn.Linear(4, embedding_dim)  # Bounding boxes (x1, y1, x2, y2)
        self.mask_conv = nn.Conv2d(1, embedding_dim, kernel_size=3, padding=1)  # Mask inputs

    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        """
        Encodes various prompt types into embeddings.

        Args:
            prompts (torch.Tensor): User-provided inputs (click coordinates, boxes, or masks).

        Returns:
            torch.Tensor: Encoded prompt embeddings.
        """
        prompt_embeddings = []
        if "clicks" in prompts:
            click_embedding = self.click_embed(prompts["clicks"].long())
            prompt_embeddings.append(click_embedding)
        if "boxes" in prompts:
            box_embedding = self.box_embed(prompts["boxes"].float())
            prompt_embeddings.append(box_embedding)
        if "masks" in prompts:
            mask_embedding = self.mask_conv(prompts["masks"].unsqueeze(1))
            prompt_embeddings.append(mask_embedding)

        # Combine prompt types into a single embedding
        return torch.stack(prompt_embeddings, dim=1).sum(dim=1)  # Sum embeddings from all types


class MemoryAttentionLayer(nn.Module):
    """
    Implements the memory attention module of SAM 2 to condition the current frame on
    past spatial memory embeddings.
    """

    def __init__(self, embedding_dim: int, num_layers: int):
        super(MemoryAttentionLayer, self).__init__()
        self.self_attention = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=4 * embedding_dim
        )
        self.cross_attention = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=4 * embedding_dim
        )
        self.num_layers = num_layers

    def forward(self, 
                current_frame_embedding: torch.Tensor, 
                previous_memory: List[torch.Tensor]) -> torch.Tensor:
        """
        Conditions current frame embeddings on past memory.

        Args:
            current_frame_embedding (torch.Tensor): Features for the current frame.
            previous_memory (list[torch.Tensor]): Stored spatial memory embeddings.

        Returns:
            torch.Tensor: Memory-conditioned embeddings for the current frame.
        """
        x = current_frame_embedding
        for _ in range(self.num_layers):
            x = self.self_attention(x)
            memory_stack = torch.stack(previous_memory, dim=0)  # Stack memory along batch dim
            x = self.cross_attention(x, memory_stack)
        return x


class MaskDecoder(nn.Module):
    """
    Decodes segmentation masks from memory-conditioned embeddings and prompt embeddings.
    """

    def __init__(self, embedding_dim: int, prompt_dim: int, memory_dim: int):
        super(MaskDecoder, self).__init__()
        self.transformer_blocks = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=4 * embedding_dim
        )
        self.output_head = nn.Conv2d(embedding_dim, 1, kernel_size=1)  # Single-channel mask logits

    def forward(self, 
                frame_embedding: torch.Tensor, 
                prompts: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode segmentation masks.

        Args:
            frame_embedding (torch.Tensor): Embeddings from frame encoder + memory attention.
            prompts (torch.Tensor, optional): Encoded prompt embeddings.

        Returns:
            tuple: Segmentation mask logits, IoU predictions.
        """
        x = frame_embedding
        if prompts is not None:
            x += prompts
        x = self.transformer_blocks(x)
        mask_logits = self.output_head(x)

        # Generate IoU estimates (as a scalar per frame)
        iou_logits = mask_logits.mean(dim=(2, 3))
        return mask_logits, iou_logits
