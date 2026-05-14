import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

# Import sub-modules
from model.image_encoder import ImageEncoder
from model.memory_attention import MemoryAttention
from model.prompt_encoder import PromptEncoder
from model.mask_decoder import MaskDecoder
from model.memory_modules import MemoryEncoder, MemoryBank


class SAM2Model(nn.Module):
    """
    The main Segment Anything Model 2 (SAM 2) class.
    It orchestrates the ImageEncoder, MemoryAttention, PromptEncoder, MaskDecoder,
    MemoryEncoder, and MemoryBank to perform promptable segmentation on images and videos.
    """

    def __init__(self, config: Any):  # Use Any to avoid circular import with config.py
        """
        Initializes the SAM2Model and its sub-modules.

        Args:
            config (Any): The global configuration object, typically an instance of config.Config.
        """
        super().__init__()
        self._config = config

        self.device: torch.device = torch.device(self._config.get("system.device", "cuda" if torch.cuda.is_available() else "cpu"))

        # Initialize sub-modules
        self.image_encoder = ImageEncoder(self._config).to(self.device)
        self.prompt_encoder = PromptEncoder(self._config).to(self.device)
        # Note: MaskDecoder's forward method is conceptually designed to return 4 items including the object pointer token.
        # This was an implicit detail in the paper's description of memory bank update,
        # which has been clarified for this implementation.
        self.mask_decoder = MaskDecoder(self._config).to(self.device)
        self.memory_encoder = MemoryEncoder(self._config).to(self.device)
        self.memory_attention = MemoryAttention(self._config).to(self.device)

        # Enable torch.compile for image_encoder if specified and available
        if self._config.get("system.torch_compile.image_encoder", False) and hasattr(torch, 'compile'):
            if torch.cuda.is_available() and self.device.type == 'cuda':
                print("Compiling ImageEncoder with torch.compile...")
                self.image_encoder = torch.compile(self.image_encoder)
            else:
                print("Skipping torch.compile for ImageEncoder: CUDA not available or device is CPU.")

        # Occlusion threshold for inference decisions (e.g., if pred_occlusion_score < threshold, object is occluded)
        self.occlusion_threshold: float = self._config.get("evaluation.occlusion_threshold", 0.5)
        # The prompt_embedding_dim should be consistent across modules
        self.prompt_embedding_dim: int = self._config.get("model.memory_attention.hidden_dim", 256)
        

    def forward(
        self,
        images: List[torch.Tensor],  # List of image tensors, each (C, H, W)
        prompts: List[Dict[str, Any]],  # List of dictionaries, each per frame, for ONE object
        memory_bank: Optional[MemoryBank] = None,
        is_training: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], MemoryBank]:
        """
        Processes a sequence of images (video frames or a single image) with prompts
        to generate segmentation masks.

        Args:
            images (List[torch.Tensor]): A list of image tensors. Each tensor is (C, H, W).
            prompts (List[Dict[str, Any]]): A list of dictionaries, one for each frame.
                                            Each dict contains prompts for the *single target object*.
                                            Expected keys: 'points' (Tensor), 'boxes' (Tensor), 'masks' (Tensor).
                                            Missing keys or empty tensors imply no prompt of that type.
            memory_bank (Optional[MemoryBank]): An optional MemoryBank instance. If None, a new one
                                                is created (e.g., for the start of a new video sequence
                                                or single image processing).
            is_training (bool, optional): Flag indicating training mode. Defaults to True.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], MemoryBank]:
                - all_pred_masks (List[torch.Tensor]): Predicted binary segmentation masks for each frame.
                                                     Each mask is (H, W).
                - all_pred_ious (List[torch.Tensor]): Predicted IoU scores for each frame. Each is (1,).
                - all_pred_occlusions (List[torch.Tensor]): Predicted occlusion probabilities for each frame.
                                                          Each is (1,).
                - updated_memory_bank (MemoryBank): The MemoryBank instance after processing all frames.
        """
        # Ensure model is in appropriate mode
        if is_training:
            self.train()
        else:
            self.eval()

        if memory_bank is None:
            memory_bank = MemoryBank(self._config).to(self.device)
        else:
            memory_bank.to(self.device) # Ensure memory bank is on correct device

        all_pred_masks: List[torch.Tensor] = []
        all_pred_ious: List[torch.Tensor] = []
        all_pred_occlusions: List[torch.Tensor] = []
        
        # 1. Image Feature Encoding (for all frames upfront)
        # The image encoder is run once for the entire interaction.
        # This produces multi-scale features for each frame.
        # List of [C1_feats, C2_feats, highest_level_feature] for each frame
        all_image_encoder_features: List[List[torch.Tensor]] = []
        # Use torch.no_grad() for image encoder if not training to save memory/computation,
        # especially if it's frozen during finetuning.
        # The paper mentions freezing image encoder during finetuning.
        # For simplicity, apply `no_grad` if `is_training` is False.
        with torch.no_grad() if not is_training else torch.enable_grad():
            for img_tensor in images:
                # Add batch dimension to the single image tensor: (C, H, W) -> (1, C, H, W)
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                features = self.image_encoder(img_tensor)
                all_image_encoder_features.append(features)

        # 2. Sequential Frame Processing
        # Ensure that `torch.enable_grad()` is explicitly used if `is_training` is True.
        # This loop will run with `grad` enabled/disabled based on the overall `is_training` flag.
        for frame_idx, (frame_image_features, frame_prompts) in enumerate(
            zip(all_image_encoder_features, prompts)
        ):
            C1_feats, C2_feats, highest_level_feature = frame_image_features
            # highest_level_feature is (1, C_img, H_img, W_img) (e.g., stride 16)

            # Determine if current frame has prompts
            is_prompted_current_frame: bool = (
                (frame_prompts.get('points') is not None and frame_prompts['points'].numel() > 0) or
                (frame_prompts.get('boxes') is not None and frame_prompts['boxes'].numel() > 0) or
                (frame_prompts.get('masks') is not None and frame_prompts['masks'].numel() > 0)
            )

            # 2.1. Prompt Encoding
            # Ensure prompt tensors are on the correct device.
            points_tensor = frame_prompts.get('points')
            if points_tensor is not None: points_tensor = points_tensor.unsqueeze(0).to(self.device)
            boxes_tensor = frame_prompts.get('boxes')
            if boxes_tensor is not None: boxes_tensor = boxes_tensor.unsqueeze(0).to(self.device)
            masks_tensor = frame_prompts.get('masks')
            if masks_tensor is not None: masks_tensor = masks_tensor.unsqueeze(0).to(self.device)

            prompt_embeddings: torch.Tensor = self.prompt_encoder(
                points=points_tensor,
                boxes=boxes_tensor,
                masks=masks_tensor,
            ) # (1, N_tokens, hidden_dim)

            # 2.2. Memory Retrieval
            memories = memory_bank.get_memories(frame_idx)
            # memories: { 'memory_features': List[Tensor(C,H,W)], 'object_pointers': List[Tensor(4, C_token)], 'temporal_pos_embeddings': List[Tensor(1)] }

            # 2.3. Memory Attention
            conditioned_frame_embedding: torch.Tensor = self.memory_attention(
                current_frame_features=highest_level_feature, # (1, C_img, H_img, W_img)
                memory_features=memories['memory_features'],
                object_pointers=memories['object_pointers'],
                temporal_pos_embeddings=memories['temporal_pos_embeddings'],
            ) # (1, H_img*W_img, hidden_dim)

            # 2.4. Mask Decoding
            # MaskDecoder expects conditioned_frame_embedding to be (B, C, H, W) for its upsampling path.
            # `conditioned_frame_embedding` from MemoryAttention is `(1, H_img*W_img, hidden_dim)`.
            # Reshape it to `(1, hidden_dim, H_img, W_img)` assuming H_img, W_img were the spatial dimensions
            # of the `highest_level_feature` before flattening for MemoryAttention.
            _, C_highest, H_highest, W_highest = highest_level_feature.shape
            conditioned_frame_embedding_spatial = conditioned_frame_embedding.transpose(1, 2).reshape(
                1, C_highest, H_highest, W_highest
            ) # (1, hidden_dim, H_highest, W_highest)

            # Skip connection features for MaskDecoder (C1, C2) must also be on device
            C1_feats = C1_feats.to(self.device)
            C2_feats = C2_feats.to(self.device)

            # MaskDecoder's forward is now expected to return 4 items, with the last being the object pointer token.
            masks_probs_multi, iou_preds_multi, occlusion_preds_probs, object_pointer_token_from_decoder = self.mask_decoder(
                image_embedding=conditioned_frame_embedding_spatial,
                prompt_embeddings=prompt_embeddings,
                multi_scale_features=[C1_feats, C2_feats], # Pass high-res skip features
            )
            # masks_probs_multi: (1, num_mask_variants, H_orig/4, W_orig/4)
            # iou_preds_multi: (1, num_mask_variants)
            # occlusion_preds_probs: (1, 1) or None
            # object_pointer_token_from_decoder: (1, object_pointer_dim) (this is the token for the selected mask)

            # 2.5. Mask Selection (for inference/propagation)
            pred_mask_t: torch.Tensor
            pred_iou_t: torch.Tensor
            pred_occlusion_t: torch.Tensor # Store as (1,) tensor

            if is_training or not self._config.get("model.mask_decoder.predict_multiple_masks", True):
                # During training or if not predicting multiple masks, take the first output.
                # The trainer will handle the supervision with ground truth.
                pred_mask_t = masks_probs_multi[:, 0, :, :].squeeze(0) # (H, W)
                pred_iou_t = iou_preds_multi[:, 0] # (1,)
                # Note: `object_pointer_token_from_decoder` is already the one selected by MaskDecoder,
                # if MaskDecoder's internal logic handles this selection. If not, it would need to be
                # selected here based on `best_idx`. Assuming MaskDecoder's output is ready.
            else:
                # During inference, select the mask with the highest predicted IoU
                best_idx = torch.argmax(iou_preds_multi, dim=1).squeeze(0) # Scalar index
                pred_mask_t = masks_probs_multi[:, best_idx, :, :].squeeze(0) # (H, W)
                pred_iou_t = iou_preds_multi[:, best_idx] # (1,)
            
            # Extract occlusion prediction
            if occlusion_preds_probs is not None:
                pred_occlusion_t = occlusion_preds_probs.squeeze(0) # (1,)
            else:
                # If occlusion head is disabled, assume object is always visible
                pred_occlusion_t = torch.tensor([1.0], device=self.device) # (1,)

            # 2.6. Determine Occlusion Status for Memory Update
            is_occluded_current_frame: bool = pred_occlusion_t.item() < self.occlusion_threshold

            # 2.7. Memory Encoding and Update
            # The predicted mask needs to be (1, 1, H, W) for MemoryEncoder
            memory_feature_t: torch.Tensor = self.memory_encoder(
                predicted_mask=pred_mask_t.unsqueeze(0).unsqueeze(0).to(self.device),
                frame_embedding=highest_level_feature,
            ) # (1, memory_feature_dim, H_feat, W_feat)

            memory_bank.update(
                frame_idx=frame_idx,
                memory_feature=memory_feature_t,
                object_pointer=object_pointer_token_from_decoder, # (1, object_pointer_dim)
                is_prompted=is_prompted_current_frame,
                is_occluded=is_occluded_current_frame,
            )

            # Store predictions (remove batch dim if any)
            all_pred_masks.append(pred_mask_t)
            all_pred_ious.append(pred_iou_t)
            all_pred_occlusions.append(pred_occlusion_t)

        return all_pred_masks, all_pred_ious, all_pred_occlusions, memory_bank

