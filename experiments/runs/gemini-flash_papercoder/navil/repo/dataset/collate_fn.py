import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from config import Config
from utils import logger


class CustomCollateFn:
    """
    A callable object used by torch.utils.data.DataLoader to process a batch of samples.
    It handles dynamic padding for both text and image sequences, constructs appropriate
    attention masks, and integrates special tokens and visual token placeholders.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: Config):
        """
        Initializes the CustomCollateFn.

        Args:
            tokenizer: The Hugging Face tokenizer for the LLM.
            config: The global configuration object.
        """
        self.tokenizer = tokenizer
        self.config = config

        # Retrieve and store IDs for special tokens from the Config object
        self.begin_of_image_id = tokenizer.convert_tokens_to_ids(config.begin_of_image)
        self.end_of_image_id = tokenizer.convert_tokens_to_ids(config.end_of_image)
        self.end_of_line_id = tokenizer.convert_tokens_to_ids(config.end_of_line)
        self.end_of_scale_id = tokenizer.convert_tokens_to_ids(config.end_of_scale)

        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            # Fallback if tokenizer does not explicitly define a pad_token_id.
            # Using EOS token or a specific unused token ID is common practice.
            # Here, we default to the tokenizer's EOS token ID if pad is None, or 0 if that also fails.
            self.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            logger.warning(f"Tokenizer does not have a pad_token_id. Using {self.pad_token_id} as default. "
                           "Ensure this does not conflict with valid token IDs.")
        
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        
        # Max sequence length for the LLM, including all tokens (text + visual placeholders)
        self.llm_max_sequence_length = config.common.llm_max_sequence_length
        
        # Stride of the visual encoder's patch embedding layer, used to calculate num_visual_tokens
        self.patch_embedding_stride = config.model_architecture.visual_encoder.patch_embedding_stride
        
        # Default label to ignore in loss calculation, common in Hugging Face
        self.labels_ignore_index = -100

        logger.info(f"CustomCollateFn initialized. Special token IDs: "
                    f"BoI={self.begin_of_image_id}, EoI={self.end_of_image_id}, "
                    f"EoL={self.end_of_line_id}, EoS={self.end_of_scale_id}, "
                    f"Pad={self.pad_token_id}, LLM Max Seq Len={self.llm_max_sequence_length}.")


    @staticmethod
    def _pad_image_to_multiple(image_tensor: torch.Tensor, multiple: int) -> torch.Tensor:
        """
        Pads a single image tensor (C, H, W) with zeros so that its height and width
        are multiples of the specified `multiple`.

        Args:
            image_tensor: A torch.Tensor of shape (C, H, W).
            multiple: The integer value to which width and height should be multiples.

        Returns:
            The padded torch.Tensor of shape (C, H_padded, W_padded).
        """
        # image_tensor shape is (C, H, W)
        _, H, W = image_tensor.shape
        
        pad_h = (multiple - (H % multiple)) % multiple
        pad_w = (multiple - (W % multiple)) % multiple

        if pad_h == 0 and pad_w == 0:
            return image_tensor # No padding needed

        # Pad with 0.0 (black) on right (W) and bottom (H)
        # (pad_left, pad_right, pad_top, pad_bottom)
        padded_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
        return padded_tensor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Processes a list of individual data samples into a batched format.

        Args:
            batch: A list of dictionaries, where each dictionary represents a single sample
                   as returned by MultimodalDataset.__getitem__. Expected keys:
                   'image_tensors': List[torch.Tensor] (pre-padded image tensors for VMP scales)
                   'text_ids': List[int] (token IDs of the original text)
                   'image_path': str (path to image for debugging)

        Returns:
            A dictionary containing batched tensors for:
                'images': torch.Tensor (all image scales stacked for visual encoder)
                'input_ids': torch.Tensor (LLM input sequence with visual placeholders)
                'attention_mask': torch.Tensor
                'labels': torch.Tensor (LLM labels with masked special tokens)
                'visual_token_lengths': torch.Tensor (padded lengths of visual token sequences per scale per sample)
                'visual_token_start_indices': torch.Tensor (padded start indices of visual tokens in combined image batch)
        """
        batched_image_tensors_for_visual_encoder: List[torch.Tensor] = []
        batched_multimodal_input_ids: List[List[int]] = []
        batched_labels: List[List[int]] = []
        
        all_sample_visual_token_lengths: List[List[int]] = []
        all_sample_visual_token_start_indices: List[List[int]] = [] # Index into batched_image_tensors_for_visual_encoder

        global_image_batch_idx = 0 # Tracks the sequential index for images in batched_image_tensors_for_visual_encoder

        for sample_idx, sample in enumerate(batch):
            sample_image_tensors = sample['image_tensors']
            sample_text_ids = sample['text_ids']

            # Determine if VMP is enabled for the current stage.
            # The `Config` object stores stage parameters directly, if `current_stage_name` is set.
            # Assuming `config.current_stage_name` is set by the trainer to indicate the current stage.
            current_stage_config_key = self.config.get("current_stage_name", "stage_1_1")
            vmp_enabled_for_current_stage = self.config.get(
                f"training_stages.{current_stage_config_key}.visual_multi_scale_packing",
                False # Default to False if config path not found, or VMP not specified for stage
            )

            current_sample_multimodal_input_ids: List[int] = []
            current_sample_labels: List[int] = []
            current_sample_visual_token_lengths: List[int] = []
            current_sample_visual_token_start_indices: List[int] = []
            
            has_image = len(sample_image_tensors) > 0 # At least one image scale means an image is present

            if has_image:
                for scale_idx, image_tensor_scale in enumerate(sample_image_tensors):
                    # Image tensors coming from MultimodalDataset are already padded to multiples of 32
                    # (which is patch_embedding_stride * 2 if stride is 16).
                    # `_pad_image_to_multiple` is called here for strict adherence to design,
                    # but it should act as a no-op if images are already correctly padded upstream.
                    # It ensures dimensions are multiples of 32, as required by the Visual Encoder's PatchEmbed.
                    padded_image_tensor = self._pad_image_to_multiple(image_tensor_scale, self.patch_embedding_stride * 2)
                    batched_image_tensors_for_visual_encoder.append(padded_image_tensor)

                    # Calculate number of visual tokens for this scale
                    # The PatchEmbed layer produces N_patches = H_out * W_out
                    # H_out = H_padded / patch_embedding_stride
                    # W_out = W_padded / patch_embedding_stride
                    _, H_padded, W_padded = padded_image_tensor.shape
                    num_visual_tokens_h = H_padded // self.patch_embedding_stride
                    num_visual_tokens_w = W_padded // self.patch_embedding_stride
                    num_visual_tokens_in_this_scale = num_visual_tokens_h * num_visual_tokens_w
                    
                    current_sample_visual_token_lengths.append(num_visual_tokens_in_this_scale)
                    current_sample_visual_token_start_indices.append(global_image_batch_idx) # Index in the final stacked images tensor

                    # Append special tokens and placeholders for visual embeddings
                    current_sample_multimodal_input_ids.append(self.begin_of_image_id)
                    current_sample_labels.append(self.labels_ignore_index) # Mask BoI from loss

                    # Add placeholder tokens for visual embeddings.
                    # As per paper "Special token <end_of_line> is inserted at the end of each row of image tokens".
                    for row_idx in range(num_visual_tokens_h):
                        current_sample_multimodal_input_ids.extend([self.pad_token_id] * num_visual_tokens_w)
                        current_sample_labels.extend([self.labels_ignore_index] * num_visual_tokens_w)

                        # Add <end_of_line> token after each row
                        if self.end_of_line_id is not None and row_idx < num_visual_tokens_h - 1: # Only for intermediate rows
                            current_sample_multimodal_input_ids.append(self.end_of_line_id)
                            current_sample_labels.append(self.labels_ignore_index) # Mask EoL

                    current_sample_multimodal_input_ids.append(self.end_of_image_id)
                    current_sample_labels.append(self.labels_ignore_index) # Mask EoI

                    # Add <end_of_scale> if VMP is enabled and it's not the last scale for this sample
                    if vmp_enabled_for_current_stage and scale_idx < len(sample_image_tensors) - 1:
                        current_sample_multimodal_input_ids.append(self.end_of_scale_id)
                        current_sample_labels.append(self.labels_ignore_index) # Mask EoS

                    global_image_batch_idx += 1
            
            # --- Text processing ---
            # Prepend BOS token if tokenizer has one and it's not already implicitly handled
            if self.bos_token_id is not None and sample_text_ids and sample_text_ids[0] != self.bos_token_id:
                current_sample_multimodal_input_ids.append(self.bos_token_id)
                current_sample_labels.append(self.labels_ignore_index) # Mask BOS from loss

            # Add actual text tokens and their labels
            current_sample_multimodal_input_ids.extend(sample_text_ids)
            current_sample_labels.extend(sample_text_ids) # Labels are the text tokens themselves for causal LM

            # Append EOS token if tokenizer has one and it's not already implicitly handled
            if self.eos_token_id is not None and (not sample_text_ids or sample_text_ids[-1] != self.eos_token_id):
                current_sample_multimodal_input_ids.append(self.eos_token_id)
                current_sample_labels.append(self.eos_token_id) # EOS token is usually part of the label for next token prediction

            # --- Truncation ---
            # Truncate if the combined sequence exceeds the max LLM sequence length
            if len(current_sample_multimodal_input_ids) > self.llm_max_sequence_length:
                logger.warning(f"Sample {sample_idx} multimodal sequence length "
                               f"({len(current_sample_multimodal_input_ids)}) exceeds "
                               f"max ({self.llm_max_sequence_length}). Truncating.")
                current_sample_multimodal_input_ids = current_sample_multimodal_input_ids[:self.llm_max_sequence_length]
                current_sample_labels = current_sample_labels[:self.llm_max_sequence_length] # Labels should be truncated accordingly

            batched_multimodal_input_ids.append(current_sample_multimodal_input_ids)
            batched_labels.append(current_sample_labels)
            all_sample_visual_token_lengths.append(current_sample_visual_token_lengths)
            all_sample_visual_token_start_indices.append(current_sample_visual_token_start_indices)

        # --- Final Batch Padding for text-related tensors ---
        max_seq_len = max(len(ids) for ids in batched_multimodal_input_ids) if batched_multimodal_input_ids else 0
        
        # Pad input_ids
        padded_input_ids = [
            ids + [self.pad_token_id] * (max_seq_len - len(ids))
            for ids in batched_multimodal_input_ids
        ]
        
        # Pad labels
        padded_labels = [
            lbls + [self.labels_ignore_index] * (max_seq_len - len(lbls))
            for lbls in batched_labels
        ]

        # Create attention mask
        attention_mask = [
            [1] * len(ids) + [0] * (max_seq_len - len(ids))
            for ids in batched_multimodal_input_ids
        ]

        # Convert to tensors
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        labels_tensor = torch.tensor(padded_labels, dtype=torch.long)
        
        # Stack image tensors
        # If no images in the batch, create an empty tensor with dummy dimensions (e.g., matching expected VE input shape)
        if batched_image_tensors_for_visual_encoder:
            images_tensor = torch.stack(batched_image_tensors_for_visual_encoder)
        else:
            # Dummy shape: 0 images, 3 channels, H_min, W_min (e.g., 32x32 based on padding multiple)
            images_tensor = torch.empty(0, 3, self.patch_embedding_stride * 2, self.patch_embedding_stride * 2, dtype=torch.float)

        # Pad and convert visual token info to tensors to match Dict[str, Tensor] return type
        max_scales_per_sample = max(len(l) for l in all_sample_visual_token_lengths) if all_sample_visual_token_lengths else 0

        padded_visual_token_lengths = [
            l + [0] * (max_scales_per_sample - len(l))
            for l in all_sample_visual_token_lengths
        ]
        padded_visual_token_start_indices = [
            idx + [0] * (max_scales_per_sample - len(idx)) # 0 is a safe padding value for index
            for idx in all_sample_visual_token_start_indices
        ]
        
        visual_token_lengths_tensor = torch.tensor(padded_visual_token_lengths, dtype=torch.long)
        visual_token_start_indices_tensor = torch.tensor(padded_visual_token_start_indices, dtype=torch.long)

        return {
            'images': images_tensor,
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'labels': labels_tensor,
            'visual_token_lengths': visual_token_lengths_tensor,
            'visual_token_start_indices': visual_token_start_indices_tensor
        }

