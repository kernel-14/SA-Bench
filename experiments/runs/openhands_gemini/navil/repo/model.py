
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import Optional, List, Union

from modules import VisualEncoder, MLPProjector, NaViLLLMTypicalBlock
from layers import RMSNorm
from config import NaViLConfig

class NaViL(nn.Module):
    def __init__(self, config: NaViLConfig):
        super().__init__()
        self.config = config
        
        # 1. Visual Encoder
        # The paper describes the visual encoder as:
        # "visual encoder nu consists of a series of transformer layers and can be defined as
        # V_d,w(I) = C o F_d^w o ... o F_2^w o F_1^w o P(I)"
        # where P is Patch Embedding Layer.
        # "For simplicity, we use the same architectures as the LLM for the visual encoder layers F but with bi-directional attention and vary the hyperparameters d and w."
        # This implies the embed_dim for the visual encoder is `width` from config.visual_encoder
        self.visual_encoder = VisualEncoder(
            img_size=224,  # Nominal input image size, actual can vary due to padding/resizing
            patch_size=self.config.visual_encoder["patch_embedding_stride"], # stride is also kernel size
            in_channels=3,
            embed_dim=self.config.visual_encoder["width"],
            depth=self.config.visual_encoder["depth"],
            num_heads=self.config.visual_encoder["attention_heads"],
            mlp_dim=self.config.visual_encoder["mlp_width"],
            stride=self.config.visual_encoder["patch_embedding_stride"]
        )

        # 2. MLP Projector (Connector C)
        self.mlp_projector = MLPProjector(
            visual_embed_dim=self.config.visual_encoder["width"],
            llm_embed_dim=self.config.llm["width"]
        )

        # 3. LLM (with MoE)
        # Load pre-trained LLM tokenizer and model
        # For simplicity, we assume the base LLM provides the initial token embeddings
        # and final classification head, and we insert our MoE blocks.
        # In a full reproduction, one would load AutoModelForCausalLM and replace its transformer layers.
        
        # Placeholder for loading tokenizer and initial embeddings
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm["init_model"])
        except Exception: # Catching a more general exception for cases where model_name isn't found
            print(f"Could not load tokenizer for {self.config.llm['init_model']}. Using a dummy tokenizer.")
            # Create a dummy tokenizer for development purposes if actual one not found
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=None, 
                model_max_length=config.training["llm_max_sequence_length"],
                eos_token="<|endoftext|>", 
                pad_token="<|pad|>", 
                unk_token="<|unk|>",
                bos_token="<|bos|>"
            )


        # Define special tokens for multimodal input
        # These are crucial for interleaving visual and text tokens as described.
        self.IMG_START_TOKEN = "<image_start>"
        self.IMG_END_TOKEN = "<image_end>"
        self.EOL_TOKEN = "<end_of_line>" # For visual tokens in a row
        self.EOS_SCALE_TOKEN = "<end_of_scale>" # For multi-scale packing

        additional_special_tokens = [
            self.IMG_START_TOKEN, self.IMG_END_TOKEN,
            self.EOL_TOKEN, self.EOS_SCALE_TOKEN
        ]
        
        # Add special tokens to tokenizer
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
        print(f"Added {num_added_tokens} special tokens to tokenizer.")

        # Assign IDs for easier access
        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_START_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_END_TOKEN)
        self.eol_token_id = self.tokenizer.convert_tokens_to_ids(self.EOL_TOKEN)
        self.eos_scale_token_id = self.tokenizer.convert_tokens_to_ids(self.EOS_SCALE_TOKEN)

        # Initialize LLM token embeddings. If loading a pretrained LLM, this would be its embedding layer.
        # We need to resize it to accommodate new special tokens.
        self.llm_token_embeddings = nn.Embedding(len(self.tokenizer), self.config.llm["width"])
        # If loading an actual LLM, copy its weights to the new larger embedding layer, then initialize new tokens.
        
        self.llm_norm = RMSNorm(self.config.llm["width"])
        self.llm_head = nn.Linear(self.config.llm["width"], len(self.tokenizer), bias=False)

        self.llm_blocks = nn.ModuleList([
            NaViLLLMTypicalBlock(
                embed_dim=self.config.llm["width"],
                num_heads=self.config.llm["attention_heads"],
                mlp_dim=self.config.llm["mlp_width"]
            )
            for _ in range(self.config.llm["depth"])
        ])

    def _prepare_multimodal_input(self, images: Optional[Union[torch.Tensor, List[torch.Tensor]]],
                                  input_ids: torch.Tensor,
                                  attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepares the multimodal input sequence by interleaving visual and text tokens.
        This is a conceptual implementation based on the paper's description of special tokens.
        A full implementation would need exact image token layout and handling of EOL_TOKEN.
        """
        batch_size = input_ids.shape[0]
        
        # Get text embeddings
        text_embeddings = self.llm_token_embeddings(input_ids)

        # Initialize modality indicator for text (0 for linguistic)
        text_modality_indicator = torch.zeros(batch_size, text_embeddings.shape[1],
                                              device=text_embeddings.device, dtype=torch.long)
        
        # Initialize final combined sequence
        final_embeddings: List[torch.Tensor] = []
        final_modality_indicator: List[torch.Tensor] = []
        final_attention_mask: List[torch.Tensor] = []

        # Prepend BOS token if tokenizer has one (and if it's not already in input_ids)
        if self.tokenizer.bos_token_id is not None and input_ids[0, 0] != self.tokenizer.bos_token_id:
            bos_embedding = self.llm_token_embeddings(torch.tensor(self.tokenizer.bos_token_id, device=input_ids.device)).unsqueeze(0).repeat(batch_size, 1, 1)
            final_embeddings.append(bos_embedding)
            final_modality_indicator.append(torch.zeros(batch_size, 1, device=input_ids.device, dtype=torch.long))
            final_attention_mask.append(torch.ones(batch_size, 1, device=input_ids.device, dtype=torch.long))

        if images is not None:
            # Insert <image_start> token
            img_start_embedding = self.llm_token_embeddings(torch.tensor(self.img_start_token_id, device=input_ids.device)).unsqueeze(0).repeat(batch_size, 1, 1)
            final_embeddings.append(img_start_embedding)
            final_modality_indicator.append(torch.zeros(batch_size, 1, device=input_ids.device, dtype=torch.long)) # Special tokens are linguistic context
            final_attention_mask.append(torch.ones(batch_size, 1, device=input_ids.device, dtype=torch.long))
            
            # Process images (single or multi-scale)
            if self.config.training["visual_multi_scale_packing"] and isinstance(images, List):
                # Assumes 'images' is a list of tensors [scale0_batch, scale1_batch, ...]
                # Each img_scale_batch: (B, C, H, W)
                for i, img_scale_batch in enumerate(images):
                    visual_features = self.visual_encoder(img_scale_batch)
                    projected_visual_tokens = self.mlp_projector(visual_features) # (B, N_patches, llm_embed_dim)
                    
                    final_embeddings.append(projected_visual_tokens)
                    final_modality_indicator.append(torch.ones(batch_size, projected_visual_tokens.shape[1],
                                                                device=input_ids.device, dtype=torch.long))
                    final_attention_mask.append(torch.ones(batch_size, projected_visual_tokens.shape[1],
                                                            device=input_ids.device, dtype=torch.long))
                    
                    if i < len(images) - 1: # Add <end_of_scale> between scales
                        eos_scale_embedding = self.llm_token_embeddings(torch.tensor(self.eos_scale_token_id, device=input_ids.device)).unsqueeze(0).repeat(batch_size, 1, 1)
                        final_embeddings.append(eos_scale_embedding)
                        final_modality_indicator.append(torch.zeros(batch_size, 1, device=input_ids.device, dtype=torch.long))
                        final_attention_mask.append(torch.ones(batch_size, 1, device=input_ids.device, dtype=torch.long))
            else: # Single scale image (or multi-scale packing disabled)
                visual_features = self.visual_encoder(images) # images is a single tensor (B, C, H, W)
                projected_visual_tokens = self.mlp_projector(visual_features)

                final_embeddings.append(projected_visual_tokens)
                final_modality_indicator.append(torch.ones(batch_size, projected_visual_tokens.shape[1],
                                                            device=input_ids.device, dtype=torch.long))
                final_attention_mask.append(torch.ones(batch_size, projected_visual_tokens.shape[1],
                                                        device=input_ids.device, dtype=torch.long))
            
            # Insert <image_end> token
            img_end_embedding = self.llm_token_embeddings(torch.tensor(self.img_end_token_id, device=input_ids.device)).unsqueeze(0).repeat(batch_size, 1, 1)
            final_embeddings.append(img_end_embedding)
            final_modality_indicator.append(torch.zeros(batch_size, 1, device=input_ids.device, dtype=torch.long))
            final_attention_mask.append(torch.ones(batch_size, 1, device=input_ids.device, dtype=torch.long))

        # Append text embeddings
        final_embeddings.append(text_embeddings)
        final_modality_indicator.append(text_modality_indicator)
        final_attention_mask.append(attention_mask) # Original text attention mask

        # Concatenate all parts
        combined_embeddings = torch.cat(final_embeddings, dim=1)
        combined_modality_indicator = torch.cat(final_modality_indicator, dim=1)
        combined_attention_mask = torch.cat(final_attention_mask, dim=1)

        # Pad or truncate to max_sequence_length if necessary
        max_len = self.config.training["llm_max_sequence_length"]
        current_len = combined_embeddings.shape[1]

        if current_len > max_len:
            combined_embeddings = combined_embeddings[:, :max_len, :]
            combined_modality_indicator = combined_modality_indicator[:, :max_len]
            combined_attention_mask = combined_attention_mask[:, :max_len]
        elif current_len < max_len:
            pad_len = max_len - current_len
            pad_embedding = torch.zeros(batch_size, pad_len, combined_embeddings.shape[-1], device=input_ids.device, dtype=combined_embeddings.dtype)
            pad_modality = torch.zeros(batch_size, pad_len, device=input_ids.device, dtype=torch.long)
            pad_attn_mask = torch.zeros(batch_size, pad_len, device=input_ids.device, dtype=torch.long)

            combined_embeddings = torch.cat([combined_embeddings, pad_embedding], dim=1)
            combined_modality_indicator = torch.cat([combined_modality_indicator, pad_modality], dim=1)
            combined_attention_mask = torch.cat([combined_attention_mask, pad_attn_mask], dim=1)

        return combined_embeddings, combined_modality_indicator, combined_attention_mask


    def forward(self, images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        
        # Prepare multimodal input
        if input_ids is None:
            raise ValueError("input_ids must be provided for the forward pass.")
        
        # Handle cases where images might be None (e.g. pure text processing for NLP tasks)
        if images is None:
            combined_embeddings = self.llm_token_embeddings(input_ids)
            combined_modality_indicator = torch.zeros(input_ids.shape[0], input_ids.shape[1],
                                                      device=input_ids.device, dtype=torch.long)
            combined_attention_mask = attention_mask
        else:
            combined_embeddings, combined_modality_indicator, combined_attention_mask = \
                self._prepare_multimodal_input(images, input_ids, attention_mask)

        # Pass through LLM blocks
        hidden_states = combined_embeddings
        for block in self.llm_blocks:
            # Combined attention mask is used for padding, causal mask is applied internally for LLM
            # The attention mask passed to the block is for masking padded tokens, so ~combined_attention_mask.bool()
            # where True means masked (attention should not be applied).
            hidden_states = block(hidden_states, combined_modality_indicator, attn_mask=~combined_attention_mask.bool())
        
        # Final LLM head
        hidden_states = self.llm_norm(hidden_states)
        logits = self.llm_head(hidden_states)

        # Loss calculation for Next-Token-Prediction (NTP)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # Labels should be padded with -100 where attention_mask is 0 (or for image tokens)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Ensure only linguistic tokens contribute to loss. Image tokens and special tokens
            # related to images should be masked out in labels, typically with -100.
            # This requires careful construction of `labels` in the data processing step
            # to align with the `combined_embeddings`
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # Use ignore_index for masked labels
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

