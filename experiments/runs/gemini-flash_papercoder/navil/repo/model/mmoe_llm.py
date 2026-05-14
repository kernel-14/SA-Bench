import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.tokenization_utils import PreTrainedTokenizer

# Specific model architectures to patch. These are common for Llama-derived and Qwen-derived models.
# We need to import them to access their forward methods.
try:
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaForCausalLM, LlamaModel
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    logger.warning("Llama models not fully available. MoE injection for Llama-like architectures might fail.")

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP, Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2Model
    _QWEN2_AVAILABLE = True
except ImportError:
    _QWEN2_AVAILABLE = False
    logger.warning("Qwen2 models not fully available. MoE injection for Qwen2-like architectures might fail.")


from config import Config
from model.components import MoELayer, RMSNorm # Assuming MoELayer and RMSNorm are defined in components.py
from utils import logger, get_numerical_precision # Assuming logging and precision utility are in utils.py


# --- Helper function to get LLM module types ---
def get_llm_module_types(llm_model: nn.Module) -> Tuple[Type[nn.Module], Type[nn.Module], Type[nn.Module], Type[nn.Module]]:
    """
    Identifies the Attention, MLP, DecoderLayer, and Model (core) module types
    for the given LLM model instance.
    This is necessary because different Hugging Face models use different class names.

    Args:
        llm_model: An instance of AutoModelForCausalLM.

    Returns:
        A tuple containing (AttentionClass, MLPClass, DecoderLayerClass, ModelClass).

    Raises:
        ValueError: If the LLM type is not recognized.
    """
    model_name = llm_model.__class__.__name__

    if "Llama" in model_name or "InternLM2" in model_name: # InternLM2 is Llama-like
        if not _LLAMA_AVAILABLE:
            raise ImportError(f"Llama (or InternLM2) modules not imported. Cannot process model {model_name}.")
        return LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaModel
    elif "Qwen2" in model_name or "Qwen3" in model_name: # Qwen3 is Qwen2-like
        if not _QWEN2_AVAILABLE:
            raise ImportError(f"Qwen2 (or Qwen3) modules not imported. Cannot process model {model_name}.")
        return Qwen2Attention, Qwen2MLP, Qwen2DecoderLayer, Qwen2Model
    else:
        raise ValueError(
            f"Unsupported LLM type: {model_name}. Only Llama-like (e.g., InternLM2) "
            "and Qwen2-like (e.g., Qwen3) models are explicitly supported for MoE injection. "
            "Please extend `get_llm_module_types` for other architectures."
        )


# --- Custom MoE-enabled Decoder Layer Class ---
class _MoEDecoderLayer(nn.Module):
    """
    A custom Decoder Layer that replaces the original LLM's decoder layers
    to integrate Modality-specific Mixture-of-Experts (MoEs) for attention and FFN.

    CRITICAL ASSUMPTION: The `MoELayer` from `model/components.py` is assumed to be
    capable of:
    1. Being a full drop-in replacement for the LLM's original Attention and MLP modules.
    2. Accepting a `torch.Tensor` for its `modality_id` argument (for token-level routing),
       despite its design signature showing `int`. It is assumed to perform per-token expert selection
       when a tensor is provided.
    3. Internally handling all arguments typically passed to the original LLM's
       Attention (`attention_mask`, `position_ids`, `past_key_value`, `output_attentions`, `use_cache`)
       and MLP modules, returning compatible outputs. This implies its internal complexity
       goes beyond a simple projection and includes full attention/FFN logic.

    This class explicitly adheres to the structure `x' = x + MHA-MMoE(RMSNorm(x))` where
    RMSNorm is handled internally by the MoELayer as per the paper.
    """
    def __init__(self, original_layer: nn.Module, config: Config):
        super().__init__()
        # Copy original configuration for compatibility (e.g., config.hidden_size)
        self.config = original_layer.config
        self.layer_idx = original_layer.layer_idx if hasattr(original_layer, 'layer_idx') else None

        # Replace self_attn and mlp with MoELayer instances from components.py
        self.self_attn = MoELayer(config, is_attention_expert=True)
        self.mlp = MoELayer(config, is_attention_expert=False)

        # The paper's formula `x'_{i,m,l} = x_{i,m,l-1} + MHA-MMoE(RMSNorm(x_{i,m,l-1}))`
        # implies RMSNorm is handled *within* the MHA-MMoE.
        # Thus, original input/post-attention layernorms of the DecoderLayer should be bypassed
        # if MoELayer is a full replacement that applies its own internal RMSNorm.
        # However, to preserve the overall structure of `LlamaDecoderLayer` etc.,
        # we keep these if they exist, but the MoELayer's internal RMSNorm is what's active.
        # The design implies `MoELayer` is the full component.

        # For models like Llama/Qwen2, residual connections and layer norms are typically like:
        # hidden_states = hidden_states + self_attn(self.input_layernorm(hidden_states))
        # hidden_states = hidden_states + mlp(self.post_attention_layernorm(hidden_states))
        # If MoELayer applies RMSNorm internally, then `self.input_layernorm` and `self.post_attention_layernorm`
        # become redundant or must be removed.
        
        # To maintain strict adherence to the *paper's formula* within the MoELayer design,
        # we will assume `MoELayer` (components.py) handles `RMSNorm` itself before its expert logic.
        # The `_MoEDecoderLayer` then simply calls these new `self_attn` and `mlp` modules.

        logger.debug(f"Initialized _MoEDecoderLayer for layer_idx={self.layer_idx}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        modality_ids: Optional[torch.Tensor] = None, # Key addition for token-level MoE
        **kwargs, # Accept any extra args for compatibility with original LLM layer signatures
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Attention part: `x' = x + MHA-MMoE(RMSNorm(x))`
        # MoELayer (components.py) is assumed to apply its own RMSNorm internally.
        # It also implicitly handles QKV projections and attention calculation based on `modality_ids`.

        residual = hidden_states
        # Call the MoE attention expert.
        # ASSUMPTION: MoELayer from components.py (when is_attention_expert=True) returns
        # a tuple compatible with original LLM attention: (attn_output, self_attn_weights, present_key_value).
        # And it handles all args (`attention_mask`, `position_ids`, `past_key_value`, etc.).
        attn_output_tuple = self.self_attn(
            hidden_states=hidden_states,
            modality_id=modality_ids, # Pass the token-level modality IDs tensor
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        attn_output = attn_output_tuple[0] # Assuming first element is transformed hidden states
        self_attn_weights = attn_output_tuple[1] if output_attentions else None
        present_key_value = attn_output_tuple[2] if use_cache else None

        hidden_states = residual + attn_output

        # FFN part: `x = x' + FFN-MMoE(RMSNorm(x'))`
        residual = hidden_states
        # Call the MoE FFN expert.
        # ASSUMPTION: MoELayer from components.py (when is_attention_expert=False) returns
        # only the transformed hidden states, and accepts `modality_ids`.
        mlp_output = self.mlp(
            hidden_states=hidden_states,
            modality_id=modality_ids, # Pass the token-level modality IDs tensor
            **kwargs,
        )
        hidden_states = residual + mlp_output

        # Return format consistent with Hugging Face `LlamaDecoderLayer`
        return (hidden_states, present_key_value) if use_cache else (hidden_states, self_attn_weights)


class MoELLM(nn.Module):
    """
    Multimodal Large Language Model (LLM) enhanced with Modality-specific Mixture-of-Experts (MoEs).
    It loads a pre-trained LLM and injects MoE layers into its transformer blocks.
    Handles multimodal input concatenation and forward pass.
    """

    def __init__(self, llm_name: str, config: Config, tokenizer: PreTrainedTokenizer):
        """
        Initializes the MoE-enhanced LLM.

        Args:
            llm_name: Name or path of the pre-trained LLM to load from Hugging Face.
            config: The global configuration object.
            tokenizer: The tokenizer for the base LLM.
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.llm_name = llm_name

        # Determine numerical precision for loading LLM
        torch_dtype = get_numerical_precision(self.config.get("common.numerical_precision", "bfloat16"))
        
        logger.info(f"Loading base LLM: {llm_name} with dtype {torch_dtype}")
        self.base_llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch_dtype,
            device_map="auto", # Automatically distribute model across available GPUs
            trust_remote_code=True, # Required for some models like Qwen
        )
        logger.info(f"Base LLM {llm_name} loaded.")

        # Add NaViL's special tokens to the tokenizer and resize token embeddings
        self._add_and_resize_special_tokens()

        # Inject MoE layers into the LLM's transformer blocks and patch forward methods
        self._inject_moe_layers()

        # Get device for embeddings (model might be sharded by device_map="auto")
        # Use a heuristic to get a representative device
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_initialized() else torch.device("cuda:0")
        if hasattr(self.base_llm, 'device'): # Try to get device from the model itself
            self.device = self.base_llm.device
        logger.info(f"MoELLM initialized on device: {self.device}")

    def _add_and_resize_special_tokens(self):
        """
        Adds NaViL's special tokens to the tokenizer and resizes the LLM's token embeddings.
        """
        special_tokens_dict = {
            "additional_special_tokens": [
                self.config.begin_of_image,
                self.config.end_of_image,
                self.config.end_of_line,
                self.config.end_of_scale,
            ]
        }
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_tokens} new special tokens to tokenizer.")

        # Resize token embeddings to accommodate new tokens
        self.base_llm.resize_token_embeddings(len(self.tokenizer))
        logger.info(f"Resized LLM token embeddings to {len(self.tokenizer)}.")

        # Store IDs of special tokens for later use
        self.begin_of_image_id = self.tokenizer.convert_tokens_to_ids(self.config.begin_of_image)
        self.end_of_image_id = self.tokenizer.convert_tokens_to_ids(self.config.end_of_image)
        self.end_of_line_id = self.tokenizer.convert_tokens_to_ids(self.config.end_of_line)
        self.end_of_scale_id = self.tokenizer.convert_tokens_to_ids(self.config.end_of_scale)
        
        logger.info(f"Special token IDs: <begin_of_image>={self.begin_of_image_id}, "
                    f"<end_of_image>={self.end_of_image_id}, <end_of_line>={self.end_of_line_id}, "
                    f"<end_of_scale>={self.end_of_scale_id}")

    def _inject_moe_layers(self):
        """
        Replaces the original LLM's decoder layers with custom `_MoEDecoderLayer` instances
        and patches the `forward` methods of the main model and the CausalLM head
        to accept and propagate `modality_ids` (tensor) to the `_MoEDecoderLayer`s.
        """
        # Determine the types of modules for the specific LLM architecture
        _, _, decoder_layer_cls, model_core_cls = get_llm_module_types(self.base_llm)

        logger.info(f"Injecting MoE layers into LLM: DecoderLayer type {decoder_layer_cls.__name__}.")
        
        # Access the main transformer model within the AutoModelForCausalLM
        if hasattr(self.base_llm, 'model') and hasattr(self.base_llm.model, 'layers'):
            llm_layers = self.base_llm.model.layers
            llm_model_core = self.base_llm.model # e.g., LlamaModel, Qwen2Model
        else:
            raise AttributeError("Could not find standard transformer layers (e.g., .model.layers) in the base LLM.")

        num_llm_layers = len(llm_layers)
        
        # Replace original decoder layers with _MoEDecoderLayer instances
        for i in range(num_llm_layers):
            original_layer = llm_layers[i]
            # Instantiate _MoEDecoderLayer with the original layer's properties and global config
            llm_layers[i] = _MoEDecoderLayer(original_layer, self.config)
            logger.debug(f"Replaced original DecoderLayer {i} with _MoEDecoderLayer.")

        # --- Patching `forward` methods to propagate `modality_ids` ---
        # This involves patching the top-level CausalLM forward and the internal Model's forward.

        # 1. Patch `self.base_llm.forward` (e.g., LlamaForCausalLM.forward)
        # This method is the entry point for inference/training calls to the whole LLM.
        _original_llm_for_causal_lm_forward = self.base_llm.forward
        def patched_llm_for_causal_lm_forward(
            self_lm: Any, # refers to self.base_llm
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            modality_ids: Optional[torch.Tensor] = None, # NEW PARAM: propagated from MoELLM.forward
            **kwargs,
        ) -> CausalLMOutputWithPast:
            
            # Call the internal model's forward (e.g., LlamaModel.forward)
            # and explicitly pass the `modality_ids` argument.
            outputs = self_lm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                modality_ids=modality_ids, # Pass the new argument
                **kwargs,
            )
            
            hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
            logits = self_lm.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss(ignore_index=self_lm.config.ignore_index)
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
        # Apply the patch to `self.base_llm.forward`
        self.base_llm.forward = patched_llm_for_causal_lm_forward.__get__(self.base_llm, self.base_llm.__class__)
        logger.info("Patched top-level LLM `forward` to accept `modality_ids`.")


        # 2. Patch `self.base_llm.model.forward` (e.g., LlamaModel.forward or Qwen2Model.forward)
        # This is the internal model that contains the decoder layers.
        _original_llm_model_core_forward = llm_model_core.forward
        def patched_llm_model_core_forward(
            self_core_model: Any, # refers to self.base_llm.model
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            modality_ids: Optional[torch.Tensor] = None, # NEW PARAM: propagated to layers
            **kwargs,
        ) -> Union[Tuple, BaseModelOutputWithPast]:
            
            # Most LLM models have a similar structure here:
            # They compute initial embeddings, then loop through layers.
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self_core_model.embed_tokens(input_ids)

            # Prepare attention_mask for layers (e.g., causal mask generation)
            # This is typically handled by the original model's forward
            # and passed to each layer. Let's assume the original logic is called.
            
            # The original `LlamaModel.forward` has more setup for `attention_mask`, `position_ids`, etc.
            # We are modifying `LlamaModel.forward` to ensure `modality_ids` is passed to each `_MoEDecoderLayer`.
            
            # This logic mimics the actual `LlamaModel.forward` / `Qwen2Model.forward` but adds `modality_ids`.
            
            # Original LlamaModel/Qwen2Model forward handles the embedding lookup,
            # then typically prepares `past_key_values`, `attention_mask` and `position_ids`.
            # Then it loops through `self.layers`.
            
            # We explicitly call the original `forward` but inject `modality_ids`.
            # This relies on the original `forward` having `**kwargs` and `_MoEDecoderLayer` accepting `modality_ids`.
            
            output = _original_llm_model_core_forward(
                self_core_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                modality_ids=modality_ids, # Pass the new argument
                **kwargs,
            )
            return output

        # Apply the patch to `self.base_llm.model.forward`
        llm_model_core.forward = patched_llm_model_core_forward.__get__(llm_model_core, llm_model_core.__class__)
        logger.info("Patched internal LLM model's core `forward` to propagate `modality_ids` to layers.")

        logger.info("MoE layers injection and patching complete.")

    def forward(
        self,
        input_ids: torch.Tensor,
        visual_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs the forward pass for the MoE-enhanced LLM with multimodal input.

        Args:
            input_ids: Token IDs of the textual input. Shape (batch_size, text_sequence_length).
                       Expected to contain special tokens like <begin_of_image>, <end_of_image>.
            visual_embeddings: Visual features from Connector. Shape (batch_size, num_visual_tokens, hidden_dim).
            attention_mask: Attention mask for the input_ids. Shape (batch_size, text_sequence_length).
            labels: Optional target labels for language modeling loss. Shape (batch_size, text_sequence_length).

        Returns:
            A tuple of (logits, loss). Loss is None if labels are not provided.
        """
        batch_size, text_seq_len = input_ids.shape
        num_visual_tokens = visual_embeddings.shape[1]
        llm_hidden_size = self.base_llm.config.hidden_size
        
        # Get LLM's token embeddings
        text_embeds = self.base_llm.get_input_embeddings()(input_ids) # (B, text_seq_len, hidden_dim)

        # Initialize lists for combined embeddings and modality IDs for each batch item
        final_embeddings: List[torch.Tensor] = []
        final_modality_ids: List[torch.Tensor] = [] # 0 for visual, 1 for linguistic

        # Process each item in the batch to construct the multimodal sequence
        for b in range(batch_size):
            current_embeddings_list: List[torch.Tensor] = []
            current_modality_ids_list: List[torch.Tensor] = []

            # Logic to splice visual embeddings into text embeddings based on special tokens.
            # Assumes the `input_ids` from `CustomCollateFn` already has placeholder tokens
            # like `<begin_of_image>` where visual content should be inserted.
            
            # This implementation assumes the input_ids have the placeholders and visual_embeddings correspond
            # to ONE block of visual content per sequence.
            
            in_image_block = False
            for i in range(text_seq_len):
                token_id = input_ids[b, i].item()
                
                if token_id == self.begin_of_image_id:
                    # Append <begin_of_image> token embedding
                    current_embeddings_list.append(text_embeds[b, i].unsqueeze(0))
                    current_modality_ids_list.append(torch.tensor([0], dtype=torch.long, device=self.device)) # Visual control token
                    
                    # Append actual visual embeddings
                    current_embeddings_list.append(visual_embeddings[b])
                    current_modality_ids_list.append(torch.full((num_visual_tokens,), 0, dtype=torch.long, device=self.device)) # Visual tokens

                    in_image_block = True
                elif token_id == self.end_of_image_id and in_image_block:
                    # Append <end_of_image> token embedding
                    current_embeddings_list.append(text_embeds[b, i].unsqueeze(0))
                    current_modality_ids_list.append(torch.tensor([0], dtype=torch.long, device=self.device)) # Visual control token
                    in_image_block = False # Reset flag
                elif token_id == self.end_of_line_id or token_id == self.end_of_scale_id:
                    # Append other visual control tokens
                    current_embeddings_list.append(text_embeds[b, i].unsqueeze(0))
                    current_modality_ids_list.append(torch.tensor([0], dtype=torch.long, device=self.device)) # Visual control token
                else:
                    # Append regular text token embedding
                    current_embeddings_list.append(text_embeds[b, i].unsqueeze(0))
                    current_modality_ids_list.append(torch.tensor([1], dtype=torch.long, device=self.device)) # Linguistic token
            
            # Concatenate all parts for the current batch item
            final_embeddings.append(torch.cat(current_embeddings_list, dim=0))
            final_modality_ids.append(torch.cat(current_modality_ids_list, dim=0))
        
        # Pad combined sequences to the maximum length for the batch, or global max_seq_len
        max_batch_seq_len = max(emb.shape[0] for emb in final_embeddings)
        model_max_seq_len = self.config.common.llm_max_sequence_length
        actual_max_seq_len = min(max_batch_seq_len, model_max_seq_len) # Ensure not to exceed model's context window

        padded_embeddings = torch.zeros(batch_size, actual_max_seq_len, llm_hidden_size,
                                        dtype=text_embeds.dtype, device=self.device)
        padded_modality_ids = torch.full((batch_size, actual_max_seq_len), 1, dtype=torch.long, device=self.device) # Default to linguistic

        # Create combined attention mask and labels
        combined_attention_mask = torch.zeros(batch_size, actual_max_seq_len, dtype=torch.long, device=self.device)
        ignore_index = self.config.get("labels_ignore_index", -100) # Default ignore_index for loss calculation
        combined_labels = torch.full((batch_size, actual_max_seq_len), ignore_index,
                                     dtype=torch.long, device=self.device) if labels is not None else None

        for b in range(batch_size):
            seq_len_b = final_embeddings[b].shape[0]
            current_seq_len = min(seq_len_b, actual_max_seq_len)

            padded_embeddings[b, :current_seq_len] = final_embeddings[b][:current_seq_len]
            padded_modality_ids[b, :current_seq_len] = final_modality_ids[b][:current_seq_len]
            combined_attention_mask[b, :current_seq_len] = 1 # Mark active tokens
            
            if labels is not None:
                # Align labels: mark visual/special tokens as ignore_index, map original text labels.
                
                # This label alignment needs to be done carefully based on how `final_embeddings` was constructed.
                # `current_embeddings_list` has entries from `text_embeds[b, i].unsqueeze(0)` and `visual_embeddings[b]`.
                
                combined_labels_for_b = []
                original_labels_idx = 0
                for i in range(text_seq_len):
                    token_id = input_ids[b, i].item()
                    
                    if token_id == self.begin_of_image_id:
                        combined_labels_for_b.append(ignore_index) # <begin_of_image> token
                        combined_labels_for_b.extend([ignore_index] * num_visual_tokens) # visual embeddings
                    elif token_id == self.end_of_image_id or token_id == self.end_of_line_id or token_id == self.end_of_scale_id:
                        combined_labels_for_b.append(ignore_index) # Visual control token
                    else:
                        # Linguistic token, its label comes from original `labels`
                        if original_labels_idx < labels.shape[1]:
                            combined_labels_for_b.append(labels[b, original_labels_idx].item())
                        else: # Should not happen if labels are aligned with input_ids length
                            combined_labels_for_b.append(ignore_index)
                        original_labels_idx += 1
                
                # Convert to tensor and pad
                combined_labels_for_b_tensor = torch.tensor(combined_labels_for_b, dtype=torch.long, device=self.device)
                if combined_labels_for_b_tensor.shape[0] > actual_max_seq_len:
                    combined_labels[b, :actual_max_seq_len] = combined_labels_for_b_tensor[:actual_max_seq_len]
                else:
                    combined_labels[b, :combined_labels_for_b_tensor.shape[0]] = combined_labels_for_b_tensor


        # Finally, make the call to the patched base LLM's forward method
        # This will now pass `modality_ids` as an extra argument, which the patched
        # `_MoEDecoderLayer.forward` will then use.
        
        output = self.base_llm.forward(
            inputs_embeds=padded_embeddings,
            attention_mask=combined_attention_mask,
            labels=combined_labels, # This is an optional argument
            modality_ids=padded_modality_ids, # Pass the new argument
            output_attentions=False, # Set to True if attention weights are needed
            output_hidden_states=False, # Set to True if hidden states are needed
            return_dict=True,
        )
        
        return output.logits, output.loss

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """
        Returns the tokenizer associated with the LLM.
        """
        return self.tokenizer
