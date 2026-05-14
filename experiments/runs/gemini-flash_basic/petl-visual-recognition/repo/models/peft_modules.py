import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.vit.modeling_vit import ViTLayer, ViTEncoder, ViTEmbeddings, ViTModelOutput
from typing import Optional, Tuple, Union

# LoRA implementation
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, lora_alpha=1):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.lora_alpha = lora_alpha

        self.linear_layer = linear_layer
        self.linear_layer.weight.requires_grad = False
        if linear_layer.bias is not None:
            self.linear_layer.bias.requires_grad = False

        self.lora_down = nn.Linear(self.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, self.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        original_output = self.linear_layer(x)
        lora_output = self.lora_up(self.lora_down(x)) * (self.lora_alpha / self.rank)
        return original_output + lora_output

def apply_lora_to_linear(model, rank, target_modules=["query", "key", "value", "dense"]):
    # Freeze all parameters of the backbone first
    for param in model.model.parameters():
        param.requires_grad = False

    for name, module in model.model.named_modules():
        if isinstance(module, nn.Linear):
            if any(target_key in name for target_key in target_modules):
                parent_module = model.model
                name_parts = name.split('.')
                for part in name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, name_parts[-1], LoRALinear(module, rank))
    
    # Unfreeze the classification head as it's always fine-tuned
    for param in model.head.parameters():
        param.requires_grad = True
    return model

# BitFit implementation
def apply_bitfit(model):
    # Freeze all parameters of the backbone first
    for param in model.model.parameters():
        param.requires_grad = False

    for name, param in model.model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True
    
    # Unfreeze the classification head
    for param in model.head.parameters():
        param.requires_grad = True
    return model

# LayerNorm tuning implementation
def apply_layernorm_tuning(model):
    # Freeze all parameters of the backbone first
    for param in model.model.parameters():
        param.requires_grad = False

    for name, param in model.model.named_parameters():
        if 'layernorm' in name.lower() or 'ln' in name.lower():
            param.requires_grad = True
    
    # Unfreeze the classification head
    for param in model.head.parameters():
        param.requires_grad = True
    return model

# Adapter implementation (Houl. Adapter)
class Adapter(nn.Module):
    def __init__(self, in_features, bottleneck_dim):
        super().__init__()
        self.down_proj = nn.Linear(in_features, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, in_features)
        # Initialize to identity mapping by default (up_proj weights to zero)
        nn.init.zeros_(self.up_proj.weight)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        return x + self.up_proj(self.activation(self.down_proj(x)))

# Custom ViTLayer that includes Adapters
class AdapterViTLayer(ViTLayer):
    def __init__(self, config, msa_adapter, mlp_adapter):
        super().__init__(config)
        self.msa_adapter = msa_adapter
        self.mlp_adapter = mlp_adapter

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        # Self-Attention block
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states), head_mask, output_attentions=output_attentions
        )
        attention_output = self_attention_outputs[0]
        # Apply MSA Adapter
        attention_output = self.msa_adapter(attention_output)
        hidden_states = hidden_states + attention_output # Residual connection

        # MLP block
        layer_output = self.intermediate(self.layernorm_after(hidden_states))
        layer_output = self.output(layer_output)
        # Apply MLP Adapter
        layer_output = self.mlp_adapter(layer_output)
        hidden_states = hidden_states + layer_output # Residual connection

        outputs = (hidden_states,)

        if output_attentions:
            outputs = outputs + (self_attention_outputs[1],)
        return outputs

def apply_adapter_to_vit(model, bottleneck_dim=64):
    # Freeze all parameters of the backbone first
    for param in model.model.parameters():
        param.requires_grad = False

    num_layers = model.model.config.num_hidden_layers
    adapters_msa = nn.ModuleList([Adapter(model.model.config.hidden_size, bottleneck_dim) for _ in range(num_layers)])
    adapters_mlp = nn.ModuleList([Adapter(model.model.config.hidden_size, bottleneck_dim) for _ in range(num_layers)])

    for i in range(num_layers):
        # Replace the original ViTLayer with our AdapterViTLayer
        model.model.encoder.layer[i] = AdapterViTLayer(
            config=model.model.config,
            msa_adapter=adapters_msa[i],
            mlp_adapter=adapters_mlp[i]
        )
        # Ensure the adapters are trainable
        for param in adapters_msa[i].parameters():
            param.requires_grad = True
        for param in adapters_mlp[i].parameters():
            param.requires_grad = True

    # Unfreeze the classification head
    for param in model.head.parameters():
        param.requires_grad = True
    return model

# VPT-Deep Implementation
class VPTDeepModel(nn.Module):
    def __init__(self, vit_model, prompt_length=10, prompt_dropout=0.0):
        super().__init__()
        self.vit = vit_model
        # Freeze the entire ViT backbone (model.model refers to the transformers.ViTModel)
        for param in self.vit.model.parameters():
            param.requires_grad = False

        self.prompt_length = prompt_length
        self.embedding_dim = self.vit.model.config.hidden_size
        self.prompt_dropout = prompt_dropout
        self.num_layers = self.vit.model.config.num_hidden_layers

        # Create learnable prompts for each transformer layer
        self.prompt_embeddings = nn.ParameterList([
            nn.Parameter(torch.zeros(prompt_length, self.embedding_dim))
            for _ in range(self.num_layers)
        ])
        for i in range(self.num_layers):
            nn.init.uniform_(self.prompt_embeddings[i], -0.1, 0.1) # Initialize prompts

        self.dropout = nn.Dropout(prompt_dropout)

        # Ensure the classification head is trainable
        for param in self.vit.head.parameters():
            param.requires_grad = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.vit.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.vit.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.vit.model.config.use_return_dict

        # Original ViT embeddings
        embedding_output = self.vit.model.embeddings(
            pixel_values, interpolate_pos_encoding=True # Paper uses interpolate_pos_encoding=True
        ) # (batch_size, 1 + num_patches, hidden_size)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        hidden_states = embedding_output # Initial hidden states for the encoder

        # Manually iterate through the encoder layers to inject prompts
        for i, layer_module in enumerate(self.vit.model.encoder.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Get prompts for this specific layer
            batch_size = hidden_states.shape[0]
            current_prompts = self.dropout(self.prompt_embeddings[i]).unsqueeze(0).expand(batch_size, -1, -1)

            # Split CLS token and patch tokens from hidden_states
            cls_token = hidden_states[:, :1, :] # (batch_size, 1, hidden_size)
            patch_tokens = hidden_states[:, 1:, :] # (batch_size, num_patches, hidden_size)

            # Concatenate CLS + prompts + patches for this layer's input
            input_to_layer = torch.cat([cls_token, current_prompts, patch_tokens], dim=1)

            layer_outputs = layer_module(
                input_to_layer,
                head_mask[i] if head_mask is not None else None,
                output_attentions,
            )
            
            # Extract CLS token and patch tokens from the layer's output for the next layer's input
            output_cls_token = layer_outputs[0][:, :1, :]
            output_patch_tokens = layer_outputs[0][:, (1 + self.prompt_length):, :]

            hidden_states = torch.cat([output_cls_token, output_patch_tokens], dim=1)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,) # Final hidden states without prompts

        # Final processing from the original ViT model (layer norm and pooling)
        # Note: self.vit.model.encoder.layernorm expects input of original sequence length (1 + num_patches)
        # So we use the `hidden_states` which have had prompts removed.
        encoded_output = self.vit.model.encoder.layernorm(hidden_states)[:, 0] # CLS token

        # Final classification head
        logits = self.vit.head(encoded_output)

        if not return_dict:
            # For consistency with ViTForImageClassificationOutput, return (logits,)
            # or (logits, hidden_states, attentions)
            return (logits, ) + tuple(v for v in [encoded_output, all_hidden_states, all_self_attentions] if v is not None)
        
        # Return a dictionary similar to ViTForImageClassificationOutput or custom
        # For simplicity, returning a dict that holds the final logits directly.
        return {'logits': logits, 'last_hidden_state': hidden_states, 'hidden_states': all_hidden_states, 'attentions': all_self_attentions}


def get_peft_model(model, peft_method, **kwargs):
    if peft_method == "lora":
        model = apply_lora_to_linear(model, **kwargs)
    elif peft_method == "bitfit":
        model = apply_bitfit(model)
    elif peft_method == "layernorm":
        model = apply_layernorm_tuning(model)
    elif peft_method == "adapter":
        model = apply_adapter_to_vit(model, **kwargs)
    elif peft_method == "vpt_deep":
        model = VPTDeepModel(model, **kwargs)
    else:
        raise ValueError(f"Unknown PEFT method: {peft_method}")
    return model

