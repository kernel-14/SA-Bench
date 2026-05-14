
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from lora_sb_layers import LoRASBLayer
from config import Config

class LoRASBModel(nn.Module):
    def __init__(self, base_model, config: Config, task_type: str = "causal_lm"):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.lora_sb_layers = nn.ModuleDict()

        if task_type == "causal_lm":
            self._target_modules = config.target_modules_llm
        elif task_type == "sequence_classification":
            self._target_modules = config.target_modules_roberta
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        self._patch_model()

    def _patch_model(self):
        for name, module in self.base_model.named_modules():
            if any(target_module_key in name for target_module_key in self._target_modules):
                if isinstance(module, (nn.Linear, nn.Conv2d)): # Assuming LoRA applies to Linear and Conv2d, common in Transformers
                    print(f"Patching module: {name} of type {type(module)}")
                    # Store original module to avoid issues if a module is processed multiple times
                    original_module = module

                    # Create a new LoRASBLayer
                    lora_sb_layer = LoRASBLayer(
                        original_module,
                        rank=self.config.rank,
                        scaling_factor=self.config.scaling_factor
                    )
                    
                    # Replace the module in the base model
                    self._replace_module(self.base_model, name, lora_sb_layer)
                    self.lora_sb_layers[name] = lora_sb_layer

        # Ensure original_weight_data in LoRASBLayer does not require grad
        for name, param in self.base_model.named_parameters():
            if "original_weight_data" in name: # This assumes original_weight_data might mistakenly be registered as a parameter
                param.requires_grad = False

    def _replace_module(self, root_module, module_path, new_module):
        parts = module_path.split('.')
        parent_module = root_module
        for part in parts[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, parts[-1], new_module)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def get_trainable_parameters(self):
        # Only the R matrix in LoRASBLayer is trainable
        trainable_params = []
        for name, param in self.named_parameters():
            if "lora_sb_layers" in name and "R" in name:
                trainable_params.append(param)
        return trainable_params

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

# Factory function to load base model
def load_base_model(model_name: str, task_type: str = "causal_lm"):
    if task_type == "causal_lm":
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    elif task_type == "sequence_classification":
        return AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16, num_labels=2) # num_labels example for GLUE
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

