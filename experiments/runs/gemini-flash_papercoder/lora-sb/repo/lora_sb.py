import torch
import torch.nn as nn
import transformers
from peft import PeftModel
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutput
from typing import Dict, List, Tuple, Iterator, Optional, Callable

from config import Config, LoRASBConfig # Assuming these are defined in config.py
from dataset_utils import DatasetLoader # For data collator, though not directly used in this file for instance


class LoRASBLinear(nn.Module):
    """
    A LoRA-SB adapted linear layer.
    Replaces a torch.nn.Linear layer and implements W = W_0 + s B R A.
    B and A are fixed, R is trainable.
    """
    def __init__(self, base_layer: nn.Linear, B_init: torch.Tensor, R_init: torch.Tensor, A_init: torch.Tensor, s: float):
        """
        Initializes the LoRA-SB specific components for a given base linear layer.

        Args:
            base_layer (nn.Linear): The original torch.nn.Linear layer.
            B_init (torch.Tensor): Initial B matrix (m x r), fixed.
            R_init (torch.Tensor): Initial R matrix (r x r), trainable.
            A_init (torch.Tensor): Initial A matrix (r x n), fixed.
            s (float): The scaling factor for the update, fixed to 1.0 for LoRA-SB.
        """
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.eval() # Ensure base layer remains in eval mode
        self.base_layer.requires_grad_(False) # Base layer weights should not be trainable

        # Register B and A as non-trainable buffers
        self.register_buffer('B', B_init.to(self.base_layer.weight.device))
        self.register_buffer('A', A_init.to(self.base_layer.weight.device))

        # Register R as a trainable parameter
        self.R = nn.Parameter(R_init.to(self.base_layer.weight.device))

        self.s = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the LoRA-SB forward pass: W = W_0 + s B R A.

        Args:
            x (torch.Tensor): Input tensor to the linear layer.

        Returns:
            torch.Tensor: Output of the LoRA-SB adapted linear layer.
        """
        # Original weight W_0
        W0 = self.base_layer.weight

        # Low-rank update matrix: s * B @ R @ A
        # W0 is (out_features, in_features).
        # B is (out_features, r), R is (r, r), A is (r, in_features).
        # So B@R@A is (out_features, in_features).
        delta_W = self.s * torch.matmul(self.B, torch.matmul(self.R, self.A))

        # Combined effective weight W = W_0 + delta_W
        effective_W = W0 + delta_W

        # Perform the linear transformation: x @ W.T
        # x is typically (batch_size, seq_len, in_features)
        # effective_W.T is (in_features, out_features)
        output = torch.matmul(x, effective_W.T)

        # Add bias if present
        if self.base_layer.bias is not None:
            output += self.base_layer.bias

        return output


class LoRASBModel(PeftModel):
    """
    Wraps a pre-trained model and replaces its target linear layers with LoRASBLinear modules.
    Inherits from peft.PeftModel to leverage PEFT utilities, but customizes layer replacement.
    """
    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        lora_sb_config: LoRASBConfig,
        initialized_lora_matrices: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        """
        Initializes the LoRASBModel.

        Args:
            base_model (transformers.PreTrainedModel): The pre-trained model to adapt.
            lora_sb_config (LoRASBConfig): Configuration for LoRA-SB.
            initialized_lora_matrices (Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
                A dictionary mapping module paths to (B_init, R_init, A_init) tensors.
        """
        # Initialize PeftModel. It will set self.base_model = base_model and self.peft_config = lora_sb_config.
        # This base call might attempt to set up default LoRA adapters. We will override these.
        super().__init__(base_model, lora_sb_config, adapter_name="lora_sb")

        self.lora_sb_config: LoRASBConfig = lora_sb_config
        self.lora_sb_modules: nn.ModuleDict = nn.ModuleDict() # Store references to LoRASBLinear instances
        self.initialized_lora_matrices = initialized_lora_matrices # Keep reference for replacement

        # Perform custom layer replacement after PeftModel's init.
        # This replaces the original Linear layers (or potentially default LoRA layers if PeftModel
        # tried to inject them) with our specific LoRASBLinear layers.
        self._replace_linear_layers()
        
        # Ensure only the R matrices are trainable, overriding default PeftModel behavior
        self.set_trainable_parameters()

    def _replace_linear_layers(self) -> None:
        """
        Iterates through the base_model, finds target linear layers, and replaces them
        with LoRASBLinear instances using the pre-initialized matrices.
        """
        for module_name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if the current module is one of the target modules for LoRA-SB
                # Target modules can be full names or end-suffixes (e.g., 'q_proj')
                is_target_module = False
                for target_key in self.lora_sb_config.target_modules:
                    # Use target_key in name for flexibility (e.g., "q_proj" matching "model.layers.0.self_attn.q_proj")
                    if target_key in module_name: 
                        is_target_module = True
                        break

                if is_target_module and module_name in self.initialized_lora_matrices:
                    B_init, R_init, A_init = self.initialized_lora_matrices[module_name]

                    # Create the LoRASBLinear layer
                    lora_sb_layer = LoRASBLinear(
                        base_layer=module,
                        B_init=B_init,
                        R_init=R_init,
                        A_init=A_init,
                        s=self.lora_sb_config.s
                    )

                    # Replace the original module in the base model's hierarchy
                    # This relies on the module_name format (e.g., "parent.child")
                    parent_name_parts = module_name.split(".")[:-1]
                    child_name = module_name.split(".")[-1]
                    
                    if parent_name_parts:
                        parent_module = self.base_model.get_submodule(".".join(parent_name_parts))
                    else: # Module is a direct child of the base model
                        parent_module = self.base_model

                    setattr(parent_module, child_name, lora_sb_layer)
                    
                    # Store reference to the new LoRASBLinear module
                    self.lora_sb_modules[module_name] = lora_sb_layer
                else:
                    # If it's a linear layer but not a target module, ensure it's not trainable
                    module.requires_grad_(False) # Explicitly freeze non-LoRA-SB linear layers

    def set_trainable_parameters(self, adapter_name: str = "default") -> None:
        """
        Sets the trainable parameters to only the R matrices of LoRASBLinear layers.
        This overrides the default behavior of PeftModel which might set lora_A/lora_B trainable.
        """
        # First, ensure all parameters in the base model are frozen
        self.base_model.requires_grad_(False)
        
        # Then, explicitly set only the R matrices in our LoRASBLinear modules to trainable
        for param in self.get_trainable_parameters():
            param.requires_grad = True

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """
        Returns an iterator over only the trainable parameters of the LoRA-SB model,
        which are the R matrices.
        """
        for lora_sb_layer in self.lora_sb_modules.values():
            yield lora_sb_layer.R

    # The forward method is implicitly handled by `PeftModel`'s `forward`
    # which calls `self.base_model.forward(*args, **kwargs)`.
    # Since we replaced layers in `self.base_model` in-place, this will correctly
    # use our `LoRASBLinear` layers.


class LoRASBInitializer:
    """
    Handles the estimation of ΔW_avg and the subsequent SVD-based initialization
    of LoRA-SB matrices using backward hooks for memory efficiency.
    """
    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: Config,
        lora_sb_config: LoRASBConfig,
        learning_rate_for_avg_grad: float
    ):
        """
        Initializes the LoRASBInitializer.

        Args:
            base_model (transformers.PreTrainedModel): The pre-trained model (W_0) for gradient estimation.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer for data collation.
            config (Config): The global configuration object.
            lora_sb_config (LoRASBConfig): The LoRA-SB specific configuration.
            learning_rate_for_avg_grad (float): The learning rate (eta) for approximating ΔW_avg.
        """
        self._base_model = base_model
        self._tokenizer = tokenizer
        self._config = config
        self._lora_sb_config = lora_sb_config
        self._learning_rate_for_avg_grad = learning_rate_for_avg_grad
        self._device = base_model.device # Assuming base_model is already on the correct device

        # Identify all linear layers that are target modules
        self._target_linear_modules: Dict[str, nn.Linear] = {}
        for name, module in self._base_model.named_modules():
            if isinstance(module, nn.Linear):
                for target_key in self._lora_sb_config.target_modules:
                    if target_key in name:
                        self._target_linear_modules[name] = module
                        break
        
        if not self._target_linear_modules:
            print("Warning: No target linear modules found for LoRA-SB initialization. "
                  "Please check `lora_sb.target_modules` in config.")

    def estimate_avg_gradient(self, dataset_for_init: torch.utils.data.Dataset) -> Dict[str, torch.Tensor]:
        """
        Computes ΔW_avg = - η * sign(sum(∇_W L(W_0, x_i))) for all target modules.
        Uses backward hooks to efficiently collect gradients layer-wise from a single backward pass.

        Args:
            dataset_for_init (torch.utils.data.Dataset): The subset of data to use for gradient approximation.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping module paths to their ΔW_avg tensors.
        """
        self._base_model.eval() # Set model to evaluation mode
        self._base_model.zero_grad() # Clear any existing gradients

        # Temporarily enable requires_grad for target layer weights to compute gradients
        # Store original states to restore them later
        original_requires_grad_states = {}
        for name, module in self._target_linear_modules.items():
            original_requires_grad_states[name] = module.weight.requires_grad
            module.weight.requires_grad = True

        # Dictionary to store accumulated signed gradients
        raw_summed_gradients: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(module.weight, device=self._device)
            for name, module in self._target_linear_modules.items()
        }
        
        # Data collator for batching the initialization dataset
        data_collator = DatasetLoader(self._tokenizer, self._config).get_data_collator(
            self._config.task_name,
            model_type="causal_lm" if self._config.task_name in ["MetaMathQA", "COMMONSENSE170K"] else "sequence_classification"
        )
        init_dataloader = torch.utils.data.DataLoader(
            dataset_for_init,
            batch_size=1, # Process one sample at a time for explicit gradient control
            shuffle=False,
            collate_fn=data_collator
        )

        print(f"Estimating average gradients over {len(dataset_for_init)} samples...")

        for i, batch in enumerate(init_dataloader):
            batch = {k: v.to(self._device) for k, v in batch.items()}
            
            # Forward pass to get loss
            with torch.enable_grad(): # Ensure gradients are computed even if model is in eval mode
                outputs = self._base_model(**batch)
                loss = outputs.loss
            
            # Backward pass
            self._base_model.zero_grad() # Clear gradients from previous sample
            if loss is not None:
                loss.backward()
            
            # Collect and accumulate signed gradients for target modules
            for name, module in self._target_linear_modules.items():
                if module.weight.grad is not None:
                    # Collect the sign of the gradient
                    raw_summed_gradients[name] += torch.sign(module.weight.grad.clone().detach())
                else:
                    # This can happen if a module's weight does not participate in the loss computation
                    # for a given sample, or if its gradient is zero.
                    print(f"Warning: No gradient found for module '{name}' for sample {i+1}. "
                          "This might indicate an issue or an inactive path for this sample.")
            
            # Gradients are already detached and added, and zero_grad() called above will clear all.
            # This implicitly "discards" them for the next sample.

        # Restore original requires_grad states
        for name, module in self._target_linear_modules.items():
            module.weight.requires_grad = original_requires_grad_states[name]

        # Compute ΔW_avg for each layer
        delta_W_avg_per_layer: Dict[str, torch.Tensor] = {}
        for name, summed_grad_tensor in raw_summed_gradients.items():
            # Paper: ΔW_avg = - η * sign(sum(∇_W L(W_0, x_i)))
            # Here, summed_grad_tensor already contains sum(sign(∇_W L)).
            # So, ΔW_avg = - η * summed_grad_tensor
            delta_W_avg_per_layer[name] = -self._learning_rate_for_avg_grad * summed_grad_tensor

        print("Average gradient estimation complete.")
        return delta_W_avg_per_layer

    def initialize_lora_matrices(
        self, avg_gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Performs truncated SVD on ΔW_avg to get B_init, R_init, A_init for each layer.

        Args:
            avg_gradients (Dict[str, torch.Tensor]): Dictionary mapping module paths to their ΔW_avg tensors.

        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                A dictionary mapping layer names to (B_init, R_init, A_init) tensors.
        """
        initialized_matrices: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        target_rank = self._lora_sb_config.r

        print(f"Initializing LoRA-SB matrices using SVD with rank {target_rank}...")

        for layer_name, delta_W_avg in avg_gradients.items():
            # Ensure the matrix is 2D for SVD. Linear layer weights are typically 2D.
            # Convert to float for SVD for numerical stability (often default to float32).
            # The SVD operation is done on the current device (CPU or GPU).
            try:
                U, S, Vh = torch.linalg.svd(delta_W_avg.float(), full_matrices=False)

                # Truncate to target rank r
                # Ensure rank is not greater than min(dimensions)
                actual_rank = min(target_rank, U.shape[1], Vh.shape[0])
                if actual_rank < target_rank:
                    print(f"Warning: Actual rank for layer '{layer_name}' is {actual_rank}, "
                          f"less than target rank {target_rank}. Using actual_rank.")

                U_r = U[:, :actual_rank].to(self._device)
                S_r = torch.diag(S[:actual_rank]).to(self._device)
                Vh_r = Vh[:actual_rank, :].to(self._device)

                # Initialize LoRA-SB matrices
                # B_init = U_r
                # R_init = S_r (since s = 1.0 for LoRA-SB, R_init = (1/s)S_r simplifies to S_r)
                # A_init = Vh_r
                
                # Check for rank mismatch when creating R_init (diag(S))
                # If target_rank is 0, S_r would be empty, need to handle.
                if actual_rank == 0:
                    B_init = torch.empty(delta_W_avg.shape[0], 0, device=self._device)
                    R_init = torch.empty(0, 0, device=self._device)
                    A_init = torch.empty(0, delta_W_avg.shape[1], device=self._device)
                else:
                    B_init = U_r
                    R_init = S_r # Diagonal matrix of singular values
                    A_init = Vh_r

                initialized_matrices[layer_name] = (B_init, R_init, A_init)

            except Exception as e:
                print(f"Error performing SVD for layer '{layer_name}': {e}")
                print(f"delta_W_avg shape: {delta_W_avg.shape}, dtype: {delta_W_avg.dtype}")
                # Re-raise the error as SVD failure is critical for initialization
                raise

        print("LoRA-SB matrix initialization complete.")
        return initialized_matrices

