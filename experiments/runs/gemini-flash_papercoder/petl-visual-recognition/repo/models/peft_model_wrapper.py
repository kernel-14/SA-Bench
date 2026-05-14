## models/peft_model_wrapper.py
import os
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Iterator, Dict, Any, Type, List, Optional
import warnings

# Import specific PEFT module classes
from models.peft_modules import (
    BasePEFTModule,
    VPTShallow, VPTDeep,
    HoulsbyAdapter, PfeifferAdapter, AdaptFormer, ConvPass, RepAdapter,
    BitFit, LayerNormTune, DiffFit, SSF,
    LoRA, # FacTTT, FacTTK  # FacT not implemented yet, placeholder
)
from models.backbone import BackboneModel
from models.classification_head import ClassificationHead
from transformers import CLIPTextModel, CLIPTokenizer # For CLIP head initialization
import re # For regex in DropPath handling

# Placeholder for ImageNet class names. In a real scenario, this would come from the dataset loader.
# For simplicity, we use a dummy list for compilation.
IMAGENET_CLASS_NAMES = [f"class_{i}" for i in range(1000)] # Dummy list for compilation, to be replaced by actual names from RobustnessLoader


class PEFTModelWrapper(nn.Module):
    """
    Orchestrates the assembly and management of the entire model, encompassing
    the backbone, the PEFT module, and the classification head. It also provides
    utilities for parameter management, saving/loading, and applying Weight-space
    Ensembles (WiSE).
    """

    def __init__(self,
                 backbone_config: Dict[str, Any],
                 peft_config: Dict[str, Any],
                 head_config: Dict[str, Any],
                 num_classes: int,
                 experiment_type: str,
                 pretrained_model_path: Optional[str] = None) -> None:
        """
        Initializes the model wrapper, stores configuration parameters, and triggers
        the construction of the model components.

        Args:
            backbone_config (Dict[str, Any]): Configuration for the backbone model.
            peft_config (Dict[str, Any]): Configuration for the PEFT method, including
                                         the method type and its specific hyperparameters.
            head_config (Dict[str, Any]): Configuration for the classification head.
            num_classes (int): The number of output classes for the classification head.
            experiment_type (str): Type of experiment ('low_shot', 'many_shot', 'robustness').
            pretrained_model_path (Optional[str]): Path to a pre-trained model checkpoint to load.
        """
        super().__init__()
        self.backbone_config = backbone_config
        self.peft_config = peft_config
        self.head_config = head_config
        self.num_classes = num_classes
        self.experiment_type = experiment_type
        self.pretrained_model_path = pretrained_model_path

        self.backbone: Optional[nn.Module] = None
        self.peft_module: Optional[BasePEFTModule] = None
        self.head: Optional[ClassificationHead] = None
        self._initial_state_dict: Optional[Dict[str, Any]] = None

        self.build_model()

        # Capture the model's initial state dictionary immediately after build_model() is complete.
        # This will serve as the "pre-fine-tuned" state for WiSE interpolation later.
        self._initial_state_dict = deepcopy(self.state_dict())
        
        if self.pretrained_model_path:
            self.load_pretrained(self.pretrained_model_path)

    def build_model(self) -> None:
        """
        Constructs the full model: loads the pre-trained backbone, applies the chosen
        PEFT method (or baseline), and initializes the classification head.
        """
        # --- 1. Load Backbone ---
        model_name = self.backbone_config.get('type', 'vit') # Assuming 'vit' as default if not specified
        pretrained_on = self.backbone_config.get('pretrained_on')
        
        if pretrained_on is None:
            raise ValueError("Backbone 'pretrained_on' must be specified in backbone_config.")

        self.backbone_model_helper = BackboneModel(model_name, pretrained_on, self.backbone_config)
        self.backbone = self.backbone_model_helper.load_model()
        backbone_feature_dim = self.backbone_model_helper.get_feature_dim()
        total_backbone_params = self.backbone_model_helper.get_num_parameters()

        # --- 2. Initialize Classification Head ---
        self.head = ClassificationHead(backbone_feature_dim, self.num_classes, self.head_config)
        
        # CLIP-specific Head Initialization for robustness experiments
        if self.experiment_type == "robustness" and pretrained_on == "clip":
            # Assuming 'openai/clip-vit-base-patch16' for text encoder too if not specified
            clip_text_model_id = self.backbone_config.get('pretrained_on_clip_checkpoint', 'openai/clip-vit-base-patch16')
            
            # TODO: In a complete system, actual ImageNet class names would be passed from RobustnessLoader.
            # Using a placeholder list here.
            class_names_for_clip_head = IMAGENET_CLASS_NAMES 

            # Create an instance of CLIPTextModel and CLIPTokenizer
            clip_text_encoder = CLIPTextModel.from_pretrained(clip_text_model_id)
            clip_tokenizer = CLIPTokenizer.from_pretrained(clip_text_model_id)
            
            # Determine device for text encoder processing (will be moved to proper device later)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.head.initialize_with_text_embeddings(clip_text_encoder, clip_tokenizer, class_names_for_clip_head, device)
            
            # Dispose of text encoder and tokenizer after use to free memory if they are not needed further
            del clip_text_encoder
            del clip_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- 3. Apply PEFT / Baselines ---
        peft_method_name = self.peft_config.get('method')
        if peft_method_name is None:
            raise ValueError("PEFT method 'method' must be specified in peft_config.")

        if peft_method_name == "linear_probing":
            # Backbone parameters are already frozen by BackboneModel.load_model()
            # Only head parameters are trainable by default.
            pass
        elif peft_method_name == "full_ft":
            # Recursively set requires_grad=True for all parameters in self.backbone
            for param in self.backbone.parameters():
                param.requires_grad = True
            # Head parameters are already trainable (initialized with requires_grad=True).
        else:
            # For specific PEFT methods, instantiate the correct module
            peft_class_map: Dict[str, Type[BasePEFTModule]] = {
                "VPT-Shallow": VPTShallow,
                "VPT-Deep": VPTDeep,
                "HoulsbyAdapter": HoulsbyAdapter,
                "PfeifferAdapter": PfeifferAdapter,
                "AdaptFormer": AdaptFormer,
                "Convpass": ConvPass,
                "RepAdapter": RepAdapter,
                "BitFit": BitFit,
                "LayerNorm": LayerNormTune, # Renamed LayerNorm to LayerNormTune for clarity matching class name
                "DiffFit": DiffFit,
                "SSF": SSF,
                "LoRA": LoRA,
                # "FacTTT": FacTTT, # Uncomment when FacT is implemented
                # "FacTTK": FacTTK, # Uncomment when FacT is implemented
            }

            peft_class = peft_class_map.get(peft_method_name)
            if peft_class is None:
                raise ValueError(f"Unknown PEFT method: {peft_method_name}")
            
            # Pass relevant peft-specific hyperparameters.
            # `peft_hyperparameter_search_key` from config indicates which block in
            # `peft_hyperparameter_search_spaces` to use.
            method_specific_hparams_key = self.peft_config.get('peft_hyperparameter_search_key', peft_method_name)
            
            # Default to an empty dict if not found, then update with specific search space values.
            # The PEFT module itself will pick parameters from this dict.
            peft_hparams_from_config = deepcopy(self.peft_config.get('peft_fixed_hyperparameters', {}))
            if method_specific_hparams_key in self.peft_config.get('peft_hyperparameter_search_spaces', {}):
                peft_hparams_from_config.update(
                    self.peft_config['peft_hyperparameter_search_spaces'][method_specific_hparams_key]
                )

            # Initialize PEFT module, passing a combined config that includes relevant HPs
            peft_module_combined_config = deepcopy(self.peft_config) # Copy original peft_config
            peft_module_combined_config['peft_hyperparameters'] = peft_hparams_from_config # Add/overwrite with resolved HPs

            self.peft_module = peft_class(self.backbone, peft_module_combined_config)
            self.peft_module.apply_to_backbone()

            # Parameter Cap Check (only relevant for PEFT methods that add params or unfroze select params)
            current_peft_trainable_params = sum(p.numel() for p in self.peft_module.get_trainable_parameters())
            cap_ratio = self.peft_config.get('cap_peft_params_ratio', 0.015) # Default 1.5% from config.yaml
            max_peft_params = cap_ratio * total_backbone_params

            if current_peft_trainable_params > max_peft_params:
                warnings.warn(f"Trainable PEFT parameters ({current_peft_trainable_params:.0f}) for method {peft_method_name} "
                              f"exceeds the cap of {cap_ratio*100:.2f}% of backbone parameters ({max_peft_params:.0f}).")
        
        # --- 4. Drop Path Rate ---
        # The paper notes the importance of drop path rate.
        # This parameter is often specified in the experiment-specific configuration.
        drop_path_rate = self.peft_config.get('drop_path_rate', 0.0) # Default to 0.0 if not specified
        
        if drop_path_rate > 0:
            # Find and set drop_prob for any ViTDropPath modules in the backbone
            found_droppath = False
            for module in self.backbone.modules():
                # Check if the module's class name contains 'DropPath' (e.g., ViTDropPath)
                if 'DropPath' in str(type(module)): 
                    if hasattr(module, 'drop_prob'):
                        module.drop_prob = drop_path_rate
                        found_droppath = True
                        # print(f"Set drop_prob for {type(module)} to {drop_path_rate}") # For debugging
                    # else:
                        # warnings.warn(f"Found a DropPath-like module ({type(module)}) but it doesn't have a 'drop_prob' attribute. "
                        #               "Drop path rate might not be applied correctly.")
            if not found_droppath and drop_path_rate > 0:
                warnings.warn(f"Drop path rate {drop_path_rate} was specified but no 'DropPath' modules were found in the backbone to apply it to.")
        
        # Ensure all components are in evaluation mode by default, training will set to train()
        self.eval()

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """
        Gathers and returns an iterator over all parameters in the model that
        have `requires_grad=True`.
        """
        # Using a list and then converting to iterator to allow for potential sorting/deduplication
        # if any complex interaction caused duplicates, though `named_parameters` usually handles it.
        trainable_params_list = [] 

        # Parameters from the classification head are generally always trainable
        for p in self.head.parameters():
            if p.requires_grad: # Check requires_grad in case it was frozen by some custom logic
                trainable_params_list.append(p)

        # Parameters specific to the PEFT module
        if self.peft_module:
            for p in self.peft_module.get_trainable_parameters():
                if p.requires_grad:
                    trainable_params_list.append(p)
        
        # If full fine-tuning, all backbone parameters are trainable
        peft_method_name = self.peft_config.get('method')
        if peft_method_name == "full_ft":
            for p in self.backbone.parameters():
                if p.requires_grad:
                    trainable_params_list.append(p)

        # Return an iterator over the collected trainable parameters
        return iter(trainable_params_list)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Defines the complete forward pass, from input image to classification logits.

        Args:
            pixel_values (torch.Tensor): The input image tensor(s) with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The raw logits for each class, with shape (batch_size, num_classes).
        """
        # The backbone (ViTModel or CLIPVisionModel) typically expects 'pixel_values' as input
        backbone_output = self.backbone(pixel_values=pixel_values)
        
        # Extract the class token's embedding from the backbone's output.
        # For HuggingFace ViT models, it's typically the first token of the last_hidden_state.
        # For CLIPVisionModel, it uses 'pooler_output' which is the pooled last hidden state (CLS token output).
        if hasattr(backbone_output, 'pooler_output') and backbone_output.pooler_output is not None:
            backbone_features = backbone_output.pooler_output
        elif hasattr(backbone_output, 'last_hidden_state'):
            # For standard ViT, the [CLS] token is the first token in the sequence
            backbone_features = backbone_output.last_hidden_state[:, 0] 
        else:
            raise AttributeError("Backbone output does not contain 'pooler_output' or 'last_hidden_state' "
                                 "as expected for feature extraction.")

        logits = self.head(backbone_features)
        return logits

    def save_pretrained(self, save_directory: str) -> None:
        """
        Saves the model's current state dictionary to a specified file path.

        Args:
            save_directory (str): The directory where the model checkpoint will be saved.
        """
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        warnings.warn(f"Model saved to {model_path}")

    def load_pretrained(self, load_path: str) -> None:
        """
        Loads a previously saved model state dictionary into the current model.

        Args:
            load_path (str): The file path to the model checkpoint.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {load_path}")
        
        # Determine device dynamically for loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded_state_dict = torch.load(load_path, map_location=device)
        
        # Ensure strict loading, but catch potential mismatches in case of model changes
        try:
            self.load_state_dict(loaded_state_dict, strict=True)
        except RuntimeError as e:
            warnings.warn(f"Strict state_dict loading failed: {e}. Attempting non-strict loading. "
                          "This might indicate a mismatch in model architecture.")
            self.load_state_dict(loaded_state_dict, strict=False)

        warnings.warn(f"Model loaded from {load_path}")

    def get_pre_fine_tuned_state_dict(self) -> Dict[str, Any]:
        """
        Returns a deep copy of the model's state dictionary as it was right after
        `build_model()` but before any fine-tuning. This is the base model for WiSE.
        """
        if self._initial_state_dict is None:
            raise RuntimeError("Initial state dictionary was not captured. Ensure build_model() was called before calling WiSE methods.")
        return deepcopy(self._initial_state_dict)

    def apply_wise_interpolation(self, fine_tuned_state_dict: Dict[str, Any], alpha: float) -> None:
        """
        Applies Weight-space Ensembles (WiSE) interpolation to the model, merging
        a pre-fine-tuned state with a fine-tuned state using `alpha`.

        Args:
            fine_tuned_state_dict (Dict[str, Any]): The state dictionary of the
                                                      fully fine-tuned model.
            alpha (float): The mixing coefficient for interpolation (0.0 for pre-trained,
                           1.0 for fine-tuned).
        """
        # First, ensure the current model reflects the fine-tuned state.
        # This is important so `self.named_parameters()` retrieves the fine-tuned values.
        self.load_state_dict(fine_tuned_state_dict, strict=True)

        pre_ft_state_dict = self.get_pre_fine_tuned_state_dict() # Get a deep copy of the initial state

        peft_method_name = self.peft_config.get('method')

        # List of PEFT methods that directly modify backbone weights or specific parameters (like biases/LN)
        # For these, we directly interpolate the weight values.
        direct_interpolation_methods = ["linear_probing", "full_ft", "BitFit", "LayerNorm", "DiffFit"] # SSF modifies features, not weights, but its parameters are themselves weights.

        # SSF is a hybrid: its internal parameters (w, b) are interpolated directly,
        # but its effect on the forward pass is a scaling/shifting.
        # The paper's formulation for WiSE on Adapter/LoRA refers to scaling their *impact* with alpha.
        # For SSF, interpolating its `w` and `b` parameters directly is more consistent with how its parameters are learned.
        # However, the paper's specific phrasing for Adapter-based and Efficient Selective "scale the adapter modules with alpha"
        # and "scale the additive residuals with alpha" implies modifying the *contribution* during forward pass.
        # Let's align SSF with direct interpolation for its parameters `w` and `b`, as they are akin to other directly tuned parameters.

        if peft_method_name in direct_interpolation_methods or peft_method_name == "SSF":
            # For these methods, directly interpolate all trainable parameters (backbone and head)
            for p_name, p_param in self.named_parameters():
                if p_param.requires_grad: # Only interpolate parameters that were trained during fine-tuning
                    if p_name not in pre_ft_state_dict:
                        warnings.warn(f"Parameter '{p_name}' not found in pre-fine-tuned state dict. Skipping direct WiSE interpolation.")
                        continue
                    W_pre_ft = pre_ft_state_dict[p_name]
                    W_ft = p_param.data
                    p_param.data.copy_((1 - alpha) * W_pre_ft + alpha * W_ft)
        else:
            # For additive PEFT methods (VPT, Adapter-based, LoRA, FacT),
            # the paper suggests scaling their *impact* by alpha.
            # We interpolate the ClassificationHead parameters directly, as it's a standard linear layer.
            # Then, we call `configure_wise_scaling` on the PEFT module itself to scale its additive contribution.

            # Interpolate ClassificationHead parameters directly
            for p_name, p_param in self.head.named_parameters():
                if p_param.requires_grad: # Head parameters are always trainable
                    if p_name not in pre_ft_state_dict:
                        warnings.warn(f"Head parameter '{p_name}' not found in pre-fine-tuned state dict. Skipping direct WiSE interpolation.")
                        continue
                    W_pre_ft_head = pre_ft_state_dict[p_name]
                    W_ft_head = p_param.data
                    p_param.data.copy_((1 - alpha) * W_pre_ft_head + alpha * W_ft_head)
            
            # Configure scaling for the PEFT module if it exists and supports WiSE scaling
            if self.peft_module and hasattr(self.peft_module, 'configure_wise_scaling'):
                self.peft_module.configure_wise_scaling(alpha)
            elif self.peft_module:
                warnings.warn(f"PEFT method '{peft_method_name}' is not in direct interpolation list, "
                              "but its module does not have a 'configure_wise_scaling' method. "
                              "WiSE might not be applied as intended for this PEFT method.")
            else:
                 warnings.warn(f"PEFT method '{peft_method_name}' is selected, but no peft_module was instantiated. "
                               "This might be an issue if PEFT-specific WiSE scaling was expected.")
