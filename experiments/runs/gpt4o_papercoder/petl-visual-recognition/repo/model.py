## model.py
import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from typing import Dict, Union
from peft_modules import PEFTModules  # For applying and integrating PEFT-specific methods


class Model(nn.Module):
    """
    Vision Transformer (ViT-B/16) Backbone with PEFT Modules integration.
    Handles the backbone initialization, PEFT module addition, and forward propagation.
    """

    def __init__(self, backbone_name: str = "ViT-B/16", peft_params: Union[Dict, None] = None):
        """
        Initialize the Model with a pre-trained Vision Transformer backbone and optional PEFT configurations.

        Args:
            backbone_name (str): Backbone model name (default: "ViT-B/16").
            peft_params (Dict | None): Parameters needed for PEFT methods (default: None).
        """
        super(Model, self).__init__()
        self.backbone_name = backbone_name
        self.peft_params = peft_params or {}

        # Initialize the pre-trained ViT-B/16 backbone
        self.backbone = self._initialize_backbone()

        # Placeholder for PEFT modules that may be added later
        self.peft_modules = nn.ModuleDict()

    def _initialize_backbone(self) -> nn.Module:
        """
        Load the Vision Transformer backbone pre-trained on ImageNet-21K.
        """

        try:
            backbone = vit_b_16(pretrained=True)  # Load ViT-B/16 model with pre-trained weights
        except Exception as e:
            raise RuntimeError(f"Error loading the backbone model '{self.backbone_name}': {e}")

        # Freeze all layers by default
        for param in backbone.parameters():
            param.requires_grad = False

        # Modify the classifier head for fine-tuning (reset weights)
        num_features = backbone.heads.head.in_features
        backbone.heads.head = nn.Linear(num_features, 1000)  # As an example, output classes can adapt later
        return backbone

    def add_peft_modules(self, module_type: str, params: Dict) -> None:
        """
        Dynamically attach PEFT modules to the Transformer backbone.

        Args:
            module_type (str): PEFT module type to apply (e.g., "LoRA", "VPT-Shallow").
            params (Dict): Configuration and hyperparameters for the chosen PEFT module.

        Returns:
            None
        """
        if not isinstance(module_type, str) or not isinstance(params, dict):
            raise ValueError("Invalid `module_type` or `params`. Ensure `module_type` is a string and `params` is a dictionary.")

        # Initialize the PEFTModules utility class to dynamically apply PEFT configurations
        peft_util = PEFTModules()

        # Apply the PEFT module based on its type (e.g., "LoRA", "VPT-Shallow")
        if module_type == "VPT-Shallow":
            self.peft_modules["vpt_shallow"] = peft_util.apply_vpt_shallow(params)
        elif module_type == "VPT-Deep":
            self.peft_modules["vpt_deep"] = peft_util.apply_vpt_deep(params)
        elif module_type == "Houl. Adapter":
            self.peft_modules["houl_adapter"] = peft_util.apply_houl_adapter(params)
        elif module_type == "LoRA":
            self.peft_modules["lora"] = peft_util.apply_lora(params)
        elif module_type.startswith("Adapter"):
            self.peft_modules[module_type.lower()] = peft_util.apply_custom_peft(module_type, params)
        else:
            raise NotImplementedError(f"The PEFT module type '{module_type}' has not been implemented yet.")

        # Validate the number of added parameters (≤ 1.5% of ViT-B/16 total parameters, approximately 1.5M max)
        total_backbone_params = sum(p.numel() for p in self.backbone.parameters())
        total_tunable_params = sum(p.numel() for p in self.peft_modules.parameters())

        if total_tunable_params > 0.015 * total_backbone_params:
            raise ValueError(
                f"The added PEFT modules exceed the parameter efficiency constraint of 1.5%. "
                f"Tunable parameters: {total_tunable_params}, Backbone parameters: {total_backbone_params}"
            )

        print(f"Successfully added PEFT module: {module_type} ({total_tunable_params} parameters added).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the model.
        Integrates PEFT components dynamically into the pre-trained ViT backbone.

        Args:
            x (torch.Tensor): Input tensor for the backbone.

        Returns:
            torch.Tensor: Predictions from the modified backbone.
        """
        # Pass through the standard backbone
        x = self.backbone(x)

        # Enable interaction with PEFT modules (if any are configured dynamically)
        for peft_name, peft_module in self.peft_modules.items():
            x = peft_module(x)  # Each PEFT module modifies x based on its specific operation

        return x

