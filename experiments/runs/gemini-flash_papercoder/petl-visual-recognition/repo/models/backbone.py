## models/backbone.py

import torch.nn as nn
from transformers import ViTModel, CLIPVisionModel
from typing import Dict, Any, Optional


class BackboneModel:
    """
    Manages loading and configuration of pre-trained Vision Transformer (ViT)
    backbone models from Hugging Face, including freezing parameters and
    extracting model properties.
    """

    def __init__(self, model_name: str, pretrained_on: str, config: Dict[str, Any]) -> None:
        """
        Initializes the BackboneModel instance.

        Args:
            model_name (str): A general identifier for the Vision Transformer model
                              (e.g., "ViT-B/16"). Used for logging/identification.
            pretrained_on (str): Specifies the pre-training source, either
                                 "imagenet21k" or "clip".
            config (Dict[str, Any]): The complete configuration loaded from config.yaml.

        Raises:
            ValueError: If 'pretrained_on' is an unrecognized value.
        """
        self.model_name: str = model_name
        self.pretrained_on: str = pretrained_on
        self.config: Dict[str, Any] = config

        backbone_config: Dict[str, Any] = self.config['model']['backbone']

        if self.pretrained_on == "imagenet21k":
            self.hf_model_id: str = backbone_config['pretrained_on_imagenet21k_checkpoint']
        elif self.pretrained_on == "clip":
            self.hf_model_id: str = backbone_config['pretrained_on_clip_checkpoint']
        else:
            raise ValueError(f"Unrecognized 'pretrained_on' value: {pretrained_on}. "
                             "Expected 'imagenet21k' or 'clip'.")

        self.freeze_backbone_weights: bool = backbone_config['freeze_backbone_weights']

        self._model: Optional[nn.Module] = None
        self._feature_dim: Optional[int] = None
        self._num_parameters: Optional[int] = None

    def load_model(self) -> nn.Module:
        """
        Loads the specified pre-trained ViT backbone model from Hugging Face
        transformers, configures it for evaluation, and optionally freezes its
        parameters. It also extracts and stores the output feature dimension
        and total parameter count.

        Returns:
            nn.Module: The loaded pre-trained backbone model.

        Raises:
            Exception: If there's an error during model loading from Hugging Face.
        """
        if self._model is not None:
            return self._model  # Model already loaded

        try:
            if self.pretrained_on == "imagenet21k":
                # ViTModel outputs BaseModelOutputWithPooling. We care about the hidden_size for [CLS] token.
                self._model = ViTModel.from_pretrained(self.hf_model_id)
                self._feature_dim = self._model.config.hidden_size
            elif self.pretrained_on == "clip":
                # CLIPVisionModel outputs CLIPVisionModelOutput. We care about the hidden_size.
                self._model = CLIPVisionModel.from_pretrained(self.hf_model_id)
                self._feature_dim = self._model.config.hidden_size
            
            self._model.eval()  # Set to evaluation mode

            # Freeze all parameters if configured
            if self.freeze_backbone_weights:
                for param in self._model.parameters():
                    param.requires_grad = False
            
            # Calculate total parameters for the backbone
            self._num_parameters = sum(p.numel() for p in self._model.parameters())
            
            return self._model

        except Exception as e:
            raise Exception(f"Error loading backbone model '{self.hf_model_id}': {e}") from e

    def get_feature_dim(self) -> int:
        """
        Provides the output feature dimension of the loaded backbone,
        which is required for initializing the classification head.

        Returns:
            int: The feature dimension.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self._feature_dim is None:
            # Attempt to load the model if not already loaded
            if self._model is None:
                self.load_model()
            if self._feature_dim is None:  # Check again in case load_model failed silently
                raise RuntimeError("Backbone model feature dimension not determined. "
                                   "Ensure 'load_model()' was called successfully and check for errors.")
        return self._feature_dim

    def get_num_parameters(self) -> int:
        """
        Provides the total count of parameters within the backbone model.

        Returns:
            int: The total number of parameters.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self._num_parameters is None:
            # Attempt to load the model if not already loaded
            if self._model is None:
                self.load_model()
            if self._num_parameters is None:  # Check again
                raise RuntimeError("Backbone model parameter count not determined. "
                                   "Ensure 'load_model()' was called successfully and check for errors.")
        return self._num_parameters

