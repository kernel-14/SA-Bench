## models/classification_head.py

import torch
import torch.nn as nn
from typing import Any, Dict, List

# CLIP specific imports from transformers
# If using transformers library, CLIPTextModel and CLIPTokenizer are typically used.
# The `tokenizer` could be an instance of CLIPTokenizer.
# The `text_encoder` could be an instance of CLIPTextModel.
# The prompt templates are often from the original OpenAI CLIP repo.
# A representative set of prompt templates for ImageNet classes are listed below,
# derived from common CLIP usage and similar to those found in OpenAI's CLIP repo.
# For the full list of 80 templates, one would refer to
# https://github.com/openai/CLIP/blob/main/clip/clip.py -> _get_templates()
CLIP_IMAGENET_TEMPLATES: List[str] = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
    "a photo of a {} in the wild.",
    "a photo of a {} on the road.",
    "a photo of a {} on the grass.",
    "a photo of a {} in the snow.",
    "a photo of a {} in the city.",
    "a photo of a {} in the forest.",
    "a photo of a {} in the jungle.",
    "a photo of a {} in the desert.",
    "a photo of a {} on the beach.",
    "a photo of a {} at the beach.",
    "a photo of a {} by the sea.",
    "a photo of a {} near the sea.",
    "a photo of a {} near the ocean.",
    "a photo of a {} in the garden.",
    "a photo of a {} in the park.",
    "a photo of a {} at the park.",
    "a photo of a {} in the field.",
    "a photo of a {} on the farm.",
    "a photo of a {} on the mountain.",
    "a photo of a {} in the mountains.",
    "a photo of a {} in the air.",
    "a photo of a {} in the sky.",
    "a photo of a {} on the water.",
    "a photo of a {} in the water.",
    "a photo of a {} in the river.",
    "a photo of a {} in the lake.",
    "a photo of a {} in the ocean.",
    "a photo of a {} in the pool.",
    "a photo of a {} in the house.",
    "a photo of a {} in the home.",
    "a photo of a {} in the building.",
    "a photo of a {} in the office.",
    "a photo of a {} in the studio.",
    "a photo of a {} in the room.",
    "a photo of a {} in the kitchen.",
    "a photo of a {} in the bathroom.",
    "a photo of a {} in the bedroom.",
    "a photo of a {} in the living room.",
    "a photo of a {} in the dining room.",
    "a photo of a {} on the table.",
    "a photo of a {} on the chair.",
    "a photo of a {} on the bed.",
    "a photo of a {} on the sofa.",
    "a photo of a {} on the couch.",
    "a photo of a {} on the floor.",
    "a photo of a {} on the wall.",
    "a photo of a {} on the ceiling.",
    "a photo of a {} on the roof.",
    "a photo of a {} on the window.",
    "a photo of a {} on the door.",
    "a photo of a {} on the shelf.",
    "a photo of a {} on the counter.",
    "a photo of a {} on the desk.",
    "a photo of a {} on the car.",
    "a photo of a {} in the car.",
    "a photo of a {} in the truck.",
    "a photo of a {} in the bus.",
    "a photo of a {} in the train.",
    "a photo of a {} in the plane.",
    "a photo of a {} in the boat.",
    "a photo of a {} in the ship.",
    "a photo of a {} on the train.",
    "a photo of a {} on the plane.",
    "a photo of a {} on the boat.",
    "a photo of a {} on the ship.",
    "a photo of a {} with a person.",
    "a photo of a {} with no person.",
    "a photo of a {} with a dog.",
    "a photo of a {} with a cat.",
    "a photo of a {} with a child.",
    "a photo of a {} with a man.",
    "a photo of a {} with a woman.",
    "a photo of a {} with a girl.",
    "a photo of a {} with a boy.",
    "a photo of a {} with a flower.",
    "a photo of a {} with a tree.",
]


class ClassificationHead(nn.Module):
    """
    A linear classification head that projects the extracted features from the
    backbone into class logits. Supports standard random initialization and
    specialized initialization using text embeddings for CLIP models.
    """

    def __init__(self, input_dim: int, num_classes: int, config: Dict[str, Any]) -> None:
        """
        Initializes the classification layer.

        Args:
            input_dim (int): The dimensionality of the feature vector output by the backbone.
            num_classes (int): The total number of classes in the downstream task dataset.
            config (Dict[str, Any]): A dictionary containing configuration parameters for the head.
        """
        super().__init__()
        self.input_dim: int = input_dim
        self.num_classes: int = num_classes
        self.config: Dict[str, Any] = config

        # The classification head is a simple linear layer
        self.fc = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the classification head to compute class logits.

        Args:
            features (torch.Tensor): The feature representation extracted from the backbone
                                     (e.g., the [CLS] token's embedding), with shape
                                     (batch_size, input_dim).

        Returns:
            torch.Tensor: The raw logits for each class, with shape (batch_size, num_classes).
        """
        return self.fc(features)

    @torch.no_grad() # No gradients needed for initialization
    def initialize_with_text_embeddings(self, text_encoder: Any, tokenizer: Any,
                                        class_names: List[str], device: str) -> None:
        """
        Initializes the classification head's weights using text embeddings of class names
        and a prompt ensemble, suitable for CLIP models in robustness studies.
        The paper references "80 prompts provided by CLIP at https://github.com/openai/CLIP".

        Args:
            text_encoder (Any): An instance of the CLIP text encoder (e.g., CLIPTextModel).
            tokenizer (Any): An instance of the CLIP tokenizer (e.g., CLIPTokenizer).
            class_names (List[str]): A list of strings, where each string is a class label.
            device (str): The device ('cpu' or 'cuda') on which to perform computations.
        
        Raises:
            ValueError: If the input_dim does not match the text encoder's output dimension.
        """
        if self.input_dim != text_encoder.config.hidden_size:
            raise ValueError(
                f"Input dimension of ClassificationHead ({self.input_dim}) "
                f"does not match CLIP text encoder's hidden size ({text_encoder.config.hidden_size})."
                "Ensure backbone's feature dimension matches text encoder's output."
            )
        
        text_encoder.eval()
        text_encoder.to(device)

        averaged_class_embeddings: List[torch.Tensor] = []

        for class_name in class_names:
            class_embeddings_sum = torch.zeros(self.input_dim, device=device)
            num_prompts_processed = 0

            for template in CLIP_IMAGENET_TEMPLATES:
                prompt = template.format(class_name)
                
                # Tokenize the prompt
                # max_length should typically be 77 for CLIP
                inputs = tokenizer(
                    prompt, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=tokenizer.model_max_length, 
                    return_tensors="pt"
                ).to(device)

                # Get text features from the text encoder
                # The 'pooler_output' is commonly used for CLIP's class embeddings
                outputs = text_encoder(**inputs)
                text_features = outputs.pooler_output # Shape: (1, hidden_size)

                # L2 normalize the text features as per CLIP's practice
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                class_embeddings_sum += text_features.squeeze(0) # Remove batch dimension
                num_prompts_processed += 1
            
            if num_prompts_processed == 0:
                raise RuntimeError(f"No prompts processed for class '{class_name}'. Check CLIP_IMAGENET_TEMPLATES.")

            averaged_embedding = class_embeddings_sum / num_prompts_processed
            averaged_class_embeddings.append(averaged_embedding)

        # Stack the averaged embeddings to form the weight matrix
        # Shape: (num_classes, input_dim)
        zero_shot_weights = torch.stack(averaged_class_embeddings, dim=0)

        # Assign to the linear layer's weight.
        # nn.Linear weights are typically (output_features, input_features),
        # so zero_shot_weights needs to be transposed.
        self.fc.weight.data = zero_shot_weights.T.contiguous()

        # Initialize bias to zeros. CLIP zero-shot classification typically doesn't use bias.
        if self.fc.bias is not None:
            self.fc.bias.data.zero_()
            # The paper states "FT only the PEFT modules and the head."
            # This implies the head parameters (weights and bias) should be trainable.
            self.fc.bias.requires_grad = True
        
        # Ensure weights are trainable for fine-tuning
        self.fc.weight.requires_grad = True

