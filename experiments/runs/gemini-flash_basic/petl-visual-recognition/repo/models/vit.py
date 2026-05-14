import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ViT(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224-in21k', num_classes=1000):
        super().__init__()
        self.model = ViTModel.from_pretrained(model_name)
        self.num_features = self.model.config.hidden_size
        self.head = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        x = self.model(x).last_hidden_state[:, 0]  # Get the [CLS] token
        x = self.head(x)
        return x

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the classification head
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True
        # Ensure the classification head is also trainable for full FT
        for param in self.head.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.num_features, num_classes)
        # Ensure new head parameters are trainable
        for param in self.head.parameters():
            param.requires_grad = True

# Example usage (for testing purposes, not part of the final submission code to be run)
if __name__ == '__main__':
    # Initialize ViT-B/16 pre-trained on ImageNet-21K
    vit_model = ViT(model_name='google/vit-base-patch16-224-in21k', num_classes=10)
    vit_model.freeze_backbone()

    # Check trainable parameters for linear probing
    trainable_params_lp = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in vit_model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters (linear probing): {trainable_params_lp}")

    # Test unfreeze for full FT
    vit_model.unfreeze_backbone()
    trainable_params_ft = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    print(f"Trainable parameters (full FT): {trainable_params_ft}")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = vit_model(dummy_input)
    print(f"Output shape: {output.shape}")
