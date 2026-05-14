
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial

class CustomVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 global_pool='token', drop_path_rate=0.0, **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                         num_classes=num_classes, global_pool=global_pool, **kwargs)
        self.num_classes = num_classes

        # Re-initialize head for downstream tasks
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Set drop path rate
        if drop_path_rate > 0.0:
            for block in self.blocks:
                block.drop_path.drop_prob = drop_path_rate

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    model = CustomVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        # Load ImageNet-21K pretrained weights
        # For actual implementation, you would load a specific timm pretrained model
        # For now, we'll simulate it or use a placeholder if directly available
        # Example: timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        print("Loading ImageNet-21K pretrained weights (simulated/placeholder)...")
        # In a real scenario, you'd use timm.create_model or load checkpoint
        # This is a placeholder for now.
        pass
    return model

def clip_vit_base_patch16_224(pretrained=False, **kwargs):
    # CLIP ViT-B/16 model structure. Assume a similar structure to timm ViT.
    # For actual implementation, you would load a specific CLIP model.
    model = CustomVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        print("Loading CLIP pretrained weights (simulated/placeholder)...")
        # In a real scenario, you'd load CLIP weights.
        pass
    return model

