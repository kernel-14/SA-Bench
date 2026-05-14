"""
LoRA-SB: Initialization using Update Approximation for Efficient Low-Rank Fine-Tuning.

LoRA-SB approximates full fine-tuning within low-rank subspaces using a carefully
designed initialization strategy. It uses the LoRA-XS architecture (W = W0 + s B R A)
with fixed B and A matrices and only R trainable.

The key innovation is initializing B and A via truncated SVD of an approximation
of the first full fine-tuning step, which provides:
- Optimal low-rank approximation of the initial gradient
- Orthonormal bases (B^T B = I, A A^T = I)
- Scaling factor independence (s can be set to 1)
- Guaranteed loss reduction at each step
"""

from .lora_sb_layer import LoRA_SB_Layer, apply_lora_sb
from .init import init_lora_sb

__all__ = ["LoRA_SB_Layer", "apply_lora_sb", "init_lora_sb"]
