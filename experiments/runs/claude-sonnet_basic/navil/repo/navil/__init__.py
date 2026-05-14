"""
NaViL: Rethinking Scaling Properties of Native Multimodal LLMs under Data Constraints
"""

from .model import NaViLModel, NaViLConfig
from .visual_encoder import VisualEncoder, VisualEncoderConfig
from .moe import ModalitySpecificMoE, MMoEConfig
from .norm import RMSNorm, get_rms_norm
from .scaling_analysis import ScalingAnalyzer, ScalingExperimentResult

__all__ = [
    "NaViLModel",
    "NaViLConfig",
    "VisualEncoder",
    "VisualEncoderConfig",
    "ModalitySpecificMoE",
    "MMoEConfig",
    "RMSNorm",
    "get_rms_norm",
    "ScalingAnalyzer",
    "ScalingExperimentResult",
]
