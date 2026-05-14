"""
PEFT methods for visual recognition.
"""

from .bitfit import apply_bitfit
from .layernorm import apply_layernorm
from .difffit import apply_difffit
from .ssf import apply_ssf
from .vpt import apply_vpt
from .adapter import apply_adapter
from .adaptformer import apply_adaptformer
from .convpass import apply_convpass
from .repadapter import apply_repadapter
from .lora import apply_lora
from .fact import apply_fact

__all__ = [
    'apply_bitfit',
    'apply_layernorm',
    'apply_difffit',
    'apply_ssf',
    'apply_vpt',
    'apply_adapter',
    'apply_adaptformer',
    'apply_convpass',
    'apply_repadapter',
    'apply_lora',
    'apply_fact',
]

METHOD_REGISTRY = {
    'linear': None,  # linear probing - handled separately
    'full': None,    # full fine-tuning - handled separately
    'bitfit': apply_bitfit,
    'layernorm': apply_layernorm,
    'difffit': apply_difffit,
    'ssf': apply_ssf,
    'vpt_shallow': apply_vpt,
    'vpt_deep': apply_vpt,
    'pfeif_adapter': apply_adapter,
    'houl_adapter': apply_adapter,
    'adaptformer': apply_adaptformer,
    'convpass': apply_convpass,
    'repadapter': apply_repadapter,
    'lora': apply_lora,
    'fact_tt': apply_fact,
    'fact_tk': apply_fact,
}
