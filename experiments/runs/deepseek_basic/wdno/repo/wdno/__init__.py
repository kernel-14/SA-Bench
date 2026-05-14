"""
Wavelet Diffusion Neural Operator (WDNO)

A framework for PDE simulation and control that performs diffusion-based
generative modeling in the wavelet domain with multi-resolution training.
"""

from .wavelet_transform import WaveletTransform1D, WaveletTransform2D, WaveletTransform3D
from .diffusion import GaussianDiffusion, DDIMSampler
from .unet import UNet1D, UNet2D, UNet3D
from .wdno_base import WDNO
from .wdno_simulation import WDNOSimulation
from .wdno_control import WDNOControl
from .super_resolution import SuperResolutionModel

__version__ = "0.1.0"
