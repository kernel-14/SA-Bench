"""
Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing.

Paper: "Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing"
Authors: Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, Long Chen

Key contributions:
  1. Causal Generation: Unidirectional temporal attention enabling KV-cache reuse
  2. Cache Sharing: Shared KV-cache across all denoising timesteps
  3. KV-Cache Queue: Queue structure for extendable long-term context
  4. Cyclic-TPEs: Cyclic temporal positional embeddings for long video generation
"""

__version__ = "1.0.0"
