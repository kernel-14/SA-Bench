# Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing Reproduction

This repository aims to reproduce the core contributions of the paper "Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing".

## Core Contributions to be Reproduced

The paper proposes Ca2-VDM, an efficient autoregressive video diffusion model with two main innovations:

1.  **Causal Generation**: Introduces unidirectional feature computation and causal temporal attention. This allows for precomputation and reuse of KV-cache in subsequent autoregression steps, eliminating redundant computations.
2.  **Cache Sharing**: Shares the KV-cache across all denoising steps by using a distinct timestep embedding (t=0) for conditional frames during both training and inference. This significantly reduces cache storage costs.

Additionally, the paper details:

*   **KV-cache queue**: A queue structure for temporal KV-cache to manage long-term context while maintaining affordable computation and storage.
*   **Cyclic-TPEs**: A cyclic shift mechanism for temporal positional embeddings to support the KV-cache queue and extendable long-term context beyond training length.
*   **Prefix-Enhanced Spatial Attention**: A mechanism to enhance guidance from prefix frames in spatial attention.

## Implemented Components

*(This section will be updated as implementation progresses)*

## Assumptions and Missing Details

*(This section will be updated as implementation progresses)*

## Usage

*(This section will be updated with instructions on how to run the reproduced model)*

