# MoE-POT: Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training Reproduction

This repository aims to reproduce the core contributions of the paper "MoE-POT: Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training".

## Implemented Components

I will focus on implementing the following key architectural components:

1.  **Input Encoding and Temporal Aggregation**: This includes the patchification layer with positional embeddings and the temporal aggregation layer as described in Section 4 of the paper.
2.  **Fourier Layer**: Implementation of the multi-head Fourier layer for kernel-based integral transformations, also detailed in Section 4.
3.  **Mixture-of-Experts Layer**: This will be the central component, encompassing:
    *   Shared Experts
    *   Routed Experts
    *   Router-Gating Network with Top-K selection
    *   Load Balancing Objective

## Assumptions and Deviations

*   **Static Implementation**: This reproduction focuses solely on providing the source code, without running any training or evaluation. Therefore, aspects like dataset loading, specific training loops, and detailed experimental setups (e.g., hyperparameter tuning) will be represented structurally but not executed.
*   **Simplification of Specifics**: Where exact implementation details (e.g., specific convolutional layer configurations, activation functions, or initialization schemes) are not explicitly provided or are common practices, reasonable defaults or placeholder implementations will be used.
*   **Appendix Details**: Details from the appendix will be incorporated if they are directly referenced and crucial for the main architectural components described in the main body of the paper.
*   **L2RE Metric**: The L2 relative error (L2RE) is mentioned as the primary evaluation metric. While the implementation will not include an evaluation loop, the loss function will reflect the squared L2 norm.
*   **Optimizer**: The Adam optimizer with a learning rate of 1e-3 is mentioned. This will be noted for potential future use.

## Directory Structure

```
moe_pot/
├── configs/             # Configuration files (e.g., model hyperparameters)
├── data/                # Placeholder for datasets (no actual data will be included)
└── model/               # Python modules for the MoE-POT model architecture
    ├── __init__.py
    ├── moe_pot.py       # Main MoE-POT model definition
    ├── layers.py        # Definitions for Patchification, Temporal Aggregation, Fourier, and MoE layers
    └── utils.py         # Utility functions (e.g., Top-K selection, load balancing)
```

## Progress

- [ ] Initial directory structure created.
- [ ] README.md drafted.
- [ ] `moe_pot.py` (Main model architecture)
- [ ] `layers.py` (Sub-layers like Patchification, Temporal Aggregation, Fourier, MoE)
- [ ] `utils.py` (Helper functions for MoE, e.g., Top-K, Load Balancing)

