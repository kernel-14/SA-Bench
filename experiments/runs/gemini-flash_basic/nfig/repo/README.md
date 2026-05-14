# NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering Reproduction

This repository aims to reproduce the core contributions of the paper "NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering".

## Core Contributions to be reproduced:

1.  **Frequency-guided Residual-quantized VAE (FR-VAE)**: This component is responsible for encoding images into frequency-guided residual quantized representations and decoding them. It includes:
    *   **Frequency-guided Decomposer**: Decomposes image latent features into multiple frequency components using FFT and frequency masks.
    *   **Frequency-guided Composer**: Reconstructs the image from frequency components by interpolating and merging them.
    *   **Frequency-guided Residual-quantization**: Extracts residual tokens and performs vector quantization with a learnable codebook.

2.  **Autoregressive Image Generation**: This involves a next-frequency prediction model that employs a frequency-aware Transformer to auto-regressively generate token sequences from low to high frequencies.

## Codebase Structure:

```
repo/
├── README.md
└── nfig/
    ├── models/
    │   ├── fr_vae.py
    │   └── autoregressive_transformer.py
    └── utils/
        └── frequency_utils.py
```

## Implementation Details and Assumptions:

*   **FR-VAE Architecture**: Will implement the encoder, decoder, decomposer, composer, and residual quantization logic as described in Section 3.1.
    *   The FR-VAE incorporates a VQGAN architecture, and the image encoder is initialized with pretrained weights from DINOv2-base. (Note: Pre-trained weights loading will be represented by appropriate code structure but not actual loading due to the static nature of this task).
    *   Frequency residual quantizer will use scaling factors `[1, 2, 3, 4, 5, 6, 8, 10, 13, 16]` for different frequency bands.
    *   FR-VAE codebook size is 4096.
*   **Autoregressive Transformer**: Will implement a VAR Transformer backbone with a depth of 16 for next-frequency image prediction (Section 3.2).
*   **Optimization**: Adam optimizer, learning rate `8e-5`, batch size 768. (These will be noted in configuration but not actively used for training).
*   **Training**: 350 epochs on ImageNet. (Not actively performed).
*   **Inference**: CFG 4.5 and top_k 990. (To be reflected in the inference logic).

## Missing Details and Future Work:

*   Specific architectures for the VQGAN encoder/decoder and VAR Transformer are not fully detailed in the paper, so standard implementations or commonly used architectures from related works will be assumed.
*   The exact implementation of DINO discriminator within VQGAN and DINOv2-base initialization will be outlined conceptually.
*   Detailed loss functions (Appendix B.1) will be referenced but not fully implemented due to potential complexity and time constraints.
*   The frequency masks $M_i$ in the Frequency-guided Decomposer are not explicitly defined in terms of their exact shape or how they are generated, so a plausible approach will be assumed.

