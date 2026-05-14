# nGPT: Normalized Transformer with Representation Learning on the Hypersphere - Reproduction

This repository aims to reproduce the core contributions of the paper "nGPT: Normalized Transformer with Representation Learning on the Hypersphere". The reproduction focuses on implementing the architectural changes and normalization strategies described in the paper, particularly in Section 2.6 "Summary of Modifications".

## Implemented Components:

The following core components of nGPT will be implemented:

1.  **Normalization Utility**: A fundamental function for unit norm normalization of tensors.
2.  **Token Embeddings and Output Logits**:
    *   Normalization of input and output embedding matrices (`E_input`, `E_output`).
    *   Integration of the trainable scaling parameter `s_z` for logits.
3.  **Transformer Block Modifications**:
    *   **Removal of Normalization Layers**: Confirmation that `RMSNorm` or `LayerNorm` are not used.
    *   **Attention Block**:
        *   Normalization of weight matrices (`W_q`, `W_k`, `W_v`, `W_o`).
        *   Normalization and scaling of `q` and `k` vectors using `s_qk`.
        *   Adjustment of the softmax scaling factor to `sqrt(d_k)`.
    *   **MLP Block**:
        *   Normalization of weight matrices (`W_u`, `W_nu`, `W_oMLP`).
        *   Scaling of intermediate vectors `u` and `nu` using `s_u` and `s_nu`.
    *   **Update Equations**: Implementation of nGPT-specific update equations for attention and MLP blocks, incorporating eigen learning rates `alpha_A` and `alpha_M`.
4.  **Parameter Initialization**: Correct initialization procedures for all new scaling parameters (`s_qk`, `s_u`, `s_nu`, `s_z`, `alpha_A`, `alpha_M`).
5.  **Optimizer Considerations**: Note on the removal of weight decay and learning rate warmup.

## Code Structure:

The code will be organized into modular Python files to reflect the components of a Transformer model, with specific nGPT modifications integrated. Expected modules include:

*   `ngpt_model.py`: Main model definition, combining blocks and handling embeddings.
*   `ngpt_block.py`: Defines a single nGPT Transformer block.
*   `ngpt_attention.py`: Implements the nGPT multi-head self-attention mechanism.
*   `ngpt_mlp.py`: Implements the nGPT MLP block.
*   `normalization.py`: Utility functions for normalization.
*   `embeddings.py`: Handles token and positional embeddings.

## Assumptions and Missing Details:

*   **Framework**: The implementation will be conceptual and focus on the mathematical operations described. No specific deep learning framework (e.g., PyTorch, TensorFlow) will be used for execution since this is a static-only benchmark. The code will be Python-like pseudocode.
*   **Dataset and Training Loop**: The paper's experiments on Open-WebText and specific training loop details (e.g., data loading, specific optimizer parameters beyond what's mentioned for Adam) are out of scope for this reproduction, as the task is to replicate the *model architecture* and core modifications.
*   **Hyperparameters**: Default hyperparameters will be used where not explicitly specified for the architecture itself, relying on the paper's descriptions for `d_model`, `d_k`, `n_heads`, etc., and the initialization values for scaling factors.
*   **RoPE**: Rotary Positional Embeddings (RoPE) will be assumed to be applied as described in the baseline Transformer section, but the specific implementation details will be abstracted if not explicitly altered by nGPT.

## Core Contributions Addressed:

This reproduction directly addresses the key contributions of the paper by implementing:

*   Optimization of network parameters on the hypersphere through extensive normalization.
*   The Normalized Transformer as a variable-metric optimizer through the introduction of eigen learning rates.
*   The architectural changes summarized in Section 2.6 to achieve faster convergence.

