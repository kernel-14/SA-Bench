# LoRA-SB: Initialization using Update Approximation for Efficient Low-Rank Fine-Tuning - Reproduction

This repository aims to reproduce the core contributions of the paper "LoRA-SB: Initialization using Update Approximation for Efficient Low-Rank Fine-Tuning".

## Implemented Components

The primary focus of this reproduction is the LoRA-SB method, which introduces a novel initialization strategy for the LoRA-XS architecture. This includes:

1.  **LoRA-SB Module**: Implementation of the weight update mechanism $W = W_0 + sBRA$, where $R$ is the trainable matrix, and $A, B$ are fixed after initialization.
2.  **LoRA-SB Initialization Strategy**:
    *   Approximation of the first full fine-tuning step's gradient, $\Delta W_{avg}$, by averaging gradients over a small subset of the training data.
    *   Application of truncated Singular Value Decomposition (SVD) to $\Delta W_{avg}$.
    *   Initialization of $B_{init}$, $A_{init}$, and $R_{init}$ based on the SVD results, ensuring $B_{init}$ and $A_{init}$ form orthonormal bases.
    *   Approximation of the AdamW first update step using the sign of the summed gradients.

## Core Contributions Addressed

*   **Optimal Gradient Approximation**: The paper highlights how LoRA-SB's initialization ensures optimal low-rank approximation of the initial gradient and preserves update directions.
*   **Initialization Strategy**: The method of using the first full FT gradient for initializing $A$ and $B$ matrices to span the most relevant subspace for the task.
*   **Scaling Factor Independence**: The theoretical elimination of the need for the scaling factor $\alpha$ due to orthonormal $A$ and $B$ matrices and optimal gradient approximation.

## Assumptions and Missing Details

*   **Environment Setup**: Assumes a PyTorch environment with necessary libraries (e.g., `transformers`, `torch`, `scipy` for SVD) available. Specific versions are not enforced to avoid dependency conflicts.
*   **Model Architecture**: The implementation assumes a `nn.Linear` layer or similar structure where `W_0` (pre-trained weights) can be identified and LoRA-SB adapters can be inserted.
*   **Dataset Handling**: The code for sampling a subset of the training data for initialization is conceptual; actual implementation would depend on the specific dataset and data loading pipeline.
*   **Gradient Computation for Initialization**: The mechanism to compute and average `∇_W L(W_0, x_i)` over a subset of data for initialization is outlined, but the exact integration into a training loop for this initial step is left to the user to adapt to their specific training framework. This initial step would involve a temporary forward/backward pass with `W_0` to collect gradients.
*   **Full Fine-Tuning Simulation**: The core idea is to simulate full fine-tuning within low-rank subspaces. The implementation provides the mechanism for this simulation but does not include a full training script that runs this simulation end-to-end.
*   **Memory Optimization for Initialization**: The paper mentions memory optimization for layer-wise gradient computation during initialization. While the concept is acknowledged, a direct implementation of `_O(1)` memory usage via specific PyTorch hooks is not explicitly provided in the core LoRA-SB and initialization modules, as it's an implementation detail that can vary.
*   **Specific Model Integration**: The code provided is a generic LoRA-SB layer and initialization logic. Integration into specific models (e.g., Llama, Mistral, RoBERTa) would require model-specific modifications to replace or adapt linear layers.

## Usage

To use this reproduction:
1.  Integrate the `LoRASBLayer` into your model's linear layers.
2.  Implement the `lora_sb_initialization` function to compute `ΔW_avg` from your specific dataset and model. This typically involves:
    *   Loading a small subset of your training data.
    *   Performing a forward pass with `W_0` (original pre-trained weights).
    *   Computing the loss and gradients with respect to `W_0`.
    *   Averaging these gradients and taking their sign (for AdamW approximation).
3.  Apply the initialization to the `LoRASBLayer` instances.

Further details on the mathematical derivations and experimental results can be found in the original paper.
