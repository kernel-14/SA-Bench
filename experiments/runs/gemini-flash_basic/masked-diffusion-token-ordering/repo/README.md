# Reproduction of "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions"

This repository aims to reproduce the core contributions of the paper "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions". The paper investigates the training complexity and inference flexibility of Masked Diffusion Models (MDMs) in comparison to Autoregressive Models (ARMs), particularly focusing on token ordering strategies.

## Core Contributions and Reproduction Plan:

The paper highlights several key contributions, which this reproduction effort will attempt to capture in a static codebase:

1.  **Training Complexity of MDMs**:
    *   **Concept**: MDMs face a more challenging training task by learning to solve an exponentially large number of infilling problems, which can be computationally intractable for certain distributions compared to ARMs.
    *   **Reproduction**:
        *   Define the loss functions for both MDMs and ARMs, highlighting the differences in their objectives (equations 1, 2, and 3 from the paper).
        *   Conceptualize how the `L&O (Latents-and-Observations) distribution` (Definition 3.1 and Example 3.2) could be used to demonstrate these hardness aspects.

2.  **Inference Flexibility and Adaptive Strategies for MDMs**:
    *   **Concept**: MDMs offer flexible decoding paths. Adaptive inference strategies can significantly improve performance by strategically selecting token unmasking order, thereby sidestepping hard subproblems.
    *   **Reproduction**:
        *   Implement the "Vanilla MDM inference" (Algorithm 1 in the paper, lines 121-123).
        *   Implement the "Adaptive MDM inference" framework, which uses an oracle `F` (lines 211-213).
        *   Implement two specific ordering oracles:
            *   **Top Probability**: Selects tokens based on the highest predicted probability (lines 225-226).
            *   **Top Probability Margin**: Selects tokens based on the largest difference between the top two predicted probabilities (lines 229-230).

3.  **Model Architectures**:
    *   **Concept**: The paper implies the use of transformer-based models for both MDMs (denoising network) and ARMs (causal attention).
    *   **Reproduction**:
        *   Create placeholder classes for `MDM` and `ARM` models, outlining their potential components (e.g., transformer blocks, positional embeddings).
        *   Specify how the denoising network `p_theta` in MDMs would function.

4.  **Evaluation Metrics**:
    *   **Concept**: The paper evaluates models using accuracy for logic puzzles (Sudoku, Zebra) and generative perplexity/entropy for natural language tasks.
    *   **Reproduction**:
        *   Define functions or conceptual code for calculating these metrics.

## Codebase Structure:

The repository will be organized into the following directories:

*   `models/`: Contains the definitions of MDM and ARM architectures.
*   `training/`: Defines the loss functions and conceptual training loops.
*   `inference/`: Implements vanilla and adaptive inference strategies, including the ordering oracles.
*   `utils/`: Contains utility functions, such as data handling or common mathematical operations.
*   `evaluation/`: Contains functions for calculating evaluation metrics.

## Assumptions and Missing Details:

*   **Specific Model Implementations**: The paper describes the *type* of models (e.g., transformer with causal attention, denoising network) but doesn't provide specific architectural details (number of layers, heads, hidden dimensions). For this reproduction, we will use generic placeholders.
*   **Hyperparameters and Training Details**: Detailed hyperparameters (learning rates, batch sizes, optimizers, specific noise schedules `alpha_t`) are often in appendices or not explicitly stated in the main text. We will assume standard practices where details are missing.
*   **Dataset Handling**: While datasets like Slimpajama and Sudoku/Zebra puzzles are mentioned, the actual data loading and preprocessing logic are not provided. We will conceptualize these.
*   **Proof Implementations**: The proofs in the appendix (e.g., Proposition 2.1, 3.3) are not directly translatable into executable code for model reproduction but are acknowledged as theoretical underpinnings.
*   **"Time-embedding-free architecture"**: The paper mentions `p_theta(x_t, t) = p_theta(x_t)`. We will reflect this in our MDM model definition.
*   **`alpha_t` noise schedule**: The paper states `alpha_0 = 1, alpha_1 = 0`. A specific function for `alpha_t` is not given, so we will use a conceptual placeholder.

This reproduction focuses on presenting the logical structure and algorithmic flow as described in the paper, rather than a fully executable and optimized implementation, given the static nature of the benchmark.
