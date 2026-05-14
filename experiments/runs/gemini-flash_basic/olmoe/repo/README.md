# OLMoE Reproduction

This repository aims to reproduce the core contributions of the paper 'OLMoE: Open Mixture-of-Experts Language Models'.

## Implemented Components

### 1. Model Architecture ()
Based on Section 2 and Table 1 of the paper.
- ****: A simple linear layer that outputs logits for expert selection.
- ****: A standard Feed-Forward Network (FFN) with ReLU activation.
- ****: Implements the Mixture-of-Experts layer.
    - Replaces the FFN in a Transformer block.
    - Uses Token Choice routing with  to select  experts.
    - Normalizes routing weights for selected experts to sum to 1.
    - Dispatches tokens to selected experts, computes weighted outputs, and sums them.
    - Returns router logits, all expert routing weights, and selected experts for auxiliary loss calculations.
- ****: Represents a single decoder-only Transformer block.
    - Includes MultiheadAttention, LayerNorm, and Dropout layers.
    - Integrates the  layer in place of the FFN.
    - Stores MoE routing information for loss computation.
- ****: The full OLMoE language model.
    - Consists of token embeddings, a stack of s, and a language modeling head.

### 2. Loss Functions ()
Based on Section 2, 4.1.6, and 4.1.7 of the paper.
- ****: Calculates the Load Balancing Loss () to ensure even expert utilization.
    - Uses one-hot encoding for expert indicators and computes the fraction of tokens () and total routing probability () for each expert as per Equation 3.
- ****: Calculates the Router Z-loss () to penalize large router logits and promote training stability.
- ****: The combined pretraining loss function.
    - Linearly combines Cross-Entropy Loss () with  (weight ) and  (weight ).

### 3. Configuration ()
Defines hyperparameters for the model and training process, based on information from Section 2, Table 1, Figure 4, and Figure 5.
- : 50257 (common choice)
- : 2048 (inferred to match active parameter counts, around 1.3B active/7B total)
- : 24 (inferred for total parameter count)
- : 16
- : 1024 (FFN dimension for *each* expert; relevant for 64 experts with 8 activated, corresponding to 1.3B active parameters)
- : 64
- : 8
- : 0.1
- Loss weights for  and .

### 4. Conceptual Training Script ()
A conceptual  script demonstrating how the  and  would be integrated into a training loop. It sets up the model, an optimizer, and includes a basic loop for forward/backward passes with dummy data generation.

## Remaining Tasks
- Implement robust data loading and preprocessing (tokenization, batching).
- Fully incorporate general pretraining settings (Section 4.2) and adaptation settings (Section 4.3), which include various hyperparameters related to dataset experiments, initialization, normalization, etc. These would primarily impact the  and potentially require modifications to  (e.g., QK-Norm, RMSNorm) and .

## Assumptions and Missing Details
- Specific implementation details of the Transformer attention mechanisms and other normalization layers within  are assumed to follow standard practices (e.g., causal masking for ) unless explicitly stated otherwise in the paper. The paper refers to standard Transformer [185] layers, which implies common architectural choices.
- The exact training dataset preprocessing and tokenization are not yet implemented, but the model expects  and .
- The  layer's token dispatching uses a conceptual Python loop; a production-ready implementation would require more efficient scatter/gather operations, potentially with custom CUDA kernels, for optimal performance.
- The  parameter for the  module in the  layer is derived from the paper's description of maintaining active parameters for varying expert granularity (e.g., for 64 experts, FFN dimension is 1,024, implying  for *each* expert is 1,024). The relationship between  and  for a single expert is crucial to match active parameter counts to dense models.

## Dependencies ()
- 
