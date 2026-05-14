# Gated Attention for Large Language Models

## Overview
This repository reproduces core contributions of the paper titled "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free." The work highlights gated mechanisms in softmax attention layers, demonstrating improvements in stability, performance, and attention dynamics.

## Key Components
1. **Gated Attention Mechanism**: Implemented head-specific sigmoid gates applied to Scaled Dot-Product Attention (SDPA) outputs.
2. **Transformer Model Integration**: Incorporates gated attention into dense transformer models supporting experimentation.
3. **Evaluation Script**: Enables metric-based comparison of perplexity and benchmarks like MMLU for gated and baseline models.

## How to Use
### Code Structure
- : Implements the gated attention mechanism.
- : Transformer model with integrated gated attention layers.
- : Script to evaluate model performance using perplexity metrics.

### Example Usage
1. Customize  with a real dataset or data loader.
2. Instantiate and train  models using the provided configuration.

## Contributions from the Paper
- Mechanistic implementation of gated attention at SDPA.
- Structural integration with dense transformer architectures.
- Perplexity evaluation aligned with benchmark datasets and hyperparameters described in the paper.

## Missing Details
- Explicit experiment replication of mixture-of-experts (MoE) models due to dataset and training setup constraints.

## References
Refer to the paper for detailed insights into gating mechanisms, sparsity analysis, and performance benchmarks.

