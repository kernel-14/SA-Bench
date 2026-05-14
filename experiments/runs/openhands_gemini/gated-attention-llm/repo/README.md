
# Gated Attention for Large Language Models

This repository contains a reproduction of the paper "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free".

## Codebase Structure:

- `config.py`: Contains all hyperparameters and configuration settings for the models and training process. Various experimental configurations from the paper are provided as classes, and the active configuration can be selected.

- `data.py`: Handles dataset loading, preprocessing, and tokenization.
- `modules.py`: Implements core building blocks of the Transformer architecture, including attention mechanisms and feed-forward networks, with specific implementations of gated attention.
- `model.py`: Defines the overall Transformer model architecture, integrating the modules from `modules.py`.
- `train.py`: Contains the main training loop, including optimization, loss calculation, and perplexity (PPL) evaluation.
- `requirements.txt`: Lists all Python dependencies required to run the code.

## Data Handling
The `data.py` script currently uses dummy data for demonstration purposes. For actual training and evaluation with specific datasets (e.g., the 3.5 trillion token dataset mentioned in the paper), this script would need to be extended to load and preprocess real-world data.

## Rotary Position Embeddings (RoPE)
The paper mentions the use of RoPE for context length extension. A full implementation of RoPE is complex and not central to the core gating mechanism contribution. Therefore, explicit RoPE layers are omitted in this reproduction for simplicity, assuming they would be handled by a base pre-trained model or an external utility in a complete LLM pipeline.

## Evaluation Notes:
The paper mentions several benchmarks like Hellaswag, MMLU, etc. For this reproduction, perplexity (PPL) is calculated during training. Full evaluation on these external benchmarks typically requires specialized datasets and external evaluation frameworks (e.g., `lm-eval-harness`), which are outside the scope of this core architectural reproduction.

