# LoRA-SB: Implementation Documentation

## Description
This project replicates the core contributions of the paper titled 'LoRA-SB: Initialization using Update Approximation for Efficient Low-Rank Fine-Tuning.' It introduces an initialization strategy that closely approximates full fine-tuning gradients within low-rank subspaces, demonstrating efficiency and performance advantages over other PEFT methods.

## Directory Structure
- **src/**: Contains source code for the LoRA-SB implementation.
- **data/**: Directory for datasets or processing scripts.
- **models/**: Directory to save model weights and checkpoints.

## Progress Made
1. Created initial directory structure for implementation.
2. Extracted key methodologies and experimental setups from the paper.
3. Planned modules for gradient approximation, initialization, and evaluation.

## Future Steps
- Implement gradient computations and truncated SVD-based initialization for LoRA-SB.
- Prepare fine-tuning and evaluation scripts compatible with transformer architectures.
- Add memory optimization scripts during backward passes for GPU efficiency.

