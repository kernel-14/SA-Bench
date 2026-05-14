# Hi-MAR: Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots

This repository provides a faithful reproduction of the Hi-MAR paper.

## Overview

Hi-MAR (Hierarchical Masked Autoregressive Models) introduces a new autoregressive design for visual generation that models a hierarchy from low-resolution image tokens to typical dense image tokens. It utilizes a multi-phase approach:
1. **Phase 1**: Predicts low-resolution image tokens as intermediary pivots to reflect global structure.
2. **Phase 2**: Uses these low-resolution pivots as additional guidance to strengthen the prediction of dense image tokens.

A novel Diffusion Transformer head is also devised to amplify global context for mask token prediction.

## Repository Structure

- `config.py`: Contains all hyperparameters and configuration settings for the model and training.
- `layers.py`: Implements fundamental building blocks such as `Attention`, `FeedForward`, `AdaLNZero`, `AdaLN`, and `DiTBlock` as described in the paper.
- `modules.py`: Defines key modules like the `Transformer` (both for Hi-MAR and Diffusion Head), `MLPDiffusionHead`, and `DiffusionTransformerHead`.
- `model.py`: Integrates the modules to form the complete `HiMAR` model, including the VAE for tokenization.
- `data.py`: Handles dataset loading and preprocessing for ImageNet and MS-COCO.
- `train.py`: Contains the training loop, evaluation logic, and inference procedures.
- `requirements.txt`: Lists all Python dependencies required to run the codebase.