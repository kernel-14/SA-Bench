# Universal Neural Operators through Multiphysics Pretraining

This repository aims to reproduce the core contributions of the paper "Towards Universal Neural Operators through Multiphysics Pretraining". The primary focus is on replicating the neural operator architectures (Mamba-SSM and Perceiver IO enhanced FNOs) and the adapter-based pretraining and fine-tuning methodology.

## Reproduction Strategy and Implemented Components

This reproduction focuses on a modular implementation of the key architectural components and the training paradigm described in the paper.

1.  **Neural Operator Core (`src/models/fno.py`, `src/models/fno_block.py`):**
    *   Implemented base Fourier Neural Operator (FNO) for 1D and 2D problems, including `SpectralConv` layers and FNO blocks.

2.  **Mamba-SSM Integration (`src/models/mamba_ssm.py`):**
    *   Integrated a simplified Mamba-like State Space Model (`MambaSSM`) after the lifting layer to enhance long-range dependency encoding, as described in the paper. This is a simplified interpretation of Mamba's mechanism, focusing on its role as a latent preconditioner. A full, highly optimized Mamba implementation is beyond the scope of this reproduction but its conceptual placement and function are captured.
    *   `MambaFNO1d` and `MambaFNO2d` combine the MambaSSM with the FNO core.

3.  **Perceiver IO Integration (`src/models/perceiver_io.py`):**
    *   Implemented a `PerceiverIOModule` incorporating cross-attention and self-attention mechanisms to abstract feature arrays. This module is placed after the lifting layer in `PerceiverFNO1d` and `PerceiverFNO2d`.

4.  **Adapter-based Training (`src/adapters/adapters.py`, `src/models/universal_neural_operator.py`):**
    *   Introduced `LiftingAdapter` and `ProjectionAdapter` classes (simple linear layers) to handle problem-specific input and output dimensions.
    *   The `UniversalNeuralOperator` class dynamically composes a core neural operator (FNO, MambaFNO, or PerceiverFNO) with these adapters. It includes methods for `freeze_core()` and `get_trainable_parameters_for_finetuning()` to support the paper's pre-training and fine-tuning stages where only adapters are trained.

5.  **Experiment Scenarios (`src/training/train.py`, `config/model_config.py`):**
    *   A `train.py` script outlines the pre-training and fine-tuning workflow, demonstrating how to load different model configurations, datasets, and manage the training process.
    *   `model_config.py` provides illustrative configuration dictionaries for various FNO variants and training setups.
    *   Placeholder data loading functions are used to simulate diverse PDE problems.

6.  **Evaluation Metric (`src/utils/metrics.py`):**
    *   Implemented the Range-Normalized Mean Absolute Error (NMAE) as specified in the paper (Equation 3).

## Directory Structure

```
/mnt/paper2any/pzw/proj/paperagent/hx/Research_space/SemanticAlign-Bench/experiments/runs/gemini-flash_basic/universal-neural-operators/universal-neural-operators-20260505-151805/workspace/repo
├── README.md
├── src/
│   ├── models/             
│   │   ├── __init__.py     # Exports all model classes
│   │   ├── fno.py          # Base FNO (1D and 2D) implementations
│   │   ├── fno_block.py    # Modular FNO blocks (SpectralConv, FNOBlock)
│   │   ├── mamba_ssm.py    # Simplified Mamba-SSM and MambaFNO variants
│   │   ├── perceiver_io.py # Perceiver IO module and PerceiverFNO variants
│   │   └── universal_neural_operator.py # Orchestrates core model with adapters
│   ├── adapters/           
│   │   ├── __init__.py     # Exports adapter classes
│   │   └── adapters.py     # Lifting and Projection adapters
│   ├── training/           
│   │   └── train.py        # Script for pre-training and fine-tuning workflow
│   └── utils/              
│       ├── __init__.py     # Exports utility functions
│       └── metrics.py      # NMAE calculation
└── config/                 
    └── model_config.py     # Configuration for models and training
```

## Missing Details and Assumptions

*   **Dataset specifics:** The paper mentions using "PDEBench dataset" and other custom datasets (e.g., Burgers' equation, Gray-Scott, Navier-Stokes). Specifics of how these datasets are generated or accessed were not provided in detail within the paper's main text. Therefore, placeholder data loading functions (`load_dataset`) are used in `train.py`.
*   **Hyperparameters:** Detailed hyperparameters for each model variant and training scenario (e.g., specific learning rates, decay schedules, exact architecture dimensions for Mamba/Perceiver, number of FNO layers) were not exhaustively listed. Reasonable defaults and common practices in the field have been adopted for `model_config.py`.
*   **Computational Environment:** Assumptions are made regarding the availability of standard deep learning libraries (e.g., PyTorch) and sufficient computational resources (GPU) for training.
*   **Exact architecture details:** While the general ideas for Mamba-SSM and Perceiver IO integration are provided, the precise implementation details (e.g., number of layers, hidden dimensions, attention heads for Perceiver, internal parameters of Mamba) have been derived based on typical usage patterns and the conceptual descriptions in the paper. The MambaSSM implementation is a simplified representation of the complex recurrent mechanism, focusing on its external behavior and role in the overall architecture.

This submission provides a comprehensive architectural framework and training pipeline based on the core contributions of the paper, with clear points for further detailed implementation and hyperparameter tuning.
