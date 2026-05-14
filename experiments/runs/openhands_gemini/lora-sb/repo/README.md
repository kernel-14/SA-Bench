
# LoRA-SB: Initialization Using Update Approximation

This repository contains a faithful reproduction of the paper "INITIALIZATION USING UPDATE APPROXIMATION IS A Silver Bullet FOR EXTREMELY EFFICIENT LOW-RANK FINE-TUNING" (LoRA-SB).

LoRA-SB proposes a novel initialization strategy for LoRA-XS architecture that approximates full fine-tuning within low-rank subspaces, achieving significant parameter efficiency without sacrificing performance.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Reproducing Experiments](#reproducing-experiments)

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

```bash
# Clone the repository (if not already cloned)
# git clone https://github.com/your-username/lora-sb-reproduction.git
# cd lora-sb-reproduction

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

The codebase is organized as follows:

```
.
├── requirements.txt         # Python dependencies
├── config.py                # All hyperparameters and configuration settings
├── lora_sb_layers.py        # Defines the custom LoRA-SB layer (W = W_0 + sBRA)
├── model.py                 # Integrates LoRA-SB layers into pre-trained models
├── data.py                  # Handles dataset loading, tokenization, and preprocessing
├── utils.py                 # Utility functions for LoRA-SB initialization and SVD
└── train.py                 # Main script for training and evaluation
└── README.md                # This file
```

## Configuration

All configurable parameters, including model names, LoRA-SB specific settings (rank, scaling factor), optimizer details, training parameters, and dataset information, are located in `config.py`.

Key configurations:
- `model_name`: Specifies the base pre-trained model (e.g., `mistralai/Mistral-7B-v0.1`, `roberta-large`).
- `rank`: The rank `r` for LoRA-SB matrices.
- `scaling_factor`: The scaling factor `s` (set to 1.0 as per paper).
- `init_data_percentage`: Percentage of data used for gradient estimation during initialization (0.1%).
- `num_initialization_samples`: Number of samples used for gradient estimation.
- `learning_rate`, `batch_size`, `epochs`: Standard training hyperparameters.
- `target_modules_llm`, `target_modules_roberta`: Specifies which modules in the base model will be adapted by LoRA-SB.
- `TASK_CONFIGS`: Predefined configurations for arithmetic, commonsense reasoning, and NLU tasks, mirroring the paper's experiments.

## Usage

### LoRA-SB Initialization

The core of LoRA-SB is its initialization strategy. This is handled automatically by `train.py`.
1. A small subset of the training data is used to estimate the first-step gradients of the original full fine-tuning.
2. Truncated SVD is performed on these estimated gradients.
3. The `B`, `A`, and `R` matrices of the `LoRASBLayer` are initialized based on the SVD results.

### Training

To run a training experiment, execute the `train.py` script. The script will automatically load the specified model, perform LoRA-SB initialization, and start the fine-tuning process.

```bash
python train.py
```

You can modify the `config.py` file or add command-line arguments parsing to `train.py` to easily switch between different models, tasks, ranks, and other hyperparameters.

### Example for Arithmetic Task (Mistral-7B)

The `train.py` script by default loads the configuration for an arithmetic task using `Mistral-7B`. You can adjust this within `train.py` to select other tasks or specific ranks from `config.TASK_CONFIGS`.

## Reproducing Experiments

To reproduce the experiments from the paper, you would typically:

1. **Select a task**: Choose between `arithmetic`, `commonsense_reasoning`, or `nlu` from `config.TASK_CONFIGS`.
2. **Set model and rank**: Update `config.py` or modify `train.py` to iterate through the ranks (`ranks = [32, 64, 96]` for LLMs, `[8, 16, 24]` for RoBERTa).
3. **Run `train.py`**: The script will handle initialization, training, and (simplified) evaluation.
4. **Detailed Evaluation**: For comprehensive evaluation as presented in the paper (e.g., GSM8K, MATH for arithmetic, multiple datasets for commonsense, GLUE metrics), you would extend the `train.py` script or create separate evaluation scripts to report the specific metrics mentioned (e.g., Pearson correlation for STS-B, Matthew's correlation for CoLA, accuracy for others).

**Note on Dataset Handling**: The `data.py` file provides functions to load common datasets. For `COMMONSENSE170K`, which is a combination of eight datasets, further integration logic would be required to merge and prepare it for training as a single dataset as implied by the paper. The `get_dataloader_for_initialization` and `get_metamath_datasets` and `get_glue_datasets` are provided as starting points.
