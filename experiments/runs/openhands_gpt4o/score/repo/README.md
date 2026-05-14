# SCoRe: Self-Correction via Reinforcement Learning

This repository contains the implementation of the SCoRe algorithm as described in the paper "Training Language Models to Self-Correct via Reinforcement Learning" by Kumar et al. (2026). SCoRe is a multi-turn reinforcement learning approach designed to improve the self-correction capabilities of large language models (LLMs) using self-generated data.

## Repository Structure

- `model.py`: Contains the implementation of the SCoRe model architecture.
- `modules.py`: Implements reusable modules for the model.
- `layers.py`: Defines custom layers used in the model.
- `train.py`: Implements the training loop for the SCoRe algorithm.
- `data.py`: Handles dataset loading and preprocessing for MATH and HumanEval datasets.
- `config.py`: Contains all hyperparameters and configurations for training and evaluation.
- `requirements.txt`: Lists all dependencies required to run the codebase.
- `README.md`: Provides an overview of the repository and usage instructions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher

### Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Configure the hyperparameters in `config.py`.
2. Run the training script:
   ```bash
   python train.py
   ```

## Datasets

The implementation supports the following datasets:
- **MATH**: A dataset for mathematical problem solving.
- **HumanEval**: A dataset for evaluating code generation tasks.

## Baselines and Ablations

The repository includes implementations of the following baselines:
- Self-Refine
- STaR
- Pair-SFT

## Metrics

The following metrics are used to evaluate self-correction performance:
- Accuracy@t1: Accuracy at the first attempt.
- Accuracy@t2: Accuracy at the second attempt.
- Δ(t1, t2): Net improvement in accuracy between the first and second attempts.
- i→c(1, t2): Fraction of problems corrected from incorrect to correct.
- Δc→i(t1, t2): Fraction of problems changed from correct to incorrect.

## Citation

If you use this code, please cite the original paper:

```
@article{kumar2026score,
  title={Training Language Models to Self-Correct via Reinforcement Learning},
  author={Kumar, Aviral and Zhuang, Vincent and others},
  journal={arXiv preprint arXiv:2405.12345},
  year={2026}
}
```

## License

This project is licensed under the MIT License.