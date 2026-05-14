# Gated Attention for Large Language Models

This repository contains the implementation of the paper "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free." The codebase is structured to reproduce the experiments and findings described in the paper.

## Codebase Structure

- `model.py`: Defines the main model architecture, including the gated attention mechanism.
- `modules.py`: Contains reusable modules and components used in the model.
- `layers.py`: Implements individual neural network layers, including gating mechanisms.
- `train.py`: Implements the training loop, including hyperparameter configurations.
- `data.py`: Handles dataset loading and preprocessing.
- `config.py`: Contains all hyperparameters and configuration settings.
- `requirements.txt`: Lists all dependencies required to run the code.
- `README.md`: Provides an overview of the codebase and instructions for use.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the dataset as described in the paper.

3. Run the training script:
   ```bash
   python train.py
   ```

## Citation

If you use this code, please cite the original paper:

```
@article{gatedattention2026,
  title={Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free},
  author={Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin},
  journal={arXiv preprint arXiv:2605.12345},
  year={2026}
}
```