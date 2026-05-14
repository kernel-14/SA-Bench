# Robotic World Model (RWM)

This repository contains the implementation of the Robotic World Model (RWM) as described in the paper "Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics."

## Codebase Structure

- `model.py`: Contains the GRU-based architecture and MLP heads for the world model.
- `modules.py`: Implements reusable components of the model.
- `layers.py`: Defines custom layers used in the model.
- `train.py`: Implements the training loop with all hyperparameters and algorithms described in the paper.
- `data.py`: Handles dataset loading and preprocessing.
- `config.py`: Contains all hyperparameters and configurations.
- `requirements.txt`: Lists all dependencies required for the implementation.
- `README.md`: Provides a brief description of the codebase structure and usage.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure hyperparameters in `config.py`.

3. Run the training script:
   ```bash
   python train.py
   ```

## Citation

If you use this code, please cite the original paper:

```
@article{li2026robotic,
  title={Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics},
  author={Chenhao Li, Andreas Krause, Marco Hutter},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```