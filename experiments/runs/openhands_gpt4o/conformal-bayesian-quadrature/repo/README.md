# Conformal Prediction as Bayesian Quadrature

This repository contains the implementation of the paper "Conformal Prediction as Bayesian Quadrature" by Jake C. Snell and Thomas L. Griffiths. The codebase reproduces the experiments and methods described in the paper.

## Codebase Structure

- `model.py`: Contains the implementation of the Bayesian quadrature-based conformal prediction framework.
- `data.py`: Handles data loading and preprocessing for synthetic and MS-COCO datasets.
- `train.py`: Implements the training and evaluation loops.
- `config.py`: Stores all hyperparameters and configuration settings.
- `requirements.txt`: Lists all dependencies required to run the code.
- `README.md`: Provides an overview of the repository and instructions for usage.

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

## Experiments

The repository includes implementations for the following experiments:

1. Synthetic Binomial Data
2. Synthetic Heteroskedastic Data
3. False Negative Rate on MS-COCO

## Citation

If you use this code, please cite the original paper:

```
@article{snell2026conformal,
  title={Conformal Prediction as Bayesian Quadrature},
  author={Snell, Jake C. and Griffiths, Thomas L.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```