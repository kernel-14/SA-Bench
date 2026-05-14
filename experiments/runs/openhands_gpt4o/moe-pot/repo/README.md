# MoE-POT: Mixture-of-Experts Pre-training Operator Transformer

This repository contains the implementation of the Mixture-of-Experts Pre-training Operator Transformer (MoE-POT) as described in the paper. MoE-POT is a sparse-activated architecture designed for efficient and scalable pre-training on large-scale PDE datasets.

## Codebase Structure

- `model.py`: Contains the main model architecture, including the MoE layer, Fourier layer, and other components.
- `modules.py`: Implements reusable modules such as the router-gating network and expert networks.
- `layers.py`: Defines low-level neural network layers used in the model.
- `train.py`: Implements the training loop, including loss functions, metrics, and optimization strategies.
- `data.py`: Handles dataset loading, preprocessing, and augmentation.
- `config.py`: Contains all hyperparameters and configuration settings.
- `requirements.txt`: Lists all dependencies required to run the code.
- `README.md`: Provides an overview of the project and instructions for usage.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the settings in `config.py` as needed.

4. Run the training script:
   ```bash
   python train.py
   ```

## Datasets

The code supports multiple PDE datasets, including FNO, PDEBench, and CFDBench. Ensure the datasets are preprocessed and placed in the appropriate directory as specified in `config.py`.

## Citation

If you use this code, please cite the original paper:

```
@article{wang2026moepot,
  title={Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training},
  author={Hong Wang and Haiyang Xim and Jie Wang and others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```