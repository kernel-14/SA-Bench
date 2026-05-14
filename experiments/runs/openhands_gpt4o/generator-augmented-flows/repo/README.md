# Generator-Augmented Flows

This repository contains the implementation of the paper "Improving Consistency Models with Generator-Augmented Flows" by Thibaut Issenhuth et al. The codebase is structured to reproduce the experiments and results presented in the paper.

## Codebase Structure

- `model.py`: Contains the main model architecture for the consistency model and generator-augmented flows.
- `modules.py`: Defines reusable modules and components used in the model.
- `layers.py`: Implements custom layers required for the model.
- `train.py`: Implements the training loop, including generator-augmented coupling and consistency training.
- `data.py`: Handles dataset loading and preprocessing for CIFAR-10, ImageNet, CelebA, and LSUN Church.
- `config.py`: Stores all hyperparameters and configurations used in the experiments.
- `evaluation.py`: Implements evaluation metrics such as FID, KID, and IS.
- `requirements.txt`: Lists all dependencies required to run the code.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd generator-augmented-flows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the experiment settings in `config.py`.

4. Run training:
   ```bash
   python train.py
   ```

5. Evaluate the model:
   ```bash
   python evaluation.py
   ```

## Datasets

The following datasets are used in the experiments:
- CIFAR-10
- ImageNet (32x32 resolution)
- CelebA (64x64 resolution)
- LSUN Church (64x64 resolution)

Ensure the datasets are downloaded and preprocessed as described in `data.py`.

## Citation

If you use this code, please cite the original paper:
```
@article{issenhuth2026generatoraugmentedflows,
  title={Improving Consistency Models with Generator-Augmented Flows},
  author={Thibaut Issenhuth and Sangchul Lee and Ludovic Dos Santos and Jean-Yves Franceschi and Chansoo Kim and Alain Rakotomamonjy},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```