# Neural Operator Flow Matching for Generative PDE Foundation Model

This repository contains the implementation of the paper "Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model" by Zituo Chen and Sili Deng. The codebase is structured to reproduce the experiments and results presented in the paper.

## Codebase Structure

- `model.py`: Contains the implementation of the neural network models, including P2VAE and FMT.
- `modules.py`: Implements reusable modules and components for the models.
- `layers.py`: Defines custom layers used in the models.
- `train.py`: Implements the training loop and evaluation logic.
- `data.py`: Handles dataset loading and preprocessing.
- `config.py`: Contains all hyperparameters and configuration settings.
- `requirements.txt`: Lists all dependencies required to run the code.
- `README.md`: Provides an overview of the repository and instructions for usage.

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

3. Prepare the dataset:
   Follow the instructions in `data.py` to download and preprocess the datasets.

4. Train the model:
   ```bash
   python train.py --config config.py
   ```

5. Evaluate the model:
   Use the evaluation scripts provided in `train.py` to reproduce the results from the paper.

## Citation

If you use this code, please cite the paper:

```
@article{chen2026bridging,
  title={Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model},
  author={Chen, Zituo and Deng, Sili},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.