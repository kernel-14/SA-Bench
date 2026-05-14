# OLMoE Reproduction Codebase

This repository contains the implementation of the OLMoE model as described in the paper. The codebase is structured to facilitate easy reproduction of the experiments and results.

## Codebase Structure

- `model.py`: Defines the main OLMoE model architecture.
- `modules.py`: Implements reusable modules like the Mixture-of-Experts (MoE) layer.
- `layers.py`: Contains individual neural network layers, including the Transformer layer.
- `train.py`: Implements the training loop and integrates all components.
- `data.py`: Handles dataset loading and preprocessing.
- `config.py`: Stores all hyperparameters and configurations.
- `requirements.txt`: Lists all dependencies required to run the code.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the dataset and update the `data_path` in `config.py`.

3. Train the model:
   ```bash
   python train.py
   ```

## Notes

- Ensure that the dataset is preprocessed and stored in the correct format.
- The model is designed to run on both CPU and GPU. For optimal performance, use a GPU.

## License

This codebase is open-source and available under the Apache 2.0 License.