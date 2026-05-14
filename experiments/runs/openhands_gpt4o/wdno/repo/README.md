# Wavelet Diffusion Neural Operator (WDNO)

This repository contains the implementation of the Wavelet Diffusion Neural Operator (WDNO) as described in the paper. WDNO is a novel framework for simulating and controlling physical systems governed by partial differential equations (PDEs).

## Codebase Structure

- `model.py`: Contains the implementation of the WDNO model, including the wavelet transform and diffusion model.
- `train.py`: Training script for the WDNO model.
- `data.py`: Dataset loading and preprocessing, including wavelet transforms.
- `config.py`: Configuration file containing hyperparameters and paths.
- `requirements.txt`: List of dependencies required to run the codebase.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your dataset and place it in the `data/` directory. Update the path in `config.py` if necessary.

3. Train the model:
   ```bash
   python train.py
   ```

4. The trained model will be saved in the `models/` directory.

## Dependencies

- PyTorch
- NumPy
- PyWavelets

## Citation

If you use this code, please cite the original paper.