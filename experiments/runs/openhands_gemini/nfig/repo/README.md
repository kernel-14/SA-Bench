
# NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering

This repository contains a faithful reproduction of the "NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering" paper.

## Project Structure
- `config.py`: Configuration and hyperparameters.
- `data.py`: Dataset loading and preprocessing utilities for ImageNet.
- `model.py`: Defines the overall NFIG model, including FR-VAE and the autoregressive Transformer.
- `modules.py`: Contains individual neural network modules like Encoder, Decoder, Quantizer, etc.
- `layers.py`: Defines custom layers used in the models (e.g., frequency masks, FFT/IFFT operations).
- `train.py`: The main script for training the NFIG model.
- `requirements.txt`: Python dependencies.
