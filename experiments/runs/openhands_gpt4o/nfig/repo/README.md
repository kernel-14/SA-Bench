# NFIG: Next-Frequency Image Generation

This repository contains the implementation of the NFIG framework for autoregressive image generation, as described in the paper "Multi-Scale Autoregressive Image Generation via Frequency Ordering."

## Codebase Structure

- `model.py`: Contains the implementation of the FR-VAE and TransformerAR models.
- `train.py`: Training script for the NFIG framework.
- `data.py`: Dataset loading and preprocessing for ImageNet.
- `config.py`: Configuration file with hyperparameters and paths.
- `requirements.txt`: List of dependencies.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the ImageNet dataset and update the `data_path` in `config.py`.

3. Train the model:
   ```bash
   python train.py
   ```

## Citation

If you use this code, please cite the original paper:

```
@article{huang2026nfig,
  title={Multi-Scale Autoregressive Image Generation via Frequency Ordering},
  author={Huang, Zhihao and others},
  journal={NeurIPS},
  year={2026}
}
```