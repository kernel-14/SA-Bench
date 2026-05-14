# MoE-POT: Mixture-of-Experts Pre-training Operator Transformer

Reproduction of "Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training"
by Hong Wang et al.

## Code Structure

- `config.py` — All hyperparameters for Tiny/Small/Medium model variants, training configs
- `layers.py` — CNN expert networks, router-gating network, patchification, temporal aggregation
- `modules.py` — Fourier layer (multi-head kernel integral) and MoE layer
- `model.py` — Full MoE-POT model with auto-regressive prediction
- `data.py` — Dataset loading, preprocessing (padding, masking, noise injection), balanced sampling
- `train.py` — Pre-training, fine-tuning, and evaluation loops

## Usage

Pre-training:
```bash
python train.py --mode pretrain --config config.py
```

Fine-tuning:
```bash
python train.py --mode finetune --config config.py --dataset NS_1e-5
```

## Requirements

See `requirements.txt`.
