# Improving Consistency Models with Generator-Augmented Flows

Reproduction of experiments from *"Improving Consistency Models with Generator-Augmented Flows"* by Issenhuth et al.

## Codebase Structure

| File | Description |
|------|-------------|
| `model.py` | ConsistencyModel with SongUNet (NCSN++) backbone, EMA helper |
| `modules.py` | Building blocks: ResBlock, Attention, Conv2d, GroupNorm, embeddings |
| `coupling.py` | Data-noise coupling strategies: IC, batch-OT, Generator-Augmented (GC) |
| `loss.py` | Loss functions: CT, CD, GC, joint IC/GC training with parameter μ |
| `data.py` | Dataset loading for CIFAR-10, ImageNet, CelebA, LSUN Church |
| `metrics.py` | FID, KID, IS evaluation using TorchMetrics |
| `train.py` | Training loop with discretization schedule, Lion optimizer, mixed precision |
| `config.py` | Configuration dataclasses with dataset-specific hyperparameters |

## Training

```bash
# Standard IC training on CIFAR-10
python train.py --dataset cifar10 --coupling ic

# GC training with joint learning (μ = 0.5) on CIFAR-10
python train.py --dataset cifar10 --coupling gc --mu 0.5

# batch-OT training on CIFAR-10
python train.py --dataset cifar10 --coupling ot

# GC training on CelebA-64
python train.py --dataset celeba --coupling gc --mu 0.5

# ImageNet-32
python train.py --dataset imagenet --coupling gc --mu 0.5
```

## Key Hyperparameters

| Parameter | CIFAR-10 | ImageNet-32 | CelebA-64 | LSUN-64 |
|-----------|----------|-------------|-----------|---------|
| Batch size | 512 | 512 | 128 | 128 |
| Total steps | 100k | 150k | 150k | 150k |
| Learning rate | 1e-4 | 8e-5 | 8e-5 | 8e-5 |
| Model channels | 128 | 128 | 128 | 128 |
| μ (GC joint) | 0.5 | 0.5 | 0.5 | 0.5 |
