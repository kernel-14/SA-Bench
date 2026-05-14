# Reproduction: Lessons Learned from a Unifying Empirical Study of PETL in Visual Recognition

This repository reproduces the key contributions of the paper "Lessons and Insights from a Unifying Study of Parameter-Efficient Fine-Tuning (PEFT) in Visual Recognition" by Mai et al.

## Overview

The paper conducts a systematic empirical study of 14 PEFT methods for Vision Transformers (ViT) on visual recognition tasks. Key contributions:

1. **Fair comparison**: All methods are carefully tuned with the same hyperparameter search protocol
2. **Prediction diversity**: Despite similar accuracy, PEFT methods make different predictions
3. **Many-shot evaluation**: PEFT is effective even with abundant training data
4. **Robustness**: PEFT better preserves CLIP's distribution shift robustness vs full fine-tuning
5. **WiSE for PEFT**: First study of weight-space ensembles applied to PEFT methods

## Repository Structure

```
├── src/
│   ├── models/
│   │   └── vit.py              # ViT backbone utilities
│   ├── methods/
│   │   ├── bitfit.py           # BitFit: bias-only tuning
│   │   ├── layernorm.py        # LayerNorm tuning
│   │   ├── difffit.py          # DiffFit: bias + LN + scale factors
│   │   ├── ssf.py              # SSF: Scale & Shift Features
│   │   ├── vpt.py              # VPT: Visual Prompt Tuning (Shallow & Deep)
│   │   ├── adapter.py          # Houlsby & Pfeiffer Adapters
│   │   ├── adaptformer.py      # AdaptFormer: parallel adapter
│   │   ├── convpass.py         # ConvPass: convolutional bypass
│   │   ├── repadapter.py       # RepAdapter: re-parameterizable adapter
│   │   ├── lora.py             # LoRA: Low-Rank Adaptation
│   │   └── fact.py             # FacT: Factor Tuning (TT & TK)
│   ├── datasets/
│   │   ├── vtab.py             # VTAB-1K dataset utilities
│   │   ├── manyshot.py         # Many-shot datasets (CIFAR-100, RESISC45, Clevr)
│   │   └── imagenet.py         # ImageNet + distribution shift datasets
│   └── utils/
│       ├── trainer.py          # Training loop with AdamW + cosine LR
│       ├── evaluator.py        # Evaluation, ensemble, prediction similarity
│       └── wise.py             # WiSE (Weight-Space Ensembles) for PEFT
├── train_vtab.py               # Main VTAB-1K training script
├── train_manyshot.py           # Many-shot training script
├── train_robustness.py         # Robustness evaluation with CLIP
├── analyze_predictions.py      # Prediction diversity analysis
├── hyperparameter_search.py    # Grid search for hyperparameters
└── requirements.txt
```

## Implemented PEFT Methods

### Direct Selective Tuning
- **BitFit**: Tunes only bias terms (~0.1M params)
- **LayerNorm**: Tunes only LayerNorm parameters (~0.04M params)
- **DiffFit**: BitFit + LayerNorm + learnable scale factors (~0.14M params)
- **SSF**: Scale & Shift intermediate features (~0.21M params)

### Prompt-based
- **VPT-Shallow**: Learnable prompts at first layer only
- **VPT-Deep**: Learnable prompts at every layer

### Adapter-based
- **Houlsby Adapter**: Sequential adapters after MSA and MLP
- **Pfeiffer Adapter**: Sequential adapter after MLP only
- **AdaptFormer**: Parallel adapter with MLP
- **ConvPass**: Convolutional bypass in parallel with MSA and MLP
- **RepAdapter**: Linear adapter with group-wise transformation

### Efficient Selective Tuning
- **LoRA**: Low-rank decomposition for Q and V projections
- **FacT-TT**: Tensor-Train decomposition across all layers
- **FacT-TK**: Tucker decomposition across all layers

## Experimental Setup

### VTAB-1K (Low-shot)
- **Backbone**: ViT-B/16 pretrained on ImageNet-21K
- **Optimizer**: AdamW with cosine decay LR scheduler
- **Epochs**: 100
- **Batch size**: 64
- **LR search**: [1e-3, 1e-2]
- **Weight decay search**: [1e-4, 1e-3]
- **Drop path rate**: 0.1 (key finding: this significantly helps!)
- **No data augmentation** (following original VTAB-1K paper)
- **PEFT size cap**: ≤1.5% of ViT-B/16 parameters

### Many-shot
- **Datasets**: CIFAR-100, RESISC45, Clevr-Distance
- **Epochs**: 40
- **Data augmentation**: Horizontal flip for CIFAR-100, H+V flip for RESISC45, none for Clevr

### Robustness (CLIP)
- **Backbone**: CLIP ViT-B/16
- **Target**: ImageNet-1K (100 shots/class)
- **Distribution shifts**: ImageNet-V2, ImageNet-R, ImageNet-S, ImageNet-A
- **LR**: 3e-5, **Weight decay**: 5e-3
- **WiSE**: Linear interpolation between fine-tuned and pretrained models

## Usage

### VTAB-1K Experiments

```bash
# Train a single method on a single dataset
python train_vtab.py --method bitfit --dataset caltech101 --data_dir /path/to/vtab

# Train all methods on all datasets
python train_vtab.py --method all --dataset all --data_dir /path/to/vtab

# With specific hyperparameters
python train_vtab.py --method lora --dataset dtd --lr 1e-3 --rank 8 --drop_path_rate 0.1
```

### Hyperparameter Search

```bash
python hyperparameter_search.py --method bitfit --dataset caltech101 --data_dir /path/to/vtab
```

### Many-shot Experiments

```bash
python train_manyshot.py --method all --dataset all --data_dir /path/to/data
```

### Robustness Evaluation

```bash
# Train and evaluate with WiSE
python train_robustness.py --method all --data_dir /path/to/imagenet --wise
```

### Prediction Diversity Analysis

```bash
python analyze_predictions.py --checkpoints_dir ./output --data_dir /path/to/vtab --dataset cifar100
```

## Key Findings Reproduced

1. **Similar accuracy**: With proper hyperparameter tuning, all PEFT methods achieve similar accuracy on VTAB-1K, including simple methods like BitFit
2. **Drop path rate**: Setting drop_path_rate=0.1 (non-zero) significantly improves all methods
3. **Diverse predictions**: Despite similar accuracy, methods make different predictions (~20% difference in DTD/Retinopathy, ~35% in DMLab)
4. **Ensemble gains**: Majority vote ensemble consistently improves over individual methods
5. **Many-shot**: PEFT with 2-5% parameters achieves comparable accuracy to full fine-tuning
6. **Robustness**: PEFT better preserves CLIP's distribution shift robustness than full fine-tuning
7. **WiSE for PEFT**: WiSE improves both target and distribution shift accuracy for PEFT methods

## Data Preparation

### VTAB-1K
The VTAB-1K dataset should be organized as:
```
vtab_data/
├── caltech101/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   └── images/
├── dtd/
│   └── ...
└── ...
```

Each split file contains lines of format: `image_path label`

### Many-shot Datasets
- CIFAR-100: Downloaded automatically via torchvision
- RESISC45: Download from https://onedrive.live.com/?authkey=AHHjFGnFHnSFqYA&id=5C5E061130630A68!107&cid=5C5E061130630A68
- Clevr-Distance: Part of VTAB-1K

### ImageNet + Distribution Shifts
```
imagenet_data/
├── train/
├── val/
├── imagenet-v2/
├── imagenet-r/
├── imagenet-sketch/
└── imagenet-a/
```

## Assumptions and Limitations

1. **VTAB-1K data format**: We assume a specific file format for VTAB-1K. The original VTAB-1K uses TensorFlow datasets; conversion to PyTorch format may be needed.

2. **CLIP model**: The robustness experiments require the CLIP package. We provide a fallback to timm's CLIP-pretrained ViT if CLIP is not available.

3. **Zero-shot head initialization**: The paper initializes the classification head using CLIP's text encoder embeddings. Our implementation uses a random head as a simplification.

4. **FacT implementation**: The FacT method involves complex tensor decomposition. Our implementation is a faithful reproduction of the mathematical formulation but may differ in minor implementation details.

5. **Hyperparameter search**: The paper performs extensive hyperparameter search. Our scripts support this but it requires significant compute.

6. **Drop path rate**: The paper uses timm's built-in drop path (stochastic depth). We use drop_path_rate=0.1 as the recommended setting.

## Citation

```bibtex
@article{mai2024lessons,
  title={Lessons and Insights from a Unifying Study of Parameter-Efficient Fine-Tuning (PEFT) in Visual Recognition},
  author={Mai, Zheda and Zhang, Ping and Tu, Cheng-Hao and Chen, Hong-You and Nguyen, Quang-Huy and Zhang, Li and Chao, Wei-Lun},
  year={2024}
}
```
