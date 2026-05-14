# PETL in Visual Recognition: A Unifying Empirical Study (Reproduction)

This repository contains a reproduction of the paper "Lessons Learned from a Unifying Empirical Study of PETL in Visual Recognition" by Mai et al.

## What was successfully reproduced

### Core Methods Implemented
All 14 PEFT methods described in the paper are implemented:
- **Prompt-based**: VPT-Shallow, VPT-Deep
- **Adapter-based**: Houlsby Adapter, Pfeiffer Adapter, AdaptFormer, ConvPass, RepAdapter
- **Direct Selective Tuning**: BitFit, LayerNorm, DiffFit, SSF
- **Efficient Selective Tuning**: LoRA, FacT_TT, FacT_TK

Additionally: Linear Probing and Full Fine-Tuning baselines are implemented.

### Evaluation Framework
- VTAB-1K low-shot evaluation (19 tasks, 3 groups)
- Many-shot evaluation (CIFAR-100, RESISC, Clevr-Distance)
- Robustness evaluation with CLIP backbone (ImageNet + 4 distribution shift datasets)
- WiSE (Weight-Space Ensembles) implementation for PEFT methods

### Analysis Tools
- Prediction similarity analysis
- Ensemble methods (majority vote)
- Confidence-based prediction overlap analysis

## Codebase Structure

```
├── methods/                  # PEFT method implementations
│   ├── __init__.py
│   ├── prompt_based.py       # VPT-Shallow, VPT-Deep
│   ├── adapter_based.py      # Houlsby, Pfeiffer, AdaptFormer, ConvPass, RepAdapter
│   ├── direct_selective.py   # BitFit, LayerNorm, DiffFit
│   ├── efficient_selective.py # LoRA, FacT_TT, FacT_TK
│   └── ssf.py                # SSF (Scale & Shift Features)
├── utils/                    # Utility functions
│   ├── data.py               # Dataset loading (VTAB-1K, many-shot, robustness)
│   ├── training.py           # Training loop, hyperparameter search
│   ├── evaluation.py         # Evaluation utilities
│   └── wise.py               # Weight-space ensembles for PEFT
├── configs/                  # Configuration files
│   ├── vtab1k.yaml           # VTAB-1K experiment config
│   ├── many_shot.yaml        # Many-shot experiment config
│   └── robustness.yaml       # Robustness experiment config
├── scripts/                  # Training and evaluation scripts
│   ├── train_vtab1k.py       # VTAB-1K training script
│   ├── train_many_shot.py    # Many-shot training script
│   ├── train_robustness.py   # Robustness training script
│   └── run_analysis.py       # Analysis script
├── analysis/                 # Analysis and plotting
│   ├── prediction_similarity.py
│   ├── ensemble.py
│   ├── ranking.py
│   └── plot_utils.py
└── README.md
```

## Implementation Details

### Backbone
- ViT-B/16 pre-trained on ImageNet-21K (via timm or HuggingFace transformers)
- CLIP ViT-B/16 for robustness experiments (via open_clip)

### Hyperparameter Tuning
Following the paper, we systematically tune:
- Learning rate: [1e-3, 1e-2] for VTAB-1K; [5e-4, 1e-3] for many-shot
- Weight decay: [1e-4, 1e-3]
- Drop path rate: 0.1 (on) or 0 (off)
- Method-specific hyperparameters as described in Table 3 of the paper

### Key Assumptions & Missing Details
- The exact 80/20 train-val split seeds for VTAB-1K are not specified; we follow standard practice
- The paper mentions a cap for PEFT parameters ≤ 1.5% of ViT-B/16 (~1.3M parameters)
- For ConvPass, the kernel size is 3x3 (as described in paper)
- For RepAdapter, group-wise transformation details follow the paper description
- AdamW optimizer with cosine decay scheduler, batch size 64, 100 epochs for VTAB-1K

## Usage

```bash
# VTAB-1K training
python scripts/train_vtab1k.py --dataset caltech101 --method lora

# Many-shot training
python scripts/train_many_shot.py --dataset cifar100 --method ssf

# Robustness training
python scripts/train_robustness.py --method bitfit

# Analysis
python scripts/run_analysis.py --results_dir ./results
```
