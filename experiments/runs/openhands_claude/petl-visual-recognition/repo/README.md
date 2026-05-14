# PEFT Visual Recognition: Unifying Study

Reproduction of "Lessons and Insights from a Unifying Study of Parameter-Efficient Fine-Tuning (PEFT) in Visual Recognition" (Mai et al., 2024).

## Overview

This codebase implements 14 PEFT methods for Vision Transformers and reproduces the paper's four main experiments:

1. **VTAB-1K low-shot** (Table 1, Figure 2): All 14 PEFT methods on 19 classification tasks
2. **Many-shot** (Figure 5): PEFT vs. full FT with varying parameter budgets
3. **Prediction diversity** (Figures 3, 4, 12, 13): Ensemble and similarity analysis
4. **Robustness** (Table 2, Figure 1c): CLIP fine-tuning + WiSE on distribution shifts

## Implemented PEFT Methods

| Category | Method | Params (M) |
|---|---|---|
| Prompt-based | VPT-Shallow | 0.07 |
| Prompt-based | VPT-Deep | 0.43 |
| Adapter-based | Houlsby Adapter | 0.77 |
| Adapter-based | Pfeiffer Adapter | 0.67 |
| Adapter-based | AdaptFormer | 0.46 |
| Adapter-based | ConvPass | 0.49 |
| Adapter-based | RepAdapter | 0.53 |
| Direct selective | BitFit | 0.10 |
| Direct selective | LayerNorm | 0.04 |
| Direct selective | DiffFit | 0.14 |
| Direct selective | SSF | 0.21 |
| Efficient selective | LoRA | 0.55 |
| Efficient selective | FacT-TT | 0.13 |
| Efficient selective | FacT-TK | 0.23 |

## Repository Structure

```
repo/
├── config.py           # All hyperparameters, search grids, dataset configs
├── data.py             # Dataset loading: VTAB-1K, CIFAR-100, RESISC45, ImageNet
├── train.py            # Training loop, hyperparameter search, experiment runners
├── evaluate.py         # Accuracy, prediction similarity, confidence overlap analysis
├── ensemble.py         # Majority vote and soft ensemble
├── wise.py             # Weight-Space Ensembles (WiSE) for PEFT
├── utils.py            # Shared utilities (AverageMeter, set_seed, etc.)
├── run_vtab.py         # Run VTAB-1K experiments (Table 1)
├── run_manyshot.py     # Run many-shot experiments (Figure 5)
├── run_robustness.py   # Run CLIP robustness experiments (Table 2, Figure 1c)
├── run_analysis.py     # Run prediction diversity analysis (Figures 3, 4)
├── models/
│   ├── vit.py          # ViT backbone wrapper + PEFTViT class
│   └── peft/
│       ├── vpt.py      # VPT-Shallow, VPT-Deep
│       ├── adapters.py # Houl., Pfeif., AdaptFormer, ConvPass, RepAdapter
│       ├── selective.py # BitFit, LayerNorm, DiffFit, SSF
│       ├── lora.py     # LoRA
│       └── fact.py     # FacT-TT, FacT-TK
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### VTAB-1K Experiments (Table 1)

```bash
# Run all methods on all 19 tasks
python run_vtab.py --data_dir ./data --output_dir ./outputs/vtab --device cuda

# Single method/task
python run_vtab.py --methods lora --tasks cifar100 dtd --device cuda

# Skip hyperparameter search (use defaults)
python run_vtab.py --no_hparam_search --device cuda
```

### Many-Shot Experiments (Figure 5)

```bash
# Run on CIFAR-100, RESISC45, Clevr-Distance
python run_manyshot.py --data_dir ./data --output_dir ./outputs/manyshot --device cuda

# With parameter size sweep
python run_manyshot.py --param_sweep --methods lora adaptformer --device cuda
```

### Robustness Experiments (Table 2, Figure 1c)

```bash
python run_robustness.py \
    --imagenet_root /path/to/imagenet \
    --imagenet_v2 /path/to/imagenet-v2 \
    --imagenet_r /path/to/imagenet-r \
    --imagenet_s /path/to/imagenet-s \
    --imagenet_a /path/to/imagenet-a \
    --output_dir ./outputs/robustness \
    --device cuda
```

### Prediction Diversity Analysis (Figures 3, 4)

```bash
# Requires trained model checkpoints from VTAB-1K run
python run_analysis.py \
    --checkpoint_dir ./outputs/vtab \
    --data_dir ./data/vtab \
    --tasks dtd retinopathy dmlab \
    --output_dir ./outputs/analysis
```

## Key Experimental Settings

### VTAB-1K (Low-Shot)
- Backbone: ViT-B/16 pretrained on ImageNet-21K
- Training: 100 epochs, AdamW, cosine LR decay, batch size 64
- LR search: {1e-3, 1e-2}, WD search: {1e-4, 1e-3}
- Drop path rate: 0 or 0.1 (paper recommends 0.1)
- No data augmentation
- PEFT parameter cap: ≤ 1.5% of ViT-B/16 (~1.29M params)

### Many-Shot
- Training: 40 epochs, AdamW, cosine LR decay, batch size 64
- LR search: {5e-4, 1e-3}, WD search: {1e-4, 1e-3}
- Augmentation: horizontal flip (CIFAR-100), h+v flip (RESISC45), none (Clevr)

### Robustness (CLIP)
- Backbone: CLIP ViT-B/16
- Training: 100 epochs, LR=3e-5, WD=5e-3, strong augmentation
- 100-shot ImageNet (100 images/class)
- WiSE α sweep: [0.0, 0.1, ..., 1.0]

## Key Findings Reproduced

1. All PEFT methods achieve similar accuracy on VTAB-1K when properly tuned
2. Drop path rate (0.1) significantly improves all methods
3. Different PEFT methods make diverse predictions (~20-35% disagreement)
4. Majority vote ensemble consistently improves over worst method
5. PEFT methods preserve CLIP's distribution shift robustness better than full FT
6. WiSE is compatible with PEFT and further improves robustness

## Citation

```bibtex
@article{mai2024lessons,
  title={Lessons and Insights from a Unifying Study of Parameter-Efficient Fine-Tuning (PEFT) in Visual Recognition},
  author={Mai, Zheda and Zhang, Ping and Tu, Cheng-Hao and Chen, Hong-You and Nguyen, Quang-Huy and Zhang, Li and Chao, Wei-Lun},
  year={2024}
}
```
