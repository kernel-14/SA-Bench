# PEFT Study for Visual Recognition

Reproduction of "Lessons and Insights from a Unifying Study of Parameter-Efficient Fine-Tuning (PEFT) in Visual Recognition" (Mai et al.).

This codebase implements a comprehensive empirical study of 14 PEFT methods for Vision Transformers, covering low-shot (VTAB-1K), many-shot, and robustness experiments.

## Codebase Structure

```
repo/
├── models/
│   ├── __init__.py
│   ├── vit.py          # Vision Transformer backbone (ViT-B/16)
│   ├── peft.py         # All 14 PEFT method implementations
│   └── peft_vit.py     # PEFT-wrapped ViT with all method integrations
├── data/
│   ├── __init__.py
│   ├── vtab.py         # VTAB-1K dataset loading
│   └── datasets.py     # Many-shot and robustness datasets
├── configs/
│   └── config.yaml     # All hyperparameters and configuration
├── train.py            # Training loops (VTAB-1K, many-shot, robustness)
├── evaluate.py         # Evaluation: WiSE, ensembles, prediction analysis
├── main.py             # Main entry point / CLI
├── requirements.txt    # Python dependencies
└── README.md
```

## PEFT Methods Implemented (14 total)

**Prompt-based:**
- VPT-Shallow - prepend prompts to first layer
- VPT-Deep - prepend prompts to every layer

**Adapter-based:**
- Pfeif. Adapter - adapter after MLP only
- Houl. Adapter - adapters after MSA and MLP
- AdaptFormer - parallel adapter after MLP
- RepAdapter - linear adapter with group-wise transform
- Convpass - convolutional adapter with visual inductive bias

**Direct Selective Tuning:**
- BitFit - bias terms only
- LayerNorm - LN parameters only
- DiffFit - BitFit + LayerNorm + scale factors
- SSF - scale & shift intermediate features

**Efficient Selective Tuning:**
- LoRA - low-rank adaptation (Q/V)
- FacT_TT - Tensor-Train decomposition
- FacT_TK - Tucker decomposition

## Baselines
- Linear Probing - train only prediction head
- Full Fine-Tuning - train all parameters

## Key Experiments

### 1. VTAB-1K (Low-Shot)
```bash
python main.py --mode vtab1k --method lora --dataset cifar100
python main.py --mode hp_sweep --method lora --dataset cifar100
```

### 2. Many-Shot
```bash
python main.py --mode many_shot --method lora --dataset cifar100
```

### 3. Robustness (CLIP ViT)
```bash
python main.py --mode robustness --method lora --backbone clip
python main.py --mode wise --method lora --checkpoint path/to/model.pt
```

### 4. Ensemble
```bash
python main.py --mode ensemble --checkpoints_dir path/to/checkpoints/
```

## Paper Highlights Reproduced
- Similar accuracy across PEFT methods on VTAB-1K after careful tuning
- Prediction diversity analysis (overlap matrices, confidence analysis)
- Ensemble methods (majority vote, logit averaging)
- WiSE (Weight-space Ensembles) for PEFT
- Drop path rate regularization analysis

## Configuration
All hyperparameters are in `configs/config.yaml`, including:
- ViT-B/16 architecture parameters
- PEFT method-specific hyperparameter grids
- Training settings for each experiment mode
- WiSE mixing coefficients
