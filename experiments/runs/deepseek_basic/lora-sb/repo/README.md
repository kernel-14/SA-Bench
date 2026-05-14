# LoRA-SB: Initialization using Update Approximation for Efficient Low-Rank Fine-Tuning

This repository implements **LoRA-SB** (LoRA Silver Bullet), a parameter-efficient fine-tuning method that approximates full fine-tuning within low-rank subspaces using a carefully designed initialization strategy.

## Paper Reference

> **LoRA-SB: Initialization using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning**
> Kaustubh Ponkshe, Raghav Singhal, Eduard Gorbunov, Alexey Tumanov, Samuel Horvath, Praneeth Vepakomma
> Mohamed bin Zayed University of Artificial Intelligence, Georgia Institute of Technology, MIT

## Core Contributions Reproduced

### 1. LoRA-SB Architecture (`lora_sb/lora_sb_layer.py`)

Implements `W = W_0 + s B R A` where:
- `B` (m×r): frozen after initialization
- `A` (r×n): frozen after initialization  
- `R` (r×r): **only trainable matrix**
- `s`: scaling factor (can be 1.0 with orthonormal B, A)

This uses the LoRA-XS architecture but with our novel initialization.

### 2. Initialization via Update Approximation (`lora_sb/init.py`)

The key innovation - approximates the first step of full fine-tuning:
1. Compute `ΔW_avg = -η · sign(Σ ∇_W L(W_0, x_i))` over ~0.1% of training data
2. Apply truncated SVD: `U, S, V^T = SVD(ΔW_avg)`
3. Initialize: `B = U[:, :r]`, `A = V[:r, :]`, `R = S[:r, :r] / s`

This provides:
- **Optimal rank-r approximation** of the first full FT update (Eckart-Young theorem)
- **Orthonormal bases**: `B^T B = I` and `A A^T = I`
- **Scaling factor independence**: `s` can be set to 1
- **Guaranteed loss reduction** at each step (Theorem 4)

### 3. Gradient Optimization (`lora_sb/gradient_opt.py`)

Implements the closed-form optimal gradient from Theorem 3:
```
g^R = (1/s²) (B^T B)^{-1} g_LoRA-XS^R (A A^T)^{-1}
```
With orthonormal initialization, this simplifies to `g^R = g_LoRA-XS^R` (no matrix inversion needed).

### 4. Memory-Efficient Initialization

Layerwise gradient computation using backward hooks ensures O(1) memory usage independent of layer count (Section 2.6).

### 5. Training Scripts

- `run_glue.py`: Fine-tuning RoBERTa-large on GLUE benchmark (Section 3.3)
- `run_math.py`: Fine-tuning LLMs on math/commonsense reasoning (Sections 3.1, 3.2)

### 6. Theoretical Verification (`tests/test_lora_sb.py`)

Tests validating the core theorems:
- **Lemma 2**: `g_R = s B^T g A^T` gradient relationship
- **Theorem 3**: Optimal gradient approximation
- **Theorem 5**: Scaling factor independence
- **Theorem 6**: Optimal first-step approximation
- Orthonormality preservation
- Parameter reduction verification

## Repository Structure

```
├── lora_sb/
│   ├── __init__.py           # Package entry point
│   ├── lora_sb_layer.py      # LoRA-SB layer implementation (W = W0 + s B R A)
│   ├── init.py               # Initialization via SVD of first full-FT step approximation
│   ├── gradient_opt.py       # Optimal gradient computation (Theorem 3)
│   └── train.py              # Training utilities (optimizer, train loop, merge)
├── tests/
│   └── test_lora_sb.py       # Unit tests for theoretical properties
├── run_glue.py               # GLUE fine-tuning script (RoBERTa-large)
├── run_math.py               # Math/commonsense fine-tuning script (Mistral/LLaMA/Gemma)
└── README.md                 # This file
```

## Usage

### Basic Usage

```python
from lora_sb import init_lora_sb
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader

# Load model
model = AutoModelForSequenceClassification.from_pretrained('roberta-large')

# Prepare dataloader (even a small subset for initialization)
train_loader = DataLoader(train_dataset, batch_size=8)

# Initialize LoRA-SB
model = init_lora_sb(
    model,
    dataloader=train_loader,
    rank=8,
    num_init_samples=50,     # 0.1% of training data
    scaling=1.0,             # s=1 with orthonormal B, A
    target_modules=['query', 'key', 'value'],  # RoBERTa self-attention
)

# Train (only R is trainable)
from lora_sb.train import get_lora_sb_optimizer, train_epoch
optimizer = get_lora_sb_optimizer(model, lr=1e-3)
train_epoch(model, train_loader, optimizer)

# Merge for inference
from lora_sb.train import merge_and_save
merge_and_save(model, 'merged_model.pt')
```

### Running Experiments

**GLUE (RoBERTa-large):**
```bash
python run_glue.py --task sst2 --rank 8 --lr 1e-3 --epochs 30
```

**Math Reasoning (Mistral-7B):**
```bash
python run_math.py --model_name mistralai/Mistral-7B-v0.1 --dataset metamath --rank 32 --lr 1e-4
```

### Running Tests
```bash
python tests/test_lora_sb.py
```

## Key Properties

### Parameter Efficiency
- LoRA: `r(m+n)` trainable parameters
- LoRA-SB: `r²` trainable parameters
- Reduction: ~27-90x fewer parameters than LoRA at competitive performance

### Scaling Factor Independence
With orthonormal B and A from SVD initialization, the scaling factor `s` can be set to 1, eliminating the need for α hyperparameter tuning (Theorem 5).

### Optimal Gradient Approximation
The equivalent gradient `\tilde{g} = s B g^R A` optimally approximates the full fine-tuning gradient `g` at each step, without requiring matrix inversions during training.

### Guaranteed Convergence
With orthonormal B and A, the loss reduction property (Theorem 4) holds throughout training since B and A remain full-rank.

## Assumptions and Unresolved Details

1. **Dataset-specific configurations**: The paper specifies different hyperparameters for different models/tasks (Tables 8, 9). These are encoded in the training scripts but may need adjustment for specific setups.

2. **Full gradient computation**: The paper mentions computing `ΔW_avg` using the sign of summed gradients (AdamW first-step approximation). The implementation follows this but the exact behavior depends on the batch composition.

3. **COMMONSENSE170K dataset**: The paper uses a consolidated dataset of 8 commonsense reasoning tasks. Our implementation expects the HuggingFace `commonsense_qa` dataset as a proxy; the full COMMONSENSE170K may need custom dataset loading.

4. **Evaluation metrics**: The paper reports accuracy, Matthews correlation, and Pearson correlation for GLUE tasks. The current training scripts compute loss during training; full evaluation metric computation (e.g., using `evaluate` library) can be added.

5. **Multiple random seeds**: The paper reports results averaged over 3 random seeds. The implementation supports seeding but users should run multiple seeds for reported-quality results.

6. **Low-rank SVD**: For very large weight matrices, `torch.svd_lowrank` is used; the paper mentions this takes <1 second per entire LLM.

## Dependencies

- PyTorch >= 1.13
- Transformers (HuggingFace)
- Datasets (HuggingFace)
- (Optional) Accelerate for model parallelism

## Theoretical Results Verified

| Theorem/Lemma | Description | Status |
|--------------|-------------|--------|
| Lemma 1 | Constrained update space of LoRA-XS | Verified |
| Lemma 2 | Gradient relationship g_R = s B^T g A^T | Verified (test) |
| Theorem 3 | Optimal gradient closed-form solution | Verified (test) |
| Theorem 4 | Guaranteed loss reduction | Implementation supports |
| Theorem 5 | Scaling factor independence | Verified (test) |
| Theorem 6 | Optimal first-step approximation | Verified (test) |

## References

- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- LoRA-XS: [Bałazy et al., 2024](https://arxiv.org/abs/2405.17604)
- PiSSA: [Meng et al., 2024](https://arxiv.org/abs/2404.02948)
- LoRA-Pro: [Wang et al., 2024](https://arxiv.org/abs/2407.18242)
- rsLoRA: [Kalajdzievski, 2023](https://arxiv.org/abs/2312.03732)
