# LoRA-SB: LoRA Silver Bullet

Reproduction of the paper "INITIALIZATION USING UPDATE APPROXIMATION IS A Silver Bullet FOR EXTREMELY EFFICIENT LOW-RANK FINE-TUNING" (Ponkshe et al., 2025).

## Overview

LoRA-SB bridges the gap between low-rank PEFT and full fine-tuning by:
1. Initializing B, A, R via truncated SVD of the first full-FT update
2. Fixing B and A as orthonormal bases while only training R (r × r)
3. Using optimal gradient approximation: g^R = g_{LoRA-XS}^R (when s=1)
4. Achieving scaling factor independence (no α tuning needed)

The method uses the LoRA-XS architecture (W = W0 + s·B·R·A) but with a carefully designed initialization that provably approximates full fine-tuning in low-rank subspaces.

## Codebase Structure

```
repo/
├── config.py      # All hyperparameters from the paper
├── model.py       # LoRA-SB, LoRA, LoRA-XS, rsLoRA, DoRA, LoRA-Pro, PiSSA
├── init.py        # LoRA-SB initialization (SVD of first AdamW step)
├── data.py        # Dataset loading (GLUE, MetaMathQA, COMMONSENSE170K)
├── train.py       # Training loops for all tasks
├── eval.py        # Evaluation on GSM8K, MATH, GLUE, commonsense
├── utils.py       # Helpers (seeding, counting, FLOPs estimation)
├── requirements.txt
└── README.md
```

## Key Algorithms

### Initialization (Algorithm 1 from paper)
1. Sample n data points from training set (0.1% = 50 samples)
2. Compute ΔW_avg = -η · sign(∑ ∇_W L(W0, xi))
3. Perform truncated SVD: U, S, V^T ← SVD(ΔW_avg)
4. Set B = U[:, :r], A = V^T[:r, :], R = diag(S[:r])

### Optimal Gradient Approximation (Theorem 3)
- Standard: g^R = 1/s² · (B^T B)^{-1} · g_{LoRA-XS}^R · (A A^T)^{-1}
- LoRA-SB (orthonormal B, A, s=1): g^R = g_{LoRA-XS}^R

## Supported Methods

| Method | Trainable Params | Gradient Approx | Description |
|--------|-----------------|-----------------|-------------|
| LoRA-SB | r² | Yes | Our method |
| LoRA-XS | r² | No | B, A fixed, R trainable |
| LoRA | r(m+n) | No | Standard LoRA |
| rsLoRA | r(m+n) | No | Rank-stabilized LoRA |
| DoRA | r(m+n)+n | No | Weight-decomposed LoRA |
| LoRA-Pro | r(m+n) | Yes | LoRA with gradient optimization |
| PiSSA | r(m+n) | No | SVD-init from pretrained weights |

## Usage

### GLUE Benchmark (RoBERTa-large)
```bash
python train.py --task_type glue --task_name mrpc --model_name roberta-large \
    --method lora_sb --rank 8 --learning_rate 1e-3 --num_epochs 30
```

### Arithmetic Reasoning (Mistral-7B / Gemma-2 9B)
```bash
python train.py --task_type math --model_name mistralai/Mistral-7B-v0.1 \
    --method lora_sb --rank 32 --learning_rate 1e-4 --num_epochs 1 \
    --batch_size 1 --gradient_accumulation_steps 32
```

### Commonsense Reasoning (Llama-3.2 3B)
```bash
python train.py --task_type commonsense --model_name meta-llama/Llama-3.2-3B \
    --method lora_sb --rank 32 --learning_rate 2e-3 --num_epochs 2 \
    --batch_size 6 --gradient_accumulation_steps 24
```

## Hyperparameters

### LoRA-SB specific
- `init_samples`: Number of samples for initialization (default: 50 = 0.1%)
- `rank`: Low-rank dimension r
- `scaling`: Fixed at 1.0 (no α tuning needed)

### From the paper
| Config | RoBERTa-large (GLUE) | Mistral-7B/Gemma-2 (Math) | Llama-3.2 3B (Commonsense) |
|--------|---------------------|--------------------------|---------------------------|
| Optimizer | AdamW | AdamW | AdamW |
| Batch size | 30/128 | 1 | 6 |
| Grad acc | 1 | 32 | 24 |
| Epochs | 30 | 1 | 2 |
| LR | 1e-3 | 1e-4 | 2e-3 |
| LR scheduler | Linear | Cosine | Linear |
| Warmup ratio | 0.06 | 0.02 | 0.02 |
| Dropout | 0 | 0 | 0.05 |
| Max seq len | 256/512 | 512 | 256 |

## Key Theoretical Results

- **Lemma 1**: LoRA-XS constrains updates to Col(B) and Row(A)
- **Lemma 2**: g_{LoRA-XS}^R = s · B^T · g · A^T
- **Theorem 3**: Closed-form optimal g^R that minimizes ||g̃ - g||_F
- **Theorem 4**: Loss reduction ΔL ≤ 0 guaranteed with optimal g^R
- **Theorem 5**: Scaling factor independence when using optimal gradient approximation
- **Theorem 6**: First update is optimal rank-r approximation of full-FT update (SGD)
