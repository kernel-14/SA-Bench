# LoRA-SB: Initialization Using Update Approximation Is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning

Reproduction of [Ponkshe et al., 2024](https://arxiv.org/abs/2410.23111).

## Overview

LoRA-SB uses the LoRA-XS architecture (`W = W_0 + B·R·A`, with B and A frozen and only R trainable) but initializes B, R, A from the first-step gradient approximation of full fine-tuning. This gives:

- **Optimal low-rank approximation** of the initial full FT update (Eckart-Young theorem)
- **Orthonormal B and A** (`B^T B = I`, `A A^T = I`), enabling scaling-factor independence (`s = 1`)
- **Guaranteed loss reduction** at every step (Theorem 4)
- **27–90× fewer parameters** than LoRA while matching or exceeding its performance

## Repository Structure

```
repo/
├── src/
│   ├── lora_layers.py      # LoRA, rsLoRA, PiSSA, DoRA, LoRA-Pro, LoRA-XS, LoRA-SB layers
│   ├── initialization.py   # Gradient approximation + truncated SVD initialization
│   ├── model.py            # Model loading and PEFT application
│   ├── data.py             # Dataset loading (MetaMathQA, COMMONSENSE170K, GLUE)
│   ├── train.py            # Training loop with AdamW + LR scheduling
│   ├── evaluate.py         # Evaluation metrics for all benchmarks
│   └── utils.py            # Ablation helpers, logging, parameter counting
├── config/
│   ├── math_reasoning.yaml       # Mistral-7B / Gemma-2 9B on MetaMathQA
│   ├── commonsense_reasoning.yaml # Llama-3.2 3B on COMMONSENSE170K
│   └── glue.yaml                 # RoBERTa-large on GLUE
├── train_math.py           # Math reasoning entry point
├── train_commonsense.py    # Commonsense reasoning entry point
├── train_glue.py           # GLUE entry point
├── run_ablations.py        # Ablation experiments (Tables 4, 5; Figure 3)
└── requirements.txt
```

## Method

### LoRA-SB Initialization (Algorithm 1)

1. Accumulate gradients over `n` samples (0.1% of dataset, ~50 samples for MetaMathQA)
2. Compute `ΔW_avg = -sign(Σ ∇_W L(W_0, x_i))` — approximates AdamW first step
3. Truncated SVD: `U, S, V^T = SVD(ΔW_avg)`
4. Set `B_init = U[:, :r]`, `A_init = V[:r, :]`, `R_init = diag(S[:r])`
5. Freeze B and A; only R is trainable; scaling `s = 1`

### Key Properties

- `B^T B = A A^T = I` (orthonormal bases from SVD)
- Optimal gradient: `g^R = g^R_{LoRA-XS}` (no matrix inversions needed, Theorem 3 + 5)
- Scaling-factor independent (Theorem 5)
- Guaranteed `ΔL ≤ 0` (Theorem 4)

## Experiments

### Math Reasoning (Table 1)

```bash
# LoRA-SB on Mistral-7B, rank=96
python train_math.py --config config/math_reasoning.yaml \
    --model_name mistralai/Mistral-7B-v0.1 --method lora_sb --rank 96

# LoRA-XS baseline
python train_math.py --config config/math_reasoning.yaml \
    --method lora_xs --rank 32

# LoRA baseline
python train_math.py --config config/math_reasoning.yaml \
    --method lora --rank 32

# Gemma-2 9B
python train_math.py --config config/math_reasoning.yaml \
    --model_name google/gemma-2-9b --method lora_sb --rank 64
```

### Commonsense Reasoning (Table 2)

```bash
# LoRA-SB on Llama-3.2 3B, rank=96
python train_commonsense.py --config config/commonsense_reasoning.yaml \
    --method lora_sb --rank 96

# All baselines
for method in lora rslora pissa dora lora_pro lora_xs; do
    python train_commonsense.py --method $method --rank 32
done
```

### GLUE (Table 3)

```bash
# LoRA-SB on RoBERTa-large, rank=24
python train_glue.py --config config/glue.yaml --method lora_sb --rank 24

# LoRA baseline, rank=8
python train_glue.py --method lora --rank 8

# Specific tasks only
python train_glue.py --method lora_sb --rank 16 --tasks cola rte mrpc
```

### Ablations

```bash
# Table 4: Initialization strategies
python run_ablations.py --ablation init_strategy --output_dir outputs/ablations

# Table 5: Number of initialization samples
python run_ablations.py --ablation n_samples --output_dir outputs/ablations

# Figure 3: Optimal gradient approximation
python run_ablations.py --ablation grad_approx --output_dir outputs/ablations
```

## Baselines

| Method | Description |
|--------|-------------|
| `lora` | Standard LoRA (Hu et al., 2021) |
| `rslora` | Rank-stabilized LoRA, `s = α/√r` (Kalajdzievski, 2023) |
| `pissa` | PiSSA: initialized from principal SVD of W_0 (Meng et al., 2024) |
| `dora` | Weight-decomposed LoRA (Liu et al., 2024) |
| `lora_pro` | Optimal gradient approximation for LoRA (Wang et al., 2024) |
| `lora_xs` | LoRA-XS: frozen B, A from SVD of W_0; trainable R (Bałazy et al., 2024) |
| `lora_sb` | **LoRA-SB (ours)**: frozen B, A from SVD of ΔW_avg; trainable R |

## Hyperparameters

| Setting | Mistral-7B / Gemma-2 9B | Llama-3.2 3B | RoBERTa-large |
|---------|------------------------|--------------|---------------|
| Optimizer | AdamW | AdamW | AdamW |
| LR | 1e-4 | 2e-3 | 1e-3 |
| Batch size | 1 | 6 | 15–30 |
| Grad acc. | 32 | 24 | 1 |
| Epochs | 1 | 2 | 30 |
| Dropout | 0 | 0.05 | 0 |
| LR schedule | Cosine | Linear | Linear |
| Warmup ratio | 0.02 | 0.02 | 0.06 |
| Init samples | 50 (0.1%) | ~170 (0.1%) | 0.1% per task |

## Citation

```bibtex
@article{ponkshe2024lora,
  title={Initialization Using Update Approximation Is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning},
  author={Ponkshe, Kaustubh and Singhal, Raghav and Gorbunov, Eduard and Tumanov, Alexey and Horvath, Samuel and Vepakomma, Praneeth},
  journal={arXiv preprint arXiv:2410.23111},
  year={2024}
}
```
