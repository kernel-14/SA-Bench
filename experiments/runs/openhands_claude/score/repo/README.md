# SCoRe: Self-Correction via Reinforcement Learning

Implementation of **"Training Language Models to Self-Correct via Reinforcement Learning"** (Kumar et al., 2024).

## Overview

SCoRe trains a single LLM to self-correct its own responses using multi-turn online RL on entirely self-generated data. It addresses two failure modes of SFT-based approaches:
1. **Distribution shift** — offline correction traces don't match the model's own error distribution
2. **Behavior collapse** — models learn to produce good first attempts and make no corrections

SCoRe uses a two-stage training procedure:
- **Stage I**: Optimize second-attempt accuracy while constraining first-attempt distribution to the base model via KL divergence
- **Stage II**: Jointly optimize both attempts with reward shaping that incentivizes self-correction progress

## Repository Structure

```
repo/
├── config.py          # All hyperparameters and experiment configurations
├── prompts.py         # All prompts and self-correction instructions from the paper
├── data.py            # Dataset loading for MATH, MBPP, HumanEval
├── rewards.py         # Reward functions: math answer checking, code test execution
├── model.py           # LLM policy wrapper with log-prob computation
├── train.py           # SCoRe Stage I and Stage II training loops
├── evaluate.py        # Evaluation metrics: Acc@t1, Acc@t2, Δ(t1,t2), Δ^(i→c), Δ^(c→i)
├── baselines/
│   ├── __init__.py
│   ├── star.py        # STaR: filter successful correction traces, run SFT
│   ├── pair_sft.py    # Pair-SFT: pair incorrect/correct responses, run SFT
│   └── self_refine.py # Self-Refine: prompting-based self-correction baseline
└── requirements.txt
```

## Method

### Stage I Objective (Equation 3)
Optimize second-attempt reward while keeping first-attempt close to base model:

```
max_θ E[r(y2, y*) - β2 * KL(π_θ(·|x1) || π_ref(·|x1))]
```

### Stage II Objective (Equation 4 + reward shaping)
Jointly optimize both attempts with shaped reward:

```
max_θ E[Σ r(yi, y*) - β1 * KL(π_θ(·|xi) || π_ref(·|xi))]
```

Shaped reward bonus at second attempt:
```
b(y2 | y1, y*) = α * (r(y2, y*) - r(y1, y*))
```

### Hyperparameters

| Hyperparameter | MATH | MBPP |
|---|---|---|
| Learning rate | 5e-6 | 1e-5 |
| Training steps | 3000 | 1500 |
| Batch size | 512 | 128 |
| α (reward shaping) | 10 | 10 |
| β1 (KL weight) | 0.01 | 0.01 |
| β2 (Stage I KL weight) | 0.1 | 0.25 |
| Temperature | 1.0 | 1.0 |

## Usage

### Training SCoRe on MATH
```bash
python train.py --task math --stage 1
python train.py --task math --stage 2 --stage1_checkpoint <path>
```

### Training SCoRe on MBPP
```bash
python train.py --task mbpp --stage 1
python train.py --task mbpp --stage 2 --stage1_checkpoint <path>
```

### Evaluation
```bash
python evaluate.py --task math --checkpoint <path>
python evaluate.py --task humaneval --checkpoint <path>
```

### Baselines
```bash
python baselines/star.py --task math
python baselines/pair_sft.py --task math
python baselines/self_refine.py --task math
```

## Datasets

- **MATH**: Following Lightman et al. (2023), augment training set with 4500 problems from test set; evaluate on remaining 500 (MATH500)
- **MBPP**: Train on MBPP, evaluate on HumanEval (zero-shot transfer)
- **MBPP-R**: Offline repair task from Ni et al. (2024)

## Metrics

- `Accuracy@t1`: First-attempt accuracy
- `Accuracy@t2`: Second-attempt accuracy  
- `Δ(t1, t2)`: Net improvement (Acc@t2 - Acc@t1)
- `Δ^(i→c)(t1, t2)`: Fraction of problems incorrect at t1 that become correct at t2
- `Δ^(c→i)(t1, t2)`: Fraction of problems correct at t1 that become incorrect at t2
