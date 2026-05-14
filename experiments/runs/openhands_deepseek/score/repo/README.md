# SCoRe: Training Language Models to Self-Correct via Reinforcement Learning

Reproduction of the paper "Training Language Models to Self-Correct via Reinforcement Learning" (Kumar et al., Google DeepMind).

## Overview

SCoRe is a multi-turn online reinforcement learning approach that trains LLMs to self-correct their own mistakes without external feedback. The method uses:

1. **Stage I**: RL training that maximizes second-attempt reward while constraining the first-turn distribution to be close to the base model via KL divergence penalty.
2. **Stage II**: Joint optimization of both attempts with reward shaping: `b(y2|y1) = α · (r(y2) - r(y1))` to reward progress towards self-correction.

## Codebase Structure

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters from Table 5, including SCoRe, STaR, Pair-SFT configurations |
| `prompts.py` | Zero-shot and self-correction prompts from Appendix C |
| `data.py` | Dataset loading for MATH500, MBPP, HumanEval, MBPP-R |
| `models.py` | LLM policy wrapper with REINFORCE/Stage I/Stage II loss computation |
| `rewards.py` | Binary correctness reward functions (math answer checking, code test execution) |
| `metrics.py` | Self-correction metrics: Acc@t1, Acc@t2, Δ, Δ^{i→c}, Δ^{c→i}, edit distance |
| `train_score.py` | SCoRe training loop (Stage I + Stage II) with evaluation |
| `train_baselines.py` | STaR, Pair-SFT, Self-Refine, standard multi-turn RL, single-turn RL baselines |
| `utils.py` | Checkpoint management, majority voting, inference-compute scaling |
| `run_experiments.py` | Main script to run all experiments (Table 2, 3, 4, Figure 1) |

## Key Methods Implemented

### SCoRe (Main method)
- Stage I objective: `max E[r(y2)] - β₂·D_KL(π_θ(·|x1) || π_ref(·|x1))`
- Stage II objective: `max E[Σ r(y_i)] + α·(r(y2) - r(y1)) - β₁·D_KL(π_θ || π_ref)`

### Baselines
- **STaR** (Zelikman et al. 2022): Iterative SFT on successful correction traces
- **Pair-SFT** (Welleck et al. 2023): SFT on synthetically paired repair traces
- **Self-Refine** (Madaan et al. 2023): Prompting-based self-correction
- **Standard multi-turn RL**: REINFORCE on turn-2 reward only
- **Single-turn RL**: Standard REINFORCE on first attempts

### Ablations (Table 4)
- Without multi-turn training
- Without Stage I
- Without reward shaping (α = 0)
- STaR instead of REINFORCE in Stage II

## Hyperparameters (Table 5)

### MATH (Gemini 1.5 Flash)
- Learning rate: 5e-6, Batch size: 512, Steps: 3000
- α = 10, β₁ = 0.01, β₂ = 0.1

### Code (Gemini 1.0 Pro)
- Learning rate: 1e-5, Batch size: 128, Steps: 1500
- α = 10, β₁ = 0.01, β₂ = 0.25

## Usage

```bash
# Run all experiments
python run_experiments.py --data_dir /path/to/data --output_dir ./results

# Run only MATH experiments
python run_experiments.py --data_dir /path/to/data --task math

# Run only SCoRe (skip baselines)
python train_score.py --data_dir /path/to/data --output_dir ./score_output --task math
```

## Expected Results

| Method | Acc@t1 | Acc@t2 | Δ(t1,t2) |
|--------|--------|--------|-----------|
| Base model | 52.6% | 41.4% | -11.2% |
| Self-Refine | 52.8% | 51.8% | -1.0% |
| STaR w/ D_STaR^+ | 53.6% | 54.0% | 0.4% |
| Pair-SFT w/ D_SFT | 52.4% | 54.2% | 1.8% |
| **SCoRe** | **60.0%** | **64.4%** | **+4.4%** |

## Dependencies

See `requirements.txt`. Core: torch, transformers, datasets.
