# SCoRe: Training Language Models to Self-Correct via Reinforcement Learning

Reproduction of "Training Language Models to Self-Correct via Reinforcement Learning" (Kumar et al., 2024).

## Overview

SCoRe is a multi-turn online reinforcement learning approach that trains LLMs to self-correct their own mistakes using entirely self-generated data. The key insight is that standard SFT and naive RL approaches suffer from either distribution shift or behavior collapse, and SCoRe addresses both via a two-stage training procedure.

## Core Contributions Implemented

### 1. SCoRe Algorithm (Section 5)

The main algorithm is implemented in `src/score_trainer.py` and `src/score_trainer_trl.py`.

**Stage I** (Section 5.1): Train second-attempt while constraining first-attempt to base model.
- Objective: `max E[r(y2, y*) - beta2 * KL(pi_theta(.|x1) || pi_ref(.|x1))]`
- Strong KL penalty (beta2) on first attempt prevents distribution shift
- Allows second attempt to improve without collapsing first attempt

**Stage II** (Section 5.2): Joint optimization with reward shaping.
- Objective: `max E[sum_i r(yi, y*) - beta1 * KL(pi_theta(.|xi) || pi_ref(.|xi))]`
- Reward shaping bonus: `b(y2|y1,y*) = alpha * (r(y2,y*) - r(y1,y*))`
- Bonus rewards i→c transitions, penalizes c→i transitions

### 2. Baselines (Section 4)

Implemented in `src/baselines.py`:
- **STaR** (Zelikman et al., 2022): SFT on successful correction traces (D_STaR and D_STaR+)
- **Pair-SFT** (Welleck et al., 2023 variant): SFT on paired incorrect/correct responses (D_SFT and D_SFT+)
- **Self-Refine** (Madaan et al., 2023): Prompting-based self-correction without training

### 3. Evaluation Metrics (Section 3)

Implemented in `src/evaluation.py`:
- **Accuracy@t1**: First-attempt accuracy
- **Accuracy@t2**: Second-attempt accuracy
- **Δ(t1,t2)**: Net improvement (t2 - t1)
- **Δ^(i→c)(t1,t2)**: Fraction of incorrect t1 that become correct t2
- **Δ^(c→i)(t1,t2)**: Fraction of correct t1 that become incorrect t2
- Edit distance ratio analysis (Figure 4)

### 4. Reward Functions (Section 3)

Implemented in `src/reward.py`:
- **MATH**: Binary reward based on answer matching with `\boxed{}` extraction
- **Code**: Binary reward based on test case execution

### 5. Prompts (Appendix C)

Implemented in `src/prompts.py`:
- MATH zero-shot CoT prompt
- MATH self-correction instruction
- MBPP 3-shot prompt
- Code self-correction instruction

### 6. Inference-Compute Scaling (Section 6.2)

Implemented in `src/inference_scaling.py`:
- Parallel sampling with majority vote
- Sequential (self-correction) + parallel sampling
- Comparison at different sample budgets

## Hyperparameters

From Appendix B, Table 5:

| Hyperparameter | MATH | MBPP |
|---|---|---|
| Base model | Gemini 1.5 Flash | Gemini 1.0 Pro |
| Learning rate | 5e-6 | 1e-5 |
| Training steps | 3000 | 1500 |
| Batch size | 512 | 128 |
| Sampling temperature | 1.0 | 1.0 |
| α (reward shaping) | 10 | 10 |
| β1 (KL Stage II) | 0.01 | 0.01 |
| β2 (KL Stage I first-turn) | 0.1 | 0.25 |

## Usage

### Training SCoRe on MATH

```bash
python train_score.py \
    --task math \
    --model_name google/gemma-2b \
    --math_data_dir /path/to/MATH \
    --output_dir ./outputs/score_math \
    --learning_rate 5e-6 \
    --training_steps 3000 \
    --batch_size 512 \
    --alpha 10.0 \
    --beta1 0.01 \
    --beta2 0.1
```

### Training SCoRe on MBPP/HumanEval

```bash
python train_score.py \
    --task code \
    --model_name google/gemma-2b \
    --mbpp_data_path /path/to/mbpp.jsonl \
    --humaneval_data_path /path/to/HumanEval.jsonl \
    --output_dir ./outputs/score_code \
    --learning_rate 1e-5 \
    --training_steps 1500 \
    --batch_size 128 \
    --alpha 10.0 \
    --beta1 0.01 \
    --beta2 0.25
```

### Training Baselines

```bash
# STaR
python train_baselines.py --method star --task math \
    --model_name google/gemma-2b --math_data_dir /path/to/MATH

# Pair-SFT
python train_baselines.py --method pair_sft --task math \
    --model_name google/gemma-2b --math_data_dir /path/to/MATH

# STaR with D+ (correct->correct pairs)
python train_baselines.py --method star --task math \
    --include_correct_to_correct \
    --model_name google/gemma-2b --math_data_dir /path/to/MATH
```

### Evaluation

```bash
python evaluate.py \
    --task math \
    --base_model_name google/gemma-2b \
    --score_model_path ./outputs/score_math/best_checkpoint \
    --star_model_path ./outputs/baselines/star_math/model \
    --pair_sft_model_path ./outputs/baselines/pair_sft_math/model \
    --math_data_dir /path/to/MATH \
    --output_dir ./outputs/evaluation
```

## Data Setup

### MATH Dataset
Download from: https://github.com/hendrycks/math
Place in `./data/MATH/` with `train/` and `test/` subdirectories.

### MBPP Dataset
Download from: https://github.com/google-research/google-research/tree/master/mbpp
Place as `./data/mbpp.jsonl`

### HumanEval Dataset
Download from: https://github.com/openai/human-eval
Place as `./data/HumanEval.jsonl`

## Key Design Decisions

### Why Two Stages?
Standard multi-turn RL collapses to a degenerate solution where the model learns to produce a good first attempt and make no changes in the second attempt (behavior collapse). Stage I prevents this by decoupling the two attempts.

### Why Reward Shaping?
The reward shaping bonus `alpha * (r2 - r1)` makes it more attractive to learn self-correction (i→c transitions) than to simply produce a good first attempt. Without this, Stage II can still collapse.

### Why On-Policy RL?
SFT on offline data suffers from distribution shift: the model learns to correct mistakes made by the data-collection policy, but these gains don't transfer to correcting its own mistakes.

## Assumptions and Unresolved Details

1. **Model**: The paper uses Gemini 1.5 Flash (MATH) and Gemini 1.0 Pro (code), which are not publicly available. This reproduction uses open-source models (e.g., Gemma-2B) as proxies.

2. **Stage I/II split**: The paper doesn't specify the exact number of steps for each stage. We use a 1/3 - 2/3 split as a reasonable default.

3. **Offline data augmentation**: Section 5.3 mentions augmenting with base model samples in Stage II, but doesn't specify the exact ratio. We include this as an option.

4. **MBPP-R dataset**: The paper uses MBPP-R (Ni et al., 2024) for offline repair evaluation, which requires pre-generated incorrect programs from PaLM 2. This dataset is not publicly available.

5. **Batch size**: The paper uses large batch sizes (512 for MATH) which require gradient accumulation on typical hardware.

## File Structure

```
├── src/
│   ├── __init__.py
│   ├── score_trainer.py      # Core SCoRe training algorithm
│   ├── score_trainer_trl.py  # TRL-based implementation
│   ├── baselines.py          # STaR, Pair-SFT, Self-Refine
│   ├── reward.py             # Reward functions (MATH, code)
│   ├── prompts.py            # Prompts from Appendix C
│   ├── data_utils.py         # Data loading utilities
│   ├── evaluation.py         # Evaluation metrics
│   └── inference_scaling.py  # Inference-compute scaling (Section 6.2)
├── configs/
│   ├── score_math.yaml       # MATH hyperparameters
│   └── score_code.yaml       # Code hyperparameters
├── train_score.py            # Main SCoRe training script
├── train_baselines.py        # Baseline training script
├── evaluate.py               # Evaluation script
├── requirements.txt
└── README.md
```
