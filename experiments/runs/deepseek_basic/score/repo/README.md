# SCoRe: Training Language Models to Self-Correct via Reinforcement Learning

A reproduction of the paper:

> **Training Language Models to Self-Correct via RL (SCoRe)**  
> Aviral Kumar*, Vincent Zhuang*, Rishabh Agarwal*, et al.  
> Google DeepMind, 2024

## Overview

SCoRe is a multi-turn online reinforcement learning (RL) approach that significantly improves an LLM's self-correction ability using entirely self-generated data. The method trains a single model to both produce responses and correct its own errors without any oracle feedback at test time.

### Core Algorithm

SCoRe consists of two stages:

**Stage I** (Section 5.1): Train an initialization that decouples the two attempts.
- Maximizes second-attempt reward while constraining first-turn distribution to stay close to the base model via KL divergence
- Objective: `max_θ E[r̂(y₂, y*) - β₂·D_KL(π_θ(·|x₁) || π_ref(·|x₁))]`

**Stage II** (Section 5.2): Multi-turn RL with reward shaping.
- Jointly optimizes both attempts with a progress bonus
- Reward shaping: `b(y₂|y₁) = α·(r̂(y₂) - r̂(y₁))` 
- Objective: `max_θ E[Σ r̂(yᵢ) + α·(r̂(y₂)-r̂(y₁)) - β₁·Σ D_KL(π_θ || π_ref)]`

The progress bonus rewards transitions that flip correctness and heavily penalizes changing correct answers to incorrect, preventing behavior collapse.

## Repository Structure

```
score/
├── __init__.py              # Package metadata
├── __main__.py              # python -m score entry point
├── train.py                 # Main training script
├── training/
│   ├── __init__.py
│   ├── reinforce.py         # REINFORCE policy gradient with KL penalty (Eq. 2)
│   ├── score_trainer.py     # SCoRe two-stage trainer (Stage I + Stage II)
│   └── sft_trainer.py       # SFT baselines: STaR and Pair-SFT (Section 4)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py           # Self-correction metrics & behavior analysis
├── data/
│   ├── __init__.py
│   └── dataset.py           # MATH, MBPP, HumanEval loading & configs
├── prompts/
│   ├── __init__.py
│   └── templates.py         # Exact prompts from Appendix C
└── utils/
    ├── __init__.py
    └── logging.py           # Training logging utilities

scripts/
├── train_math.sh            # MATH training (Gemini 1.5 Flash config)
└── train_code.sh            # Code training (Gemini 1.0 Pro config)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers (HuggingFace)
- datasets (HuggingFace)
- numpy, tqdm

Install:
```bash
pip install torch transformers datasets numpy tqdm
```

## Usage

### Training SCoRe on MATH
```bash
python -m score.train \
    --task math \
    --model_name_or_path "google/gemma-2-9b-it" \
    --output_dir ./outputs/score_math \
    --batch_size 512 \
    --learning_rate 5e-6 \
    --stage1_beta2 0.1 \
    --stage2_alpha 10.0
```

Or use the provided script:
```bash
bash scripts/train_math.sh
```

### Training SCoRe on Code (MBPP → HumanEval)
```bash
python -m score.train \
    --task code \
    --model_name_or_path "google/gemma-2-9b-it" \
    --output_dir ./outputs/score_code \
    --batch_size 128 \
    --learning_rate 1e-5 \
    --stage1_beta2 0.25 \
    --stage2_alpha 10.0
```

### Running Ablations (Section 6.3, Table 4)
```bash
# Without Stage I
python -m score.train --task math --ablation no_stage1 ...

# Without reward shaping
python -m score.train --task math --ablation no_reward_shaping ...

# Single-turn training
python -m score.train --task math --ablation single_turn ...

# STaR instead of REINFORCE in Stage II
python -m score.train --task math --ablation star_stage2 ...
```

### Running SFT Baselines (Section 4, Table 1)
```bash
python -m score.train --task math --run_sft_baselines ...
```

## Key Implementation Details

### Hyperparameters (Table 5, Appendix B)

**MATH (Gemini 1.5 Flash):**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 5e-6 |
| Training steps (total) | 3000 |
| Batch size | 512 |
| Sampling temperature | 1.0 |
| α (Stage II) | 10 |
| β₁ | 0.01 |
| β₂ (Stage I) | 0.1 |

**MBPP (Gemini 1.0 Pro):**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-5 |
| Training steps (total) | 1500 |
| Batch size | 128 |
| Sampling temperature | 1.0 |
| α (Stage II) | 10 |
| β₁ | 0.01 |
| β₂ (Stage I) | 0.25 |

### Evaluation Metrics (Section 3)
- **Accuracy@t1**: First-attempt accuracy
- **Accuracy@t2**: Second-attempt accuracy
- **Δ(t1, t2)**: Net improvement
- **Δ(i→c)**: Incorrect→correct rate
- **Δ(c→i)**: Correct→incorrect rate
- **Edit distance ratio**: Used to detect behavior collapse (Section 4)

### Prompts (Appendix C)
All prompts are implemented exactly as specified:
- MATH zero-shot CoT for evaluation
- MATH self-correction instruction (does NOT reveal correctness)
- MBPP 3-shot for training
- HumanEval zero-shot for evaluation

## What Was Reproduced

✅ **SCoRe Algorithm**: Full two-stage multi-turn RL implementation
- Stage I: Second-attempt optimization with first-turn KL constraint (Eq. 3)
- Stage II: Joint optimization with reward shaping progress bonus (Eq. 4)
- REINFORCE policy gradient with KL penalty (Ahmadian et al., 2024)

✅ **SFT Baselines (Section 4)**:
- STaR with iterative filtering and retraining
- Pair-SFT on synthetic correction pairs
- D_STaR^+ and D_SFT^+ variants with correct-to-correct data

✅ **Evaluation Metrics**: All five metrics from Section 3
- Accuracy@t1, Accuracy@t2, Δ(t1,t2), Δ(i→c), Δ(c→i)
- Edit distance ratio analysis for behavior collapse detection

✅ **Self-Consistency Evaluation (Section 6.2)**:
- Parallel vs. sequential sampling comparison

✅ **Prompt Templates (Appendix C)**:
- Exact prompts for MATH, MBPP, and HumanEval

✅ **Hyperparameters (Appendix B, Table 5)**:
- Full MATH and MBPP configurations

✅ **Ablation Studies (Section 6.3, Table 4)**:
- Without Stage I
- Without reward shaping
- Single-turn RL
- STaR in Stage II

## Assumptions and Missing Details

1. **Model**: The paper uses proprietary Gemini 1.0 Pro and 1.5 Flash models. We use open-source alternatives (e.g., Gemma) with HuggingFace integration. The code is model-agnostic and can work with any causal LM.

2. **Infrastructure**: The paper uses Google's internal RL infrastructure. We implement the REINFORCE algorithm from scratch using PyTorch and HuggingFace transformers.

3. **Data scale**: The paper uses MATH training set augmented with 4500 test problems. Our data loader supports both HuggingFace datasets and local JSON files.

4. **Adaptive β₂**: The paper mentions using adaptive β₂ in some experiments to balance KL and policy objective magnitudes. We implement a fixed β₂ as the default, matching the reported values in Table 5.

5. **Offline data mixing**: The paper mentions amplifying coverage with base model first-attempt data in Stage II. This is configurable via `mix_offline_first_attempts`.

6. **Multi-round correction**: The paper only trains for 2 attempts. We follow this exactly; the code supports l=2 turns.

7. **Checkpoint selection**: The paper selects checkpoints with highest training reward. Our implementation saves checkpoints periodically; final evaluation uses the last checkpoint.

## Paper Reference

```
@article{kumar2024training,
  title={Training Language Models to Self-Correct via Reinforcement Learning},
  author={Kumar, Aviral and Zhuang, Vincent and Agarwal, Rishabh and others},
  journal={arXiv preprint},
  year={2024}
}
```
