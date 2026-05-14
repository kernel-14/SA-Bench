# MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions

Implementation of the MA-RLHF framework from the paper "MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions" (Chai et al., 2024).

## Overview

MA-RLHF integrates macro actions — sequences of tokens or higher-level language constructs — into the RLHF framework. By operating at a coarser temporal scale, it reduces the temporal distance between actions and rewards, improving credit assignment and learning efficiency.

## Repository Structure

```
repo/
├── config.py           # All hyperparameters and configuration dataclasses
├── data.py             # Dataset loading and preprocessing (TL;DR, HH-RLHF, WebGPT, APPS)
├── model.py            # PolicyModel, CriticModel, RewardModel
├── macro_actions.py    # Macro action termination strategies and value/loss computation
├── trainer.py          # PPOTrainer and MAPPOTrainer with GAE
├── train_sft.py        # Supervised fine-tuning stage
├── train_rm.py         # Reward model training stage
├── train_ppo.py        # PPO / MA-PPO training stage
├── evaluate.py         # RM scoring, pass@k, best-of-N, GPT-4 evaluation
├── utils.py            # Shared utilities
├── requirements.txt
└── README.md
```

## Three-Stage Training Pipeline

### Stage 1: Supervised Fine-Tuning
```bash
python train_sft.py --task tldr --model_name google/gemma-2b --output_dir outputs/sft
```

### Stage 2: Reward Modeling
```bash
python train_rm.py --task tldr --sft_model_path outputs/sft --output_dir outputs/rm
```

### Stage 3: MA-PPO Training
```bash
python train_ppo.py \
    --task tldr \
    --policy_model_path outputs/sft \
    --critic_model_path outputs/rm \
    --reward_model_path outputs/rm \
    --use_macro_actions \
    --termination ngram \
    --n_gram 5 \
    --output_dir outputs/ma_ppo
```

For vanilla PPO (baseline), omit `--use_macro_actions`.

## Macro Action Termination Strategies

| Strategy | Description | Key Hyperparameter |
|---|---|---|
| `ngram` | Fixed-length n-gram grouping | `--n_gram N` (default: 5) |
| `randomized_ngram` | Random lengths from {2,3,5,10} | — |
| `parser` | Constituent tree DFS, cutoff C=5 | `--parser_cutoff 5` |
| `ppl` | Perplexity-monotone grouping | — |

## Supported Tasks and Datasets

| Task | Dataset | Base Model |
|---|---|---|
| Text Summarization | TL;DR (OpenAI) | Gemma-2B / 7B / 27B |
| Dialogue Generation | Anthropic HH-RLHF | Gemma-2B / 7B |
| Question Answering | WebGPT Comparisons | Gemma-2B / 7B |
| Code Generation | APPS | CodeGemma-2B / 7B |

## Key Results

MA-PPO achieves 1.7–2× faster convergence than vanilla PPO and up to:
- +30% on TL;DR summarization (7B)
- +18% on HH-RLHF dialogue
- +8% on WebGPT QA
- +35% pass@1 on APPS code generation (7B)

## Value Function Estimation

Three σ assignment strategies for macro action value aggregation:
- `equal`: uniform average over tokens in macro action (default)
- `unit`: use only the last token's value
- `position_decayed`: harmonic position-weighted average

## Citation

```bibtex
@article{chai2024marlhf,
  title={MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions},
  author={Chai, Yekun and Sun, Haoran and Fang, Huang and Wang, Shuohuan and Sun, Yu and Wu, Hua},
  journal={arXiv preprint},
  year={2024}
}
```
