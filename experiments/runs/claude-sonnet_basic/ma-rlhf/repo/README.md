# MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions

Reproduction of the paper:
> **MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions**  
> Yekun Chai, Haoran Sun, Huang Fang, Shuohuan Wang, Yu Sun, Hua Wu (Baidu Inc.)

## Overview

MA-RLHF proposes integrating **macro actions** — sequences of tokens or higher-level language constructs — into the RLHF framework. By operating at a coarser temporal scale, MA-RLHF reduces the temporal distance between actions and rewards, improving credit assignment and learning efficiency.

Key contributions reproduced:
1. **MA-PPO algorithm** (Algorithm 1): PPO with macro-action-level policy gradient and value estimation
2. **Four termination strategies** for macro actions: fixed n-gram, randomized n-gram, perplexity-based, parsing-based
3. **Three value assignment strategies**: equal, unit, position-decayed
4. **Full training pipeline**: SFT → Reward Model → MA-PPO
5. **Evaluation**: RM scores, Best-of-N, pass@k (code), GPT-4 pairwise evaluation

## Repository Structure

```
.
├── src/
│   ├── macro_actions.py      # Core MA-RLHF: termination strategies, loss functions
│   ├── ma_ppo_trainer.py     # MA-PPO training loop (Algorithm 1)
│   ├── reward_model.py       # Reward model architecture and training
│   └── data_utils.py         # Dataset loading for TL;DR, HH-RLHF, WebGPT, APPS
├── scripts/
│   ├── train_sft.py          # Stage 1: Supervised Fine-Tuning
│   ├── train_rm.py           # Stage 2: Reward Model training
│   ├── train_ma_ppo.py       # Stage 3: MA-PPO training
│   ├── evaluate.py           # RM score / Best-of-N / pass@k evaluation
│   └── gpt4_eval.py          # GPT-4 pairwise win-rate evaluation
├── configs/
│   ├── tldr_2b.yaml          # TL;DR + Gemma-2B (main experiment)
│   ├── tldr_7b.yaml          # TL;DR + Gemma-7B
│   ├── hh_rlhf_2b.yaml       # HH-RLHF + Gemma-2B
│   ├── webgpt_2b.yaml        # WebGPT + Gemma-2B
│   └── apps_2b.yaml          # APPS + CodeGemma-2B
└── requirements.txt
```

## Core Algorithm

### Macro Action Termination Strategies (Section 3.2.1)

1. **Fixed n-gram** (default, n=5): Groups tokens into fixed-length chunks
2. **Randomized n-gram**: Randomly selects lengths from {2, 3, 5, 10}
3. **Perplexity-based**: Terminates when perplexity increases (monotonic decrease broken)
4. **Parsing-based**: Uses constituency parse tree with DFS, cutoff C=5

### MA-PPO Objective (Equation 3)

The policy gradient is computed at the macro-action level:

```
L^MA-PPO(θ) = E_τ [min(r_τ * Â_τ, clip(r_τ, 1-ε, 1+ε) * Â_τ)]
```

where `r_τ = π_θ(ω_τ|s_τ) / π_θ_old(ω_τ|s_τ)` is the macro-action importance ratio.

### Value Function Estimation (Section D.1)

The macro-action value is estimated from token-level values:

```
V^π(s_τ, ω_τ) = Σ_{t ∈ ω_τ} σ_t * V^π(s_t, a_t)
```

Three assignment strategies: equal (default), unit (last token), position-decayed.

## Usage

### Step 1: Supervised Fine-Tuning

```bash
python scripts/train_sft.py --config configs/tldr_2b.yaml
```

### Step 2: Reward Model Training

```bash
python scripts/train_rm.py --config configs/tldr_2b.yaml
```

### Step 3: MA-PPO Training

```bash
python scripts/train_ma_ppo.py --config configs/tldr_2b.yaml
```

### Evaluation

```bash
# RM score evaluation
python scripts/evaluate.py \
    --config configs/tldr_2b.yaml \
    --checkpoint output/tldr_2b/ma_ppo/checkpoint-4600 \
    --rm_checkpoint output/tldr_2b/rm \
    --eval_type rm_score

# Best-of-N evaluation
python scripts/evaluate.py \
    --config configs/tldr_2b.yaml \
    --checkpoint output/tldr_2b/ma_ppo/final \
    --rm_checkpoint output/tldr_2b/rm \
    --eval_type best_of_n \
    --n_samples 8

# GPT-4 pairwise evaluation
python scripts/gpt4_eval.py \
    --task tldr \
    --articles data/tldr_val_articles.jsonl \
    --model_a_responses responses_ma_ppo.jsonl \
    --model_b_responses responses_ppo.jsonl \
    --n_samples 50
```

## Datasets

The paper uses the following datasets (download separately):

| Dataset | Task | HuggingFace ID |
|---------|------|----------------|
| OpenAI TL;DR | Summarization | `openai/summarize_from_feedback` |
| Anthropic HH-RLHF | Dialogue | `Anthropic/hh-rlhf` |
| WebGPT Comparisons | QA | `openai/webgpt_comparisons` |
| APPS | Code Generation | `codeparrot/apps` |

## Hyperparameters

Key hyperparameters from Table 5 of the paper:

| Parameter | Gemma-2B | Gemma-7B |
|-----------|----------|----------|
| Policy LR | 1.5e-5 | 1e-6 |
| Critic LR | 1.5e-5 | 1e-6 |
| KL coef (β) | 0.05 | 0.01 (TL;DR) |
| Clip ratio (ε) | 0.2 | 0.2 |
| λ (GAE) | 0.95 | 0.95 |
| γ (discount) | 1.0 | 1.0 |
| Temperature | 0.8 | 0.8 |
| n-gram (n) | 5 | 5 |

## Expected Results

From Table 2 of the paper:

| Model | Task | Vanilla PPO | MA-PPO | Improvement |
|-------|------|-------------|--------|-------------|
| Gemma-2B | TL;DR | 0.84 | 1.41 | +68% |
| Gemma-2B | HH-RLHF | 1.31 | 1.55 | +18% |
| Gemma-2B | WebGPT | -0.62 | -0.60 | +3% |
| Gemma-7B | TL;DR | 1.90 | 2.47 | +30% |
| Gemma-7B | HH-RLHF | 1.05 | 1.24 | +18% |
| Gemma-7B | WebGPT | -0.61 | -0.56 | +8% |

MA-PPO also achieves parity with vanilla PPO **1.7-2x faster** in training steps.

## Assumptions and Unresolved Details

1. **DeepSpeed integration**: The paper uses DeepSpeed-Chat for distributed training. This reproduction uses standard PyTorch; DeepSpeed can be added via `deepspeed` launcher.

2. **Parsing-based termination**: Requires `nltk` with constituency parser (e.g., `benepar`). Falls back to standard PPO when parsing fails.

3. **Critic architecture**: The paper initializes the critic from the reward model. We implement this as a `RewardModel` wrapper with a linear value head on top of the base LM.

4. **Data splits**: The paper uses 20%/40%/40% for SFT/RM/PPO. For code generation (APPS), 80% is used for PPO (no RM stage).

5. **KL penalty**: Applied per-token as `β * (log π_θ - log π_ref)`, with the reward model score added at the final response token.
