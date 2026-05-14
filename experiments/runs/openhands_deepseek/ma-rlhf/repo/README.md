# MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions

Reproduction of "MA-RLHF: REINFORCEMENT LEARNING FROM HUMAN FEEDBACK WITH MACRO ACTIONS" (Chai et al., Baidu Inc.).

## Paper Overview

MA-RLHF incorporates macro actions — sequences of tokens or higher-level language constructs — into the RLHF framework. By operating at a higher level of abstraction, it reduces temporal distance between actions and rewards, facilitating faster and more accurate credit assignment. This results in more stable policy gradient estimates and enhances learning efficiency without increasing computational complexity.

## Codebase Structure

```
repo/
├── README.md          # This file
├── requirements.txt   # Python dependencies
├── config.py          # All hyperparameters and configuration
├── config.yaml        # YAML configuration files for each dataset
├── data.py            # Dataset loading and preprocessing
├── macro_actions.py   # Macro action termination strategies
├── model.py           # Model architectures (SFT, Reward, Policy, Critic)
├── ppo.py             # MA-PPO and vanilla PPO algorithms
├── train.py           # Full training pipeline
├── evaluate.py        # Evaluation utilities (RM scores, GPT-4, human eval)
└── utils.py           # Utility functions
```

## Key Components

### Macro Action Termination Strategies
- **Fixed n-gram**: Group tokens into fixed-length n-grams (default: n=5)
- **Randomized n-gram**: Randomly sample n ∈ {2, 3, 5, 10}
- **Parsing-based**: Use constituent tree DFS with threshold C=5
- **Perplexity-based**: Terminate when perplexity increases

### Supported Tasks
- TL;DR Text Summarization
- HH-RLHF Dialogue Generation
- WebGPT Question Answering
- APPS Code Generation

### Model Sizes
- Gemma-2B, Gemma-7B, Gemma-2-27B
- CodeGemma-1.1-2B, CodeGemma-1.1-7B-it

## Usage

```bash
# Train MA-PPO on TL;DR
python train.py --config configs/tldr_2b.yaml --method ma_ppo --n_gram 5

# Train vanilla PPO
python train.py --config configs/tldr_2b.yaml --method vanilla_ppo

# Evaluate
python evaluate.py --checkpoint /path/to/checkpoint --task tldr
```
