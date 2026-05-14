# MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions

Reproduction of the paper **"MA-RLHF: RL from Human Feedback with Macro Actions"** (ICLR 2025, Baidu Inc.).

## Paper Summary

MA-RLHF introduces macro actions (sequences of tokens or higher-level language constructs) into the RLHF framework to address the credit assignment problem in long sequences. By operating at a coarser temporal scale, MA-RLHF reduces the temporal distance between actions and rewards, leading to:

- **1.7× to 2× faster** learning efficiency in reward scores
- Performance gains of up to **30%** in text summarization and code generation
- **No additional computational costs** during training or inference
- Robust scalability from 2B to 27B parameter models

## Core Contributions Reproduced

### 1. Macro Action Framework (Section 3)

The core idea: instead of optimizing at the token level, MA-RLHF groups tokens into "macro actions" and optimizes at this coarser level.

- **Macro action formalization** (Section 3.2.1): Each macro action ω_τ = {a_{t_τ}, ..., a_{t_{τ+1}-1}} is a sequence of consecutive tokens
- **Joint probability**: π_θ(ω_τ | s_τ) = ∏ π_θ(a_t | a_<t)
- **Macro rewards**: R_τ = Σ ρ^i r_{t_τ+i} with ρ=1
- **Connection to prior methods** (Section 3.2.3): When n=1, reduces to standard token-level RLHF; when n→∞, approaches REINFORCE/RLOO/GRPO

### 2. Four Termination Strategies (Section 3.2.1, Appendix B.4)

| Strategy | Description | Best for |
|----------|-------------|----------|
| **Fixed n-gram** | Groups tokens into fixed-length sequences of n | Default, best overall performance |
| **Randomized n-gram** | Randomly selects from {2, 3, 5, 10} | Best relevance/coherence/consistency |
| **Parsing-based** | Uses constituent tree DFS with cutoff C=5 | Complex grammar, HH-RLHF |
| **Perplexity-based** | Terminates when token increases perplexity | Fluency-focused tasks |

### 3. MA-PPO Algorithm (Section 3.2.2, Appendix E)

The PPO algorithm adapted for macro actions:

```
L^{MA-PPO}(θ) = E_τ[min(π_θ(ω_τ|s_τ)/π_θold(ω_τ|s_τ) * Â_τ,
                          clip(π_θ(ω_τ|s_τ)/π_θold(ω_τ|s_τ), 1-ε, 1+ε) * Â_τ)]
```

Key implementation details:
- Maintains original action space (no vocabulary changes)
- GAE applied at macro action level
- Advantages broadcast to all tokens within each macro action

### 4. Value Function Estimation (Appendix D.1)

Three σ assignments for estimating macro action values:

| Assignment | Formula | Best for |
|------------|---------|----------|
| **Equal** (default) | σ_i = 1/|ω_τ| | Highest RM scores |
| **Unit** | σ = [0,0,...,0,1] | Best consistency/fluency |
| **Position decayed** | σ_i ∝ 1/(|ω_τ|-i) | Balanced |

### 5. RLHF Utilities (Section 2.2)

- KL divergence penalty: D_KL(π_θ(·|x) || π_sft(·|x))
- Shaped reward: R(x,y) = r_φ(x,y) - β·D_KL
- Reward model training loss: -log σ(log(r_φ(x,y_+)) - log(r_φ(x,y_-)))
- Program synthesis reward (Appendix B.5): adaptive compiler signal

## Repository Structure

```
ma_rlhf/
├── __init__.py
├── termination.py       # Four macro action termination strategies
├── value_estimation.py  # Three σ assignments for value estimation
├── ma_ppo.py           # MA-PPO policy and critic losses
├── rlhf_utils.py       # KL penalty, reward shaping, RM loss
├── trainer.py          # Full training pipeline (MAConfig, MacroActionScheduler, MARLHFTrainer)
└── evaluation.py       # GPT-4 prompts, Best-of-N, pass@k, win rates

utils/
├── __init__.py
└── data_utils.py       # Dataset loading, formatting, splitting

configs/
├── default_config.yaml  # Default hyperparameters (Table 5)
└── experiments.yaml     # Experiment matrix covering all paper experiments

scripts/
└── run_ma_ppo.py        # Main training entry point

tests/
└── test_macro_actions.py # Comprehensive unit tests
```

## Key Design Decisions

1. **Same action space**: Following the paper (Section 3.2.2), we maintain the original token vocabulary rather than adding macro actions to the vocabulary, which would require retraining the model.

2. **Macro action joint probability**: Computed as the product of token-level probabilities within each macro action: π_θ(ω_τ|s_τ) = ∏ π_θ(a_t|a_<t).

3. **GAE at macro level**: Advantages and returns are computed at the macro action level using standard GAE, then broadcast to tokens for optimization.

4. **Two policy loss variants**:
   - `policy_loss_macro_action`: Applies macro-level advantages to each token's individual ratio
   - `policy_loss_macro_action_joint`: Uses true macro-action joint probability ratio

5. **Reduction to vanilla PPO**: When n=1, each token is its own macro action, and MA-PPO is mathematically equivalent to standard token-level PPO.

## Usage

### Installation

```bash
pip install torch transformers
```

### Quick Start

```python
from ma_rlhf.termination import get_macro_action_positions
from ma_rlhf.value_estimation import get_macro_action_values
from ma_rlhf.ma_ppo import policy_loss_macro_action

# Define macro action boundaries
sequence = get_macro_action_positions(
    start=prompt_len - 1,
    mask=attention_mask,
    termination='ngram',
    n_gram=5
)

# Compute macro action values
macro_values = get_macro_action_values(
    values=token_values,
    mask=mask,
    start=0,
    sequence=sequence,
    value_assignment='equal'
)

# Compute MA-PPO policy loss
loss = policy_loss_macro_action(
    logprobs=current_logprobs,
    old_logprobs=old_logprobs,
    advantages=macro_advantages,
    mask=response_mask,
    sequence=sequence,
    cliprange=0.2
)
```

### Training

```bash
# Vanilla PPO (n=1, baseline)
python scripts/run_ma_ppo.py --macro_termination ngram --n_gram 1

# MA-PPO with n=5 (default, best overall)
python scripts/run_ma_ppo.py --macro_termination ngram --n_gram 5

# MA-PPO with n=10 (best for HH-RLHF)
python scripts/run_ma_ppo.py --macro_termination ngram --n_gram 10

# MA-PPO with randomized n-gram
python scripts/run_ma_ppo.py --macro_termination randomized_ngram

# MA-PPO with perplexity-based termination
python scripts/run_ma_ppo.py --macro_termination ppl

# MA-PPO with parsing-based termination
python scripts/run_ma_ppo.py --macro_termination parser
```

### Running Tests

```bash
python tests/test_macro_actions.py
```

## Experiments Covered

The `configs/experiments.yaml` file defines configurations for all experiments described in the paper:

- **Section 4.2**: Main results (TL;DR, HH-RLHF, WebGPT) for 2B/7B models
- **Section 4.3.1**: Termination strategy comparison (Figure 5)
- **Section 4.3.2**: Varying n values n∈{3,5,10,∞} (Figures 6, 7)
- **Section 4.4**: Scaling to 27B models (Figure 9)
- **Section 4.5**: Code generation on APPS (Table 3)
- **Appendix D.1**: Value function estimation variants (Figure 19)

## Datasets

| Dataset | Task | Train Size | Test Size |
|---------|------|-----------|-----------|
| TL;DR | Summarization | 92.9k | 86.1k |
| HH-RLHF | Dialogue | 112k | 12.5k |
| WebGPT | QA | 18.5k | 979 |
| APPS | Code Generation | 5k | 5k |

## Hyperparameters

Defaults follow Table 5 in the paper:

| Parameter | 2B | 7B | 27B |
|-----------|----|----|------|
| PPO batch size | 256 | 256 | 256 |
| Policy LR | 1.5e-5 | 1e-6 | 7e-7 |
| Critic LR | 1.5e-5 | 1e-6 | 1e-6 |
| KL coefficient (β) | 0.05 | 0.05* | 0.1 |
| Clip ratio (ε) | 0.2 | 0.2 | 0.2 |
| γ (GAE) | 1.0 | 1.0 | 1.0 |
| λ (GAE) | 0.95 | 0.95 | 0.95 |
| Temperature | 0.8 | 0.8 | 0.8 |

*For TL;DR 7B, KL coefficient reduced to 0.01 for stability (see Appendix B.2).

## Assumptions & Unresolved Details

1. **Model initialization**: The paper initializes the reward model from the SFT model and the critic model from the reward model. We follow this convention.

2. **KL penalty application**: The KL penalty is applied per-token, with the RM score added only at the final token position, following standard RLHF practice.

3. **Data split**: 20% SFT, 40% RM, 40% PPO for most datasets; for APPS, 20% SFT, 0% RM, 80% PPO.

4. **Parsing-based termination**: Requires a constituent parser (e.g., Berkeley Neural Parser, Stanza). The paper uses depth-first search with cutoff C=5.

5. **Perplexity-based termination**: Uses the reference model's logits to compute perplexity at each position, avoiding additional forward passes.

6. **Training stability**: The paper notes that for 7B models on TL;DR, the KL coefficient was reduced from 0.05 to 0.01 due to training instability.

7. **DeepSpeed integration**: The paper uses DeepSpeed-Chat (Yao et al., 2023). Our implementation is structured to be compatible with DeepSpeed but can run without it.

## References

Key references from the paper:
- Sutton et al. (1999b) - Semi-MDP framework for temporal abstraction
- Schulman et al. (2017) - PPO algorithm
- Ouyang et al. (2022) - InstructGPT / RLHF
- Stiennon et al. (2020) - Learning to summarize from human feedback
- Yao et al. (2023) - DeepSpeed-Chat
