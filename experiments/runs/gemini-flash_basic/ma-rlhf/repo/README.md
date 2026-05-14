# MA-RLHF: RL from Human Feedback with Macro Actions - Reproduction

This repository aims to reproduce the core contributions of the paper "MA-RLHF: RL from Human Feedback with Macro Actions". The implementation focuses on replicating the MA-PPO algorithm with n-gram based macro actions.

## Implemented Core Contributions:

1.  **Macro Action Generation**: Implemented `generate_ngram_macro_actions` and `generate_randomized_ngram_macro_actions` as described in Section 3.2.1 of the paper. The fixed `n`-gram approach is used as the default.
2.  **MA-PPO Objective**: The `PPO` class in `marlhf/ppo.py` implements the Proximal Policy Optimization algorithm adapted for macro actions, as detailed in Section 3.2.2 (Equation 4) of the paper. This includes:
    -   Calculation of macro-level advantages using Generalized Advantage Estimation (GAE).
    -   Clipping mechanism for policy updates.
    -   Combination of policy and value losses.
3.  **RLHF Framework Integration**: The `MARLHF` class in `marlhf/ma_rlhf.py` orchestrates the entire training process:
    -   **Policy and Reference Models**: Uses `transformers.AutoModelForCausalLM` for the active policy (`policy_lm`) and the reference policy (`policy_lm_ref`).
    -   **Response Generation**: Generates sequences of tokens from the policy model while collecting token-level log probabilities.
    -   **Reward Calculation**: Includes a placeholder for the reward model, which in a full implementation would provide a scalar reward for generated responses.
    -   **KL Divergence Penalty**: Calculates a KL divergence penalty between the current policy and a Supervised Fine-Tuning (SFT) model (or reference policy), which is incorporated into the reward signal as per Equation 3 in the paper.
    -   **Hidden State Extraction**: Extracts the last layer hidden states from the LLM at appropriate points to serve as `s_tau` (states for macro actions) for the value function.
    -   **Training Loop**: Manages the training steps, including optimizer updates for both the policy LLM and a separate value head.

## Repository Structure:

-   `repo/main.py`: Entry point for demonstrating the MA-RLHF training process.
-   `repo/marlhf/`: Contains the core MA-RLHF implementation.
    -   `__init__.py`: Makes `marlhf` a Python package.
    -   `ma_rlhf.py`: Defines the `MARLHF` class, coordinating the overall RLHF training with macro actions.
    -   `ppo.py`: Implements the MA-PPO algorithm, including `ValueModel` for state-value estimation.
    -   `macro_actions.py`: Contains functions for generating different types of macro actions.

## Assumptions and Simplifications:

Due to the static nature of this benchmark and time constraints, certain aspects are simplified or left as placeholders:

-   **Reward Model**: The `_get_reward` function in `ma_rlhf.py` returns a random dummy reward. In a real scenario, this would involve a separately trained reward model (e.g., a sentiment classifier or a preference model) that provides meaningful feedback.
-   **KL Penalty Approximation**: The KL divergence penalty (`_calculate_kl_penalty`) is approximated by the mean difference of log probabilities of the *sampled* tokens. While common in practice, this is an approximation of the true distribution-level KL divergence.
-   **SFT Model**: The `sft_model` in `MARLHF` can be the same as the reference policy (`policy_lm_ref`) for initial setup. In a full pipeline, this would typically be a distinct model resulting from the supervised fine-tuning stage.
-   **Parsing-based and Perplexity-based Termination**: These macro action termination conditions (Section 3.2.1) are acknowledged but not implemented (`NotImplementedError`) due to their increased complexity and the prioritization of the n-gram approach which is stated as the default and best-performing in the paper.
-   **Optimizers and Learning Rate Schedules**: Simple Adam optimizers are used. Real-world RLHF often employs more sophisticated optimizers and learning rate schedules (e.g., linear warmup and decay) for large language models.
-   **Batching**: The `train_step` processes one prompt at a time. A full-scale implementation would involve batching multiple prompts for efficiency.
-   **Device Management**: Basic `.to(device)` calls are used, but more advanced distributed training setups are not covered.

## How to Use (Conceptual):

The `main.py` script provides a conceptual outline for running the MA-RLHF training:

```python
import torch
from marlhf.ma_rlhf import MARLHF
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # ... (configuration and setup)
    ma_rlhf_agent = MARLHF(...
    ma_rlhf_agent.run_training(prompts=prompts, num_epochs=num_epochs, sft_model=sft_model)
    # ... (demonstrate generation)

if __name__ == "__main__":
    main()
```

To conceptually run this, you would need to have `torch` and `transformers` libraries installed. The script demonstrates how to initialize the `MARLHF` agent, provide example prompts, and start the training process. Due to the static nature of this benchmark, direct execution for actual model training is not performed.

This reproduction provides a solid foundation for understanding and further developing MA-RLHF, with clear connections to the original paper's methodology.
