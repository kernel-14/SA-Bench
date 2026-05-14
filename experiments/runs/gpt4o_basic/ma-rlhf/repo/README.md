-e # MA-RLHF Replication

## Overview
This repository is dedicated to reproducing the core contributions of the paper 'MA-RLHF: RL from Human Feedback with Macro Actions.' The paper proposes a novel framework for Reinforcement Learning from Human Feedback (RLHF) that incorporates macro actions. This approach addresses the credit assignment problem in long sequences and accelerates training convergence.

## Main Contributions
1. **Integration of Macro Actions:** RLHF is augmented with temporally extended actions using fixed gram, randomized gram, parsing-based, and perplexity-based strategies.
2. **Adaptation of PPO Optimization:** MA-PPO modifies Proximal Policy Optimization to evaluate and optimize macro actions.
3. **Extensive Evaluation:** Performance validated on datasets including TL;DR (summarization), HH-RLHF (dialogue generation), WebGPT Comparison (question answering), and APPS (code generation).
4. **Scalability and Efficiency:** Demonstrated faster training (.7 \sim 2$ times faster) with higher reward scores across model sizes (2B to 27B).

## Repository Structure
- **datasets/**: Scripts for downloading and preprocessing TL;DR, HH-RLHF, WebGPT, and APPS datasets.
- **models/**: Code for implementing Gemma base models and reward models.
- **src/**: Core implementation of MA-RLHF, macro action termination strategies, MA-PPO, etc.
- **scripts/**: Utility scripts for evaluation, metrics computation, and task-specific workflows.
- **results/**: Placeholder for storing replication results, including RM scores and evaluation outputs.

## Dependencies
- PyTorch
- Transformers
- Deepspeed
- Datasets (Hugging Face)
- Other RL libraries

## Next Steps
The implementation will proceed by:
1. Developing dataset loaders and preprocessing scripts.
2. Coding the MA-RLHF framework with macro action logic and MA-PPO optimization.
3. Running ablation studies for macro action termination strategies.
4. Evaluating with RM scores, GPT-4 evaluation, and human feedback.

For questions or issues, please reach out to the repository maintainer.
