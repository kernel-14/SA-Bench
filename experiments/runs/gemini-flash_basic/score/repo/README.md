# SCoRe: Training Language Models to Self-Correct via RL

This repository aims to reproduce the core contributions of the paper "Training Language Models to Self-Correct via Reinforcement Learning (SCoRe)".

## Paper Summary

The paper introduces SCoRe, a multi-turn online reinforcement learning (RL) approach designed to improve a Large Language Model's (LLM) self-correction ability using entirely self-generated data. The authors identify two main shortcomings of existing Supervised Fine-Tuning (SFT) based methods for self-correction:
1. **Distribution Mismatch**: A discrepancy between mistakes made by the data-collection policy and the model's own responses.
2. **Behavior Collapse**: The learning process implicitly favors a certain mode of correction behavior that is often ineffective.

SCoRe addresses these issues by:
- Training under the model’s own distribution of self-generated correction traces.
- Using a two-stage regularization process to steer the learning:
    - **Stage I**: An initial phase of multi-turn RL on a base model to generate a policy initialization less susceptible to collapse. This stage focuses on decoupling the first and second attempts by improving second-attempt accuracy while constraining the first attempt to be close to the base model using a KL-divergence penalty.
    - **Stage II**: Training on both attempts to maximize reward, employing a reward bonus to amplify self-correction by rewarding "progress" (i.e., changing an incorrect answer to a correct one).

SCoRe demonstrates state-of-the-art self-correction performance on MATH and HumanEval datasets, significantly improving the base models' self-correction capabilities.

## Core Contributions to Replicate

The core contributions of this paper revolve around the SCoRe algorithm itself, which involves a two-stage multi-turn RL training process with specific reward shaping.

1.  **SCoRe Algorithm Implementation**:
    *   **Stage I: Decoupling Attempts**: Implementing the RL objective for Stage I, which involves maximizing the reward of the second attempt while applying a KL-divergence penalty to keep the first attempt close to the reference policy.
    *   **Stage II: Multi-Turn RL with Reward Shaping**: Implementing the RL objective for Stage II, which jointly optimizes both attempts with a reward shaping term that encourages progress in self-correction.
2.  **Evaluation Metrics**: Implementing the evaluation metrics used in the paper, specifically:
    *   Accuracy@t1
    *   Accuracy@t2
    *   Δ(t1, t2)
    *   Δi→c(t1, t2)
    *   Δc→i(t1, t2)
3.  **Experimental Setup (as much as possible within the static nature of this task)**:
    *   Defining the problem setup for intrinsic self-correction.
    *   Specifying the prompt structure for initial attempts and self-correction instructions.
    *   Outlining the use of an oracle reward function for evaluation.

## Out of Scope / Not Replicated

-   Any experiments introduced only in the Appendix.
-   Running the actual training or evaluation of models. This is a static-only benchmark.
-   Specific model architectures (Gemini 1.0 Pro, Gemini 1.5 Flash) and their fine-tuning details beyond the algorithm itself.
-   The generation of self-generated data beyond defining the process.
-   Detailed hyperparameter tuning (unless explicitly stated in the main paper and critical to the algorithm).

## Repository Structure (Planned)

```
repo/
├── README.md
├── score/
│   ├── __init__.py
│   ├── model.py            # Defines the SCoRe model architecture and forward pass
│   ├── training.py         # Implements the two-stage RL training process (Stage I and Stage II objectives)
│   ├── rewards.py          # Defines reward functions, including the shaped reward for Stage II
│   ├── datasets.py         # Handles data loading and preparation (e.g., MATH, HumanEval)
│   └── utils.py            # Utility functions (e.g., metric calculation, prompting)
└── config.yaml           # Configuration file for hyperparameters and settings
```

## Assumptions and Unresolved Details

-   The paper mentions using a "REINFORCE policy gradient training approach with a KL-divergence penalty against a fixed model (Ahmadian et al., 2024)". We assume a standard implementation of REINFORCE with a KL penalty will be used.
-   Specific details of the "oracle reward" function (e.g., how exactly `r(y, y*)` is computed for MATH and HumanEval beyond "answer matches ground truth" or "passes all test cases") will need to be approximated or left as placeholders if not fully specified.
-   The exact implementation of the "multi-turn MDP" and "hierarchical framework of Zhou et al. (2024)" will be based on the provided equations and descriptions.
-   The exact prompts and self-correction instructions are mentioned to be in Appendix C, which is out of scope. We will use placeholder prompts based on the examples in the main text.

This `README.md` will be updated as the reproduction progresses.
