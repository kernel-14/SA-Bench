# Robotic World Model: A Neural Network Simulator for Robust Policy Optimization

This repository contains a reproduction attempt of the paper "Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics" (robotic-world-model).

## Core Contributions Reproduced

This reproduction focuses on the core contributions of the paper, including:

1.  **Robotic World Model (RWM) Architecture:** Implementation of the GRU-based world model with MLP heads for predicting observations and privileged information, as described in Section 3.2 and Appendix A.2.1.
2.  **Self-supervised Autoregressive Training:** Implementation of the training framework for RWM, including the multi-step prediction error loss function (Equation 2) and the dual-autoregressive mechanism for long-horizon robustness (Section 3.2, Figure S6).
3.  **Model-Based Policy Optimization with PPO (MBPO-PPO):** Outline of the policy optimization framework, combining model-based imagination with PPO for efficient and robust policy learning (Section 3.3, Algorithm 1). This includes the policy and value function network architectures (Appendix A.2.3).
4.  **Environment Configuration:** Definition of observation, action, and privileged information spaces, as well as the reward functions (Appendix A.1).

## Codebase Structure

The codebase is organized as follows:

-   `rwm/`: Contains the implementation of the Robotic World Model.
    -   `model.py`: Defines the RWM network architecture (GRU base, MLP heads).
    -   `trainer.py`: Implements the self-supervised autoregressive training logic.
-   `policy/`: Contains the implementation of the policy and value networks for MBPO-PPO.
    -   `model.py`: Defines the MLP-based policy and value function architectures.
-   `mbpo_ppo/`: Contains the MBPO-PPO training loop and related utilities.
    -   `agent.py`: Outlines the MBPO-PPO algorithm (Algorithm 1).
-   `env/`: Defines the environment-specific configurations.
    -   `spaces.py`: Specifies observation, action, and privileged information spaces (Tables S2, S3, S4, S5).
    -   `rewards.py`: Implements the reward functions (Section A.1.2, Table S6).
-   `config.py`: Centralized configuration for RWM and MBPO-PPO training parameters (Tables S10, S11).

## Assumptions and Missing Details

Due to the static nature of this benchmark (no code execution), the following assumptions and simplifications have been made:

*   **No Execution:** The provided code is a direct translation of the paper's descriptions into Python classes and functions. It is not executable and lacks actual training loops, data loading, or environment interaction implementations. The focus is on the architectural and algorithmic definitions.
*   **Placeholder for Environment Interaction:** Functions related to interacting with the environment (e.g., `collect_data`, `rollout_imagination`) are abstract representations based on the paper's descriptions.
*   **Detailed Hyperparameters:** All hyperparameters mentioned in the paper (e.g., learning rates, batch sizes, network hidden sizes, network architecture details, reward weights) are included in the `config.py` file or directly within the model definitions.
*   **No GPU Specifics:** While the paper mentions PyTorch and CUDA, the code does not include device-specific logic (e.g., `.to('cuda')`) as it's not executable.
*   **No External Libraries (beyond standard Python):** The code aims to be self-contained within Python built-in types and basic data structures, without external libraries like PyTorch itself, as per the rules. However, I will represent the *structure* of a PyTorch-like implementation. This means I will use Python classes to define the network layers and forward passes, but not import `torch` or use actual `torch.nn.Module` objects. This is a significant deviation from a runnable implementation but adheres to the "no imports" constraint outside the API context.

This reproduction attempts to provide a clear and structured representation of the proposed methods, serving as a blueprint for a functional implementation.
