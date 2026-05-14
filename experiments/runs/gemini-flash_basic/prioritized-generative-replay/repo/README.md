# Reproduction of "Prioritized Generative Replay"

This repository contains a conceptual code-replication attempt of the paper "Prioritized Generative Replay" by Wang et al. (2024). The goal was to reproduce as many of the core contributions of the paper as possible within the given time constraints and the static-only nature of the benchmark.

## Paper Summary

"Prioritized Generative Replay" (PGR) proposes a novel approach to sample-efficient online reinforcement learning by augmenting traditional replay buffers with a conditional generative model, specifically a diffusion model. This generative model learns to capture and densify past experience, allowing the agent to generate new, relevant training data. The key innovation is the use of "relevance functions" to guide the generation process towards transitions that are most beneficial for policy learning, thereby addressing the limitations of uniform replay and the potential overfitting issues of simple prioritized experience replay. The paper demonstrates that conditioning on intrinsic curiosity metrics effectively promotes diversity in generated transitions and reduces overfitting, leading to improved performance and sample efficiency in both state- and pixel-based RL tasks.

## Replicated Core Contributions

This codebase conceptually replicates the following core contributions and components described in the paper:

1.  **Prioritized Generative Replay (PGR) Framework:** The central `pgr.PrioritizedGenerativeReplay` module encapsulates the interaction between the generative model and the relevance function, managing the training of the diffusion model and the generation of synthetic transitions.
2.  **Conditional Diffusion Model for Replay:** A `models.DiffusionModel` is implemented, representing the conditional generative model used to capture and densify online experience. It includes the forward diffusion process (`q_sample`) and a simplified reverse (sampling) process (`sample`) that incorporates classifier-free guidance.
3.  **Relevance Functions (`\mathcal{F}`):** The codebase includes implementations for three key relevance functions discussed in the paper:
    *   **`pgr.relevance_functions.ReturnRelevanceFunction` (Eq. 3):** Based on the Q-value estimate of the learned Q-function and current policy.
    *   **`pgr.relevance_functions.TDErrorRelevanceFunction` (Eq. 4):** Based on the temporal difference error.
    *   **`pgr.relevance_functions.CuriosityRelevanceFunction` (Eq. 5):** Leverages an intrinsic curiosity module (ICM) consisting of a `FeatureEncoder` and `ForwardDynamicsModel` to measure prediction error, promoting novelty.
4.  **Classifier-Free Guidance (CFG):** The `DiffusionModel` and `PrioritizedGenerativeReplay` classes incorporate the mechanisms for classifier-free guidance during both the training of the generative model (randomly dropping conditions with `p_uncond`) and during sampling (using a `guidance_scale` for conditional and unconditional predictions).
5.  **Integration with an Online RL Algorithm:** The framework is designed to work with an off-policy RL algorithm. A `agents.SACAgent` (Soft Actor-Critic) is implemented as a placeholder, demonstrating how the generated synthetic data would be used to train the policy.
6.  **Replay Buffer Management:** Separate `utils.ReplayBuffer` instances are used for real and synthetic experiences, and the `main.py` demonstrates how these buffers would be mixed for policy training.
7.  **Conceptual Training Loop (Algorithm 1):** The `main.py` script provides a high-level structure of the outer and inner loops, illustrating data collection, relevance function updates, generative model training, synthetic data generation, and policy optimization.

## Codebase Structure

The repository is organized into the following directories:

-   `pgr/`: Contains the core Prioritized Generative Replay logic.
    -   `pgr.py`: Defines the `PrioritizedGenerativeReplay` class, orchestrating generative model training and synthetic data generation using relevance functions.
    -   `relevance_functions.py`: Implements the abstract `RelevanceFunction` base class and concrete subclasses for Return, TD-Error, and Curiosity-based relevance metrics.
-   `models/`: Contains neural network architectures.
    -   `diffusion.py`: Implements the `DiffusionModel` responsible for the conditional generation of transitions, including the noise prediction network and sampling procedure.
    -   `networks.py`: Defines common neural network modules like `MLP`, `FeatureEncoder`, `QNetwork`, `PolicyNetwork`, and `ForwardDynamicsModel` that are building blocks for the diffusion model, relevance functions, and the RL agent.
-   `agents/`: Contains the reinforcement learning agent implementations.
    -   `sac_agent.py`: Implements a Soft Actor-Critic (SAC) agent, a standard off-policy algorithm suitable for integration with PGR.
-   `utils/`: Contains utility functions and classes.
    -   `replay_buffer.py`: A basic replay buffer implementation for storing and sampling transitions.
-   `config.py`: Stores all configurable hyperparameters for the training process and model architectures.
-   `main.py`: The main script that orchestrates the overall training process, showcasing the interaction between the environment (mocked), PGR framework, and the SAC agent according to the paper's Algorithm 1.

## Implementation Details and Assumptions

Given the static nature of this benchmark, the code is primarily conceptual and illustrates the architecture and data flow rather than being fully runnable for experiments. Key assumptions and simplifications include:

*   **Environment Interaction:** Environment steps are mocked in `main.py` (e.g., `torch.randn` for observations, actions, rewards). A real implementation would require a `gymnasium` or similar environment setup.
*   **Dimensions:** `obs_dim` and `action_dim` are set as placeholders in `config.py`. Actual values would depend on the specific environment.
*   **Diffusion Model Architecture:** The `DiffusionModel` uses a simple MLP for its noise prediction network (`epsilon_theta`). While sufficient for demonstrating the diffusion process and guidance mechanisms, a full-scale replication, especially for pixel-based tasks, would likely require more complex architectures like U-Nets (as often used in diffusion models) or specific convolutional encoders mentioned in the paper's experimental setup (e.g., for pixel-based DMC).
*   **Relevance Function Dependencies:**
    *   `ReturnRelevanceFunction` and `TDErrorRelevanceFunction` assume access to trained `QNetwork` and `PolicyNetwork` instances. In a real setup, these would be part of the active RL agent.
    *   `TDErrorRelevanceFunction`'s calculation of target Q-values for continuous actions is based on the current policy's actions (as per SAC). The paper's mention of `argmax_a' Q(s', a')` is interpreted in the context of continuous control algorithms (like SAC/REDQ) where the policy typically provides the 'optimal' next action for target calculation.
    *   `CuriosityRelevanceFunction` relies on a `FeatureEncoder` and `ForwardDynamicsModel` that are trained via an MSE loss to predict the next latent state. This training is conceptually integrated into the PGR's `update_relevance_function` method.
*   **Condition Sampling for Generation:** In `main.py`, the conditions (`conditions_for_generation`) for generating synthetic transitions are obtained by calculating relevance scores for a batch of real transitions and then selecting the top-k most relevant scores. This aligns with the paper's description of sampling conditioning values from the highest-relevance transitions in the real replay buffer.
*   **Training Loop Details:** The `main.py` outlines the periodic updates for the generative model and the policy. The ratio of real to synthetic data (`synthetic_data_ratio`) and the guidance scale are configurable.
*   **Replay Buffer in SAC Agent:** The `SACAgent.update_parameters` method expects a `ReplayBuffer` object. For mixing real and synthetic data, a `DummyReplayBuffer` is used in `main.py` to wrap the combined batch, simplifying the integration without modifying the `SACAgent`'s interface directly.
*   **Optimizers:** Basic Adam optimizers are used for all trainable components.

## Missing Details and Future Work

For a fully functional and experimentally reproducible system, the following would need further development:

*   **Actual Environment Integration:** Replace mock environment interactions with a real `gymnasium` environment setup.
*   **Hyperparameter Tuning:** Systematically tune hyperparameters for specific environments, including those for the diffusion model, RL agent, and relevance functions.
*   **Detailed Network Architectures:** Implement more sophisticated network architectures, especially for the diffusion model (e.g., U-Net based) and feature encoders for pixel-based environments, as per the paper's experimental details.
*   **Full Evaluation Suite:** Implement a comprehensive evaluation system to track episode rewards, sample efficiency, and other metrics mentioned in the paper.
*   **Multiple Relevance Functions:** Extend `main.py` to allow easy switching and experimentation with different relevance functions as the guiding signal.
*   **Distributed Training:** For large-scale experiments, implement distributed training mechanisms as often used in state-of-the-art RL setups.
*   **More Robust Logging and Experiment Tracking:** Integrate with tools like Weights & Biases or TensorBoard for better experiment management.

## How to Use/Extend This Codebase

This codebase serves as a blueprint for implementing "Prioritized Generative Replay." To make it runnable and conduct experiments:

1.  **Replace Mock Environment:** Integrate a real `gymnasium` environment, ensuring `obs_dim` and `action_dim` in `config.py` match the environment.
2.  **Populate Replay Buffer:** Implement the initial data collection phase to fill the `real_replay_buffer` before starting the main training loop.
3.  **Refine Network Architectures:** Depending on the complexity of the environment (e.g., pixel observations), replace the `MLP`s with more suitable convolutional networks as described in the paper.
4.  **Implement Training and Evaluation:** Complete the `evaluate` function and ensure proper logging of training progress.
5.  **Run Experiments:** Execute `main.py` and observe the agent's performance with PGR. Experiment with different relevance functions and hyperparameters as described in the paper.

This reproduction provides a strong conceptual foundation for understanding and further developing the Prioritized Generative Replay framework.
