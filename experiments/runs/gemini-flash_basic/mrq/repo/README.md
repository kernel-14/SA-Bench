# MR.Q Reproduction

This repository aims to reproduce the core contributions of the paper "Towards General-Purpose Model-Free RL (MR.Q)" by Fujimoto et al. The paper introduces MR.Q, a model-free deep RL algorithm that leverages model-based representations for general-purpose problem-solving across diverse domains with a single set of hyperparameters.

## Core Contributions to Reproduce

We aim to implement the following core components of the MR.Q algorithm as described in the paper:

1.  **Encoder Network**:
    *   **State Encoder (\$\mathit{f}_{\omega}\$)**: Maps raw states (\$\mathit{s}\$) to state embeddings (\$\mathbf{z}_s\$). This will include separate architectures for image-based and vector-based observations.
    *   **State-Action Encoder (\$\mathit{g}_{\omega}\$)**: Combines state embeddings and actions to produce state-action embeddings (\$\mathbf{z}_{sa}\$). It also incorporates a linear MDP predictor.

2.  **Value Networks (\$\mathit{Q}_{\theta}\$)**:
    *   Two separate Q-networks as per TD3, taking state-action embeddings (\$\mathbf{z}_{sa}\$) as input to predict Q-values.

3.  **Policy Network (\$\pi_{\phi}\$)**:
    *   Maps state embeddings (\$\mathbf{z}_s\$) to actions (\$\mathit{a}\$). Will handle both continuous and discrete action spaces.

4.  **Loss Functions**:
    *   **Encoder Loss**: A composite loss including:
        *   Reward Loss (\$\mathcal{L}_{Reward}\$): Categorical cross-entropy with Two-Hot encoding for rewards.
        *   Dynamics Loss (\$\mathcal{L}_{Dynamics}\$): MSE between predicted and target next state embeddings.
        *   Terminal Loss (\$\mathcal{L}_{Terminal}\$): MSE for the terminal signal.
    *   **Value Loss (\$\mathcal{L}_{Value}\$)**: Modified TD3 loss with multi-step returns, Huber loss, and reward scaling.
    *   **Policy Loss (\$\mathcal{L}_{Policy}\$)**: Deterministic policy gradient with a pre-activation regularization term.

5.  **Key Mechanisms**:
    *   Target Networks for encoder, value, and policy, with periodic updates.
    *   Reward scaling for value targets.
    *   Prioritized Experience Replay (LAP) for sampling transitions.
    *   Exploration strategies for discrete and continuous action spaces (Gaussian noise, Gumbel-Softmax for discrete actions).
    *   Unrolled dynamics for the encoder loss over a short horizon.

## Assumptions and Missing Details

*   **Reward Bins and Range**: The paper specifies `Reward bins: 65` and `Reward range: [-10, 10] (effective: [-22k, 22k])`. The mapping from `[-10, 10]` to `[-22k, 22k]` and the precise implementation of `symexp` for two-hot encoding is implemented based on DreamerV3 and assumes `symexp` scaling for mapping.
*   **Action Space Scaling**: The paper mentions scaling action noise and clipping according to the range of the action space, but doesn't specify the exact scaling factor. We assume a default range of [-1, 1] for continuous actions and scale noise/clipping accordingly.
*   **`output_dim` in State-Action Encoder**: The `output_dim` for the linear MDP predictor is inferred to be the sum of `MRQConfig.REWARD_BINS`, `MRQConfig.ZS_DIM` (for next state embedding), and `1` (for terminal signal).
*   **Layer Normalization (`LinearNormalizedActivation`)**: `LinearNormalizedActivation` class is implemented to apply `LayerNorm` followed by the activation function, as interpreted from the pseudo-code.
*   **Replay Ratio**: Not explicitly given in Table 3. Assumed to be `1` (meaning one gradient update per environment step).
*   **Optimizer Details**: `AdamW` is specified, but specific parameters like `eps` or `betas` are not provided. Standard PyTorch defaults for `AdamW` will be used.
*   **Environment Specifics**: Details about state channels for image observations and state dimensions for vector observations will need to be passed as parameters to the model. We use 84x84 input for CNN as mentioned in the paper for image observations.
*   **Multi-step Returns and Unrolled Dynamics (Simplification)**: The current `agent.py` implementation performs a simplified version of multi-step returns and encoder unrolling. A full multi-step implementation would require a replay buffer capable of sampling sequences of transitions. Currently, the agent updates using single-step transitions sampled from the replay buffer. For the encoder, only the first step of the unroll is used to calculate the loss for the dynamics and reward prediction, assuming that the `ENCODER_HORIZON` effectively means the model predicts `H_Enc` steps *ahead*, but is currently only trained on the next-step prediction based on available data. For value loss, the `MULTI_STEP_RETURNS_HORIZON` is considered when calculating the target Q-value, but assumes that the `rewards_batch` already contains the sum of rewards over that horizon, which is a simplification for a single-transition replay buffer. 
*   **Missing Hyperparameters from Table 3**: 
    *   `Dynamics loss weight λDynamics`: Not explicitly stated, assumed to be `1.0`.
    *   `Encoder horizon HEnc`: Not explicitly stated, assumed to be `3`.
    *   `Pre-activation loss weight λpre-activ`: Given as "1e-5 5", interpreted as `1e-5` for the regularization term.

## Codebase Structure

The codebase is organized into a `mrq_code` directory, containing Python files for each major component:

*   `models.py`: Contains the definitions for the State Encoder, State-Action Encoder, Value Networks, and Policy Network.
*   `losses.py`: Implements the various loss functions (Reward, Dynamics, Terminal, Value, Policy).
*   `agent.py`: Orchestrates the training and interaction with the environment, combining the models and loss functions.
*   `replay_buffer.py`: Implementation of the replay buffer, including prioritized sampling.
*   `utils.py`: Helper functions (e.g., for reward scaling, two-hot encoding, Gumbel-Softmax, initialization).
*   `config.py`: Centralized configuration for hyperparameters.
*   `main.py`: Main script for running the MR.Q agent and a basic training loop.

