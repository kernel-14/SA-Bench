# Reproduction of Interpreting Emergent Planning in Model-Free RL

This repository aims to reproduce the core contributions of the paper Interpreting Emergent Planning in Model-Free RL. The primary goal is to demonstrate that model-free RL agents can learn to plan internally, using a methodology based on concept-based interpretability.

## Core Contributions to be Reproduced

1.  **Concept Probing**: Show that the DRC agent internally represents planning-relevant concepts ($: Agent Approach Direction, $: Box Push Direction) using linear probes.
2.  **Plan Formation Analysis**: Provide qualitative and quantitative evidence that the agent uses these concepts to iteratively form and refine plans, resembling parallelized bidirectional search.
3.  **Causal Intervention**: Demonstrate that intervening on these concept representations causally influences the agent's behavior, steering it to execute specific plans.

## Codebase Structure

The codebase will be organized into the following main components:

*   : Contains the implementation of the Sokoban environment, including its dynamics, reward structure, and symbolic observation generation.
*   : Houses the Deep Repeated ConvLSTM (DRC) agent architecture, including the convolutional encoder, ConvLSTM layers with skip connections, and policy/value heads.
*   : Scripts and configurations for training the DRC agent using IMPALA, based on the parameters specified in the paper.
*   : Definitions and operationalization of $ and $, along with the logic for generating ground truth labels by simulating agent behavior.
*   : Implementation of linear probes (1x1 and 3x3 convolutional probes) for predicting concept classes from agent activations. Includes probe training and evaluation scripts.
*   : Scripts for data collection (agent trajectories, activations, ground truth concept labels) for probe training and analysis.
*   : Contains tools for analyzing plan formation, visualizing internal plans, and performing intervention experiments.
*   : General utility functions.

## Assumptions and Unresolved Details

*   **Boxoban Dataset**: The paper references the Boxoban unfiltered training dataset and validation dataset. I assume these datasets are publicly available or can be generated using a standard Boxoban environment setup. Details on generating these specific level sets will be investigated further.
*   **IMPALA Implementation**: A robust implementation of the IMPALA algorithm will be required for agent training. I will aim for a standard, well-tested implementation.
*   **ConvLSTM Details**: While the paper provides architectural details, some fine-grained implementation specifics of the ConvLSTM (e.g., exact gating mechanisms if they deviate from standard LSTMs) might require careful inference from the text or reliance on common ConvLSTM implementations.
*   **Random Seed Management**: Proper management of random seeds will be crucial for reproducibility across all experiments (agent training, probe training, data collection).
*   **Hardware/Computational Resources**: The paper's experiments (e.g., 250 million transitions for agent training, 3000 episodes for probe data) imply significant computational resources. The reproduction will focus on the code structure and logic, assuming suitable hardware availability for execution.
*   **Visualizations**: Reproducing the exact visualizations (e.g., Figure 1, Figure 5) will be a goal, but the primary focus is on the underlying data generation and analysis methods.

This README will be updated as the reproduction effort progresses.
