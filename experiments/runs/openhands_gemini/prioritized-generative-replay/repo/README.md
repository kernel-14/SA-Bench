
# Prioritized Generative Replay (PGR)

This repository contains a faithful reproduction of the "Prioritized Generative Replay" (PGR) paper by Wang et al. (2024).

## Overview

PGR proposes a scalable, guidable generative replay framework for online Reinforcement Learning. It leverages conditional diffusion models to capture online experience and generate new, relevant transitions, guided by "relevance functions" such as curiosity- or value-based metrics. This approach aims to improve sample efficiency and performance in both state- and pixel-based RL domains.

## Codebase Structure

- `config.py`: Contains all hyperparameters and configuration settings for the experiments.
- `data.py`: Handles data loading, preprocessing, and replay buffer management.
- `modules.py`: Defines core neural network modules like encoders, MLPs, and the diffusion model components.
- `model.py`: Implements the overall PGR framework, including the policy network, Q-functions, relevance functions, and the generative model.
- `train.py`: Contains the main training loop, including environment interaction, replay buffer updates, generative model training, and policy optimization.

## Key Contributions Implemented

- **Conditional Diffusion Models for Replay:** Implementation of conditional diffusion models to generate synthetic experience.
- **Relevance Functions:** Support for various relevance functions, including:
    - Return-based
    - Temporal Difference (TD) error-based
    - Intrinsic Curiosity Module (ICM) based (Pathak et al., 2017)
    - Random Network Distillation (RND) based (Burda et al., 2018)
    - Context-Tree Switching (CTS) Density Model based (Bellemare et al., 2016)
    - Episodic Curiosity (ECO) based (Savinov et al., 2018)
- **Prioritized Generative Replay (PGR) Framework:** Integration of the generative model with relevance functions to guide the generation of useful transitions.
- **Policy Learning with Synthetic Data:** Training off-policy RL algorithms (SAC/REDQ) using a mix of real and synthetic data.
- **Baselines:** Inclusion of baselines such as SYNTHER (unconditional generative replay), and model-free RL algorithms (SAC, REDQ).

## Usage

(Further instructions on how to run experiments will be provided here.)
