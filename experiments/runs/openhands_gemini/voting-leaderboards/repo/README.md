
# Reproducing "EXPLORING AND MITIGATING ADVERSARIAL MANIPULATION OF VOTING-BASED LEADERBOARDS"

This repository contains a reproduction of the core contributions of the paper "EXPLORING AND MITIGATING ADVERSARIAL MANIPULATION OF VOTING-BASED LEADERBOARDS". The implementation focuses on the de-anonymization methods (Identity-probing and Training-based detectors) and the simulation setup for estimating adversarial votes.

## Codebase Structure

- `config.py`: Contains all hyperparameters and configuration settings for the detectors and simulations.
- `data.py`: Handles data generation and preprocessing for training the detectors. This includes creating synthetic responses for models based on specified features (Length, TF-IDF, BoW).
- `models.py`: Implements the `IdentityProbingDetector` and `TrainingBasedDetector` classes, including feature extraction (TF-IDF, BoW) and the Logistic Regression classifier.
- `simulation.py`: Implements the Chatbot Arena simulation environment, including Bradley-Terry model updates and the logic for adversarial voting.
- `main.py`: The main script to orchestrate data generation, detector training, and simulation runs.
- `requirements.txt`: Lists all Python dependencies required to run the code.
