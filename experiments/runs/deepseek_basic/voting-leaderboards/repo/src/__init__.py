"""Source code for reproducing "Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards".

This package implements the three main components of the paper:
1. De-anonymization of model responses (Section 2)
   - Identity-probing detector
   - Training-based detector (BoW, TF-IDF, Length features with logistic regression)

2. Estimating adversarial votes needed (Section 3)
   - Bradley-Terry model simulation
   - Simulation pipeline with attacker behavior

3. Mitigations (Section 4)
   - Cost model for attacks
   - Malicious user identification (likelihood-based anomaly detection)
   - Perturbed leaderboard release (Scenario 2)
"""
