# Reproduction of "Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards"

This repository contains a reproduction attempt of the paper:

> **Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards**
> Yangsibo Huang, Milad Nasr, Anastasios Angelopoulos, Nicholas Carlini, Wei-Lin Chiang, Christopher A. Choquette-Choo, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Ken Ziyu Liu, Ion Stoica, Florian Tramer, Chiyuan Zhang
> ICML 2025

## Paper Summary

The paper studies the security of voting-based LLM leaderboards (specifically Chatbot Arena). They show that:

1. **De-anonymization (Section 2)**: An attacker can identify which model generated a given response with >95% accuracy using simple classifiers (BoW/TF-IDF features + logistic regression) and identity-probing prompts.

2. **Adversarial Voting (Section 3)**: Using simulations, they estimate that a few thousand adversarial votes can significantly shift model rankings on the leaderboard.

3. **Mitigations (Section 4)**: They propose and analyze several defenses including authentication, rate limiting, malicious user detection (likelihood-based anomaly detection), and perturbed leaderboard release.

## Repository Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── run_experiments.py           # Main experiment runner
├── src/                         # Source code
│   ├── __init__.py
│   ├── deanonymization.py       # Section 2: De-anonymization detectors
│   ├── simulation.py            # Section 3: Bradley-Terry simulation & attack
│   ├── mitigations.py           # Section 4: Defenses & cost model
│   ├── models.py                # Model registry & data utilities
│   ├── visualization.py         # Figure reproduction
│   └── experiments.py           # Experiment orchestration
├── notebooks/                   # Jupyter notebooks for analysis
├── configs/                     # Configuration files
├── figures/                     # Generated figures
└── data/                        # Data directory
```

## What Has Been Implemented

### Section 2: De-anonymization of Model Responses

- **Identity-Probing Detector** (Section 2.4.1): Implementation of the five prompts from the paper ("Who are you?", "Which model are you?", etc.) with keyword-based detection for model families.
- **Training-Based Detector** (Section 2.4.2): 
  - Text feature extraction: Length (word/char), BoW, TF-IDF
  - Logistic regression classifier with 80/20 train/test split (random_state=42)
  - Prompt selection utilities
  - PCA visualization of BoW features (Figure 2)
- Cross-category evaluation across 8 prompt types (Table 1): English, Chinese, Spanish, Indonesian, Persian, Coding, Math, Safety-violating

### Section 3: Estimating Adversarial Votes

- **Bradley-Terry Model**: Elo-like rating system with scaling factor
- **Attack Simulation**: 
  - Configurable attacker with detection accuracy, false positive/negative rates
  - Non-target strategies: abstain, random upvote, vote tie, vote both bad
  - Objectives: Up(M, x) and Down(M, x) for rank manipulation
- **Vote Estimation**: 
  - High-ranked and low-ranked model analysis (Tables 4, 5)
  - Ablation on detector accuracy (Table 8)
  - Ablation on non-detection strategies (Table 9)

### Section 4: Mitigations

- **Cost Model** (Section 4.1): Attack cost = detector_cost + ceil(N/m) * c_account + N * c_action
- **Malicious User Detection - Scenario 1** (Section 4.2.3): 
  - Likelihood ratio test with known benign distribution
  - Empirical p-value computation via simulation
  - Detection of both naive and smart adversaries (Figure 4)
- **Perturbed Leaderboard - Scenario 2** (Section 4.2.3):
  - Gaussian noise perturbation of Bradley-Terry ratings
  - Neyman-Pearson optimal detection
  - Detection rate vs noise analysis (Figure 5)
  - Utility loss vs noise analysis (Figure 6)
- **Authentication & Rate Limiting**: Analysis of how they affect attack cost
- **CAPTCHA & Prompt Uniqueness**: Cost estimation for these defenses

## Assumptions and Missing Details

Since this is a static reproduction without access to real API keys or the actual Chatbot Arena dataset, the following assumptions were made:

1. **Synthetic data**: Model responses are generated synthetically with distinctive patterns per model family. Real experiments require API access to the actual models.

2. **Voting data**: Historical voting data is synthetically generated based on the statistics reported in the paper (1.67M votes, 34.5% tie rate, 6,895 unique comparisons).

3. **Bradley-Terry coefficients**: The paper does not specify the exact Elo K-factor used; we use K=32 (standard) and scale=400.

4. **Simulation parameters**: The paper mentions using a simulation platform from Chatbot Arena team. We implement our own simulation with the same principles.

5. **The exact scaling factor and coefficient initialization** for Bradley-Terry were not specified; we use reasonable defaults.

6. **The exact noise distributions** for perturbed leaderboard analysis were not fully specified beyond "scaled Gaussian noise."

7. **API costs**: We use the pricing stated in Appendix A.3 ($5.00/1M tokens for proprietary, $1.80/1M for open-source via Together AI).

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running Experiments

```bash
# Run all experiments (uses synthetic data by default)
python run_experiments.py

# Run with specific number of models
python run_experiments.py --n_models 10

# Specify output directory
python run_experiments.py --output_dir my_results
```

### Using Individual Modules

```python
from src.deanonymization import TrainingBasedDetector, IdentityProbingDetector
from src.simulation import BradleyTerryModel, LeaderboardSimulation, AttackerConfig
from src.mitigations import AttackCost, MaliciousUserDetector, PerturbedLeaderboard

# Example: Train a de-anonymization detector
detector = TrainingBasedDetector(
    target_model="gpt-4o-mini-2024-07-18",
    feature_type="bow",
    random_state=42
)
result = detector.train(target_responses, other_responses)
print(f"Detection accuracy: {result['test_accuracy']:.3f}")

# Example: Simulate an attack
sim = LeaderboardSimulation(model_names, initial_votes=historical_votes)
config = AttackerConfig(
    target_model="claude-3-5-sonnet-20240620",
    direction="up",
    detector_accuracy=0.95
)
result = sim.run_attack(config, target_position_change=1)

# Example: Analyze cost with defenses
cost = AttackCost(n_actions=3000, max_actions_per_account=100, 
                   cost_per_account=1.0, cost_per_action=0.002)
print(f"Total attack cost: ${cost.total_cost():.2f}")
```

## Key Results (from Synthetic Data)

The synthetic data experiments demonstrate the feasibility of the approach:
- BoW features achieve highest detection accuracy among simple features
- ~1000-5000 adversarial votes are needed to shift rankings by 1-4 positions
- Malicious user detection is effective against naive adversaries but harder against smart ones
- Perturbing the leaderboard creates a security-utility tradeoff

## Limitations

This reproduction is limited by:
1. No access to real model APIs for response generation
2. No access to the actual Chatbot Arena voting logs
3. No access to the Chatbot Arena simulation platform mentioned in the paper
4. The experiments in this repository are for demonstration and validation of the methodology only

## References

- Chatbot Arena: https://chat.lmsys.org/
- LMSYS-Chat-1M dataset (Zheng et al., 2023)
- Bradley-Terry models (Hunter, 2004)
