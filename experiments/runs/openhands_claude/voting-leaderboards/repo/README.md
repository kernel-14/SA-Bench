# Adversarial Manipulation of Voting-Based Leaderboards

Reproduction of **"Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards"** (Huang et al., 2024).

## Overview

This codebase reproduces the three core contributions of the paper:

1. **De-anonymization of model responses** (Section 2): Shows that LLM responses can be attributed to their source model with >95% accuracy using simple text features.
2. **Adversarial voting simulation** (Section 3): Estimates that ~1,000 adversarial votes suffice to shift a model's leaderboard rank by several positions.
3. **Mitigations** (Section 4): Implements and evaluates defenses including likelihood-based malicious user detection and perturbed leaderboard release.

## Repository Structure

```
repo/
├── config.py          # All hyperparameters, model lists, and configuration dataclasses
├── features.py        # Text feature extraction: BoW, TF-IDF, Length (Section 2.3)
├── data.py            # Data loading, synthetic data generation, dataset construction
├── detector.py        # Identity-probing and training-based model detectors (Section 2.2)
├── bradley_terry.py   # Bradley-Terry model with MM algorithm (Section 3.1)
├── simulation.py      # Adversarial voting simulation pipeline (Section 3)
├── mitigation.py      # Defense mechanisms: likelihood tests, perturbed leaderboard (Section 4)
├── train.py           # End-to-end training and experiment orchestration
├── evaluate.py        # Metrics, formatting, and visualization utilities
├── requirements.txt
└── README.md
```

## Key Algorithms

### Training-Based Detector (Section 2.3)
- Binary logistic regression classifier per (prompt, target_model) pair
- Features: Bag-of-Words (primary), TF-IDF, word/character length
- 80/20 train/test split, `random_state=42`, sklearn defaults
- Achieves >95% accuracy on most models with BoW features

### Bradley-Terry Model (Section 3.1)
- MM (Minorization-Maximization) algorithm from Hunter (2004)
- Win probability: `Pr(i > j) = 1 / (1 + exp(-(Q_i - Q_j) / s))`
- Ties handled as 0.5 wins for each model

### Adversarial Voting Simulation (Section 3.1)
- Attacker uses detector (95% accuracy, symmetric 5% FP/FN rates)
- Votes for/against target model when detected; abstains otherwise
- BT coefficients recalculated every 1,000 interactions
- Tracks cumulative votes and interactions to achieve target rank

### Malicious User Detection (Section 4.2.3)
- **Scenario 1**: Likelihood ratio test against known benign distribution; empirical p-value with α=0.01
- **Scenario 2**: Neyman-Pearson test using perturbed leaderboard ratings; Gaussian noise added to BT coefficients

## Usage

### Run all experiments (synthetic data)
```bash
python train.py --experiment all --output_dir results/
```

### Run only the detector experiment
```bash
python train.py --experiment detector --num_prompts 200 --num_responses 50
```

### Run simulation with real voting data
```bash
python train.py --experiment simulation --votes_file path/to/arena_votes.json
```

### Run with real model responses
```bash
python train.py --experiment detector --responses_file path/to/responses.json
```

### Skip ablations (faster)
```bash
python train.py --experiment simulation --no_ablations
```

## Data Formats

### Voting data (JSON)
```json
[
  {"model_a": "gpt-4o", "model_b": "claude-3-5-sonnet", "winner": "model_a", "user_id": "u1"},
  {"model_a": "llama-3.1-70b", "model_b": "gemini-1.5-pro", "winner": "tie"}
]
```

### Model responses (JSON)
```json
{
  "category_name": {
    "prompt text": {
      "model-name": ["response1", "response2", ...]
    }
  }
}
```

## Reproducing Paper Results

| Paper Result | Function | Output |
|---|---|---|
| Table 2/7: Identity-probing accuracy | `evaluate_identity_probing_detector()` in `detector.py` | Per-model, per-prompt accuracy |
| Table 3: Feature comparison | `run_detector_experiment()` in `train.py` | `results/detector_results.json` |
| Figure 3: Category accuracy | `evaluate_detector_by_category()` in `detector.py` | `results/detector_results.json` |
| Tables 4/5: Attack votes needed | `run_simulation_experiment()` in `train.py` | `results/simulation_results.json` |
| Table 8: Detector accuracy ablation | `simulate_varying_detector_accuracy()` in `simulation.py` | `results/simulation_results.json` |
| Table 9: Non-target strategy ablation | `simulate_varying_non_target_strategy()` in `simulation.py` | `results/simulation_results.json` |
| Figure 4: Scenario 1 detection | `simulate_detection_scenario1()` in `mitigation.py` | `results/mitigation_results.json` |
| Figures 5/6: Perturbed leaderboard | `evaluate_noise_scales()` in `mitigation.py` | `results/mitigation_results.json` |
| Section 4.1: Attack cost | `compute_attack_cost()` in `mitigation.py` | `results/mitigation_results.json` |

## Notes

- All experiments run on **synthetic data** by default (no API keys required).
- To reproduce exact paper numbers, provide real model responses collected from the 22 models listed in Table 6 (Appendix A.1) and real Chatbot Arena voting data (Appendix A.4: 1,670,250 votes).
- The simulation uses a smaller vote scale by default for tractability; increase `--num_votes` for closer approximation to paper results.
- The paper's simulation dataset contains 1,670,250 votes from 477,322 unique users across 6,895 model pair combinations.

## Citation

```bibtex
@article{huang2024exploring,
  title={Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards},
  author={Huang, Yangsibo and Nasr, Milad and Angelopoulos, Anastasios and Carlini, Nicholas
          and Chiang, Wei-Lin and Choquette-Choo, Christopher A. and Ippolito, Daphne
          and Jagielski, Matthew and Lee, Katherine and Liu, Ken Ziyu and Stoica, Ion
          and Tramer, Florian and Zhang, Chiyuan},
  year={2024}
}
```
