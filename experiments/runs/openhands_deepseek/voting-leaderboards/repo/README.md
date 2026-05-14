# Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards

Reproduction of the paper "Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards" by Huang et al. (2025).

This repository implements the complete experimental pipeline described in the paper, including:

## Structure

```
repo/
├── config.yaml           # All hyperparameters and configuration
├── config.py             # Configuration dataclasses and loader
├── requirements.txt      # Python dependencies
├── README.md             # This file
│
├── data.py               # Data loading, preprocessing, synthetic generation
├── detector.py           # De-anonymization detectors (Section 2)
├── models.py             # Statistical model wrappers (Bradley-Terry, Neyman-Pearson)
├── simulation.py         # Voting simulation and leaderboard manipulation (Section 3)
├── mitigations.py        # Defense strategies and cost model (Section 4)
│
├── train_detector.py     # Entry point: train and evaluate detectors
├── run_simulation.py     # Entry point: run voting simulations
└── run_mitigations.py    # Entry point: run mitigation experiments
```

## Paper Components Reproduced

### Section 2: De-anonymization of Model Responses
- **Identity-probing detector**: 5 prompts tested across 22 models (Table 2)
- **Training-based detector**: Logistic regression with 4 feature types (Table 3)
  - Length(R)_word, Length(R)_character, BoW(R), TF-IDF(R)
- **Prompt category analysis**: 8 categories (English, Chinese, Spanish, Indonesian, Persian, Coding, Math, Safety-violating) (Figure 3)

### Section 3: Estimating Adversarial Votes
- **Bradley-Terry ranking model** with MM algorithm (Hunter, 2004)
- **Vote simulation** for Up(M, x) and Down(M, x) objectives
- **Tables 4 & 5**: Votes/interactions needed for high- and low-ranked models
- **Table 8**: Ablation over detector accuracy (1.0, 0.95, 0.9)
- **Table 9**: Ablation over non-detection strategies (do_nothing, random_upvote, vote_tie, vote_both_bad)

### Section 4: Mitigations
- **Cost model** (Section 4.1): Attack cost = ceil(N/m)*c_account + N*c_action + c_detector
- **Authentication** (Section 4.2.1): Account cost through identity verification
- **Rate limiting** (Section 4.2.2): Per-account action caps
- **Malicious user detection** (Section 4.2.3):
  - Scenario 1: Likelihood ratio test with known benign distribution (Figure 4)
  - Scenario 2: Neyman-Pearson detector with perturbed leaderboard (Figures 5, 6)
- **Action cost increases** (Section 4.2.4): CAPTCHA, prompt uniqueness

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train de-anonymization detectors
python train_detector.py --config config.yaml

# Run voting simulations
python run_simulation.py --config config.yaml

# Run mitigation experiments
python run_mitigations.py --config config.yaml
```

## Key Results

- Identity-probing: "Who are you?" achieves >90% detection accuracy for all models
- Training-based: BoW features achieve >95% detection accuracy for most models
- Specialized prompts (Math, Coding) and non-English languages achieve highest accuracy
- Moving a model up 1 position requires <1,000 votes for high-ranked models
- Low-ranked models require ~30% of votes compared to high-ranked models
- Without mitigations, total attack cost is dominated by ~$440 detector training cost
- Perturbed leaderboards (Scenario 2) enable detection of adversaries using public rankings, at a utility cost
