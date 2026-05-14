# Reproduction: Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards

This repository reproduces the core contributions of the paper:

> **Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards**  
> Yangsibo Huang, Milad Nasr, Anastasios Angelopoulos, Nicholas Carlini, Wei-Lin Chiang, et al.  
> ICLR 2025

## Overview

The paper demonstrates that voting-based LLM leaderboards (like Chatbot Arena) are vulnerable to adversarial manipulation. The attack has two steps:
1. **Re-identification**: An adversary identifies which model generated a given response with >95% accuracy
2. **Reranking**: The adversary selectively votes for/against the target model to manipulate rankings

The paper also proposes and evaluates mitigations to increase the cost of such attacks.

## Repository Structure

```
submission/
├── src/
│   ├── model_query.py                    # LLM API querying (OpenAI, Anthropic, Google, Together)
│   ├── identity_probing_detector.py      # Identity-probing attack (Section 2.2)
│   ├── training_based_detector.py        # Training-based detector (Section 2.2)
│   ├── bradley_terry.py                  # Bradley-Terry model for leaderboard ranking
│   ├── adversarial_simulation.py         # Adversarial voting simulation (Section 3)
│   ├── mitigations.py                    # Mitigations analysis (Section 4)
│   ├── collect_responses.py              # Script to collect model responses
│   ├── run_deanonymization_experiment.py # Run Section 2 experiments
│   ├── run_simulation_experiment.py      # Run Section 3 experiments
│   ├── run_mitigations_experiment.py     # Run Section 4 experiments
│   └── visualize_bow_features.py         # Generate Figure 2 (BoW PCA visualization)
├── configs/
│   └── models.yaml                       # Model list and configuration
├── data/
│   └── prompts/                          # Prompt files (to be populated)
├── notebooks/                            # Jupyter notebooks for analysis
├── requirements.txt
└── README.md
```

## Core Contributions Reproduced

### 1. De-anonymization of Model Responses (Section 2)

**Identity-Probing Detector** (`src/identity_probing_detector.py`):
- Implements the attack where the adversary asks "Who are you?" or similar prompts
- Detects the model by checking if its name/organization appears in the response
- Achieves >90% accuracy for most models with the "Who are you?" prompt (Table 2)

**Training-Based Detector** (`src/training_based_detector.py`):
- Implements supervised learning to distinguish model responses
- Features: Length (word/character), TF-IDF, Bag-of-Words (BoW)
- Uses logistic regression from scikit-learn with default hyperparameters, random_state=42
- 80/20 train/test split, 50 responses per model per prompt
- BoW achieves >95% accuracy for most models (Table 3, Figure 3)

### 2. Adversarial Voting Simulation (Section 3)

**Bradley-Terry Model** (`src/bradley_terry.py`):
- Implements the MM algorithm for computing Bradley-Terry coefficients
- Used by Chatbot Arena to rank models

**Adversarial Simulator** (`src/adversarial_simulation.py`):
- Simulates adversarial voting attacks on the leaderboard
- Estimates votes and interactions needed to achieve Up(M, x) and Down(M, x)
- Default: 95% detection accuracy, symmetric 5% FP/FN rates
- Recalculates rankings every 1,000 interactions
- Replicates Tables 4 and 5 from the paper

Key findings:
- Moving a model up/down 1 position requires <1,000 adversarial votes
- Low-ranked models require ~30% fewer votes than high-ranked models
- The number of interactions is much higher due to uniform model sampling

### 3. Mitigations (Section 4)

**Cost Model** (`src/mitigations.py`):
- Implements the attack cost formula: `ceil(N/m) * c_account + N * c_action + c_detector`
- Estimates detector training cost at ~$440 (200 prompts × $2.20/prompt)

**Malicious User Detection** (`src/mitigations.py`):
- **Scenario 1** (Known benign distribution): Likelihood test with empirical p-value
  - Detects naive adversaries but sophisticated ones can evade by mimicking benign behavior
- **Scenario 2** (Known benign + malicious distributions): Neyman-Pearson likelihood ratio test
  - Defender releases perturbed leaderboard to reduce attacker's knowledge
  - Increasing noise scale improves detection but hurts utility (Figures 5 & 6)

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

Set API keys as environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export TOGETHER_API_KEY="your-key"
```

### Step 1: Collect Model Responses

```bash
# Collect English prompts (200 prompts, 50 responses per model)
python src/collect_responses.py \
    --config configs/models.yaml \
    --prompts_dir data/prompts \
    --output_dir data/responses \
    --category english

# Repeat for other categories: chinese, spanish, indonesian, persian, coding, math, safety_violating
```

### Step 2: Run De-anonymization Experiment (Section 2)

```bash
python src/run_deanonymization_experiment.py \
    --data_dir data/responses \
    --output_dir results/deanonymization
```

### Step 3: Run Adversarial Voting Simulation (Section 3)

```bash
# With real Chatbot Arena data (if available)
python src/run_simulation_experiment.py \
    --data_path data/chatbot_arena_votes.csv \
    --output_dir results/simulation

# With synthetic leaderboard (for testing)
python src/run_simulation_experiment.py \
    --output_dir results/simulation \
    --n_models 130
```

### Step 4: Run Mitigations Experiment (Section 4)

```bash
python src/run_mitigations_experiment.py \
    --output_dir results/mitigations
```

### Step 5: Generate Visualizations

```bash
# Figure 2: BoW PCA visualization
python src/visualize_bow_features.py \
    --data_dir data/responses \
    --output_dir results/figures
```

## Data Requirements

### Model Responses
The de-anonymization experiment requires collecting responses from 22 models across 8 prompt categories. The paper uses:
- **200 prompts per category** from LMSYS-Chat-1M, Alpaca Code, MATH, and AdvBench datasets
- **50 responses per model per prompt** (with default decoding parameters)
- **512 max output tokens**

### Chatbot Arena Voting Data
The simulation experiment uses historical voting data from Chatbot Arena:
- 1,670,250 votes from 477,322 unique users
- 1,093,875 wins, 576,375 ties
- 6,895 unique model comparison combinations

This data is not publicly available in the paper, but the simulation can run with synthetic data.

## Key Implementation Details

### Training-Based Detector
- **Logistic regression** from scikit-learn with default hyperparameters
- **Random state = 42** (as specified in the paper)
- **Balanced dataset**: 50 positive (target model) + 50 negative (other models) samples
- **80/20 train/test split**
- **BoW features** consistently outperform TF-IDF and length features

### Adversarial Simulation
- **Detection accuracy**: 95% (symmetric 5% FP/FN rates)
- **Recalculation interval**: Every 1,000 interactions
- **Non-detection strategy**: "Do nothing" (attacker abstains when target not detected)
- **Model sampling**: Uniform random pair selection

### Malicious User Detection
- **Significance level α = 0.01** for hypothesis tests
- **Scenario 1**: Empirical p-value from 10,000 simulated sequences
- **Scenario 2**: Gaussian noise added to Bradley-Terry ratings before release

## Assumptions and Unresolved Details

1. **Chatbot Arena voting data**: The paper uses a private dataset of 1.67M votes. Our simulation uses synthetic data when real data is unavailable.

2. **Prompt sources**: The paper uses LMSYS-Chat-1M, Alpaca Code, MATH, and AdvBench datasets. Users need to download these separately.

3. **Model sampling in Arena**: The paper assumes uniform random pair selection. The actual Arena may use non-uniform sampling.

4. **Tie handling**: The paper mentions 576,375 ties but doesn't specify how they're handled in the Bradley-Terry model. We treat ties as 0.5 wins for each model.

5. **Figure 3 (heatmap)**: The paper shows a heatmap of detection accuracy across models and prompt categories. This requires running the full experiment with all 22 models and 8 categories.

6. **Figures 4-6 (mitigation experiments)**: These require the full leaderboard data and may differ from paper results when using synthetic data.

## Results Summary

Based on the paper's reported results:

| Experiment | Key Finding |
|-----------|-------------|
| Identity-probing | "Who are you?" achieves >90% accuracy for all models |
| Training-based (BoW) | >95% accuracy for most models on English prompts |
| Adversarial votes (high-ranked) | <1,000 votes to move ±1 position |
| Adversarial votes (low-ranked) | ~30% fewer votes than high-ranked models |
| Detector training cost | ~$440 for 200 prompts |
| Mitigation (Scenario 1) | Naive adversaries detectable; sophisticated ones evade |
| Mitigation (Scenario 2) | Perturbed leaderboard improves detection at utility cost |
