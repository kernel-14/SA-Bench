# Reproduction of "Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards"

This repository attempts to reproduce the core contributions of the paper "Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards".

## Core Contributions to be Reproduced

The paper identifies two main attack vectors for manipulating voting-based leaderboards:

1.  **De-anonymization of Model Responses (Section 2):** This involves identifying which LLM generated a particular response, even when presented anonymously.
    *   **Identity-Probing Detector (Section 2.2, 2.3, 2.4.1):** Crafting prompts to elicit identifying information from models.
    *   **Training-Based Detector (Section 2.2, 2.3, 2.4.2):** Training a classifier to distinguish model responses based on linguistic features.
2.  **Estimating the Number of Adversarial Votes (Section 3):** Simulating the impact of adversarial voting on leaderboard rankings using de-anonymization.

## Reproduction Plan

### Phase 1: De-anonymization of Model Responses

#### 1.1 Identity-Probing Detector

This detector relies on crafting specific prompts designed to make a model reveal its identity. The paper describes five such prompts and evaluates their effectiveness. The core idea is to check for the presence of model names (e.g., "Llama") or organization names (e.g., "Meta") in the response. While the paper notes its limitations and practical challenges (e.g., Chatbot Arena filtering), a conceptual implementation would involve:

*   **Prompt Generation:** A list of identity-probing prompts.
*   **Response Simulation:** A simulated mechanism for models to respond, where some models might reveal their identity with a certain probability or in response to specific prompts.
*   **Keyword Matching:** A simple classifier that searches for predefined keywords (model names, organization names) in the simulated responses.

#### 1.2 Training-Based Detector

This is a more robust de-anonymization method. The reproduction will focus on implementing the components described in the paper:

*   **Prompt Categories (Table 1):** Define the different categories and types of prompts: normal chat (high/low-resource languages) and specialty chat (coding, math, safety-violating).
*   **Model Response Generation (Simulated):** Since live interaction with LLMs is not permitted, we will need to simulate model responses. This simulation will aim to capture the distributional differences between models that the paper exploits (e.g., differences in response length, vocabulary, and stylistic elements).
    *   For each prompt category, we will simulate responses for a set of target models and other models, ensuring that the simulated responses for different models exhibit distinct characteristics that allow for classification.
*   **Feature Extraction (Section 2.3):** Implement the three text features mentioned:
    *   `Length(R)`: Response length in words or characters.
    *   `TF-IDF(R)`: Term Frequency-Inverse Document Frequency.
    *   `BoW(R)`: Bag-of-Words representations.
*   **Classifier Training and Evaluation (Section 2.3):**
    *   Use a logistic regression classifier (as mentioned in the paper, using `scikit-learn` defaults).
    *   Construct balanced datasets with 50 positive samples (target model responses) and 50 negative samples (other model responses) per prompt-model pair.
    *   Perform an 80/20 train/test split.
    *   Evaluate using average test accuracy across all prompts.

### Phase 2: Estimating the Number of Adversarial Votes (Section 3)

This phase involves simulating the Chatbot Arena leaderboard and the impact of adversarial votes. The core components would be:

*   **Leaderboard Simulation:** A system that maintains model rankings based on Elo scores, updated by votes.
*   **Adversarial Voting Logic:** Incorporating the de-anonymization detector to simulate an attacker's behavior:
    *   An attacker submits a prompt.
    *   Two models are randomly selected.
    *   The de-anonymization detector is used to identify if the target model is present in either response and which response belongs to it.
    *   If identified, an adversarial vote is cast (up-vote for promotion, down-vote for demotion).
*   **Vote and Interaction Tracking:** Track the cumulative votes and interactions required to achieve specific ranking shifts (e.g., rise/fall by `x` positions).
*   **Bradley-Terry Coefficients:** Implement the update mechanism for model rankings based on these coefficients.

## Directory Structure

The final submission will have a structure similar to:

```
repo/
├── README.md
├── src/
│   ├── de_anonymization.py      # Contains code for identity-probing and training-based detectors
│   └── simulation.py            # Contains code for leaderboard simulation and adversarial voting
├── data/
│   ├── prompts.json             # Definitions of prompt categories and examples
│   └── simulated_responses.json # Simulated model responses for various prompts
└── config.yaml                  # Configuration file for parameters
```

## Assumptions and Limitations

*   **No Live LLM Interaction:** All model responses will be simulated or pre-defined, as direct interaction with LLMs is outside the scope of this task.
*   **Simplified Model Behavior:** Simulated model responses will be designed to mimic the characteristics described in the paper (e.g., varying response lengths, distinct vocabulary) to enable the reproduction of detector accuracies.
*   **Scikit-learn for ML:** The paper explicitly mentions using `scikit-learn` for logistic regression, and this will be adhered to.
*   **Appendix Details:** If an experiment is described in the main body but details are in the appendix, those details are in scope. However, experiments *only* introduced in the appendix are out of scope. For example, Appendix A.1 (Models, decoding parameters) and A.2 (Example Prompts for Figure 2) are in scope. A.4 (Historical Voting Data) is in scope for the simulation part. B.2 (Ablation studies) are out of scope if they are presented only in appendix. 

This `README.md` will be updated as the reproduction progresses to reflect the implemented components and any further insights or deviations from the original plan.
