# Reproduction of "Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards"

This repository contains a reproduction attempt of the research paper "Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards." The paper demonstrates vulnerabilities in current voting-based leaderboards for large language models (LLMs), devises adversarial attacks, and proposes mitigations to counteract these vulnerabilities.

## Repository Structure

- **src/**: Contains scripts implementing the core contributions of the paper, including adversarial attack pipelines and mitigation strategies.
- **config/**: Configuration files for experimental setup.
- **docs/**: Documentation related to paper reproduction.
- **results/**: Placeholder for outputs from simulations (not executed in this benchmark).


## Implementation Plan

1. **Identity-Probing Detector**
   - Prompts (e.g., "Who are you?") that directly query identifying information.
   - Implementation based on the detection workflow described in Section 2.2.

2. **Training-Based Detector**
   - Creating a classifier ensemble using text features like TF-IDF, bag-of-words, and response length.
   - Implementing training pipelines using scikit-learn logistic regression for each prompt-model pair.

3. **Reranking Attack Simulation**
   - Simulating the required adversarial votes and interactions to influence model rankings.
   - Employing Bradley-Terry coefficients for ranking adjustments as described in Section 3.

4. **Mitigation Strategies**
   - Implementing defenses like authentication, rate limiting, and malicious user identification based on Section 4.


## Implementation Status

- **Identity-Probing Detector**: Implemented using keyword matching. Prompts simulate detecting model identifiers in responses.
- **Training-Based Detector**: Logistic regression models implemented using Bag-of-Words (BoW) and TF-IDF features.
- **Reranking Attack Simulation**: Rankings computed using Bradley-Terry coefficients to estimate adversarial interactions and votes.
- **Mitigation Strategies**: Included authentication, rate limiting, and malicious voting pattern detection.

## Notes

This reproduction adheres to the experimental setup described in the main body of the paper, manually adapting its methods to relevant workflows. Experiments to reproduce further statistics or iterative improvements are pending execution.

## Next Steps

Future iterations could:

1. **Expand Training Dataset**: Experiment with advanced prompts or classifiers proposed in the discussion sections.
2. **Improve Visualizations**: Incorporate clustering techniques for BoW separability measures.
3. **Optimize Mitigations**: Enhance anomaly detection with historical data.
