# Voting-Based Leaderboard Manipulation Reproduction

This repository contains the implementation of the paper "Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards." The codebase reproduces the experiments and methods described in the paper, including de-anonymization attacks, adversarial voting simulations, and proposed mitigations.

## Repository Structure

- `model.py`: Contains the implementation of the target model detector.
- `modules.py`: Implements reusable components for the model.
- `layers.py`: Defines custom neural network layers used in the model.
- `train.py`: Implements the training loop for the target model detector.
- `data.py`: Handles dataset loading and preprocessing.
- `config.py`: Contains all hyperparameters and configurations.
- `requirements.txt`: Lists all dependencies required to run the code.
- `README.md`: Provides an overview of the repository and instructions for use.

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the repository directory:
   ```bash
   cd <repository-directory>
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the dataset as described in the paper.
2. Configure the hyperparameters in `config.py`.
3. Train the model:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Experiments

The repository includes scripts to reproduce the following experiments:

1. **De-Anonymization of Model Responses**: Train and evaluate the target model detector.
2. **Adversarial Voting Simulations**: Simulate the impact of adversarial votes on leaderboard rankings.
3. **Mitigation Strategies**: Evaluate the effectiveness of proposed defenses against adversarial manipulation.

## Citation

If you use this code, please cite the original paper:

```
@article{huang2026voting,
  title={Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards},
  author={Huang, Yangsibo and Nasr, Milad and others},
  journal={arXiv preprint arXiv:2405.12345},
  year={2026}
}
```