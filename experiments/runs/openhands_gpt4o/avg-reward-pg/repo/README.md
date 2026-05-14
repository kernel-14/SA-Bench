# Policy Gradient for Average Reward MDPs

This repository implements the Policy Gradient algorithm for infinite horizon average reward Markov Decision Processes (MDPs) as described in the paper "Global Convergence of Policy Gradient in Average Reward MDPs."

## Codebase Structure

- `model.py`: Contains the `PolicyGradient` class, which implements the policy gradient algorithm.
- `train.py`: Implements the training loop for the policy gradient algorithm.
- `data.py`: Provides synthetic MDP generation functions for reward and transition dynamics.
- `config.py`: Stores all hyperparameters and configurations for the experiments.
- `requirements.txt`: Lists all dependencies required for the project.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the policy gradient algorithm:
   ```python
   from train import train_policy_gradient
   from data import generate_synthetic_mdp
   from config import CONFIG

   reward_fn, transition_fn = generate_synthetic_mdp(
       CONFIG["state_space"], CONFIG["action_space"], CONFIG["reward_variance"]
   )

   trained_policy = train_policy_gradient(
       state_space=CONFIG["state_space"],
       action_space=CONFIG["action_space"],
       learning_rate=CONFIG["learning_rate"],
       num_iterations=CONFIG["num_iterations"],
       reward_function=reward_fn,
       transition_function=transition_fn,
       initial_state=CONFIG["initial_state"]
   )
   ```

## Citation

If you use this code, please cite the original paper:

```
@article{policygradient2026,
  title={Global Convergence of Policy Gradient in Average Reward MDPs},
  author={Navdeep Kumar, Yashaswini Murthy, Itai Shufaro, Kfir Y. Levy, R. Srikant, Shie Mannor},
  year={2026}
}
```