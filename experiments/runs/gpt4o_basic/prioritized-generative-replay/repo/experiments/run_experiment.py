# Experiment Runner for PGR Framework

from src.pgr_framework import PrioritizedGenerativeReplay
from src.train_policy import PolicyTrainer
from src.relevance_functions import curiosity_relevance, return_based_relevance, td_error_relevance

def main():
    """Runs experiments for Prioritized Generative Replay."""
    policy = ...  # Load or define the RL policy
    optimizer = ...  # Define optimizer
    trainer = PolicyTrainer(policy, optimizer)

    # Example experiment
    real_data = ...  # Gather initial transition samples
    trainer.train(real_data, curiosity_relevance, guidance_scale=1.0)

    # Repeat for different relevance functions
    trainer.train(real_data, return_based_relevance, guidance_scale=0.8)
    trainer.train(real_data, td_error_relevance, guidance_scale=0.5)

if __name__ == "__main__":
    main()


