
import numpy as np
import matplotlib.pyplot as plt
import os
from config import PGConfig, SimulationConfig, REWARD_VARIANTS, TRANSITION_VARIANTS
from model import PolicyGradient
from data import generate_mdp_s4_1, generate_mdp_s4_2, generate_mdp_s4_3

def run_simulation(sim_config: SimulationConfig):
    """
    Runs the policy gradient simulation based on the provided configuration.
    """
    os.makedirs(sim_config.plot_dir, exist_ok=True)
    print(f"Running simulation: {sim_config.plot_dir.split('/')[-1]}")

    for i, mdp_config in enumerate(sim_config.mdp_configs):
        print(f"  MDP Config: S={mdp_config.S}, A={mdp_config.A}, Name={mdp_config.name}")

        if "section_4_1" in sim_config.plot_dir:
            mdp = generate_mdp_s4_1(mdp_config.S, mdp_config.A)
            label = f"S={mdp_config.S}, A={mdp_config.A}"
        elif "section_4_2" in sim_config.plot_dir:
            # For section 4.2, we iterate over reward variance types
            for reward_type in REWARD_VARIANTS:
                print(f"    Reward Type: {reward_type}")
                mdp = generate_mdp_s4_2(mdp_config.S, mdp_config.A, reward_type, seed=sim_config.pg_config.seed)
                pg = PolicyGradient(mdp, sim_config.pg_config.learning_rate, sim_config.pg_config.initial_policy_type)
                avg_rewards, _, _, _ = pg.train(sim_config.pg_config.iterations)
                plt.plot(avg_rewards, label=f"Reward: {reward_type.replace('_', ' ').title()}")
            label = "Fixed S, A" # Label for the overall plot title
        elif "section_4_3" in sim_config.plot_dir:
            # For section 4.3, we iterate over transition kernel types
            for transition_type in TRANSITION_VARIANTS:
                print(f"    Transition Type: {transition_type}")
                mdp = generate_mdp_s4_3(mdp_config.S, mdp_config.A, transition_type, seed=sim_config.pg_config.seed)
                pg = PolicyGradient(mdp, sim_config.pg_config.learning_rate, sim_config.pg_config.initial_policy_type)
                avg_rewards, _, _, _ = pg.train(sim_config.pg_config.iterations)
                plt.plot(avg_rewards, label=f"Kernel: {transition_type.replace('_', ' ').title()}")
            label = "Fixed S, A, High Variance Reward" # Label for the overall plot title
        else:
            raise ValueError(f"Unknown plot directory type: {sim_config.plot_dir}")

        if not ("section_4_2" in sim_config.plot_dir or "section_4_3" in sim_config.plot_dir):
            pg = PolicyGradient(mdp, sim_config.pg_config.learning_rate, sim_config.pg_config.initial_policy_type)
            avg_rewards, _, _, _ = pg.train(sim_config.pg_config.iterations)
            plt.plot(avg_rewards, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title(f"Policy Gradient Convergence - {sim_config.plot_dir.split('/')[-1].replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(sim_config.plot_dir, "convergence.png"))
    plt.close() # Close the plot to free memory

if __name__ == "__main__":
    from config import sim_config_s41, sim_config_s42, sim_config_s43

    # Set a global seed for reproducibility across all simulations
    np.random.seed(sim_config_s41.pg_config.seed)

    # Run simulations for Section 4.1
    run_simulation(sim_config_s41)

    # Run simulations for Section 4.2
    run_simulation(sim_config_s42)

    # Run simulations for Section 4.3
    run_simulation(sim_config_s43)

    print("All simulations completed and plots saved.")
