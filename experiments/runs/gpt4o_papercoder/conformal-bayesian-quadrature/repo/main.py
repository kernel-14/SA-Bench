# main.py

import yaml
import numpy as np
from dataset_loader import DatasetLoader
from loss_functions import LossFunctions
from decision_rules import DecisionRules
from posterior_calculator import PosteriorCalculator
from evaluation import Evaluation
from visualization import Visualization

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration settings from the given YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration settings as a dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    """
    Main entry point for executing the workflow for reproducing experiments from the paper.
    """
    # Step 1: Load configuration
    config = load_config()
    alpha = config["experiment"]["alpha"]
    beta = config["experiment"]["beta"]
    synthetic_n = config.get("synthetic_n", 10)  # Default for synthetic data
    synthetic_k = config.get("synthetic_k", 4)   # Default number of trials for binomial
    ms_coco_path = config.get("ms_coco_path", "./ms_coco_data")
    num_calibration = config.get("num_calibration", 1000)
    num_test = config.get("num_test", 3952)
    monte_carlo_trials = config.get("monte_carlo_trials", 10000)

    # Step 2: Initialize modules
    dataset_loader = DatasetLoader(config)
    loss_functions = LossFunctions()
    decision_rules = DecisionRules()
    posterior_calculator = PosteriorCalculator()
    visualization = Visualization()

    # Step 3: Reproduce experiments
    print("Starting experiments...\n")

    ### Experiment 1: Synthetic Binomial Data ###
    print("Experiment 1: Synthetic Binomial Data")
    X, Y = dataset_loader.load_binomial_data(n=synthetic_n, K=synthetic_k)

    # Monte Carlo simulations for binomial experiments
    binomial_results = {"CRC": [], "RCPS": [], "Bayesian HPD": []}
    for _ in range(monte_carlo_trials):
        # Simulate binomial losses
        V = np.random.uniform(0, 1, (synthetic_n, synthetic_k))  # Simulate new data
        binomial_losses = loss_functions.binomial_loss(V, lambda_=0.5)

        # Apply decision rules
        lambda_crc = decision_rules.conformal_risk_control(binomial_losses, alpha, B=1.0)
        lambda_rcps = decision_rules.split_conformal_prediction(binomial_losses, alpha)
        lambda_hpd = decision_rules.bayesian_hpd_interval(binomial_losses, alpha, beta)

        # Store results
        binomial_results["CRC"].append(lambda_crc)
        binomial_results["RCPS"].append(lambda_rcps)
        binomial_results["Bayesian HPD"].append(lambda_hpd)

    # Visualize binomial results
    visualization.plot_loss_histogram(
        losses=np.array([binomial_results["CRC"], binomial_results["RCPS"], binomial_results["Bayesian HPD"]]).T,
        method_names=["CRC", "RCPS", "Bayesian HPD"],
    )

    ### Experiment 2: Synthetic Heteroskedastic Data ###
    print("\nExperiment 2: Synthetic Heteroskedastic Data")
    X, Y = dataset_loader.load_heteroskedastic_data(n=200)

    # Evaluate heteroskedastic losses
    heteroskedastic_losses = []
    for _ in range(monte_carlo_trials):
        Y = np.random.normal(0, X**2)  # Regenerate data based on heteroskedastic condition
        miscoverage_loss = loss_functions.miscoverage_loss(Y, lower=-0.5, upper=0.5)  # Example bounds

        # Apply Bayesian HPD rule
        lambda_hpd = decision_rules.bayesian_hpd_interval(miscoverage_loss, alpha, beta)
        heteroskedastic_losses.append(lambda_hpd)

    # Plot heteroskedastic results
    visualization.plot_loss_histogram(
        losses=np.array(heteroskedastic_losses).reshape(-1, 1),
        method_names=["Bayesian HPD"],
    )

    ### Experiment 3: MS-COCO Dataset ###
    print("\nExperiment 3: MS-COCO False Negative Rates")
    (calibration_X, calibration_Y), (test_X, test_Y) = dataset_loader.load_ms_coco(
        path=ms_coco_path,
        num_calibration=num_calibration,
        num_test=num_test,
    )

    # Evaluate MS-COCO losses and Bayesian intervals
    ms_coco_losses = []
    for _ in range(monte_carlo_trials):
        miscoverage_loss = loss_functions.miscoverage_loss(calibration_Y, lower=-0.5, upper=0.5)
        lambda_hpd = decision_rules.bayesian_hpd_interval(miscoverage_loss, alpha, beta)
        ms_coco_losses.append(lambda_hpd)

    # Plot MS-COCO results
    visualization.plot_loss_histogram(
        losses=np.array(ms_coco_losses).reshape(-1, 1),
        method_names=["Bayesian HPD"],
    )

    print("\nExperiments completed. Visualizations generated.")

if __name__ == "__main__":
    main()
