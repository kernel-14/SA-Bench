import numpy as np
import yaml
from typing import Dict, Any

from config import Config
from src.data_generators import SyntheticBinomialDataGenerator, HeteroskedasticDataGenerator, CocoDataLoader
from src.loss_functions import BinomialLoss, MiscoverageLoss, FalseNegativeLoss
from src.our_method import OurMethod
from src.baselines import ConformalRiskControl, RCPSBaseline
from src.experiment_runner import ExperimentRunner


def main():
    """
    Main function to load configuration, set up experiments, and run them sequentially.
    It orchestrates data generation, method execution, and result aggregation for
    each experiment defined in the configuration.
    """
    # 1. Configuration Loading
    try:
        with open('config.yaml', 'r') as f:
            yaml_config = yaml.safe_load(f)
        config = Config.from_yaml(yaml_config)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please ensure the configuration file is in the root directory.")
        return
    except Exception as e:
        print(f"Error loading or parsing config.yaml: {e}")
        return

    # 2. Global Random Seed Setup (for the ExperimentRunner's internal Generator)
    # The ExperimentRunner will initialize its own np.random.Generator with this seed.
    # This ensures all subsequent random operations within experiments are reproducible.
    print(f"Using global random seed: {config.random_seed}")

    # 3. Method Class Mapping
    # This dictionary maps descriptive names to the actual class types for each method.
    method_classes: Dict[str, Any] = {
        "Our Method": OurMethod,
        "CRC": ConformalRiskControl,
        "RCPS": RCPSBaseline  # Note: RCPS is a placeholder and will raise NotImplementedError
    }

    # Store the original lambda_search_range from config.yaml, as it will be modified
    # for each experiment below to fit the expected range of lambda values.
    original_lambda_search_range = config.lambda_search_range

    # --- Synthetic Binomial Experiment (Section 5.1) ---
    print("\n--- Running Synthetic Binomial Experiment ---")
    sb_exp_config = config.synthetic_binomial_config

    # Set experiment-specific lambda search range. For binomial loss, individual losses
    # and thus lambda values, are typically between 0 and 1.
    config.lambda_search_range = (0.0, 1.0)

    # Initialize and run the ExperimentRunner for Synthetic Binomial data.
    # The runner expects class types, not instances, for DataGenerator and LossFunction.
    sb_runner = ExperimentRunner(
        config=config,
        experiment_config=sb_exp_config,
        data_generator_class=SyntheticBinomialDataGenerator,
        loss_function_class=BinomialLoss,
        method_classes=method_classes
    )
    sb_results = sb_runner.run_trials()
    print("\nSynthetic Binomial Results:")
    for method_name, res in sb_results.items():
        print(f"  {method_name}:")
        freq = res.get('relative_frequency_exceeding_alpha', np.nan)
        ci = res.get('relative_frequency_exceeding_alpha_95_ci', (np.nan, np.nan))
        print(f"    Relative Freq. Exceeding Alpha: {freq:.2f}% (95% CI: {ci[0]:.2f}%, {ci[1]:.2f}%)")
        
        # For this experiment, the paper also reports the mean risk (1-lambda)
        if 'chosen_lambdas_all_trials' in res and res['chosen_lambdas_all_trials']:
            # Filter out NaN values before calculating mean
            valid_lambdas = [l for l in res['chosen_lambdas_all_trials'] if not np.isnan(l)]
            if valid_lambdas:
                mean_lambda = np.mean(valid_lambdas)
                # The paper notes for CRC that mean risk was 0.3363, which means mean lambda was 1 - 0.3363 = 0.6637
                # For Ours (beta=0.95), mean risk 0.1758 means mean lambda 1 - 0.1758 = 0.8242
                print(f"    Mean Lambda Chosen: {mean_lambda:.4f}")
                print(f"    Mean Risk (1-Lambda): {np.clip(1.0 - mean_lambda, 0.0, 1.0):.4f}")
            else:
                print(f"    No valid lambdas chosen for {method_name}.")


    # --- Synthetic Heteroskedastic Experiment (Section 5.2) ---
    print("\n--- Running Synthetic Heteroskedastic Experiment ---")
    sh_exp_config = config.synthetic_heteroskedastic_config
    
    # Set experiment-specific lambda search range. For miscoverage loss with N(0, X^2),
    # the prediction interval half-width (lambda) can be larger than 1.
    config.lambda_search_range = (0.0, 20.0) # An empirically reasonable upper bound

    # Initialize and run the ExperimentRunner for Synthetic Heteroskedastic data.
    sh_runner = ExperimentRunner(
        config=config,
        experiment_config=sh_exp_config,
        data_generator_class=HeteroskedasticDataGenerator,
        loss_function_class=MiscoverageLoss,
        method_classes=method_classes
    )
    sh_results = sh_runner.run_trials()
    print("\nSynthetic Heteroskedastic Results:")
    for method_name, res in sh_results.items():
        print(f"  {method_name}:")
        freq = res.get('relative_frequency_exceeding_alpha', np.nan)
        ci = res.get('relative_frequency_exceeding_alpha_95_ci', (np.nan, np.nan))
        mean_pil = res.get('mean_prediction_interval_length', np.nan)
        print(f"    Relative Freq. Exceeding Alpha: {freq:.2f}% (95% CI: {ci[0]:.2f}%, {ci[1]:.2f}%)")
        print(f"    Mean Prediction Interval Length: {mean_pil:.2f}")


    # --- MS-COCO Experiment (Section 5.3) ---
    print("\n--- Running MS-COCO Experiment ---")
    mscoco_exp_config = config.ms_coco_config

    # Validate essential configurations for MS-COCO as they are marked as UNCLEAR in the plan.
    if mscoco_exp_config.get('alpha') is None:
        raise ValueError(
            "MS-COCO experiment 'alpha' is not specified in config.yaml. "
            "Please clarify from Angelopoulos & Bates (2023, Section 5.1) "
            "and update config.yaml."
        )
    if mscoco_exp_config.get('model_name') is None:
        # A warning is provided here as CocoDataLoader has dummy data for basic testing.
        # However, a full implementation would require a specified model.
        print(
            "Warning: MS-COCO experiment 'model_name' is not specified in config.yaml. "
            "The CocoDataLoader will operate with dummy data. True risk calculation "
            "and realistic prediction set sizes will not be accurate without a real model. "
            "Please clarify from Angelopoulos & Bates (2023, Section 5.1) and update config.yaml."
        )
    # The `CocoDataLoader` will also check for `model_weights_path` and `dataset_path`.

    # Set experiment-specific lambda search range. For false negative rate, lambda is typically
    # a threshold on scores, which often fall within [0, 1] for probabilities/normalized scores.
    config.lambda_search_range = (0.0, 1.0) 

    # Initialize and run the ExperimentRunner for MS-COCO data.
    mscoco_runner = ExperimentRunner(
        config=config,
        experiment_config=mscoco_exp_config,
        data_generator_class=CocoDataLoader,
        loss_function_class=FalseNegativeLoss,
        method_classes=method_classes
    )
    # Note: This call might raise NotImplementedError from CocoDataLoader if true_risk calculation
    # is attempted, or from RCPSBaseline.
    try:
        mscoco_results = mscoco_runner.run_trials()
        print("\nMS-COCO Results:")
        for method_name, res in mscoco_results.items():
            print(f"  {method_name}:")
            freq = res.get('relative_frequency_exceeding_alpha', np.nan)
            ci = res.get('relative_frequency_exceeding_alpha_95_ci', (np.nan, np.nan))
            mean_pss = res.get('mean_prediction_set_size', np.nan)
            print(f"    Relative Freq. Exceeding Alpha: {freq:.2f}% (95% CI: {ci[0]:.2f}%, {ci[1]:.2f}%)")
            print(f"    Mean Prediction Set Size: {mean_pss:.2f}")
    except NotImplementedError as e:
        print(f"\nSkipping MS-COCO experiment due to NotImplementedError: {e}")
        print("Please address the 'UNCLEAR' points in the design related to MS-COCO "
              "(e.g., RCPS implementation, or true risk/model details for CocoDataLoader).")
    except Exception as e:
        print(f"\nAn unexpected error occurred during MS-COCO experiment: {e}")


    # Restore original lambda search range (good practice, though not strictly necessary here)
    config.lambda_search_range = original_lambda_search_range


if __name__ == '__main__':
    main()
