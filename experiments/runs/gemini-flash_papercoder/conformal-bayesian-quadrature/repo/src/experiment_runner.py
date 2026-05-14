import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Type, Dict, Any, List, Tuple

# Import classes from other modules as per design
from config import Config
from src.data_generators import DataGenerator, SyntheticBinomialDataGenerator, HeteroskedasticDataGenerator, CocoDataLoader
from src.loss_functions import LossFunction, BinomialLoss, MiscoverageLoss, FalseNegativeLoss
from src.baselines import Method, ConformalRiskControl, RCPSBaseline
from src.our_method import OurMethod
from src.metrics import EvaluationMetrics


class ExperimentRunner:
    """
    Orchestrates the entire experimental process, handling data generation,
    method execution, and metric aggregation across multiple trials.
    """

    def __init__(self,
                 config: Config,
                 experiment_config: Dict[str, Any],
                 data_generator_class: Type[DataGenerator],
                 loss_function_class: Type[LossFunction],
                 method_classes: Dict[str, Type[Method]]):
        """
        Initializes the experiment runner with global and experiment-specific
        configurations, along with the class types for data generation,
        loss calculation, and the methods to be evaluated.

        Args:
            config: An instance of the global Config class.
            experiment_config: A dictionary containing parameters specific to the
                               current experiment (e.g., synthetic_binomial_config).
            data_generator_class: The class type for data generation.
            loss_function_class: The class type for loss calculation.
            method_classes: A dictionary mapping method names to their class types.
        """
        self.config = config
        self.experiment_config = experiment_config
        self.data_generator_class = data_generator_class
        self.loss_function_class = loss_function_class
        self.method_classes = method_classes
        self.rng: np.random.Generator = None # Will be initialized in run_trials


    def run_trials(self) -> Dict[str, Any]:
        """
        Executes the specified number of trials, running all configured methods,
        collecting their results, and then aggregating these results into
        actionable metrics.

        Returns:
            A dictionary of all aggregated results for plotting/reporting.
        """
        # 1. Configuration Retrieval
        num_trials = self.config.num_trials
        random_seed = self.config.random_seed
        global_B = self.config.B
        lambda_search_range = self.config.lambda_search_range
        num_dirichlet_samples_for_our_method = self.config.num_dirichlet_samples

        # Experiment-specific parameters, with fallbacks to global config defaults
        num_calibration_samples = self.experiment_config.get(
            'num_calibration_samples', self.config.num_calibration_samples
        )
        num_test_samples = self.experiment_config.get(
            'num_test_samples', 0 # Test samples might not be needed for all experiments
        )
        current_alpha = self.experiment_config.get('alpha', self.config.alpha)
        current_beta = self.experiment_config.get('beta', self.config.beta)

        # 2. Random Number Generator Initialization
        self.rng = np.random.default_rng(random_seed)

        # 3. Result Data Structures Initialization
        lambdas_per_method = defaultdict(list)
        true_risks_per_method = defaultdict(list)
        
        # Collect test_data for each trial for later use in metrics (e.g., prediction set size)
        all_test_data_per_trial: List[List[Tuple]] = []
        
        # Store the last data_generator instance for CocoDataLoader's get_model_output
        last_data_generator_instance: DataGenerator = None

        # 4. Trial Loop
        for trial_idx in tqdm(range(num_trials), desc="Running trials"):
            # --- Data Generator Instantiation ---
            data_generator: DataGenerator
            if self.data_generator_class is CocoDataLoader:
                data_generator = self.data_generator_class(
                    model_config=self.experiment_config, random_state=self.rng
                )
            elif self.data_generator_class is SyntheticBinomialDataGenerator:
                data_generator = self.data_generator_class(
                    K=self.experiment_config['K'], random_state=self.rng
                )
            else:  # HeteroskedasticDataGenerator
                data_generator = self.data_generator_class(random_state=self.rng)
            
            last_data_generator_instance = data_generator # Keep track of the last one for metrics

            cal_data = data_generator.generate_calibration_data(num_calibration_samples)
            test_data = data_generator.generate_test_data(num_test_samples)
            
            if test_data:
                all_test_data_per_trial.append(test_data)


            # --- Loss Function Instantiation ---
            loss_function: LossFunction
            if self.loss_function_class is FalseNegativeLoss:
                # FalseNegativeLoss requires model_output_func from the data_generator
                loss_function = self.loss_function_class(model_output_func=data_generator.get_model_output)
            elif self.loss_function_class is BinomialLoss:
                loss_function = self.loss_function_class(K=self.experiment_config['K'])
            else:  # MiscoverageLoss
                loss_function = self.loss_function_class()

            # --- Method Execution Loop ---
            for method_name, method_class in self.method_classes.items():
                method_instance: Method
                # Common constructor arguments for all methods
                common_args = {
                    'alpha': current_alpha,
                    'B': global_B,
                    'lambda_search_range': lambda_search_range
                }

                if method_class is OurMethod:
                    method_instance = method_class(
                        **common_args,
                        beta=current_beta,
                        num_dirichlet_samples=num_dirichlet_samples_for_our_method,
                        random_state=self.rng
                    )
                elif method_class is RCPSBaseline:
                    # RCPSBaseline needs beta_rcps for its Hoeffding bound
                    method_instance = method_class(
                        **common_args,
                        beta_rcps=current_beta
                    )
                else:  # ConformalRiskControl
                    method_instance = method_class(**common_args)
                
                lambda_chosen: float = np.nan # Default to NaN if compute_lambda fails or is not implemented
                try:
                    lambda_chosen = method_instance.compute_lambda(cal_data=cal_data, loss_fn=loss_function)
                except NotImplementedError:
                    tqdm.write(f"Warning: Method '{method_name}' is not implemented. Skipping for this trial.")
                except Exception as e:
                    tqdm.write(f"Error computing lambda for method '{method_name}' in trial {trial_idx}: {e}")

                lambdas_per_method[method_name].append(lambda_chosen)

                # Calculate True Risk only if a valid lambda was chosen
                if not np.isnan(lambda_chosen):
                    true_risk = data_generator.calculate_true_risk(lambda_val=lambda_chosen)
                    true_risks_per_method[method_name].append(true_risk)
                else:
                    true_risks_per_method[method_name].append(np.nan)


        # 5. Metrics Aggregation
        results = {}
        evaluation_metrics = EvaluationMetrics(
            target_alpha=current_alpha, target_failure_rate=(1 - current_beta)
        )

        for method_name in self.method_classes.keys():
            method_results = {}
            
            # Filter out NaN values from failed compute_lambda calls
            valid_true_risks = [r for r in true_risks_per_method[method_name] if not np.isnan(r)]
            valid_lambdas = [l for l in lambdas_per_method[method_name] if not np.isnan(l)]

            # Relative Frequency of Exceeding Target Risk
            if valid_true_risks:
                exceed_freq, exceed_ci = evaluation_metrics.calculate_exceedance_frequency(valid_true_risks)
                method_results['relative_frequency_exceeding_alpha'] = exceed_freq
                method_results['relative_frequency_exceeding_alpha_95_ci'] = exceed_ci
            else:
                method_results['relative_frequency_exceeding_alpha'] = np.nan
                method_results['relative_frequency_exceeding_alpha_95_ci'] = (np.nan, np.nan)

            # Conditional Metrics
            if self.data_generator_class is HeteroskedasticDataGenerator:
                if valid_lambdas:
                    mean_pil = evaluation_metrics.calculate_mean_prediction_interval_length(valid_lambdas)
                    method_results['mean_prediction_interval_length'] = mean_pil
                else:
                    method_results['mean_prediction_interval_length'] = np.nan
            elif self.data_generator_class is CocoDataLoader:
                if valid_lambdas and all_test_data_per_trial and last_data_generator_instance:
                    # Re-instantiate FalseNegativeLoss with model_output_func from the *last* data_generator.
                    # This relies on the structure of z_item being consistent across all trials.
                    coco_loss_fn_for_metrics = FalseNegativeLoss(model_output_func=last_data_generator_instance.get_model_output)
                    mean_pss = evaluation_metrics.calculate_mean_prediction_set_size(
                        lambda_vals=valid_lambdas,
                        test_data_per_trial=all_test_data_per_trial,
                        loss_fn=coco_loss_fn_for_metrics
                    )
                    method_results['mean_prediction_set_size'] = mean_pss
                else:
                    method_results['mean_prediction_set_size'] = np.nan
            
            method_results['chosen_lambdas_all_trials'] = lambdas_per_method[method_name]

            results[method_name] = method_results

        return results

