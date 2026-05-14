import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import os
from copy import deepcopy

from torch.utils.data import DataLoader

from models.peft_model_wrapper import PEFTModelWrapper
from evaluation import metrics
from utils.logger import Logger
# No need to import ConfigManager here, self.config already holds the dict


class Evaluator:
    """
    Evaluator class for computing various performance metrics and conducting specialized analyses
    as described in the research paper.
    """

    def __init__(self, model: PEFTModelWrapper, config: dict, logger: Logger):
        """
        Initializes the Evaluator instance.

        Args:
            model (PEFTModelWrapper): The model wrapper instance to be evaluated.
            config (dict): The loaded global configuration dictionary.
            logger (Logger): The logger instance for recording information.
        """
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

        self.logger.info(f"Evaluator initialized. Model moved to device: {self.device}")

    def evaluate_accuracy(self, data_loader: DataLoader, prefix: str = '') -> dict:
        """
        Evaluates the Top-1 accuracy of the model on a given DataLoader.

        Args:
            data_loader (DataLoader): DataLoader for the dataset to be evaluated.
            prefix (str): String prefix for the metric names in the returned dictionary (e.g., 'val_', 'test_').

        Returns:
            dict: A dictionary containing the computed Top-1 accuracy.
        """
        self.model.eval()  # Ensure model is in eval mode
        total_correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc=f"Evaluating {prefix.strip('_')} accuracy"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted_classes = torch.max(outputs.data, 1)

                total_samples += labels.size(0)
                total_correct_predictions += (predicted_classes == labels).sum().item()

        accuracy = (total_correct_predictions / total_samples) * 100 if total_samples > 0 else 0.0
        results = {f'{prefix}top1_accuracy': accuracy}
        self.logger.info(f"Accuracy for {prefix.strip('_')} set: {accuracy:.2f}% (Total samples: {total_samples}, Correct: {total_correct_predictions})")
        return results

    def compute_prediction_overlap(self, models_dict: dict[str, PEFTModelWrapper], data_loader: DataLoader,
                                   topk_confident: int = None, leastk_confident: int = None) -> dict:
        """
        Computes prediction overlaps between multiple PEFTModelWrapper instances.
        Supports filtering by most/least confident samples for correct/wrong predictions.

        Args:
            models_dict (dict[str, PEFTModelWrapper]): Dictionary mapping model names to their instances.
            data_loader (DataLoader): DataLoader for the dataset.
            topk_confident (int, optional): Number of most confident samples to consider for correct predictions.
                                            (Read from config.yaml: prediction_overlap_config.topk_confident).
            leastk_confident (int, optional): Number of least confident samples to consider for wrong predictions.
                                              (Read from config.yaml: prediction_overlap_config.leastk_confident).

        Returns:
            dict: Dictionary containing prediction overlap percentages.
        """
        self.logger.info("Computing prediction overlaps...")
        all_predictions_dict = {}
        all_confidences_dict = {}
        ground_truths = []
        
        # Prepare all models for inference
        for model_name, model_instance in models_dict.items():
            model_instance.eval()
            model_instance.to(self.device)

        # Collect predictions and confidences from all models
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Collecting predictions for overlap analysis"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Collect ground truths once per batch
                if not ground_truths: # Only append ground truths from the first batch once
                    ground_truths.append(labels.cpu())
                else: # For subsequent batches, just append the labels
                    ground_truths.append(labels.cpu())


                for model_name, model_instance in models_dict.items():
                    outputs = model_instance(inputs)
                    softmax_outputs = torch.softmax(outputs, dim=1)
                    confidences, predicted_classes = torch.max(softmax_outputs, 1)

                    if model_name not in all_predictions_dict:
                        all_predictions_dict[model_name] = []
                        all_confidences_dict[model_name] = []
                    
                    all_predictions_dict[model_name].append(predicted_classes.cpu())
                    all_confidences_dict[model_name].append(confidences.cpu())
        
        # Concatenate batch results into single tensors
        for model_name in all_predictions_dict:
            all_predictions_dict[model_name] = torch.cat(all_predictions_dict[model_name], dim=0)
            all_confidences_dict[model_name] = torch.cat(all_confidences_dict[model_name], dim=0)
        
        ground_truths_tensor = torch.cat(ground_truths, dim=0)

        # Use the metrics module to calculate overlaps
        overlap_results = metrics.calculate_prediction_overlaps(
            predictions_dict=all_predictions_dict,
            confidences_dict=all_confidences_dict,
            targets=ground_truths_tensor, # Changed from ground_truths to targets to match metrics.py
            topk_confident=topk_confident,
            leastk_confident=leastk_confident
        )
        self.logger.info("Prediction overlap analysis complete.")
        # self.logger.info(json.dumps(overlap_results, indent=2)) # Log as JSON for readability
        return overlap_results

    def compute_ensemble_accuracy(self, models_dict: dict[str, PEFTModelWrapper], data_loader: DataLoader) -> dict:
        """
        Computes the accuracy of an ensemble of models using their logits.
        Also calculates the baseline accuracy of the worst-performing PEFT method for comparison.

        Args:
            models_dict (dict[str, PEFTModelWrapper]): Dictionary mapping model names to their instances.
            data_loader (DataLoader): DataLoader for the dataset.

        Returns:
            dict: Dictionary containing ensemble accuracy and worst PEFT accuracy.
        """
        self.logger.info("Computing ensemble accuracy...")
        all_logits_batches = []
        ground_truths = []
        
        # Prepare all models for inference
        for model_name, model_instance in models_dict.items():
            model_instance.eval()
            model_instance.to(self.device)

        # Collect logits from all models
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Collecting logits for ensemble"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                batch_logits_per_model = []
                for model_name, model_instance in models_dict.items():
                    outputs = model_instance(inputs)
                    batch_logits_per_model.append(outputs.cpu())
                
                # Stack logits for the current batch from all models
                # Resulting in: (num_models, batch_size, num_classes)
                all_logits_batches.append(torch.stack(batch_logits_per_model, dim=0)) 
                ground_truths.append(labels.cpu())
        
        # Concatenate results across batches
        # After this, `all_logits_batches` is a list of (num_models, batch_size, num_classes) tensors.
        # We need to concatenate along batch_size to get (num_models, total_samples, num_classes).
        # Then, split this into a List[torch.Tensor] where each element is (total_samples, num_classes).
        
        if not all_logits_batches:
            self.logger.warning("No data to compute ensemble accuracy.")
            return {'ensemble_top1_accuracy': 0.0, 'worst_peft_top1_accuracy_baseline': 0.0}

        stacked_all_logits = torch.cat(all_logits_batches, dim=1) # (num_models, total_samples, num_classes)
        
        # Convert to List[torch.Tensor] for metrics.calculate_ensemble_predictions
        logits_list_for_ensemble = [stacked_all_logits[i] for i in range(stacked_all_logits.shape[0])]
        
        ground_truths_tensor = torch.cat(ground_truths, dim=0)

        # Calculate ensemble predictions
        ensemble_predictions = metrics.calculate_ensemble_predictions(logits_list_for_ensemble)
        ensemble_accuracy = metrics.calculate_top1_accuracy(ensemble_predictions.to(self.device), ground_truths_tensor.to(self.device)) # Ensure predictions and targets are on same device for metric calculation

        # Calculate accuracy for each individual model to find the worst PEFT baseline
        min_peft_accuracy = float('inf')
        # individual_accuracies = {} # Not requested in return type, but useful for debugging

        self.logger.info("Calculating individual model accuracies for ensemble baseline...")
        for model_name, model_instance in models_dict.items():
            # Create a temporary evaluator to get individual model accuracy (to ensure consistent eval logic)
            # The current Evaluator's model is self.model, not model_instance. We need to evaluate model_instance.
            # To avoid creating a new Evaluator and thus new device/logger copies, we can temporarily switch self.model.
            original_model = self.model
            self.model = model_instance # Temporarily use this model instance for evaluation
            self.model.to(self.device) # Ensure it's on the correct device
            
            acc_results = self.evaluate_accuracy(data_loader, prefix=f"{model_name}_")
            individual_acc = acc_results[f"{model_name}_top1_accuracy"]
            # individual_accuracies[model_name] = individual_acc # Not needed for final return

            # Filter for PEFT methods only for worst baseline. Linear and Full FT are excluded
            # based on common interpretation and paper's context.
            if model_name not in ["linear_probing", "full_ft"]: 
                min_peft_accuracy = min(min_peft_accuracy, individual_acc)
            
            # Restore original model
            self.model = original_model
            self.model.to(self.device)

        
        # Handle case where no PEFT methods were evaluated, or min_peft_accuracy remains inf
        if min_peft_accuracy == float('inf'):
            min_peft_accuracy = 0.0
            self.logger.warning("No PEFT methods found in models_dict to establish a baseline for worst PEFT accuracy.")


        results = {
            'ensemble_top1_accuracy': ensemble_accuracy,
            'worst_peft_top1_accuracy_baseline': min_peft_accuracy,
        }
        self.logger.info(f"Ensemble accuracy: {ensemble_accuracy:.2f}%")
        self.logger.info(f"Worst PEFT method accuracy baseline: {min_peft_accuracy:.2f}%")
        return results

    def compute_ranking_frequency(self, results_per_task: dict) -> dict:
        """
        Processes a dictionary of {task: {method: accuracy}} to generate ranking frequencies
        across methods and groups as per Figure 2 in the paper.

        Args:
            results_per_task (dict): A dictionary like {'task_name': {'method_name': accuracy}}.
                                     Task names are expected to be lowercase.

        Returns:
            dict: A dictionary containing the ranking frequencies and mean ranks per method.
        """
        self.logger.info("Computing ranking frequencies...")
        
        # Define VTAB-1K task groups (based on paper Section 3 and Table 1 for the 19 tasks)
        # Using lowercase for consistency with typical dataset loading keys
        vtab_task_groups = {
            'Natural': [
                'caltech101', 'cifar100', 'dtd', 'flowers102', 'pets', 'sun397'
            ],
            'Specialized': [
                'eurosat', 'resisc45', 'retinopathy'
            ],
            'Structured': [ # 10 tasks here to make up 19 total (6+3+10)
                'clevr_count', 'clevr_distance', 'dmlab', 'kitti',
                'dsprites_color', 'dsprites_full', 'dsprites_orient',
                'snorb_azimuth', 'snorb_distance', 'snorb_elevation'
            ]
        }

        # Collect all unique method names that appeared in any task
        all_method_names = sorted(list(set(method for task_res in results_per_task.values() for method in task_res.keys())))
        num_methods = len(all_method_names)

        # Store {group_name: {method_name: [rank_1_count, rank_2_count, ...]}}
        # Initialize frequency lists with zeros for all ranks
        group_ranking_frequencies = defaultdict(lambda: defaultdict(lambda: [0] * num_methods))
        method_total_ranks = defaultdict(float)  # Sum of ranks for mean calculation
        method_task_count = defaultdict(int)     # Number of tasks method participated in

        for group_name, tasks_in_group_def in vtab_task_groups.items():
            # Filter tasks to only those present in results_per_task
            actual_tasks_in_group = [t for t in tasks_in_group_def if t in results_per_task]
            
            if not actual_tasks_in_group:
                self.logger.info(f"No tasks found in results_per_task for group: {group_name}. Skipping ranking for this group.")
                continue

            for task_name in actual_tasks_in_group:
                method_accuracies = results_per_task[task_name]
                
                # Sort methods by accuracy in descending order
                # Filter out methods that might not have results for this specific task
                filtered_method_accuracies = {m: acc for m, acc in method_accuracies.items() if m in all_method_names}
                
                if not filtered_method_accuracies:
                    continue # Skip if no methods with results for this task

                sorted_methods = sorted(filtered_method_accuracies.items(), key=lambda item: item[1], reverse=True)
                
                for rank_idx, (method_name, _) in enumerate(sorted_methods): # rank_idx is 0-based
                    group_ranking_frequencies[group_name][method_name][rank_idx] += 1
                    method_total_ranks[method_name] += (rank_idx + 1) # 1-based rank
                    method_task_count[method_name] += 1
        
        # Calculate mean ranks
        mean_ranks_per_method = {}
        for method_name in all_method_names:
            if method_task_count[method_name] > 0:
                mean_ranks_per_method[method_name] = method_total_ranks[method_name] / method_task_count[method_name]
            else:
                mean_ranks_per_method[method_name] = float('nan') # Method not present in any evaluated task

        # Convert defaultdict to regular dict for final output and sort mean ranks
        final_ranking_frequencies = {
            group: {
                method: freqs
                for method, freqs in methods.items()
            }
            for group, methods in group_ranking_frequencies.items()
        }
        
        # Sort methods by their mean rank
        sorted_mean_ranks = dict(sorted(mean_ranks_per_method.items(), key=lambda item: item[1] if not np.isnan(item[1]) else float('inf')))

        results = {
            'ranking_frequencies_by_group': final_ranking_frequencies,
            'mean_ranks_per_method': sorted_mean_ranks
        }
        self.logger.info("Ranking frequency computation complete.")
        self.logger.info(json.dumps(results, indent=2))
        return results


    def evaluate_wise_robustness(self, base_model_sd: dict, fine_tuned_model_wrapper: PEFTModelWrapper,
                                 target_loader: DataLoader, ood_loaders: dict[str, DataLoader],
                                 alphas: list[float]) -> dict:
        """
        Evaluates the robustness of PEFT methods using Weight-space Ensembles (WiSE)
        across a range of alpha values.

        Args:
            base_model_sd (dict): The state dictionary of the initial pre-trained model
                                  (before any fine-tuning).
            fine_tuned_model_wrapper (PEFTModelWrapper): The PEFTModelWrapper instance
                                                       that has been fine-tuned on the target dataset.
            target_loader (DataLoader): DataLoader for the in-distribution target dataset test set.
            ood_loaders (dict[str, DataLoader]): A dictionary mapping OOD dataset names to their DataLoaders.
            alphas (list[float]): A list of mixing coefficients (alpha) for WiSE.

        Returns:
            dict: A dictionary containing WiSE results for each alpha and dataset.
        """
        self.logger.info("Starting WiSE robustness evaluation...")
        all_wise_results = defaultdict(dict)

        # Get the state dictionary of the fine-tuned model once
        fine_tuned_sd = fine_tuned_model_wrapper.state_dict()

        # Create a temporary model wrapper instance for WiSE interpolation
        # It needs the full configuration used to build the fine_tuned_model_wrapper
        wise_model_evaluator_instance = PEFTModelWrapper(
            backbone_config=fine_tuned_model_wrapper.backbone_config,
            peft_config=fine_tuned_model_wrapper.peft_config,
            head_config=fine_tuned_model_wrapper.head_config,
            num_classes=fine_tuned_model_wrapper.num_classes,
            experiment_type=fine_tuned_model_wrapper.experiment_type,
            pretrained_model_path=None # We will load state_dicts manually
        )
        wise_model_evaluator_instance.to(self.device)

        for alpha in tqdm(alphas, desc="Evaluating WiSE alphas"):
            # Load the base model state dict into the temporary instance
            # strict=False is often needed when loading base_model_sd as the model_wrapper might have PEFT params
            # that are not in the raw base_model_sd (e.g., initial state of adapters are not in base_model_sd)
            wise_model_evaluator_instance.load_state_dict(base_model_sd, strict=False) 
            wise_model_evaluator_instance.eval() # Set to eval mode before interpolation logic

            # Apply WiSE interpolation using the PEFTModelWrapper's method
            # This will modify the parameters of wise_model_evaluator_instance
            wise_model_evaluator_instance.apply_wise_interpolation(fine_tuned_sd, alpha)
            wise_model_evaluator_instance.to(self.device) # Ensure it's on device after modification
            wise_model_evaluator_instance.eval() # Re-set eval mode after potential weight changes

            # Create a temporary Evaluator for the interpolated model to run evaluations
            # No need to create a new Evaluator. Just use the wise_model_evaluator_instance with self.evaluate_accuracy.
            # Temporarily replace self.model to evaluate the interpolated model.
            original_model = self.model
            self.model = wise_model_evaluator_instance # Temporarily set this as the model to evaluate
            
            # self.model.to(self.device) # Already done above
            # self.model.eval() # Already done above

            # Evaluate on target dataset
            target_acc_dict = self.evaluate_accuracy(target_loader, prefix='wise_target_')
            all_wise_results[alpha]['target_accuracy'] = target_acc_dict['wise_target_top1_accuracy']
            self.logger.info(f"--- WiSE with alpha={alpha:.2f} ---")
            self.logger.info(f"  Target Accuracy: {all_wise_results[alpha]['target_accuracy']:.2f}%")

            # Evaluate on OOD datasets
            current_alpha_ood_accuracies = []
            for ood_name, ood_loader in ood_loaders.items():
                ood_acc_dict = self.evaluate_accuracy(ood_loader, prefix=f'wise_{ood_name}_')
                ood_accuracy = ood_acc_dict[f'wise_{ood_name}_top1_accuracy']
                all_wise_results[alpha][f'{ood_name}_accuracy'] = ood_accuracy
                current_alpha_ood_accuracies.append(ood_accuracy)
                self.logger.info(f"  {ood_name} Accuracy: {ood_accuracy:.2f}%")

            if current_alpha_ood_accuracies:
                avg_ood_acc = np.mean(current_alpha_ood_accuracies)
                all_wise_results[alpha]['avg_ood_accuracy'] = avg_ood_acc
                self.logger.info(f"  Average OOD Accuracy: {avg_ood_acc:.2f}%")
            else:
                all_wise_results[alpha]['avg_ood_accuracy'] = 0.0 # No OOD datasets evaluated
                self.logger.warning("  No OOD datasets evaluated for this alpha.")
            
            # Restore original model
            self.model = original_model
            self.model.to(self.device)

        self.logger.info("WiSE robustness evaluation complete.")
        return all_wise_results

