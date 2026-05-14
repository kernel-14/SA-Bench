import argparse
import itertools
import os
import torch
from typing import Dict, Any, List, Tuple
from copy import deepcopy

from utils.config_manager import ConfigManager
from utils.logger import Logger
from utils.seed_manager import SeedManager
from datasets.vtab_loader import VTABLoader, BaseDatasetLoader
from datasets.many_shot_loader import ManyShotLoader
from datasets.robustness_loader import RobustnessLoader
from models.peft_model_wrapper import PEFTModelWrapper
from training.trainer import Trainer
from evaluation.evaluator import Evaluator


def main(config_path: str):
    """
    Main entry point for the PEFT visual recognition reproduction study.
    Loads configuration, initializes utilities, and orchestrates experiment execution.
    """
    # 1. Load Configuration
    config_manager = ConfigManager(config_path)
    global_config = config_manager.get_config()

    # 2. Initialize Logger and SeedManager
    log_dir = os.path.join(global_config['logging']['log_dir'], global_config['logging']['experiment_name'])
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir, filename=f"{global_config['logging']['experiment_name']}.log")
    seed_manager = SeedManager()
    seed_manager.set_seed(global_config['seed'])
    logger.info(f"Initialized with seed: {global_config['seed']}")
    logger.save_config(global_config) # Save the entire config for reproducibility

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Determine backbone total parameters for PEFT cap calculation
    # Load a dummy PEFTModelWrapper to get the base backbone's parameter count.
    # This involves instantiating the backbone (which is already efficient via HF `from_pretrained` caching)
    # and then querying its parameter count.
    try:
        dummy_backbone_config = global_config['model']['backbone'].copy()
        # Ensure a valid 'pretrained_on' is set for the dummy model to load correctly
        dummy_backbone_config['pretrained_on'] = 'imagenet21k' # Default to a common one
        
        # We need a num_classes, but it's not strictly used for parameter counting of backbone.
        # Use a placeholder, as PEFTModelWrapper needs it for the head.
        dummy_num_classes = 10 

        dummy_model_wrapper = PEFTModelWrapper(
            backbone_config=dummy_backbone_config,
            peft_config={'method': 'linear_probing'}, # Use a simple method for dummy
            head_config=global_config['model']['head'],
            num_classes=dummy_num_classes,
            experiment_type='dummy_init',
            # Pass the checkpoint path to allow PEFTModelWrapper to load pretrained weights into backbone
            pretrained_model_path=dummy_backbone_config['pretrained_on_imagenet21k_checkpoint']
        )
        # Ensure the actual backbone model is loaded by accessing it
        _ = dummy_model_wrapper.backbone
        total_backbone_params = dummy_model_wrapper.backbone_model_helper.get_num_parameters()
        peft_param_cap = total_backbone_params * global_config['peft']['cap_peft_params_ratio']
        logger.info(f"Total backbone parameters (ViT-B/16 reference): {total_backbone_params / 1e6:.2f}M")
        logger.info(f"PEFT parameter cap ({global_config['peft']['cap_peft_params_ratio'] * 100:.1f}% of backbone): {peft_param_cap / 1e6:.2f}M")
        
        # Clean up dummy model to free memory
        dummy_model_wrapper.cpu()
        del dummy_model_wrapper
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Failed to initialize dummy model to determine backbone parameters: {e}")
        logger.error("Proceeding without PEFT parameter cap check. This might lead to unexpected behavior.")
        total_backbone_params = 1 # Placeholder to avoid ZeroDivisionError
        peft_param_cap = float('inf') # Effectively disable cap check


    # Dictionary to store results for ranking frequency analysis
    all_vtab_results_for_ranking: Dict[str, Dict[str, float]] = {} # {task_name: {method_name: accuracy}}

    # 3. Iterate through experiment configurations
    for exp_config in global_config['experiments']:
        exp_name = exp_config['name']
        exp_type = exp_config['type']
        logger.info(f"\n--- Starting Experiment: {exp_name} ({exp_type}) ---")

        if exp_type in ["low_shot", "many_shot", "robustness"]:
            # --- Setup Dataset Loader ---
            if exp_type == "low_shot":
                if exp_config['dataset'] != "vtab1k":
                    logger.error(f"Mismatch: exp_type is 'low_shot' but dataset is not 'vtab1k'. Skipping experiment {exp_name}.")
                    continue
                
                # Get all VTAB task names from the VTABLoader static method
                task_names: List[str] = VTABLoader.get_vtab_task_names()
                
                for task_name in task_names:
                    logger.info(f"  -- Running Low-Shot experiment for task: {task_name} --")
                    current_dataset_loader = VTABLoader(
                        dataset_config=global_config['datasets'],
                        task_name=task_name,
                        split_seed=global_config['seed'],
                        data_augmentation_policy_name=exp_config['data_augmentation_policy']
                    )
                    # Pass the full global_config to run_single_experiment
                    run_single_experiment(exp_config, current_dataset_loader, logger, device, peft_param_cap,
                                          total_backbone_params, global_config, all_vtab_results_for_ranking, f"{exp_name}/{task_name}")

            elif exp_type == "many_shot":
                dataset_name = exp_config['dataset']
                current_dataset_loader = ManyShotLoader(
                    dataset_config=global_config['datasets'],
                    dataset_name=dataset_name,
                    data_augmentation_policy_name=exp_config['data_augmentation_policy']
                )
                run_single_experiment(exp_config, current_dataset_loader, logger, device, peft_param_cap,
                                      total_backbone_params, global_config, all_vtab_results_for_ranking, exp_name)

            elif exp_type == "robustness":
                current_dataset_loader = RobustnessLoader(
                    dataset_config=global_config['datasets'],
                    data_augmentation_policy_name=exp_config['data_augmentation_policy']
                )
                run_single_experiment(exp_config, current_dataset_loader, logger, device, peft_param_cap,
                                      total_backbone_params, global_config, all_vtab_results_for_ranking, exp_name)

        elif exp_type == "analysis":
            logger.info(f"  -- Running Analysis experiment: {exp_name} --")
            
            # --- Model Loading for Analysis ---
            models_to_analyze_loaded: Dict[str, PEFTModelWrapper] = {}
            analysis_config = exp_config.get('prediction_overlap_config', {})
            
            # For analysis, we need a DataLoader to infer num_classes and for evaluation.
            # Assuming analysis experiments are usually for VTAB-1K.
            analysis_sample_task = exp_config.get('analysis_task_for_overlap', VTABLoader.get_vtab_task_names()[0])
            analysis_dataset_loader = VTABLoader(
                dataset_config=global_config['datasets'],
                task_name=analysis_sample_task,
                split_seed=global_config['seed'],
                data_augmentation_policy_name="vtab1k_default" # Assuming default for analysis loading
            )
            analysis_num_classes = analysis_dataset_loader.get_num_classes()

            for model_alias, checkpoint_path in exp_config['models_to_analyze'].items():
                logger.info(f"    Loading model {model_alias} from {checkpoint_path}")
                
                # Heuristic to infer method from alias for PEFTModelWrapper construction
                # This assumes model_alias like "vpt_deep" or "lora"
                inferred_method_name = model_alias.split('_')[0] 
                if inferred_method_name == "full": # Handle full_ft alias
                    inferred_method_name = "full_ft"
                elif inferred_method_name == "linear": # Handle linear_probing alias
                    inferred_method_name = "linear_probing"
                # Capitalize first letter to match PEFTMethod names if needed
                inferred_method_name = inferred_method_name[0].upper() + inferred_method_name[1:]
                # Adjust for specific names
                if inferred_method_name == "Convpass": inferred_method_name = "ConvPass"
                if inferred_method_name == "Fact_tt": inferred_method_name = "FacTTT"
                if inferred_method_name == "Fact_tk": inferred_method_name = "FacTTK"

                model_backbone_config = global_config['model']['backbone'].copy()
                model_backbone_config['pretrained_on'] = "imagenet21k" # Assuming VTAB-1K for analysis

                model_peft_config = {'method': inferred_method_name}
                # For analysis, specific HPs are part of the saved model,
                # we don't need to pass search space HPs here for model construction.
                # However, PEFTModelWrapper needs a full peft_config.
                # This is a known ambiguity. For simplicity, we pass an empty dict for method-specific HPs.
                model_peft_config['peft_hyperparameters'] = {} 

                model_head_config = global_config['model']['head'].copy()

                loaded_model_wrapper = PEFTModelWrapper(
                    backbone_config=model_backbone_config,
                    peft_config=model_peft_config,
                    head_config=model_head_config,
                    num_classes=analysis_num_classes,
                    experiment_type="low_shot", # Assuming low-shot for VTAB analysis context
                    pretrained_model_path=checkpoint_path # Load state_dict immediately after build
                )
                loaded_model_wrapper.to(device)
                loaded_model_wrapper.eval()
                models_to_analyze_loaded[model_alias] = loaded_model_wrapper

            analysis_evaluator = Evaluator(None, exp_config, logger) # Evaluator can operate without a primary model
            
            if "prediction_overlap" in exp_config['analysis_tasks']:
                logger.info("    Computing Prediction Overlap...")
                overlap_task_name = exp_config.get('analysis_task_for_overlap', analysis_sample_task)
                overlap_dataset_loader = VTABLoader(
                    dataset_config=global_config['datasets'],
                    task_name=overlap_task_name,
                    split_seed=global_config['seed'],
                    data_augmentation_policy_name="vtab1k_default"
                )
                test_loader_overlap = overlap_dataset_loader.load_test_data()

                overlap_results = analysis_evaluator.compute_prediction_overlap(
                    models_to_analyze_loaded,
                    test_loader_overlap,
                    topk_confident=analysis_config.get('topk_confident'),
                    leastk_confident=analysis_config.get('leastk_confident')
                )
                logger.log_metrics({"message": f"Prediction overlap for {overlap_task_name}"}, prefix=f"{exp_name}_overlap")
                logger.save_results_to_json(overlap_results, f"{exp_name}_overlap_{overlap_task_name}.json")

            if "ensemble_accuracy" in exp_config['analysis_tasks']:
                logger.info("    Computing Ensemble Accuracy...")
                ensemble_task_name = exp_config.get('analysis_task_for_ensemble', analysis_sample_task)
                ensemble_dataset_loader = VTABLoader(
                    dataset_config=global_config['datasets'],
                    task_name=ensemble_task_name,
                    split_seed=global_config['seed'],
                    data_augmentation_policy_name="vtab1k_default"
                )
                test_loader_ensemble = ensemble_dataset_loader.load_test_data()
                
                ensemble_accuracy_results = analysis_evaluator.compute_ensemble_accuracy(
                    models_to_analyze_loaded,
                    test_loader_ensemble
                )
                logger.log_metrics(ensemble_accuracy_results, prefix=f"{exp_name}_ensemble_{ensemble_task_name}")
                logger.save_results_to_json(ensemble_accuracy_results, f"{exp_name}_ensemble_{ensemble_task_name}.json")

            if "ranking_frequency" in exp_config['analysis_tasks']:
                logger.info("    Computing Ranking Frequency...")
                if not all_vtab_results_for_ranking:
                    logger.warning("    No VTAB-1K results collected for ranking frequency analysis. Skipping.")
                else:
                    ranking_results = analysis_evaluator.compute_ranking_frequency(all_vtab_results_for_ranking)
                    logger.log_metrics({"message": "VTAB-1K Ranking Frequencies"}, prefix=f"{exp_name}_ranking")
                    logger.save_results_to_json(ranking_results, f"{exp_name}_ranking.json")
        else:
            logger.warning(f"Unknown experiment type: {exp_type}. Skipping.")

    logger.info("All experiments completed.")


def run_single_experiment(exp_config: Dict[str, Any], dataset_loader: BaseDatasetLoader, logger: Logger, device: torch.device,
                          peft_param_cap: float, total_backbone_params: int, global_config: Dict[str, Any],
                          all_vtab_results_for_ranking: Dict[str, Dict[str, float]],
                          experiment_sub_path: str):
    """
    Helper function to encapsulate the logic for running a single training/evaluation experiment.
    Handles hyperparameter search, model instantiation, training, and evaluation.
    """
    method_name = exp_config['method']
    exp_type = exp_config['type']
    epochs = exp_config['epochs']
    drop_path_rate = exp_config.get('drop_path_rate', global_config['training']['drop_path_rate_default'])
    backbone_pretrained_on = exp_config['backbone_pretrained_on']

    best_val_accuracy: float = -1.0
    best_hp_config: Dict[str, Any] = {}
    best_model_state_dict: Dict[str, Any] = {}
    best_run_log_dir: str = ""

    # Prepare hyperparameter combinations for search
    hp_search_params: Dict[str, List[Any]] = {
        'learning_rate': exp_config.get('learning_rate_search', [exp_config.get('learning_rate')]),
        'weight_decay': exp_config.get('weight_decay_search', [exp_config.get('weight_decay')]),
    }

    peft_method_specific_hps_values: Dict[str, List[Any]] = {}
    if 'peft_hyperparameter_search_key' in exp_config:
        peft_method_key = exp_config['peft_hyperparameter_search_key']
        peft_method_specific_hps_values = global_config['peft_hyperparameter_search_spaces'].get(peft_method_key, {})
    
    # Combine all hyperparameters for iteration using itertools.product
    # Need to handle cases where search space is a single value (e.g., fixed LR/WD)
    lr_values = hp_search_params['learning_rate']
    wd_values = hp_search_params['weight_decay']

    # Generate PEFT method-specific HP combinations if any
    peft_hps_keys = list(peft_method_specific_hps_values.keys())
    peft_hps_product = itertools.product(*peft_method_specific_hps_values.values()) if peft_hps_keys else [()]

    hp_combinations_list = []
    hp_keys_list = ['learning_rate', 'weight_decay'] + peft_hps_keys

    for lr in lr_values:
        for wd in wd_values:
            for peft_hps in peft_hps_product:
                current_combo = (lr, wd) + peft_hps
                hp_combinations_list.append(dict(zip(hp_keys_list, current_combo)))
    
    if not hp_combinations_list:
        logger.warning(f"No valid hyperparameter combinations found for {experiment_sub_path}. Skipping.")
        return

    logger.info(f"  -- Found {len(hp_combinations_list)} hyperparameter combinations for tuning. --")

    # Load data for current task/dataset
    train_loader = dataset_loader.load_train_data()
    val_loader = dataset_loader.load_val_data()
    test_loader = dataset_loader.load_test_data()
    num_classes = dataset_loader.get_num_classes()

    # Store the initial state of the model for WiSE interpolation later
    # This captures the weights of the backbone (pre-trained) and head/PEFT (initialized)
    pre_fine_tuned_model_state_dict: Optional[Dict[str, Any]] = None
    if exp_type == "robustness" and exp_config.get('wise_interpolation', {}).get('enable', False):
        # Create a PEFTModelWrapper instance to capture its initial state
        # The 'pretrained_model_path' should point to the correct backbone checkpoint
        backbone_checkpoint = global_config['model']['backbone']['pretrained_on_clip_checkpoint'] if backbone_pretrained_on == 'clip' else global_config['model']['backbone']['pretrained_on_imagenet21k_checkpoint']
        
        # We need to instantiate the PEFTModelWrapper with a placeholder PEFT config to get the initial state
        # of the *same PEFT method* that will be trained, so its initialized parameters are part of the base state.
        initial_state_peft_config = {'method': method_name, 'drop_path_rate': drop_path_rate}
        if 'peft_hyperparameter_search_key' in exp_config:
            initial_state_peft_config.update(global_config['peft_hyperparameter_search_spaces'].get(exp_config['peft_hyperparameter_search_key'], {}))

        base_model_for_wise = PEFTModelWrapper(
            backbone_config=global_config['model']['backbone'],
            peft_config=initial_state_peft_config,
            head_config=global_config['model']['head'],
            num_classes=num_classes,
            experiment_type=exp_type,
            pretrained_model_path=backbone_checkpoint # Pass backbone path to load backbone weights
        )
        base_model_for_wise.to(device)
        pre_fine_tuned_model_state_dict = base_model_for_wise.get_pre_fine_tuned_state_dict()
        base_model_for_wise.cpu()
        del base_model_for_wise
        torch.cuda.empty_cache()


    for i, current_hp_config in enumerate(hp_combinations_list):
        logger.info(f"    Running HP combination {i+1}/{len(hp_combinations_list)}: {current_hp_config}")

        # --- Model Initialization for current HP combination ---
        model_peft_config: Dict[str, Any] = {
            'method': method_name, 
            'drop_path_rate': drop_path_rate,
            'peft_hyperparameters': current_hp_config # Pass all HPs for the PEFT module to use
        }
        
        backbone_checkpoint = global_config['model']['backbone']['pretrained_on_clip_checkpoint'] if backbone_pretrained_on == 'clip' else global_config['model']['backbone']['pretrained_on_imagenet21k_checkpoint']

        current_model_wrapper = PEFTModelWrapper(
            backbone_config=global_config['model']['backbone'],
            peft_config=model_peft_config,
            head_config=global_config['model']['head'],
            num_classes=num_classes,
            experiment_type=exp_type,
            pretrained_model_path=backbone_checkpoint # Load backbone weights during construction
        )
        
        trainable_params_count = sum(p.numel() for p in current_model_wrapper.get_trainable_parameters())
        
        # Check PEFT parameter cap
        if method_name not in ["full_ft", "linear_probing"] and trainable_params_count > peft_param_cap:
            logger.info(f"      Skipping HP combination due to parameter cap violation: {trainable_params_count/1e6:.2f}M > {peft_param_cap/1e6:.2f}M")
            current_model_wrapper.cpu()
            del current_model_wrapper
            torch.cuda.empty_cache()
            continue
        
        # Check PEFT target params ratio for many_shot if specified
        if exp_type == "many_shot" and 'peft_target_params_ratio' in exp_config:
            min_ratio, max_ratio = exp_config['peft_target_params_ratio']
            min_cap_params = total_backbone_params * min_ratio
            max_cap_params = total_backbone_params * max_ratio
            if not (min_cap_params <= trainable_params_count <= max_cap_params):
                 logger.info(f"      Skipping HP combination due to target parameter ratio violation: {trainable_params_count/1e6:.2f}M not in [{min_ratio*100:.1f}% ({min_cap_params/1e6:.2f}M) - {max_ratio*100:.1f}% ({max_cap_params/1e6:.2f}M)]")
                 current_model_wrapper.cpu()
                 del current_model_wrapper
                 torch.cuda.empty_cache()
                 continue


        current_model_wrapper.to(device)
        logger.info(f"      Trainable parameters for current run: {trainable_params_count / 1e6:.2f}M")

        # --- Training ---
        current_run_log_dir = os.path.join(log_dir, experiment_sub_path, f"hp_run_{i+1}")
        os.makedirs(current_run_log_dir, exist_ok=True)
        current_run_logger = Logger(current_run_log_dir, filename="training.log")
        current_run_logger.info(f"Current HP config: {current_hp_config}")
        
        trainer = Trainer(current_model_wrapper, global_config['training'], current_run_logger)
        
        # Pass specific LR and WD from current HP config to trainer.train
        trained_model, training_results = trainer.train(
            train_loader, val_loader, 
            lr=current_hp_config['learning_rate'], 
            weight_decay=current_hp_config['weight_decay'], 
            epochs=epochs
        )

        val_accuracy = training_results.get('best_val_accuracy', -1.0)
        logger.info(f"    Validation accuracy for current HP run: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_hp_config = current_hp_config
            best_model_state_dict = deepcopy(trained_model.state_dict())
            # best_metrics = training_results # Not directly used for this loop, but good to store
            best_run_log_dir = current_run_log_dir
            logger.info(f"    New best validation accuracy: {best_val_accuracy:.4f} with HP: {best_hp_config}")

        # Clear model and cache for next HP iteration
        current_model_wrapper.cpu()
        del current_model_wrapper
        del trainer
        torch.cuda.empty_cache()
    
    # --- Post-HP-Search: Final Evaluation with Best Model ---
    if not best_hp_config:
        logger.error(f"No successful hyperparameter run for {experiment_sub_path}. Skipping final evaluation.")
        return

    logger.info(f"\n  --- Best Hyperparameters for {experiment_sub_path}: {best_hp_config} ---")
    
    # Re-initialize model with best hyperparameters and load best state
    final_model_peft_config: Dict[str, Any] = {
        'method': method_name, 
        'drop_path_rate': drop_path_rate,
        'peft_hyperparameters': best_hp_config # Use best HPs
    }
    
    backbone_checkpoint = global_config['model']['backbone']['pretrained_on_clip_checkpoint'] if backbone_pretrained_on == 'clip' else global_config['model']['backbone']['pretrained_on_imagenet21k_checkpoint']

    final_model_wrapper = PEFTModelWrapper(
        backbone_config=global_config['model']['backbone'],
        peft_config=final_model_peft_config,
        head_config=global_config['model']['head'],
        num_classes=num_classes,
        experiment_type=exp_type,
        pretrained_model_path=backbone_checkpoint
    )
    final_model_wrapper.load_state_dict(best_model_state_dict) # Load the best model's trained state
    final_model_wrapper.to(device)
    final_model_wrapper.eval() # Set to eval mode for evaluation

    # --- Evaluation ---
    evaluator = Evaluator(final_model_wrapper, global_config, logger) # Evaluator needs global_config
    test_results = evaluator.evaluate_accuracy(test_loader, prefix='test_')
    logger.log_metrics(test_results, prefix=f"{experiment_sub_path}_final")
    logger.save_results_to_json(test_results, os.path.join(best_run_log_dir, "final_test_results.json"))

    # For VTAB-1K low-shot experiments, collect results for ranking frequency
    if exp_type == "low_shot":
        # VTABLoader has a task_name attribute
        task_name = dataset_loader.task_name
        if task_name not in all_vtab_results_for_ranking:
            all_vtab_results_for_ranking[task_name] = {}
        all_vtab_results_for_ranking[task_name][method_name] = test_results['test_top1_accuracy']


    # --- Robustness-specific evaluation (WiSE) ---
    if exp_type == "robustness" and exp_config.get('wise_interpolation', {}).get('enable', False):
        logger.info(f"\n  --- Running WiSE Interpolation for {experiment_sub_path} ---")
        ood_loaders = dataset_loader.load_test_data_ood() # RobustnessLoader provides OOD loaders
        
        if pre_fine_tuned_model_state_dict is None:
            logger.error("pre_fine_tuned_model_state_dict is None, cannot perform WiSE. Skipping.")
        else:
            wise_results = evaluator.evaluate_wise_robustness(
                pre_fine_tuned_model_state_dict, # The base state
                final_model_wrapper,             # The fine-tuned state
                dataset_loader.load_test_data_target(), # ImageNet-1K test data
                ood_loaders,
                exp_config['wise_interpolation']['alphas']
            )
            logger.log_metrics(wise_results, prefix=f"{experiment_sub_path}_wise")
            logger.save_results_to_json(wise_results, os.path.join(best_run_log_dir, "wise_results.json"))

    # --- Save Best Model ---
    if exp_config.get('save_best_model', False):
        checkpoint_dir = os.path.join(best_run_log_dir, "checkpoints")
        final_model_wrapper.save_pretrained(checkpoint_dir)
        logger.info(f"  Best model checkpoint saved to: {checkpoint_dir}")

    final_model_wrapper.cpu()
    del final_model_wrapper
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reproduce PEFT Visual Recognition experiments.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the main configuration YAML file.")
    args = parser.parse_args()

    main(args.config)
