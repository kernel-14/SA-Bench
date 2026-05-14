
import jax
import ml_collections
import os
from datetime import datetime

from luno.config import get_config
from luno.train import train_model
from luno.evaluate import load_fno_model, evaluate_uq_method, calibrate_hyperparameters
from luno.data.pde_datasets import PDEDataset

def main(config: ml_collections.ConfigDict):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(config.log_dir, config.data.pde_name, config.uq.method, current_time)
    os.makedirs(log_dir, exist_ok=True)

    rng = jax.random.PRNGKey(config.seed)

    # 1. Train FNO model
    print("Starting FNO model training...")
    trained_fno_params = train_model(config)
    print("FNO model training complete.")

    # 2. Load the trained FNO model for UQ evaluation
    # For simplicity, we assume `train_model` saved the best params
    # We need the path to the saved parameters
    pde_checkpoint_dir = os.path.join(config.save_dir, config.data.pde_name)
    fno_params_path = os.path.join(pde_checkpoint_dir, 'best_fno_params.msgpack')
    
    fno_model, loaded_fno_params = load_fno_model(config, fno_params_path)
    print("Trained FNO model loaded for UQ.")

    # 3. Calibrate UQ hyperparameters (if applicable)
    rng, calib_rng = jax.random.split(rng)
    val_dataset = PDEDataset(config, 'validation')
    calibrated_config = calibrate_hyperparameters(fno_model, loaded_fno_params, config, val_dataset, calib_rng)
    print(f"Calibrated UQ config: {calibrated_config.uq}")

    # 4. Evaluate UQ method
    rng, eval_rng = jax.random.split(rng)
    test_dataset = PDEDataset(config, 'test')
    evaluation_metrics = evaluate_uq_method(fno_model, loaded_fno_params, calibrated_config, test_dataset, eval_rng)
    
    print("\n--- Evaluation Results ---")
    for metric_name, value in evaluation_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Save results
    results_path = os.path.join(log_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"UQ Method: {config.uq.method}\n")
        for metric_name, value in evaluation_metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")
    print(f"Evaluation results saved to {results_path}")


if __name__ == '__main__':
    config = get_config()
    main(config)
