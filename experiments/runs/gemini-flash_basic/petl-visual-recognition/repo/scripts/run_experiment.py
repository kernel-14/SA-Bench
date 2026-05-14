import yaml
import argparse
import os
from datetime import datetime

from training.train import train_model

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create a unique experiment name based on config and timestamp
    peft_method = config['peft']['method']
    scenario = config['scenario']
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{scenario}-{peft_method}-{timestamp}"
    
    output_dir = os.path.join("experiments", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save the config used for this experiment
    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        yaml.safe_dump(config, f)

    print(f"Starting experiment: {experiment_name}")
    results = train_model(config)
    print(f"Experiment {experiment_name} finished. Results: {results}")

    with open(os.path.join(output_dir, "results.txt"), 'w') as f:
        for task_name, res in results.items():
            f.write(f"Task: {task_name}, Test Accuracy: {res['test_accuracy']:.4f}, Best Val Accuracy: {res['best_val_accuracy']:.4f}
")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PEFT experiments.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help="Path to the config file.")
    args = parser.parse_args()

    main(args.config)
