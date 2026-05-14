import yaml
from score.model import SCoReModel
from score.training import SCoReTrainer
from score.datasets import SCoReDataset

def main():
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    print("--- SCoRe Reproduction Script ---")
    print(f"Configuration: {config}")

    # Initialize models
    # The main model being trained
    model = SCoReModel(
        model_name=config['model']['name'],
        base_model_path=config['model']['base_model_path']
    )
    # The reference model for KL-divergence. Can be the same as base_model or a fixed checkpoint.
    ref_model = SCoReModel(
        model_name=config['model']['name'] + "_ref", # Differentiate name if needed
        base_model_path=config['model']['ref_model_path']
    )

    # Initialize dataset
    dataset = SCoReDataset(
        dataset_name=config['task']['dataset'],
        data_dir=config['paths']['data_dir']
    )
    problems, ground_truths = dataset.get_data()

    # Initialize trainer
    trainer = SCoReTrainer(model=model, ref_model=ref_model, config=config)

    # --- Training Stages ---
    # Stage I Training
    trainer.train_stage_i(problems, ground_truths)

    # Stage II Training
    trainer.train_stage_ii(problems, ground_truths)

    # --- Evaluation ---
    print("
--- Final Evaluation ---")
    metrics = trainer.evaluate(problems, ground_truths)
    print("
Final Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("
--- SCoRe Reproduction Script Finished ---")

if __name__ == "__main__":
    main()
