import torch
from model import SCoReModel
from transformers import AutoTokenizer
import config

def evaluate_baseline(model_name, dataset_loader):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SCoReModel(base_model_name=model_name, kl_beta=config.KL_BETA)
    model.eval()

    total_accuracy = 0
    with torch.no_grad():
        for batch in dataset_loader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            logits, _ = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)

            # Calculate accuracy
            total_accuracy += (predictions == labels).float().mean().item()

    return total_accuracy / len(dataset_loader)

def run_baselines():
    from data import load_dataset
    from torch.utils.data import DataLoader

    train_dataset, val_dataset = load_dataset(config.DATASET_NAME)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    baselines = ["gpt2", "gpt2-medium", "gpt2-large"]
    for baseline in baselines:
        accuracy = evaluate_baseline(baseline, val_loader)
        print(f"Baseline {baseline} Validation Accuracy: {accuracy}")

if __name__ == "__main__":
    run_baselines()