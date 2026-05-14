import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SCoReModel
from modules import RewardShapingModule
from layers import KLDivergencePenalty
from transformers import AutoTokenizer
import config
from data import load_dataset

def train():
    # Load dataset
    train_dataset, val_dataset = load_dataset(config.DATASET_NAME)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # Initialize model, tokenizer, and optimizer
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    model = SCoReModel(base_model_name=config.BASE_MODEL_NAME, kl_beta=config.KL_BETA)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Reward shaping module
    reward_shaper = RewardShapingModule(alpha=config.REWARD_ALPHA)

    # Training loop
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            optimizer.zero_grad()

            # Forward pass
            logits, loss = model(input_ids, attention_mask, labels=labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, Loss: {total_loss / len(train_loader)}")

        # Validation
        validate(model, val_loader, tokenizer)

def validate(model, val_loader, tokenizer):
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            logits, _ = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)

            # Calculate accuracy
            total_accuracy += (predictions == labels).float().mean().item()

    print(f"Validation Accuracy: {total_accuracy / len(val_loader)}")

if __name__ == "__main__":
    train()