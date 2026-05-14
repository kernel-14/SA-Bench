import torch
import torch.nn as nn
from models.transformer_with_gated_attention import TransformerWithGatedAttention

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model.output_dim), targets.view(-1))
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

    perplexity = torch.exp(total_loss / total_tokens)
    return perplexity

if __name__ == "__main__":
    # Example configuration
    model = TransformerWithGatedAttention(model_dim=512, num_heads=8, ff_dim=2048, num_layers=12)
    criterion = nn.CrossEntropyLoss()

    # Initialize dummy data loader
    data_loader = ... # Load test data (to be integrated with real dataset)

    perplexity = evaluate_model(model, data_loader, criterion)
    print(f"Model Perplexity: {perplexity}")

