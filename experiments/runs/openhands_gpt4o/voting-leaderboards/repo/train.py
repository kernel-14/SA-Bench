import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import TargetModelDetector
from data import TextDataset
from config import Config

def train_model():
    # Load configuration
    config = Config()

    # Load dataset
    dataset = TextDataset(config.data_path)
    train_data, val_data = train_test_split(dataset, test_size=config.val_split, random_state=42)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    model = TargetModelDetector(feature_type=config.feature_type)
    optimizer = optim.Adam(model.classifier.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(config.epochs):
        model.classifier.train()
        train_loss = 0.0
        for prompts, responses, labels in train_loader:
            optimizer.zero_grad()
            predictions = model.predict(responses)
            loss = criterion(torch.tensor(predictions, dtype=torch.float32), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            for prompts, responses, labels in val_loader:
                predictions = model.predict(responses)
                loss = criterion(torch.tensor(predictions, dtype=torch.float32), labels.float())
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    train_model()