import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import calculate_accuracy, get_device

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            correct, samples = calculate_accuracy(outputs, labels)
            total_correct += correct
            total_samples += samples

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

