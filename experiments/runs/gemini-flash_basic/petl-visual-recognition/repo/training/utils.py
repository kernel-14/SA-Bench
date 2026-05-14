import torch

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct, total

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

