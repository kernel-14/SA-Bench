import torch
import numpy as np

def compute_initial_gradient(model, data_loader):
    """
    Compute the first gradient approximation from the model given a subset of the dataset.
    """
    model.eval()
    gradients = []
    for data, labels in data_loader:
        outputs = model(data)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone().detach())
    return gradients

def truncated_svd(matrix, rank):
    """
    Perform truncated SVD on a given matrix and return the low-rank components.
    """
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    U = U[:, :rank]
    S = torch.diag(S[:rank])
    Vt = Vt[:rank, :]
    return U, S, Vt

def initialize_lora(model, gradients, rank):
    """
    Initialize LoRA components ($, $, $) based on truncated SVD of averaged gradients.
    """
    gradient_matrix = torch.stack(gradients).mean(dim=0)
    U, S, Vt = truncated_svd(gradient_matrix, rank)

    B = U
    A = Vt
    R = torch.eye(rank) / S
    return B, A, R

# Example Usage:
# model = ...  # Your pre-trained model
# data_loader = ...  # DataLoader with initialization dataset
# gradients = compute_initial_gradient(model, data_loader)
# B, A, R = initialize_lora(model, gradients, rank=32)

