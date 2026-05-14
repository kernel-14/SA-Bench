import torch
import torch.nn as nn
import torch.optim as optim
from sc_fno import SCFNO

# SC-FNO Training Loop (Algorithm 2)
def train_sc_fno(
    model, dataloader, optimizer, criterion_u, criterion_s, max_epochs, device
):
    model.to(device)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss_u = 0.0
        epoch_loss_s = 0.0

        for batch in dataloader:
            params, u_true, du_true = batch  # Input parameters, ground truth for u and sensitivity du
            params, u_true, du_true = params.to(device), u_true.to(device), du_true.to(device)

            # Forward pass
            u_pred = model(params)  # Predicted solution path

            # Compute losses
            loss_u = criterion_u(u_pred, u_true)

            # Compute sensitivities with Autograd (Jacobian)
            u_pred.requires_grad_(True)
            jacobian_pred = torch.autograd.grad(
                outputs=u_pred.sum(),
                inputs=params,
                allow_unused=True,
                retain_graph=True,
                create_graph=True,
            )[0]

            # Sensitivity loss
            loss_s = criterion_s(jacobian_pred, du_true)

            # Total loss
            loss_total = loss_u + loss_s

            # Backpropagation
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # Record losses
            epoch_loss_u += loss_u.item()
            epoch_loss_s += loss_s.item()

        print(fEpoch {epoch + 1}: Loss_u - {epoch_loss_u}, Loss_s - {epoch_loss_s})

# Example usage (placeholder)
if __name__ == __main__:
    # Create dataset (placeholder)
    dataset = None  # Synthetic data must be prepared
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Model setup
    sc_fno = SCFNO(input_dim=10, output_dim=1, modes=16, width=64)

    # Optimizer and criteria
    optimizer = optim.Adam(sc_fno.parameters(), lr=1e-3)
    criterion_u = nn.MSELoss()
    criterion_s = nn.MSELoss()

    # Training
    train_sc_fno(sc_fno, dataloader, optimizer, criterion_u, criterion_s, max_epochs=20, device=cuda)
