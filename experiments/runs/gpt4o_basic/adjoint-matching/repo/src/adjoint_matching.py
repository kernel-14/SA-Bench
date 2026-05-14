import torch
import torch.nn as nn

def lean_adjoint_state(x_t, t, base_drift, cost_terminal):
    """
    Computes the lean adjoint state given the conditions.
    """
    # Backpropagate adjoint ODE iteratively
    grad_terminal = torch.autograd.grad(cost_terminal, x_t, retain_graph=True)[0]
    adjoint_state = grad_terminal
    return adjoint_state

def adjoint_matching(model, trajectories, cost_fn):
    """
    Implements the adjoint matching algorithm.
    """
    loss = 0
    for trajectory in trajectories:
        x_t, t = trajectory
        adjoint_state = lean_adjoint_state(x_t, t, model, cost_fn(x_t))
        target = model(x_t, t) + adjoint_state
        loss += nn.MSELoss()(model(x_t, t), target)
    return loss

