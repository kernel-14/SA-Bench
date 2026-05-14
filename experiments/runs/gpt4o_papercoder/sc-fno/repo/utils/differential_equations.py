# utils/differential_equations.py

# Import required libraries
import torch
from torch import Tensor
import math
from typing import Dict, Callable

# Registry for differential equations
PDE_REGISTRY: Dict[str, Callable] = {}

def harmonic_oscillator(alpha: float, beta: float, gamma: float, t: Tensor) -> Tensor:
    """
    Composite Harmonic Oscillator ODE
    Equation: du/dt = alpha * sin(alpha * π * t) + beta * cos(beta * π * t)
    Initial Condition: u(0) = sin(gamma * π)
    
    Args:
    - alpha (float): Parameter affecting frequency of sine component.
    - beta (float): Parameter affecting frequency of cosine component.
    - gamma (float): Initial condition parameter.
    - t (Tensor): Time tensor [0, 1] uniformly spaced.
    
    Returns:
    - Tensor: Solution tensor `u(t)` evaluated at discretized time `t`.
    """
    u_initial = math.sin(gamma * math.pi)
    u = -1 / math.pi * torch.cos(alpha * math.pi * t) + \
        1 / math.pi * torch.sin(beta * math.pi * t) + u_initial + 1 / math.pi
    return u

# Register harmonic oscillator
PDE_REGISTRY["composite_harmonic_oscillator"] = harmonic_oscillator

def duffing_oscillator(delta: float, alpha: float, beta: float, gamma: float, omega: float, epsilon: float, zeta: float, t: Tensor) -> Tensor:
    """
    Duffing Oscillator Equation
    Equation: d^2x/dt^2 + delta * dx/dt + alpha * t + beta * t^3 = gamma * cos(omega * t)
    Initial Conditions: x(0) = epsilon, dx/dt(0) = zeta
    
    Args:
    - delta (float): Damping coefficient.
    - alpha (float): Linear stiffness term.
    - beta (float): Non-linear stiffness term.
    - gamma (float): Driving amplitude.
    - omega (float): Driving frequency.
    - epsilon (float): Initial x value.
    - zeta (float): Initial dx/dt value.
    - t (Tensor): Time tensor [0, 1] uniformly spaced.
    
    Returns:
    - Tensor: Solution tensor `x(t)` evaluated at discretized time `t`.
    """
    # Numerically integrate Duffing equation using PyTorch
    x = epsilon * torch.ones_like(t)
    v = zeta * torch.ones_like(t)  # dx/dt
    
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        # Second-order equation discretized into two first-order equations
        a = -delta * v[i - 1] - alpha * t[i - 1] - beta * t[i - 1] ** 3 + gamma * torch.cos(omega * t[i - 1])
        v[i] = v[i - 1] + dt * a
        x[i] = x[i - 1] + dt * v[i - 1]
    
    return x

# Register Duffing oscillator
PDE_REGISTRY["duffing_oscillator"] = duffing_oscillator

def generalized_nonlinear_damped_wave(c: float, alpha: float, beta: float, gamma: float, omega: float, x: Tensor, t: Tensor) -> Tensor:
    """
    Generalized Nonlinear Damped Wave Equation
    Equation: ∂²u/∂t² = c² ∂²u/∂x² + alpha * ∂u/∂t + beta * u + gamma * sin(omega * u)
    Initial Conditions: u(x, 0) = u_initial
    
    Args:
    - c (float): Wave speed.
    - alpha (float): Damping coefficient affecting ∂u/∂t.
    - beta (float): Linear stiffness of the wave.
    - gamma (float): Forcing term amplitude.
    - omega (float): Frequency of the sinusoidal forcing.
    - x (Tensor): Spatial domain tensor [0, 1] uniformly spaced.
    - t (Tensor): Temporal domain tensor [0, 1] uniformly spaced.
    
    Returns:
    - Tensor: Solution tensor `u[x, t]` (2D shape: [space_steps, time_steps]).
    """
    nx, nt = len(x), len(t)
    u = torch.zeros((nx, nt), dtype=torch.float32)
    u[:, 0] = torch.sin(x * math.pi)  # Initial condition
    
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    for j in range(1, nt):
        for i in range(1, nx - 1):
            u[i, j] = (
                c**2 * (u[i + 1, j - 1] - 2 * u[i, j - 1] + u[i - 1, j - 1]) / dx**2
                + alpha * (u[i, j - 1] - u[i, j - 2]) / dt
                + beta * u[i, j - 1]
                + gamma * torch.sin(omega * u[i, j - 1])
            )
        u[0, j] = u[-1, j]  # Periodic boundary condition
    
    return u

# Register generalized nonlinear damped wave equation
PDE_REGISTRY["generalized_nonlinear_damped_wave"] = generalized_nonlinear_damped_wave

def forced_burgers(alpha: float, gamma: float, delta: float, omega: float, x: Tensor, t: Tensor) -> Tensor:
    """
    Forced Burgers' Equation
    Equation: (1/π) ∂u/∂t + alpha * u * ∂u/∂x = gamma * ∂²u/∂x² + delta * sin(omega * t)
    Args:
    - alpha (float): Advection coefficient.
    - gamma (float): Diffusion coefficient.
    - delta (float): Amplitude of the sinusoidal forcing.
    - omega (float): Frequency of the sinusoidal forcing.
    - x (Tensor): Spatial domain [0, 1].
    - t (Tensor): Temporal domain [0, π].
    
    Returns:
    - Tensor: Solution tensor `u[x, t]`.
    """
    nx, nt = len(x), len(t)
    u = torch.zeros((nx, nt), dtype=torch.float32)
    u[:, 0] = torch.exp(-(x - 0.5) ** 2 / (2.0 * 0.3**2)) + torch.sin(0.5 * math.pi * x)
    
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    for j in range(1, nt):
        for i in range(1, nx - 1):
            u[i, j] = (
                u[i, j - 1]
                + dt * (
                    -alpha * u[i, j - 1] * (u[i + 1, j - 1] - u[i - 1, j - 1]) / (2 * dx)
                    + gamma * (u[i + 1, j - 1] - 2 * u[i, j - 1] + u[i - 1, j - 1]) / dx**2
                    + delta * math.sin(omega * t[j - 1])
                )
            )
        u[0, j] = u[-1, j]  # Periodic boundary condition
    
    return u

# Register forced Burgers' equation
PDE_REGISTRY["forced_burgers_equation"] = forced_burgers
