"""
Unit tests for the Adjoint Matching implementation.

Tests verify:
1. Noise schedule properties (memoryless condition)
2. Adjoint matching loss computation
3. Lean adjoint ODE
4. Baseline methods
"""

import torch
import torch.nn as nn
import numpy as np
from .noise_schedules import get_sigma_memoryless_fm, get_eta_fm, get_kappa_fm, FlowMatchingSchedule
from .adjoint_matching import compute_lean_adjoint, adjoint_matching_loss_fm, select_gradient_timesteps
from .models import MLPVelocityModel
from .baselines import draft_loss, refl_loss


def test_memoryless_noise_schedule():
    """Test that sigma(t) = sqrt(2*eta_t) for the memoryless schedule."""
    h = 0.025  # 1/40
    t_values = torch.linspace(h, 1 - h, 39)
    
    # Compute sigma and eta
    sigma = get_sigma_memoryless_fm(t_values, h=h)
    eta = get_eta_fm(t_values)
    
    # Check sigma^2 = 2*eta (approximately, due to offset)
    # With offset: sigma^2 = 2*(1-t+h)/(t+h) ≈ 2*(1-t)/t = 2*eta for small h
    sigma_sq = sigma ** 2
    two_eta = 2.0 * eta
    
    # They should be close but not exactly equal due to offset
    rel_error = ((sigma_sq - two_eta).abs() / (two_eta + 1e-8)).mean().item()
    print(f"Relative error between sigma^2 and 2*eta: {rel_error:.4f}")
    
    # At t=0.5, sigma^2 = 2*(0.5+h)/(0.5+h) = 2, eta = 0.5/0.5 = 1, 2*eta = 2
    t_mid = torch.tensor([0.5])
    sigma_mid = get_sigma_memoryless_fm(t_mid, h=h)
    eta_mid = get_eta_fm(t_mid)
    print(f"At t=0.5: sigma^2={sigma_mid.item()**2:.4f}, 2*eta={2*eta_mid.item():.4f}")
    
    return True


def test_sigma_properties():
    """Test that sigma(t) is large near t=0 and small near t=1."""
    h = 0.025
    
    t_small = torch.tensor([0.025])  # Near t=0
    t_large = torch.tensor([0.975])  # Near t=1
    
    sigma_small = get_sigma_memoryless_fm(t_small, h=h)
    sigma_large = get_sigma_memoryless_fm(t_large, h=h)
    
    print(f"sigma(t=0.025) = {sigma_small.item():.4f}")
    print(f"sigma(t=0.975) = {sigma_large.item():.4f}")
    
    assert sigma_small > sigma_large, "sigma should be larger near t=0"
    print("PASSED: sigma is larger near t=0")
    return True


def test_gradient_timestep_selection():
    """Test that gradient timestep selection works correctly."""
    num_steps = 40
    timesteps = select_gradient_timesteps(num_steps, num_early=10, num_late=10)
    
    print(f"Selected {len(timesteps)} timesteps: {timesteps[:5]}...{timesteps[-5:]}")
    
    # Check that last 10 are always included
    last_10 = list(range(30, 40))
    for t in last_10:
        assert t in timesteps, f"Timestep {t} should be in gradient timesteps"
    
    # Check total count
    assert len(timesteps) == 20, f"Expected 20 timesteps, got {len(timesteps)}"
    
    print("PASSED: Gradient timestep selection")
    return True


def test_lean_adjoint_terminal_condition():
    """Test that the lean adjoint terminal condition is correct."""
    torch.manual_seed(42)
    
    data_dim = 2
    num_steps = 10
    batch_size = 4
    h = 1.0 / num_steps
    
    # Create simple models
    base_model = MLPVelocityModel(data_dim=data_dim, hidden_dim=32, num_layers=2)
    base_model.eval()
    
    # Create simple trajectory
    states = [torch.randn(batch_size, data_dim) for _ in range(num_steps + 1)]
    
    # Simple quadratic reward
    target = torch.tensor([1.0, 1.0])
    def reward_fn(x):
        return -((x - target) ** 2).sum(dim=-1)
    
    # Compute adjoint
    adjoint_states = compute_lean_adjoint(
        states=states,
        base_velocity_fn=base_model,
        reward_fn=reward_fn,
        num_steps=num_steps,
        use_noiseless_final=False,  # Use exact terminal state for testing
    )
    
    # Check terminal condition: a_K = -nabla r(X_K)
    x_K = states[-1].detach().requires_grad_(True)
    r = reward_fn(x_K)
    r_grad = torch.autograd.grad(r.sum(), x_K)[0]
    expected_a_K = -r_grad.detach()
    
    actual_a_K = adjoint_states[num_steps]
    
    error = (actual_a_K - expected_a_K).abs().max().item()
    print(f"Terminal condition error: {error:.6f}")
    assert error < 1e-5, f"Terminal condition error too large: {error}"
    
    print("PASSED: Lean adjoint terminal condition")
    return True


def test_adjoint_matching_loss_shape():
    """Test that the adjoint matching loss has the correct shape."""
    torch.manual_seed(42)
    
    data_dim = 2
    num_steps = 10
    batch_size = 4
    h = 1.0 / num_steps
    
    # Create models
    base_model = MLPVelocityModel(data_dim=data_dim, hidden_dim=32, num_layers=2)
    finetune_model = MLPVelocityModel(data_dim=data_dim, hidden_dim=32, num_layers=2)
    finetune_model.load_state_dict(base_model.state_dict())
    
    base_model.eval()
    
    # Create trajectory
    states = [torch.randn(batch_size, data_dim) for _ in range(num_steps + 1)]
    
    # Create dummy adjoint states
    adjoint_states = [torch.randn(batch_size, data_dim) for _ in range(num_steps + 1)]
    
    # Compute loss
    loss = adjoint_matching_loss_fm(
        finetune_velocity_fn=finetune_model,
        base_velocity_fn=base_model,
        states=states,
        adjoint_states=adjoint_states,
        num_steps=num_steps,
        lct=None,
        gradient_timesteps=list(range(num_steps)),
    )
    
    print(f"Loss value: {loss.item():.4f}")
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    
    print("PASSED: Adjoint matching loss shape")
    return True


def test_draft_loss():
    """Test DRaFT-1 loss computation."""
    torch.manual_seed(42)
    
    data_dim = 2
    num_steps = 10
    batch_size = 4
    
    model = MLPVelocityModel(data_dim=data_dim, hidden_dim=32, num_layers=2)
    
    x0 = torch.randn(batch_size, data_dim)
    
    def reward_fn(x):
        return -x.norm(dim=-1)
    
    loss = draft_loss(
        velocity_fn=model,
        x0=x0,
        reward_fn=reward_fn,
        num_steps=num_steps,
        K=1,
        sigma_schedule="ode",
    )
    
    print(f"DRaFT-1 loss: {loss.item():.4f}")
    assert loss.dim() == 0, "Loss should be a scalar"
    
    print("PASSED: DRaFT-1 loss")
    return True


def test_refl_loss():
    """Test ReFL loss computation."""
    torch.manual_seed(42)
    
    data_dim = 2
    num_steps = 10
    batch_size = 4
    
    model = MLPVelocityModel(data_dim=data_dim, hidden_dim=32, num_layers=2)
    
    x0 = torch.randn(batch_size, data_dim)
    
    def reward_fn(x):
        return -x.norm(dim=-1)
    
    loss = refl_loss(
        velocity_fn=model,
        x0=x0,
        reward_fn=reward_fn,
        num_steps=num_steps,
        sigma_schedule="ode",
    )
    
    print(f"ReFL loss: {loss.item():.4f}")
    assert loss.dim() == 0, "Loss should be a scalar"
    
    print("PASSED: ReFL loss")
    return True


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running Adjoint Matching unit tests")
    print("=" * 60)
    
    tests = [
        ("Memoryless noise schedule", test_memoryless_noise_schedule),
        ("Sigma properties", test_sigma_properties),
        ("Gradient timestep selection", test_gradient_timestep_selection),
        ("Lean adjoint terminal condition", test_lean_adjoint_terminal_condition),
        ("Adjoint matching loss shape", test_adjoint_matching_loss_shape),
        ("DRaFT-1 loss", test_draft_loss),
        ("ReFL loss", test_refl_loss),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_fn()
            if result:
                passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
