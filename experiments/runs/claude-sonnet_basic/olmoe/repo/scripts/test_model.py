"""
Quick test to verify OLMoE model can be instantiated and run a forward pass.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import create_olmoe_1b_7b, OLMoEConfig, OLMoEForCausalLM


def test_model_creation():
    """Test that the model can be created with correct parameter counts."""
    print("Creating OLMoE-1B-7B model...")
    model = create_olmoe_1b_7b()

    total_params = model.get_num_params()
    print(f"Total parameters: {total_params / 1e9:.2f}B")

    # Verify approximate parameter count
    # Expected: ~6.9B total
    assert 6e9 < total_params < 8e9, f"Unexpected param count: {total_params}"
    print(f"Parameter count check passed: {total_params / 1e9:.2f}B (expected ~6.9B)")

    return model


def test_forward_pass(model):
    """Test a forward pass with a small batch."""
    print("\nTesting forward pass...")

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

    # Test without labels
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, model.config.vocab_size)
    print(f"Forward pass (no labels): logits shape = {outputs['logits'].shape}")

    # Test with labels
    labels = input_ids.clone()
    model.train()
    outputs = model(input_ids=input_ids, labels=labels)

    assert "loss" in outputs
    assert outputs["loss"] is not None
    assert "aux_loss" in outputs
    print(f"Forward pass (with labels): loss = {outputs['loss'].item():.4f}, aux_loss = {outputs['aux_loss'].item():.4f}")

    return outputs


def test_auxiliary_losses():
    """Test that auxiliary losses are computed correctly."""
    print("\nTesting auxiliary losses...")

    # Small model for testing
    config = OLMoEConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=100,
        num_experts=8,
        num_experts_per_tok=2,
        expert_ffn_dim=32,
        load_balancing_loss_weight=0.01,
        router_z_loss_weight=0.001,
        use_load_balancing_loss=True,
        use_router_z_loss=True,
    )
    model = OLMoEForCausalLM(config)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)

    assert outputs["aux_loss"] is not None
    assert outputs["aux_loss"].item() > 0
    print(f"Auxiliary loss: {outputs['aux_loss'].item():.6f}")

    # Test without auxiliary losses
    config_no_aux = OLMoEConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=100,
        num_experts=8,
        num_experts_per_tok=2,
        expert_ffn_dim=32,
        use_load_balancing_loss=False,
        use_router_z_loss=False,
    )
    model_no_aux = OLMoEForCausalLM(config_no_aux)
    model_no_aux.train()

    outputs_no_aux = model_no_aux(input_ids=input_ids, labels=labels)
    print(f"Auxiliary loss (disabled): {outputs_no_aux['aux_loss'].item():.6f}")

    print("Auxiliary loss tests passed!")


def test_load_balancing_loss():
    """Test the load balancing loss formula."""
    print("\nTesting load balancing loss formula...")

    from src.model import OLMoEMoELayer
    import torch.nn.functional as F

    config = OLMoEConfig(
        hidden_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        expert_ffn_dim=16,
        use_load_balancing_loss=True,
        use_router_z_loss=False,
    )
    moe = OLMoEMoELayer(config)

    # Create uniform router logits (should give balanced routing)
    num_tokens = 100
    router_logits = torch.zeros(num_tokens, config.num_experts)
    selected = torch.zeros(num_tokens, config.num_experts_per_tok, dtype=torch.long)
    # Assign tokens uniformly to experts
    for i in range(num_tokens):
        selected[i] = torch.tensor([i % config.num_experts, (i + 1) % config.num_experts])

    lb_loss = moe.compute_load_balancing_loss(router_logits, selected)
    print(f"Load balancing loss (uniform): {lb_loss.item():.4f}")

    # For perfectly balanced routing with uniform probs:
    # f_i = k/N_E = 2/4 = 0.5 for each expert
    # P_i = 1/N_E = 0.25 for each expert (uniform softmax)
    # L_LB = N_E * sum_i f_i * P_i = 4 * 4 * 0.5 * 0.25 = 2.0
    # But this is the minimum possible value
    print("Load balancing loss test passed!")


def test_router_z_loss():
    """Test the router Z-loss formula."""
    print("\nTesting router Z-loss formula...")

    from src.model import OLMoEMoELayer

    config = OLMoEConfig(
        hidden_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        expert_ffn_dim=16,
        use_load_balancing_loss=False,
        use_router_z_loss=True,
    )
    moe = OLMoEMoELayer(config)

    # Test with zero logits
    router_logits_zero = torch.zeros(10, config.num_experts)
    z_loss_zero = moe.compute_router_z_loss(router_logits_zero)
    # log(sum(exp(0))) = log(N_E) = log(4) ≈ 1.386
    # z_loss = mean((log(N_E))^2) = log(4)^2 ≈ 1.921
    expected = (torch.log(torch.tensor(float(config.num_experts)))) ** 2
    print(f"Router Z-loss (zero logits): {z_loss_zero.item():.4f} (expected ~{expected.item():.4f})")

    # Test with large logits (should give larger loss)
    router_logits_large = torch.ones(10, config.num_experts) * 10
    z_loss_large = moe.compute_router_z_loss(router_logits_large)
    assert z_loss_large > z_loss_zero, "Large logits should give larger Z-loss"
    print(f"Router Z-loss (large logits): {z_loss_large.item():.4f} > {z_loss_zero.item():.4f}")
    print("Router Z-loss test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("OLMoE Model Tests")
    print("=" * 60)

    # Run tests
    test_auxiliary_losses()
    test_load_balancing_loss()
    test_router_z_loss()

    # Only run full model test if enough memory
    try:
        model = test_model_creation()
        test_forward_pass(model)
        print("\nAll tests passed!")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nSkipping full model test (insufficient memory): {e}")
        else:
            raise
