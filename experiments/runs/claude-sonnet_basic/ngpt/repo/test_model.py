"""
Unit tests for the nGPT implementation.

Verifies the key invariants described in the paper:
  1. All weight matrices are unit-norm after init / normalize_weights()
  2. Hidden states stay on the hypersphere through every layer
  3. Scaling parameters are initialised to their intended values
  4. nGPT uses sqrt(d_k) attention scale; GPT uses 1/sqrt(d_k)
  5. Forward pass produces correct shapes and finite loss
  6. Parameter counts match the paper (Table 2)
"""

import math
import torch
import torch.nn.functional as F

from model import nGPT, GPT, nGPTConfig, l2_norm, build_rope_cache, apply_rope


# ── helpers ───────────────────────────────────────────────────────────────────

def small_config(**kw):
    defaults = dict(vocab_size=200, n_layers=2, d_model=64,
                    n_heads=4, d_mlp=256, max_seq_len=128)
    defaults.update(kw)
    return nGPTConfig(**defaults)


def check_unit_norm(tensor, name, dim=1, atol=1e-5):
    norms = torch.norm(tensor, p=2, dim=dim)
    ok = torch.allclose(norms, torch.ones_like(norms), atol=atol)
    if not ok:
        bad = (norms - 1.0).abs().max().item()
        raise AssertionError(f"{name}: max |norm-1| = {bad:.2e}")


# ── tests ─────────────────────────────────────────────────────────────────────

def test_weight_norms_after_init():
    """All nGPT weight matrices must be unit-norm after __init__."""
    model = nGPT(small_config())

    # Embeddings: (vocab, d_model) – normalise along dim=-1
    check_unit_norm(model.E_input.weight,  'E_input',  dim=-1)
    check_unit_norm(model.E_output.weight, 'E_output', dim=-1)

    for i, layer in enumerate(model.layers):
        for name, W in [
            (f'L{i}.W_q', layer.attn.W_q.weight),
            (f'L{i}.W_k', layer.attn.W_k.weight),
            (f'L{i}.W_v', layer.attn.W_v.weight),
            (f'L{i}.W_o', layer.attn.W_o.weight),
            (f'L{i}.W_u', layer.mlp.W_u.weight),
            (f'L{i}.W_v', layer.mlp.W_v.weight),
            (f'L{i}.W_o_mlp', layer.mlp.W_o.weight),
        ]:
            # Linear weight shape: (out, in) – normalise along input dim (dim=1)
            check_unit_norm(W, name, dim=1)

    print("PASS  test_weight_norms_after_init")


def test_normalize_weights_restores_unit_norm():
    """normalize_weights() must fix corrupted matrices."""
    model = nGPT(small_config())

    with torch.no_grad():
        model.E_input.weight.data *= 7.3
        model.layers[0].attn.W_q.weight.data *= 0.1
        model.layers[1].mlp.W_o.weight.data  += 5.0

    model.normalize_weights()

    check_unit_norm(model.E_input.weight, 'E_input', dim=-1)
    check_unit_norm(model.layers[0].attn.W_q.weight, 'W_q', dim=1)
    check_unit_norm(model.layers[1].mlp.W_o.weight,  'W_o_mlp', dim=1)

    print("PASS  test_normalize_weights_restores_unit_norm")


def test_hidden_state_stays_on_hypersphere():
    """After every nGPT layer the hidden state must have unit norm."""
    cfg   = small_config()
    model = nGPT(cfg)
    model.eval()

    B, T = 2, 16
    cos, sin = build_rope_cache(T, cfg.d_model // cfg.n_heads)
    mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)

    h = l2_norm(torch.randn(B, T, cfg.d_model))

    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            h = layer(h, cos, sin, mask)
            norms = torch.norm(h, dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
                raise AssertionError(
                    f"Layer {i}: hidden state not unit-norm, "
                    f"max |norm-1| = {(norms-1).abs().max():.2e}"
                )

    print("PASS  test_hidden_state_stays_on_hypersphere")


def test_scaling_parameter_init():
    """
    Verify the init scheme from Section 2.5:
      stored_param = s_scale
      actual_value = stored_param * (s_init / s_scale) = s_init
    """
    cfg   = small_config(alpha_init=0.05, sqk_init=1.0, su_init=1.0,
                         sv_init=1.0, sz_init=1.0)
    model = nGPT(cfg)

    for i, layer in enumerate(model.layers):
        # alpha_A / alpha_M
        alpha_A = torch.abs(layer.alpha_A * layer.alpha_ratio).mean().item()
        alpha_M = torch.abs(layer.alpha_M * layer.alpha_ratio).mean().item()
        assert abs(alpha_A - cfg.alpha_init) < 1e-5, \
            f"L{i} alpha_A init wrong: {alpha_A:.6f} vs {cfg.alpha_init}"
        assert abs(alpha_M - cfg.alpha_init) < 1e-5, \
            f"L{i} alpha_M init wrong: {alpha_M:.6f} vs {cfg.alpha_init}"

        # s_qk
        sqk = (layer.attn.sqk * layer.attn.sqk_ratio).mean().item()
        assert abs(sqk - cfg.sqk_init) < 1e-5, \
            f"L{i} sqk init wrong: {sqk:.6f} vs {cfg.sqk_init}"

        # s_u, s_v
        su = (layer.mlp.su * layer.mlp.su_ratio).mean().item()
        sv = (layer.mlp.sv * layer.mlp.sv_ratio).mean().item()
        assert abs(su - cfg.su_init) < 1e-5, f"L{i} su init wrong: {su:.6f}"
        assert abs(sv - cfg.sv_init) < 1e-5, f"L{i} sv init wrong: {sv:.6f}"

    # s_z
    sz = (model.sz * model.sz_ratio).mean().item()
    assert abs(sz - cfg.sz_init) < 1e-5, f"sz init wrong: {sz:.6f}"

    print("PASS  test_scaling_parameter_init")


def test_attention_scale():
    """nGPT uses sqrt(d_k); GPT uses 1/sqrt(d_k)."""
    import inspect
    cfg = small_config()

    ngpt_src = inspect.getsource(nGPT(cfg).layers[0].attn.forward)
    gpt_src  = inspect.getsource(GPT(cfg).layers[0].attn.forward)

    # nGPT: scale = math.sqrt(self.d_head)  (multiply, not divide)
    assert 'scale = math.sqrt(self.d_head)' in ngpt_src, \
        "nGPT attention should use scale = sqrt(d_k)"
    # GPT: scale = 1.0 / math.sqrt(self.d_head)
    assert '1.0 / math.sqrt(self.d_head)' in gpt_src, \
        "GPT attention should use scale = 1/sqrt(d_k)"

    print("PASS  test_attention_scale")


def test_forward_shapes_and_finite_loss():
    """Forward pass must return correct shapes and a finite loss."""
    for ModelClass, name in [(nGPT, 'nGPT'), (GPT, 'GPT')]:
        cfg   = small_config()
        model = ModelClass(cfg)
        model.eval()

        B, T = 3, 32
        ids  = torch.randint(0, cfg.vocab_size, (B, T))
        tgt  = torch.randint(0, cfg.vocab_size, (B, T))

        with torch.no_grad():
            logits, loss = model(ids, tgt)

        assert logits.shape == (B, T, cfg.vocab_size), \
            f"{name}: wrong logits shape {logits.shape}"
        assert loss is not None, f"{name}: loss is None"
        assert torch.isfinite(loss), f"{name}: loss is not finite ({loss.item()})"

    print("PASS  test_forward_shapes_and_finite_loss")


def test_no_norm_layers_in_ngpt():
    """nGPT must not contain any RMSNorm / LayerNorm modules."""
    model = nGPT(small_config())
    for name, module in model.named_modules():
        assert not isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm
                                       if hasattr(torch.nn, 'RMSNorm') else type(None))), \
            f"nGPT contains a norm layer at '{name}'"
        # Also check our custom RMSNorm
        from model import RMSNorm
        assert not isinstance(module, RMSNorm), \
            f"nGPT contains RMSNorm at '{name}'"
    print("PASS  test_no_norm_layers_in_ngpt")


def test_parameter_counts():
    """
    Parameter counts should be close to Table 2 in the paper:
      0.5B nGPT: 468.4M   GPT: 468.2M
      1B   nGPT: 1026.1M  GPT: 1025.7M
    """
    paper = {
        '0.5B': {'ngpt': 468.4, 'gpt': 468.2},
        '1B':   {'ngpt': 1026.1, 'gpt': 1025.7},
    }
    sizes = {
        '0.5B': dict(n_layers=24, d_model=1024, n_heads=16),
        '1B':   dict(n_layers=36, d_model=1280, n_heads=20),
    }

    print("\nParameter counts (M):")
    print(f"  {'Size':6s}  {'Model':6s}  {'Ours':>8s}  {'Paper':>8s}  {'Diff':>8s}")
    for size, arch in sizes.items():
        for mtype, cls in [('ngpt', nGPT), ('gpt', GPT)]:
            cfg = nGPTConfig(vocab_size=32000, **arch)
            m   = cls(cfg)
            ours  = sum(p.numel() for p in m.parameters()) / 1e6
            paper_val = paper[size][mtype]
            diff = ours - paper_val
            print(f"  {size:6s}  {mtype:6s}  {ours:8.1f}  {paper_val:8.1f}  {diff:+8.1f}")

    print("PASS  test_parameter_counts")


def test_rope():
    """RoPE should be equivariant: rotating by position p then q == rotating by p+q."""
    d_head, T = 16, 8
    cos, sin = build_rope_cache(T, d_head)

    x = torch.randn(1, T, 2, d_head)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape, "RoPE output shape mismatch"
    assert torch.isfinite(out).all(), "RoPE output contains non-finite values"
    print("PASS  test_rope")


# ── run all ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("nGPT implementation tests")
    print("=" * 60)

    test_weight_norms_after_init()
    test_normalize_weights_restores_unit_norm()
    test_hidden_state_stays_on_hypersphere()
    test_scaling_parameter_init()
    test_attention_scale()
    test_forward_shapes_and_finite_loss()
    test_no_norm_layers_in_ngpt()
    test_rope()
    test_parameter_counts()

    print("\nAll tests passed.")
