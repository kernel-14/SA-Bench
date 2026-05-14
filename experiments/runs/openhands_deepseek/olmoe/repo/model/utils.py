import yaml
from .olmoe_model import OLMoEModel


def create_model_from_config(config_path: str) -> OLMoEModel:
    """Create an OLMoE model from a YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    return OLMoEModel(
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        vocab_size=model_cfg["vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        num_experts=model_cfg["moe"]["num_experts"],
        num_activated_experts=model_cfg["moe"]["num_activated_experts"],
        ffn_dim=model_cfg["moe"]["ffn_dim"],
        dropout=config.get("pretraining", {}).get("dropout", 0.0),
        qk_norm=model_cfg["qk_norm"],
        layer_norm_eps=model_cfg["layer_norm_eps"],
        rope_theta=model_cfg["rope_theta"],
    )


def create_dense_model_from_config(config_path: str) -> OLMoEModel:
    """
    Create a dense model equivalent from config (for MoE vs. Dense ablations, Section 4.1.1).

    Uses equivalent active parameters by scaling FFN dimension.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    moe_cfg = model_cfg["moe"]

    # Dense FFN dimension = num_activated * ffn_dim to match active params
    dense_ffn_dim = moe_cfg["num_activated_experts"] * moe_cfg["ffn_dim"]

    return OLMoEModel(
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        vocab_size=model_cfg["vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        num_experts=1,  # single expert = dense
        num_activated_experts=1,
        ffn_dim=dense_ffn_dim,
        dropout=config.get("pretraining", {}).get("dropout", 0.0),
        qk_norm=model_cfg["qk_norm"],
        layer_norm_eps=model_cfg["layer_norm_eps"],
        rope_theta=model_cfg["rope_theta"],
    )


def create_moe_variant(
    config_path: str,
    num_experts: int,
    num_activated_experts: int,
) -> OLMoEModel:
    """
    Create MoE variant for granularity experiments (Section 4.1.2).

    Maintains same active parameters by scaling ffn_dim proportionally.
    Base: 64 experts * 1024 ffn_dim * 8 activated = 64*1024 total, 8*1024 active
    For any expert count, ffn_dim is scaled to keep total params constant.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    base_ffn_dim = model_cfg["moe"]["ffn_dim"]
    base_experts = model_cfg["moe"]["num_experts"]
    base_activated = model_cfg["moe"]["num_activated_experts"]

    # Scale ffn_dim: total FFN params = num_experts * ffn_dim^2 * 3 (for SwiGLU)
    # Keep constant: num_experts * ffn_dim^2 = base_experts * base_ffn_dim^2
    import math
    scaled_ffn_dim = int(math.sqrt(base_experts * base_ffn_dim ** 2 / num_experts))

    return OLMoEModel(
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        vocab_size=model_cfg["vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        num_experts=num_experts,
        num_activated_experts=num_activated_experts,
        ffn_dim=scaled_ffn_dim,
        dropout=config.get("pretraining", {}).get("dropout", 0.0),
        qk_norm=model_cfg["qk_norm"],
        layer_norm_eps=model_cfg["layer_norm_eps"],
        rope_theta=model_cfg["rope_theta"],
    )
