import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from .layers import RMSNorm, RotaryEmbedding, MultiHeadAttention, SwiGLUMLP, create_causal_mask
from .moe import MoELayer, compute_load_balancing_loss, compute_router_z_loss


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer with attention + MoE FFN (or dense FFN)."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_moe: bool = True,
        num_experts: int = 64,
        num_activated_experts: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        qk_norm: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.use_moe = use_moe

        self.attn_norm = RMSNorm(d_model, eps=layer_norm_eps)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            layer_norm_eps=layer_norm_eps,
        )
        self.ffn_norm = RMSNorm(d_model, eps=layer_norm_eps)

        if use_moe:
            self.moe = MoELayer(
                d_model=d_model,
                num_experts=num_experts,
                num_activated_experts=num_activated_experts,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            self.ffn = None
        else:
            self.moe = None
            self.ffn = SwiGLUMLP(
                d_model=d_model,
                ffn_dim=ffn_dim * num_activated_experts,  # Equivalent active params
                dropout=dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Pre-norm attention
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, rope, attention_mask)
        x = x + residual

        # Pre-norm FFN/MoE
        residual = x
        x = self.ffn_norm(x)

        router_logits, router_probs = None, None
        if self.use_moe:
            x, router_logits, router_probs = self.moe(x)
        else:
            x = self.ffn(x)

        x = x + residual
        return x, router_logits, router_probs


class OLMoEModel(nn.Module):
    """
    OLMoE-1B-7B: Decoder-only MoE language model.

    Architecture (Section 2, Appendix B):
        - 16 transformer layers
        - d_model = 2048
        - 16 attention heads
        - MoE in every layer: 64 experts, 8 activated, ffn_dim = 1024
        - RMSNorm (parametric)
        - QK-Norm
        - RoPE with theta = 10000
        - Truncated normal init (std=0.02, truncation at 3*std)
        - No biases
        - No weight tying
    """
    def __init__(
        self,
        d_model: int = 2048,
        n_layers: int = 16,
        n_heads: int = 16,
        vocab_size: int = 50304,
        max_seq_len: int = 4096,
        num_experts: int = 64,
        num_activated_experts: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        qk_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_experts = num_experts
        self.num_activated_experts = num_activated_experts
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(
            dim=d_model // n_heads,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                use_moe=True,
                num_experts=num_experts,
                num_activated_experts=num_activated_experts,
                ffn_dim=ffn_dim,
                dropout=dropout,
                qk_norm=qk_norm,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model, eps=layer_norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Truncated normal initialization with std=0.02, truncated at 3*std."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=0.02,
                a=-0.06,
                b=0.06,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=0.02,
                a=-0.06,
                b=0.06,
            )
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) token indices
            attention_mask: optional mask
        Returns:
            logits: (B, T, vocab_size)
            router_logits_list: list of (B, T, num_experts) per layer
            router_probs_list: list of (B, T, num_experts) per layer
            aux_losses: placeholder for auxiliary loss contributions per layer
        """
        B, T = input_ids.shape
        x = self.token_embedding(input_ids)

        causal_mask = create_causal_mask(T, x.device)
        if attention_mask is not None:
            causal_mask = causal_mask + attention_mask.unsqueeze(1).unsqueeze(2)

        router_logits_list = []
        router_probs_list = []

        for layer in self.layers:
            x, router_logits, router_probs = layer(x, self.rope, causal_mask)
            if router_logits is not None:
                router_logits_list.append(router_logits)
            if router_probs is not None:
                router_probs_list.append(router_probs)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, router_logits_list, router_probs_list

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        for _ in range(max_new_tokens):
            seq_len = input_ids.shape[1]
            if seq_len > 4096:
                input_ids = input_ids[:, -4096:]

            logits, _, _ = self(input_ids)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, top_k, dim=-1)
                next_logits[next_logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids

    def get_num_params(self) -> Tuple[int, int]:
        """Returns (active_params, total_params)."""
        total = sum(p.numel() for p in self.parameters())
        active = total
        for layer in self.layers:
            if layer.moe is not None:
                inactive_experts = layer.moe.num_experts - layer.moe.num_activated_experts
                expert_params = sum(p.numel() for p in layer.moe.experts.parameters())
                active -= (inactive_experts / layer.moe.num_experts) * expert_params
        return int(active), total

    def save_pretrained(self, path: str):
        """Save model weights in a format compatible with HF."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        config_dict = {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.layers[0].attn.n_heads if self.layers else 16,
            "vocab_size": self.vocab_size,
            "num_experts": self.num_experts,
            "num_activated_experts": self.num_activated_experts,
        }
        import json
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "OLMoEModel":
        """Load model from checkpoint directory."""
        import os, json
        config_path = os.path.join(path, "config.json")
        weights_path = os.path.join(path, "pytorch_model.bin")
        checkpoint_path = os.path.join(path, "checkpoint.pt")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        model_kwargs = {
            "d_model": config.get("d_model", 2048),
            "n_layers": config.get("n_layers", 16),
            "n_heads": config.get("n_heads", 16),
            "vocab_size": config.get("vocab_size", 50304),
            "max_seq_len": config.get("max_seq_len", 4096),
            "num_experts": config.get("num_experts", 64),
            "num_activated_experts": config.get("num_activated_experts", 8),
            "ffn_dim": config.get("ffn_dim", 1024),
            "qk_norm": config.get("qk_norm", True),
            "layer_norm_eps": config.get("layer_norm_eps", 1e-5),
            "rope_theta": config.get("rope_theta", 10000.0),
        }
        model_kwargs.update(kwargs)
        model = cls(**model_kwargs)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
        elif os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)

        return model
