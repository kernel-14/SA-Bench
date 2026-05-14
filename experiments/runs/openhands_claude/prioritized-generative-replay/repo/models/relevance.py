import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Optional, Tuple

from models.networks import MLP, CNNEncoder, RNDCNNEncoder, ResNet18Encoder


class BaseRelevance(ABC, nn.Module):
    """Abstract base class for relevance functions F(s, a, s', r) = c."""

    @abstractmethod
    def compute(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return scalar relevance values of shape (batch_size, 1)."""
        ...

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Optional gradient update for learnable relevance functions."""
        return {}

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.compute(batch)


class RewardRelevance(BaseRelevance):
    """Relevance based on raw reward signal.

    F(s, a, s', r) = r
    Naive conditioning on high reward; shown to underperform in the paper.
    """

    def compute(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["rewards"]

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {}


class ReturnRelevance(BaseRelevance):
    """Relevance based on Q-value estimate (Eq. 3 from paper).

    F(s, a, s', r) = Q(s, π(s))

    Pushes generations to be more on-policy. Requires a trained Q-function.
    """

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def compute(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        states = batch["states"]
        with torch.no_grad():
            return self.agent.get_q_value(states)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {}


class TDErrorRelevance(BaseRelevance):
    """Relevance based on TD error (Eq. 4 from paper).

    F(s, a, s', r) = r + γ Q_target(s', argmax_a' Q(s', a')) - Q(s, a)

    First proposed for replay prioritization by Schaul et al. (2015).
    """

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def compute(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.agent.get_td_error(
            batch["states"],
            batch["actions"],
            batch["next_states"],
            batch["rewards"],
            batch["dones"],
        )

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {}


class ICMRelevance(BaseRelevance):
    """Intrinsic Curiosity Module relevance function (Eq. 5 from paper).

    F(s, a, s', r) = 1/2 ||g(h(s), a) - h(s')||²

    where h is a feature encoder and g is a forward dynamics model.
    Inspired by Pathak et al. (2017).

    Updated for only 5% of all policy gradient steps (per paper Section 5).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int = 64,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        self.encoder = MLP(state_dim, feature_dim, hidden_dim, n_hidden=2).to(device)
        self.forward_model = MLP(
            feature_dim + action_dim, feature_dim, hidden_dim, n_hidden=2
        ).to(device)
        self.inverse_model = MLP(
            feature_dim * 2, action_dim, hidden_dim, n_hidden=2
        ).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.forward_model.parameters())
            + list(self.inverse_model.parameters()),
            lr=lr,
        )

    def compute(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        with torch.no_grad():
            h_s = self.encoder(states)
            h_sp = self.encoder(next_states)
            h_sp_pred = self.forward_model(torch.cat([h_s, actions], dim=-1))
            curiosity = 0.5 * ((h_sp_pred - h_sp) ** 2).sum(dim=-1, keepdim=True)
        return curiosity

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]

        h_s = self.encoder(states)
        h_sp = self.encoder(next_states)

        h_sp_pred = self.forward_model(torch.cat([h_s, actions], dim=-1))
        forward_loss = 0.5 * F.mse_loss(h_sp_pred, h_sp.detach())

        a_pred = self.inverse_model(torch.cat([h_s, h_sp], dim=-1))
        inverse_loss = F.mse_loss(a_pred, actions)

        loss = forward_loss + inverse_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "icm_forward_loss": forward_loss.item(),
            "icm_inverse_loss": inverse_loss.item(),
        }


class RNDRelevance(BaseRelevance):
    """Random Network Distillation relevance function (Eq. 6 from paper).

    F(s, a, s', r) = 1/2 ||f_hat(s') - f(s')||²

    Fixed target network f and trainable predictor f_hat.
    For pixel-based tasks, uses 3-layer CNN + 2-layer MLP (Appendix A.1).
    For state-based tasks, uses MLP.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 512,
        lr: float = 1e-3,
        use_cnn: bool = False,
        obs_shape: Optional[Tuple[int, ...]] = None,
        latent_dim: int = 64,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.use_cnn = use_cnn

        if use_cnn:
            assert obs_shape is not None
            self.target = RNDCNNEncoder(obs_shape, latent_dim, output_dim).to(device)
            self.predictor = RNDCNNEncoder(obs_shape, latent_dim, output_dim).to(device)
        else:
            self.target = MLP(state_dim, output_dim, hidden_dim, n_hidden=2).to(device)
            self.predictor = MLP(state_dim, output_dim, hidden_dim, n_hidden=2).to(device)

        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def compute(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        next_states = batch["next_states"] if not self.use_cnn else batch["next_observations"]
        with torch.no_grad():
            target_feat = self.target(next_states)
            pred_feat = self.predictor(next_states)
            rnd_error = 0.5 * ((pred_feat - target_feat) ** 2).sum(dim=-1, keepdim=True)
        return rnd_error

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        next_states = batch["next_states"] if not self.use_cnn else batch["next_observations"]
        with torch.no_grad():
            target_feat = self.target(next_states)
        pred_feat = self.predictor(next_states)
        loss = 0.5 * F.mse_loss(pred_feat, target_feat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"rnd_loss": loss.item()}


class CTSRelevance(BaseRelevance):
    """Context-Tree Switching pseudo-count relevance function (Eq. 7 from paper).

    F(s, a, s', r) = (N_hat(s, a) + 0.01)^{-1/2}

    where N_hat is the pseudo-count from a CTS density model.
    Following Bellemare et al. (2016), Theorem 1 and Strehl & Littman (2008).
    Observations resized to 42×42 pixels with 8 context bins.
    """

    def __init__(
        self,
        n_context_bins: int = 8,
        image_size: int = 42,
        epsilon: float = 0.01,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_context_bins = n_context_bins
        self.image_size = image_size
        self.epsilon = epsilon
        self.device = device
        self._cts_model = None

    def _init_cts(self, obs_dim: int):
        """Lazily initialize the CTS density model."""
        self._cts_model = CTSDensityModel(
            obs_dim=obs_dim,
            n_bins=self.n_context_bins,
        )

    def compute(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        states = batch["states"]
        actions = batch["actions"]
        batch_size = states.shape[0]

        if self._cts_model is None:
            self._init_cts(states.shape[-1])

        states_np = states.cpu().numpy()
        actions_np = actions.cpu().numpy()
        pseudo_counts = np.array([
            self._cts_model.query(states_np[i], actions_np[i])
            for i in range(batch_size)
        ], dtype=np.float32)

        relevance = (pseudo_counts + self.epsilon) ** (-0.5)
        return torch.FloatTensor(relevance).unsqueeze(1).to(self.device)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = batch["states"]
        actions = batch["actions"]
        states_np = states.cpu().numpy()
        actions_np = actions.cpu().numpy()

        if self._cts_model is None:
            self._init_cts(states.shape[-1])

        for i in range(len(states_np)):
            self._cts_model.update(states_np[i], actions_np[i])
        return {}


class CTSDensityModel:
    """Simplified CTS density model for pseudo-count estimation.

    Implements a factored density model over discretized state-action space.
    Following Bellemare et al. (2016), Equation 2.
    """

    def __init__(self, obs_dim: int, n_bins: int = 8):
        self.obs_dim = obs_dim
        self.n_bins = n_bins
        self.counts: Dict[Tuple, int] = {}
        self.total = 0

    def _discretize(self, obs: np.ndarray, action: np.ndarray) -> Tuple:
        obs_bins = np.digitize(obs, np.linspace(-3, 3, self.n_bins - 1))
        action_bins = np.digitize(action, np.linspace(-1, 1, self.n_bins - 1))
        return tuple(obs_bins.tolist() + action_bins.tolist())

    def query(self, obs: np.ndarray, action: np.ndarray) -> float:
        key = self._discretize(obs, action)
        count = self.counts.get(key, 0)
        if self.total == 0:
            return 0.0
        rho = count / self.total
        rho_prime = (count + 1) / (self.total + 1)
        if rho_prime <= rho or rho_prime >= 1.0:
            return 0.0
        pseudo_count = rho * (1 - rho_prime) / (rho_prime - rho)
        return max(pseudo_count, 0.0)

    def update(self, obs: np.ndarray, action: np.ndarray):
        key = self._discretize(obs, action)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.total += 1


class ECORelevance(BaseRelevance):
    """Episodic Curiosity through reachability (Eq. 8 from paper).

    F(s, a, s', r) = α(β - F(C(E(s), E(s_i)))) for all s_i in M

    where E is an embedder, C is a comparator, M is a memory buffer.
    Following Savinov et al. (2018).

    Hyperparameters: α=0.03, β=0.5, |M|=200, F=percentile-90.
    Embedder: ResNet-18 + 4-layer MLP (output dim 512).
    """

    def __init__(
        self,
        state_dim: int,
        embed_dim: int = 512,
        memory_size: int = 200,
        alpha: float = 0.03,
        beta: float = 0.5,
        percentile: float = 90.0,
        comparator_lr: float = 1e-3,
        embedder_lr: float = 1e-3,
        use_resnet: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.percentile = percentile
        self.memory_size = memory_size
        self.device = device

        if use_resnet:
            self.embedder = ResNet18Encoder(embed_dim).to(device)
        else:
            self.embedder = MLP(state_dim, embed_dim, 512, n_hidden=3).to(device)

        self.comparator = MLP(embed_dim * 2, 1, 512, n_hidden=2, output_activation="sigmoid").to(device)
        self.memory: deque = deque(maxlen=memory_size)

        self.optimizer = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.comparator.parameters()),
            lr=comparator_lr,
        )

    def _get_reachability(self, embed_s: torch.Tensor) -> torch.Tensor:
        if len(self.memory) == 0:
            return torch.zeros(embed_s.shape[0], 1, device=self.device)

        memory_embeds = torch.stack(list(self.memory), dim=0)
        batch_size = embed_s.shape[0]
        mem_size = memory_embeds.shape[0]

        embed_s_exp = embed_s.unsqueeze(1).expand(-1, mem_size, -1)
        mem_exp = memory_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        pairs = torch.cat([embed_s_exp, mem_exp], dim=-1)
        pairs_flat = pairs.view(batch_size * mem_size, -1)

        with torch.no_grad():
            similarity = self.comparator(pairs_flat).view(batch_size, mem_size)

        aggregated = torch.quantile(similarity, self.percentile / 100.0, dim=1, keepdim=True)
        return aggregated

    def compute(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        states = batch["states"]
        with torch.no_grad():
            embed_s = self.embedder(states)
            reachability = self._get_reachability(embed_s)
            relevance = self.alpha * (self.beta - reachability)
        return relevance

    def update_memory(self, states: torch.Tensor):
        with torch.no_grad():
            embeds = self.embedder(states)
        for i in range(embeds.shape[0]):
            self.memory.append(embeds[i].detach())

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = batch["states"]
        next_states = batch["next_states"]
        batch_size = states.shape[0]

        embed_s = self.embedder(states)
        embed_sp = self.embedder(next_states)

        pos_pairs = torch.cat([embed_s, embed_sp], dim=-1)
        neg_idx = torch.randperm(batch_size)
        neg_pairs = torch.cat([embed_s, embed_sp[neg_idx]], dim=-1)

        pos_labels = torch.ones(batch_size, 1, device=self.device)
        neg_labels = torch.zeros(batch_size, 1, device=self.device)

        all_pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)

        pred = self.comparator(all_pairs)
        loss = F.binary_cross_entropy(pred, all_labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_memory(states[:min(10, batch_size)])

        return {"eco_loss": loss.item()}


def build_relevance_fn(
    relevance_type: str,
    state_dim: int,
    action_dim: int,
    agent=None,
    device: str = "cuda",
    **kwargs,
) -> BaseRelevance:
    """Factory function to build relevance functions by name."""
    if relevance_type == "reward":
        return RewardRelevance()
    elif relevance_type == "return":
        assert agent is not None, "Return relevance requires an agent"
        return ReturnRelevance(agent)
    elif relevance_type == "td_error":
        assert agent is not None, "TD-error relevance requires an agent"
        return TDErrorRelevance(agent)
    elif relevance_type == "curiosity":
        feature_dim = kwargs.get("feature_dim", 64)
        hidden_dim = kwargs.get("hidden_dim", 256)
        lr = kwargs.get("lr", 1e-3)
        return ICMRelevance(state_dim, action_dim, feature_dim, hidden_dim, lr, device)
    elif relevance_type == "rnd":
        hidden_dim = kwargs.get("hidden_dim", 512)
        output_dim = kwargs.get("output_dim", 512)
        lr = kwargs.get("lr", 1e-3)
        use_cnn = kwargs.get("use_cnn", False)
        obs_shape = kwargs.get("obs_shape", None)
        return RNDRelevance(state_dim, hidden_dim, output_dim, lr, use_cnn, obs_shape, device=device)
    elif relevance_type == "cts":
        n_bins = kwargs.get("n_context_bins", 8)
        image_size = kwargs.get("image_size", 42)
        return CTSRelevance(n_bins, image_size, device=device)
    elif relevance_type == "eco":
        embed_dim = kwargs.get("embed_dim", 512)
        memory_size = kwargs.get("memory_size", 200)
        alpha = kwargs.get("alpha", 0.03)
        beta = kwargs.get("beta", 0.5)
        percentile = kwargs.get("percentile", 90.0)
        use_resnet = kwargs.get("use_resnet", False)
        return ECORelevance(
            state_dim, embed_dim, memory_size, alpha, beta, percentile,
            use_resnet=use_resnet, device=device
        )
    else:
        raise ValueError(f"Unknown relevance function: {relevance_type}")
