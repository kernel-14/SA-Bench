from typing import Tuple, Optional

import torch
import torch.nn as nn


class IndependentCoupling:
    """
    Standard independent coupling (IC): q_I(x_*, z) = p_*(x_*) p_z(z).

    Constructs pairs (x_{t_i}, x_{t_{i+1}}) from independent data and noise samples.
    """

    def construct_pairs(
        self,
        x_star: torch.Tensor,
        z: torch.Tensor,
        sigma_i: torch.Tensor,
        sigma_i1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct consecutive noisy pairs from IC.

        Args:
            x_star: clean data (B, C, H, W)
            z: noise (B, C, H, W)
            sigma_i: lower noise level (B,)
            sigma_i1: upper noise level (B,)

        Returns:
            (x_{t_i}, x_{t_{i+1}}): noisy pairs
        """
        sigma_i_bc = sigma_i[:, None, None, None]
        sigma_i1_bc = sigma_i1[:, None, None, None]
        x_ti = x_star + sigma_i_bc * z
        x_ti1 = x_star + sigma_i1_bc * z
        return x_ti, x_ti1


class GeneratorAugmentedCoupling:
    """
    Generator-Augmented Coupling (GC) from Issenhuth et al. (2024).

    Uses the consistency model to predict the endpoint x_hat from an IC intermediate
    point, then constructs new pairs using (x_hat, z):

        x_hat_{t_i} = sg(f_θ(x_{t_i}, σ_{t_i}))
        x_tilde_{t_i} = x_hat_{t_i} + σ_{t_i} * z
        x_tilde_{t_{i+1}} = x_hat_{t_i} + σ_{t_{i+1}} * z
    """

    def construct_pairs(
        self,
        x_hat: torch.Tensor,
        z: torch.Tensor,
        sigma_i: torch.Tensor,
        sigma_i1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct consecutive noisy pairs from GC.

        Args:
            x_hat: predicted endpoint from consistency model (B, C, H, W), stop-gradient applied
            z: noise (B, C, H, W)
            sigma_i: lower noise level (B,)
            sigma_i1: upper noise level (B,)

        Returns:
            (x_tilde_{t_i}, x_tilde_{t_{i+1}}): GC noisy pairs
        """
        sigma_i_bc = sigma_i[:, None, None, None]
        sigma_i1_bc = sigma_i1[:, None, None, None]
        x_tilde_ti = x_hat + sigma_i_bc * z
        x_tilde_ti1 = x_hat + sigma_i1_bc * z
        return x_tilde_ti, x_tilde_ti1


class BatchOTCoupling:
    """
    Minibatch Optimal Transport coupling (batch-OT) from Pooladian et al. (2023),
    applied to consistency models by Dou et al. (2024).

    Solves the assignment problem within each minibatch to find the optimal
    pairing between data points and noise vectors.
    """

    def __init__(self, reg: float = 0.05, use_sinkhorn: bool = True):
        """
        Args:
            reg: regularization parameter for Sinkhorn algorithm
            use_sinkhorn: if True use Sinkhorn, else use exact Hungarian matching
        """
        self.reg = reg
        self.use_sinkhorn = use_sinkhorn

    def _compute_cost_matrix(
        self, x_star: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise squared L2 cost matrix between x_star and z."""
        B = x_star.shape[0]
        x_flat = x_star.view(B, -1)
        z_flat = z.view(B, -1)
        cost = torch.cdist(x_flat, z_flat, p=2) ** 2
        return cost

    def get_permutation(
        self, x_star: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute optimal permutation of noise vectors to match data points.

        Args:
            x_star: clean data (B, C, H, W)
            z: noise (B, C, H, W)

        Returns:
            Permutation indices (B,) such that z[perm[i]] is matched to x_star[i]
        """
        try:
            import ot
        except ImportError:
            raise ImportError("POT (Python Optimal Transport) is required for batch-OT. Install with: pip install pot")

        B = x_star.shape[0]
        cost = self._compute_cost_matrix(x_star, z).detach().cpu().numpy()
        a = torch.ones(B).numpy() / B
        b = torch.ones(B).numpy() / B

        if self.use_sinkhorn:
            transport_plan = ot.sinkhorn(a, b, cost, reg=self.reg)
        else:
            transport_plan = ot.emd(a, b, cost)

        # Extract assignment: for each data point, find the best matched noise
        perm = transport_plan.argmax(axis=1)
        return torch.from_numpy(perm).long().to(x_star.device)

    def construct_pairs(
        self,
        x_star: torch.Tensor,
        z: torch.Tensor,
        sigma_i: torch.Tensor,
        sigma_i1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct consecutive noisy pairs using OT-matched (x_star, z) pairs.

        Args:
            x_star: clean data (B, C, H, W)
            z: noise (B, C, H, W)
            sigma_i: lower noise level (B,)
            sigma_i1: upper noise level (B,)

        Returns:
            (x_{t_i}, x_{t_{i+1}}): OT-coupled noisy pairs
        """
        perm = self.get_permutation(x_star, z)
        z_matched = z[perm]

        sigma_i_bc = sigma_i[:, None, None, None]
        sigma_i1_bc = sigma_i1[:, None, None, None]
        x_ti = x_star + sigma_i_bc * z_matched
        x_ti1 = x_star + sigma_i1_bc * z_matched
        return x_ti, x_ti1


class JointCoupling:
    """
    Joint learning coupling that mixes IC and GC trajectories.

    At each training step, for each sample in the batch:
    - With probability μ: use GC trajectory
    - With probability (1-μ): use IC trajectory

    This implements the loss:
        L_{GC-μ}(θ) = μ * L_GC(θ) + (1-μ) * L_CT(θ)

    In practice, a binary mask m ~ Binomial(μ, batch_size) selects which
    samples use GC vs IC trajectories.
    """

    def __init__(self, mu: float = 0.5):
        self.mu = mu
        self.ic = IndependentCoupling()
        self.gc = GeneratorAugmentedCoupling()

    def construct_pairs(
        self,
        x_star: torch.Tensor,
        z: torch.Tensor,
        sigma_i: torch.Tensor,
        sigma_i1: torch.Tensor,
        model: nn.Module,
        x_ti_ic: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct pairs mixing IC and GC trajectories.

        Args:
            x_star: clean data (B, C, H, W)
            z: noise (B, C, H, W)
            sigma_i: lower noise level (B,)
            sigma_i1: upper noise level (B,)
            model: consistency model for GC endpoint prediction
            x_ti_ic: pre-computed IC intermediate points (optional)

        Returns:
            (x_lower, x_upper): mixed IC/GC noisy pairs
        """
        B = x_star.shape[0]
        device = x_star.device

        # Sample binary mask: m_j ~ Binomial(μ)
        m = torch.bernoulli(torch.full((B,), self.mu, device=device)).bool()

        # Compute IC intermediate points
        sigma_i_bc = sigma_i[:, None, None, None]
        sigma_i1_bc = sigma_i1[:, None, None, None]

        if x_ti_ic is None:
            x_ti_ic = x_star + sigma_i_bc * z

        # Predict endpoint using the model (stop-gradient)
        with torch.no_grad():
            x_hat = model(x_ti_ic, sigma_i).detach()

        # Mix: GC uses x_hat, IC uses x_star
        # x_hat_mixed[j] = x_hat[j] if m[j] else x_star[j]
        m_bc = m[:, None, None, None].float()
        x_hat_mixed = m_bc * x_hat + (1 - m_bc) * x_star

        # Construct pairs from mixed endpoints
        x_lower = x_hat_mixed + sigma_i_bc * z
        x_upper = x_hat_mixed + sigma_i1_bc * z

        return x_lower, x_upper
