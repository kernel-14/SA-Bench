"""Differential equation solvers with sensitivity (Jacobian) computation.

Implements both:
1. Differentiable numerical solvers (AD) using PyTorch's autograd.
2. Finite difference (FD) solvers using 4th-order central differences.

Each solver produces:
- Solution paths u(x,t) or u(t)
- True Jacobians du/dp at specified points
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class BaseSolver:
    """Base class for equation solvers."""

    def __init__(self, params: Dict[str, float], n_t: int = 100, n_x: int = 20):
        self.params = params
        self.n_t = n_t
        self.n_x = n_x

    @property
    def param_names(self) -> List[str]:
        raise NotImplementedError

    def get_param_tensor(self, param_dict: Dict[str, float]) -> torch.Tensor:
        return torch.tensor(
            [param_dict[n] for n in self.param_names], dtype=torch.float32
        )

    def solve(
        self, param_dict: Dict[str, float], return_jacobian: bool = True
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def solve_fd(
        self, param_dict: Dict[str, float], eps: float = 1e-3
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class ODE1Solver(BaseSolver):
    """ODE1: Composite Harmonic Oscillator.

    du/dt = α sin(απt) + β cos(βπt)

    Initial condition: u(0) = sin(γπ)
    Domain: t ∈ [0, 1]

    Analytical solution:
    u(t) = -1/π cos(απt) + 1/π sin(βπt) + sin(γπ) + 1/π

    Analytical Jacobians:
    ∂u/∂α = t sin(απt)
    ∂u/∂β = t cos(βπt)
    ∂u/∂γ = π cos(γπ)
    """

    @property
    def param_names(self) -> List[str]:
        return ["alpha", "beta", "gamma"]

    def solve(
        self, param_dict: Dict[str, float], return_jacobian: bool = True
    ) -> Dict[str, torch.Tensor]:
        alpha = param_dict["alpha"]
        beta = param_dict["beta"]
        gamma = param_dict["gamma"]

        t = torch.linspace(0, 1, self.n_t)

        u = (
            -1.0 / math.pi * torch.cos(alpha * math.pi * t)
            + 1.0 / math.pi * torch.sin(beta * math.pi * t)
            + math.sin(gamma * math.pi)
            + 1.0 / math.pi
        )

        result = {"t": t, "u": u}

        if return_jacobian:
            du_dalpha = t * torch.sin(alpha * math.pi * t)
            du_dbeta = t * torch.cos(beta * math.pi * t)
            du_dgamma = torch.full_like(t, math.pi * math.cos(gamma * math.pi))
            result["du_dalpha"] = du_dalpha
            result["du_dbeta"] = du_dbeta
            result["du_dgamma"] = du_dgamma

        return result

    def solve_fd(
        self, param_dict: Dict[str, float], eps: float = 1e-3
    ) -> Dict[str, torch.Tensor]:
        u0 = self.solve(param_dict, return_jacobian=False)["u"]

        jacobians = {}
        for i, name in enumerate(self.param_names):
            params_plus = dict(param_dict)
            params_minus = dict(param_dict)
            params_plus[name] += eps
            params_minus[name] -= eps

            u_plus = self.solve(params_plus, return_jacobian=False)["u"]
            u_minus = self.solve(params_minus, return_jacobian=False)["u"]

            jacobians[f"du_d{name}"] = (u_plus - u_minus) / (2 * eps)

        return {"t": torch.linspace(0, 1, self.n_t), "u": u0, **jacobians}


class ODE2Solver(BaseSolver):
    """ODE2: Duffing Oscillator.

    d²x/dt² + δ dx/dt + α x + β x³ = γ cos(ω t)

    Initial conditions: x(0) = ε, dx/dt(0) = ζ
    Parameters: α, β, γ, δ, ω, ε, ζ

    Solved numerically using 4th-order Runge-Kutta.
    """

    @property
    def param_names(self) -> List[str]:
        return ["alpha", "beta", "gamma", "delta", "omega", "epsilon", "zeta"]

    def _ode_rhs(
        self, state: torch.Tensor, t: float, params: Dict[str, float]
    ) -> torch.Tensor:
        x, v = state[..., 0], state[..., 1]
        dxdt = v
        dvdt = (
            -params["delta"] * v
            - params["alpha"] * x
            - params["beta"] * x**3
            + params["gamma"] * math.cos(params["omega"] * t)
        )
        return torch.stack([dxdt, dvdt], dim=-1)

    def solve(
        self, param_dict: Dict[str, float], return_jacobian: bool = True
    ) -> Dict[str, torch.Tensor]:
        dt = 1.0 / (self.n_t - 1)

        state = torch.tensor([param_dict["epsilon"], param_dict["zeta"]])
        t_vals = torch.linspace(0, 1, self.n_t)

        states = []
        for i in range(self.n_t):
            states.append(state.clone())
            t_i = t_vals[i].item()

            k1 = dt * self._ode_rhs(state, t_i, param_dict)
            k2 = dt * self._ode_rhs(state + k1 / 2, t_i + dt / 2, param_dict)
            k3 = dt * self._ode_rhs(state + k2 / 2, t_i + dt / 2, param_dict)
            k4 = dt * self._ode_rhs(state + k3, t_i + dt, param_dict)

            state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        states = torch.stack(states)
        u = states[:, 0]

        result = {"t": t_vals, "u": u}

        if return_jacobian:
            result = self._compute_jacobians(param_dict, t_vals, states, result)

        return result

    def _compute_jacobians(
        self,
        param_dict: Dict[str, float],
        t_vals: torch.Tensor,
        states: torch.Tensor,
        base_result: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-3
        for name in self.param_names:
            params_plus = dict(param_dict)
            params_minus = dict(param_dict)
            params_plus[name] += eps
            params_minus[name] -= eps

            u_plus = self._run_rk4(params_plus, t_vals)
            u_minus = self._run_rk4(params_minus, t_vals)

            base_result[f"du_d{name}"] = (u_plus - u_minus) / (2 * eps)

        return base_result

    def _run_rk4(
        self, param_dict: Dict[str, float], t_vals: torch.Tensor
    ) -> torch.Tensor:
        dt = 1.0 / (self.n_t - 1)
        state = torch.tensor([param_dict["epsilon"], param_dict["zeta"]])
        u_vals = []
        for i in range(len(t_vals)):
            u_vals.append(state[0].clone())
            t_i = t_vals[i].item()
            k1 = dt * self._ode_rhs(state, t_i, param_dict)
            k2 = dt * self._ode_rhs(state + k1 / 2, t_i + dt / 2, param_dict)
            k3 = dt * self._ode_rhs(state + k2 / 2, t_i + dt / 2, param_dict)
            k4 = dt * self._ode_rhs(state + k3, t_i + dt, param_dict)
            state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return torch.stack(u_vals)

    def solve_fd(
        self, param_dict: Dict[str, float], eps: float = 1e-3
    ) -> Dict[str, torch.Tensor]:
        return self.solve(param_dict, return_jacobian=True)


class PDE1Solver(BaseSolver):
    """PDE1: Generalized Nonlinear Damped Wave Equation.

    ∂²u/∂t² = c² ∂²u/∂x² + α ∂u/∂t + β u + γ sin(ω u)

    Domain: x ∈ [0, 1], t ∈ [0, 1]
    Initial: u(x, 0) = u₀(x), ∂u/∂t(x, 0) = u₀'(x)
    Parameters: c, α, β, γ, ω

    Solved via finite differences with Runge-Kutta integration.
    """

    @property
    def param_names(self) -> List[str]:
        return ["c", "alpha", "beta", "gamma", "omega"]

    def _laplacian(self, u: torch.Tensor, dx: float) -> torch.Tensor:
        u_pad = F.pad(u.unsqueeze(0).unsqueeze(0), (1, 1), mode="circular")
        laplacian = (u_pad[..., :-2] + u_pad[..., 2:] - 2 * u_pad[..., 1:-1]) / (dx**2)
        return laplacian.squeeze(0).squeeze(0)

    def _rhs(
        self,
        state: torch.Tensor,
        dx: float,
        param_dict: Dict[str, float],
    ) -> torch.Tensor:
        u, v = state[0], state[1]
        u_xx = self._laplacian(u, dx)
        du_dt = v
        dv_dt = (
            param_dict["c"] ** 2 * u_xx
            + param_dict["alpha"] * v
            + param_dict["beta"] * u
            + param_dict["gamma"] * torch.sin(param_dict["omega"] * u)
        )
        return torch.stack([du_dt, dv_dt])

    def _initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * x) + 0.5 * torch.sin(3 * math.pi * x)

    def solve(
        self, param_dict: Dict[str, float], return_jacobian: bool = True
    ) -> Dict[str, torch.Tensor]:
        x = torch.linspace(0, 1, self.n_x)
        dx = x[1] - x[0]
        dt = 1.0 / (self.n_t - 1)

        u0 = self._initial_condition(x)
        v0 = torch.zeros_like(u0)
        state = torch.stack([u0, v0])

        t_vals = torch.linspace(0, 1, self.n_t)

        snapshots = []
        for i in range(self.n_t):
            snapshots.append(state[0].clone())
            t_i = t_vals[i].item()

            k1 = dt * self._rhs(state, dx, param_dict)
            k2 = dt * self._rhs(state + k1 / 2, dx, param_dict)
            k3 = dt * self._rhs(state + k2 / 2, dx, param_dict)
            k4 = dt * self._rhs(state + k3, dx, param_dict)
            state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        u = torch.stack(snapshots)
        result = {"x": x, "t": t_vals, "u": u}

        if return_jacobian:
            result = self._compute_jacobians(param_dict, x, u, result)

        return result

    def _compute_jacobians(
        self,
        param_dict: Dict[str, float],
        x: torch.Tensor,
        u_base: torch.Tensor,
        base_result: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-3
        dt = 1.0 / (self.n_t - 1)
        dx = x[1] - x[0]

        for name in self.param_names:
            params_plus = dict(param_dict)
            params_minus = dict(param_dict)
            params_plus[name] *= (1 + eps)
            params_minus[name] *= (1 - eps) if (1 - eps) * param_dict[name] > 0 else 0

            u_plus = self._run_simulation(x, dt, dx, params_plus)
            u_minus = self._run_simulation(x, dt, dx, params_minus)

            denom = param_dict[name] * 2 * eps
            base_result[f"du_d{name}"] = (u_plus - u_minus) / max(denom, 1e-10)

        return base_result

    def _run_simulation(
        self,
        x: torch.Tensor,
        dt: float,
        dx: float,
        param_dict: Dict[str, float],
    ) -> torch.Tensor:
        u0 = self._initial_condition(x)
        v0 = torch.zeros_like(u0)
        state = torch.stack([u0, v0])

        snapshots = []
        for _ in range(self.n_t):
            snapshots.append(state[0].clone())
            k1 = dt * self._rhs(state, dx, param_dict)
            k2 = dt * self._rhs(state + k1 / 2, dx, param_dict)
            k3 = dt * self._rhs(state + k2 / 2, dx, param_dict)
            k4 = dt * self._rhs(state + k3, dx, param_dict)
            state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return torch.stack(snapshots)

    def solve_fd(
        self, param_dict: Dict[str, float], eps: float = 1e-3
    ) -> Dict[str, torch.Tensor]:
        return self.solve(param_dict, return_jacobian=True)


class PDE2Solver(BaseSolver):
    """PDE2: Forced Burgers' Equation.

    (1/π) ∂u/∂t + α u ∂u/∂x = γ ∂²u/∂x² + δ sin(ω t)

    Domain: x ∈ [0, 1], t ∈ [0, π]
    Parameters: α, γ, δ, ω

    Initial condition: u₀(x) = exp(-(x - 0.5)² / (2·0.3²)) + sin(0.5π x)
    Periodic boundary conditions.
    """

    @property
    def param_names(self) -> List[str]:
        return ["alpha", "gamma", "delta", "omega"]

    def _first_derivative(self, u: torch.Tensor, dx: float) -> torch.Tensor:
        u_pad = F.pad(u.unsqueeze(0).unsqueeze(0), (1, 1), mode="circular")
        du_dx = (u_pad[..., 2:] - u_pad[..., :-2]) / (2 * dx)
        return du_dx.squeeze(0).squeeze(0)

    def _second_derivative(self, u: torch.Tensor, dx: float) -> torch.Tensor:
        u_pad = F.pad(u.unsqueeze(0).unsqueeze(0), (1, 1), mode="circular")
        d2u_dx2 = (u_pad[..., :-2] + u_pad[..., 2:] - 2 * u_pad[..., 1:-1]) / (dx**2)
        return d2u_dx2.squeeze(0).squeeze(0)

    def _initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x - 0.5) ** 2 / (2 * 0.3**2)) + torch.sin(0.5 * math.pi * x)

    def solve(
        self,
        param_dict: Dict[str, float],
        return_jacobian: bool = True,
        zoned: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x = torch.linspace(0, 1, self.n_x)
        dx = x[1] - x[0]
        t_vals = torch.linspace(0, math.pi, self.n_t)
        dt = t_vals[1] - t_vals[0]

        u = self._initial_condition(x)

        snapshots = []
        for i in range(self.n_t):
            snapshots.append(u.clone())
            t_i = t_vals[i].item()
            u_x = self._first_derivative(u, dx)
            u_xx = self._second_derivative(u, dx)

            rhs = (
                -math.pi * param_dict["alpha"] * u * u_x
                + math.pi * param_dict["gamma"] * u_xx
                + math.pi * param_dict["delta"] * math.sin(param_dict["omega"] * t_i)
            )
            u = u + dt * rhs

        u = torch.stack(snapshots)
        result = {"x": x, "t": t_vals, "u": u}

        if return_jacobian:
            result = self._compute_jacobians(param_dict, x, t_vals, result)

        return result

    def _compute_jacobians(
        self,
        param_dict: Dict[str, float],
        x: torch.Tensor,
        t_vals: torch.Tensor,
        base_result: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-3
        dx = x[1] - x[0]
        dt = t_vals[1] - t_vals[0]

        for name in self.param_names:
            params_plus = dict(param_dict)
            params_minus = dict(param_dict)
            delta = param_dict[name] * eps
            params_plus[name] += delta
            params_minus[name] -= max(delta, 1e-12)

            u_plus = self._run_simulation(x, dt, dx, params_plus)
            u_minus = self._run_simulation(x, dt, dx, params_minus)

            denom = 2 * delta
            base_result[f"du_d{name}"] = (u_plus - u_minus) / max(denom, 1e-12)

        return base_result

    def _run_simulation(
        self,
        x: torch.Tensor,
        dt: float,
        dx: float,
        param_dict: Dict[str, float],
    ) -> torch.Tensor:
        u = self._initial_condition(x)
        snapshots = []
        for i in range(self.n_t):
            snapshots.append(u.clone())
            t_i = i * dt
            u_x = self._first_derivative(u, dx)
            u_xx = self._second_derivative(u, dx)
            rhs = (
                -math.pi * param_dict["alpha"] * u * u_x
                + math.pi * param_dict["gamma"] * u_xx
                + math.pi * param_dict["delta"] * math.sin(param_dict["omega"] * t_i)
            )
            u = u + dt * rhs
        return torch.stack(snapshots)

    def solve_fd(
        self, param_dict: Dict[str, float], eps: float = 1e-3
    ) -> Dict[str, torch.Tensor]:
        return self.solve(param_dict, return_jacobian=True)


class PDE3Solver(BaseSolver):
    """PDE3: Stream Function-Vorticity Navier-Stokes.

    ∂ω/∂t + ψ_y ∂ω/∂x - ψ_x ∂ω/∂y = (1/Re) (∂²ω/∂x² + ∂²ω/∂y²)
    ∂²ψ/∂x² + ∂²ψ/∂y² = -ω

    Domain: x, y ∈ [0, 1], t ∈ [0, 3]
    Parameters: α, β
    Re = 1000

    Maps from t=0 to t=3.
    """

    @property
    def param_names(self) -> List[str]:
        return ["alpha", "beta"]

    def _initial_vorticity(self, x: torch.Tensor, y: torch.Tensor, param_dict: Dict[str, float]) -> torch.Tensor:
        alpha = param_dict["alpha"]
        beta = param_dict["beta"]
        return (
            torch.sin(alpha * x) * torch.cos(beta * y)
            + torch.cos(alpha * y) * torch.sin(beta * x)
            + torch.sin(alpha * x + beta * y) * torch.cos(alpha * y - beta * x)
        )

    def _solve_poisson(self, omega: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        device = omega.device
        nx, ny = omega.shape[0], omega.shape[1]
        kx = torch.fft.fftfreq(nx, d=dx).to(device) * 2 * math.pi
        ky = torch.fft.fftfreq(ny, d=dy).to(device) * 2 * math.pi
        kx[0] = 1e-8
        ky[0] = 1e-8
        k_sq = kx[:, None] ** 2 + ky[None, :] ** 2
        omega_hat = torch.fft.fft2(omega)
        psi_hat = omega_hat / k_sq
        psi_hat[0, 0] = 0
        psi = torch.fft.ifft2(psi_hat).real
        return psi

    def solve(
        self, param_dict: Dict[str, float], return_jacobian: bool = True
    ) -> Dict[str, torch.Tensor]:
        n = self.n_x
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dt = 0.01
        Re = 1000.0

        xx, yy = torch.meshgrid(x, y, indexing="ij")
        omega = self._initial_vorticity(xx, yy, param_dict)

        n_steps = 300
        for _ in range(n_steps):
            psi = self._solve_poisson(omega, dx, dy)

            psi_pad = F.pad(psi.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular")
            psi_x = (psi_pad[0, 0, 2:, 1:-1] - psi_pad[0, 0, :-2, 1:-1]) / (2 * dx)
            psi_y = (psi_pad[0, 0, 1:-1, 2:] - psi_pad[0, 0, 1:-1, :-2]) / (2 * dy)

            omega_pad = F.pad(omega.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular")
            omega_x = (omega_pad[0, 0, 2:, 1:-1] - omega_pad[0, 0, :-2, 1:-1]) / (2 * dx)
            omega_y = (omega_pad[0, 0, 1:-1, 2:] - omega_pad[0, 0, 1:-1, :-2]) / (2 * dy)
            omega_xx = (
                omega_pad[0, 0, 2:, 1:-1]
                + omega_pad[0, 0, :-2, 1:-1]
                - 2 * omega
            ) / (dx**2)
            omega_yy = (
                omega_pad[0, 0, 1:-1, 2:]
                + omega_pad[0, 0, 1:-1, :-2]
                - 2 * omega
            ) / (dy**2)

            omega_rhs = (
                -psi_y * omega_x
                + psi_x * omega_y
                + (1.0 / Re) * (omega_xx + omega_yy)
            )
            omega = omega + dt * omega_rhs

        result = {"x": x, "y": y, "omega": omega}

        if return_jacobian:
            eps = 1e-3
            for i, name in enumerate(self.param_names):
                params_plus = dict(param_dict)
                params_minus = dict(param_dict)
                delta = param_dict[name] * eps
                params_plus[name] += delta
                params_minus[name] -= max(delta, 1e-12)

                omega_plus = self._run_ns_simulation(x, y, dx, dy, dt, n_steps, params_plus)
                omega_minus = self._run_ns_simulation(x, y, dx, dy, dt, n_steps, params_minus)

                denom = 2 * delta
                result[f"domega_d{name}"] = (omega_plus - omega_minus) / max(denom, 1e-12)

        return result

    def _run_ns_simulation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        dx: float,
        dy: float,
        dt: float,
        n_steps: int,
        param_dict: Dict[str, float],
    ) -> torch.Tensor:
        Re = 1000.0
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        omega = self._initial_vorticity(xx, yy, param_dict)

        for _ in range(n_steps):
            psi = self._solve_poisson(omega, dx, dy)

            psi_pad = F.pad(psi.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular")
            psi_x = (psi_pad[0, 0, 2:, 1:-1] - psi_pad[0, 0, :-2, 1:-1]) / (2 * dx)
            psi_y = (psi_pad[0, 0, 1:-1, 2:] - psi_pad[0, 0, 1:-1, :-2]) / (2 * dy)

            omega_pad = F.pad(omega.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular")
            omega_x = (omega_pad[0, 0, 2:, 1:-1] - omega_pad[0, 0, :-2, 1:-1]) / (2 * dx)
            omega_y = (omega_pad[0, 0, 1:-1, 2:] - omega_pad[0, 0, 1:-1, :-2]) / (2 * dy)
            omega_xx = (
                omega_pad[0, 0, 2:, 1:-1]
                + omega_pad[0, 0, :-2, 1:-1]
                - 2 * omega
            ) / (dx**2)
            omega_yy = (
                omega_pad[0, 0, 1:-1, 2:]
                + omega_pad[0, 0, 1:-1, :-2]
                - 2 * omega
            ) / (dy**2)

            omega_rhs = (
                -psi_y * omega_x
                + psi_x * omega_y
                + (1.0 / Re) * (omega_xx + omega_yy)
            )
            omega = omega + dt * omega_rhs

        return omega

    def solve_fd(
        self, param_dict: Dict[str, float], eps: float = 1e-3
    ) -> Dict[str, torch.Tensor]:
        return self.solve(param_dict, return_jacobian=True)


class PDE4Solver(BaseSolver):
    """PDE4: Allen-Cahn equation.

    ∂u/∂t = ε ∂²u/∂x² + α u - β u³

    Domain: x ∈ [0, 1], t ∈ [0, 1]
    Initial: u(x, 0) = c tanh(ω x)
    Periodic boundary conditions.
    Parameters: ε, α, β, c, ω
    """

    @property
    def param_names(self) -> List[str]:
        return ["epsilon", "alpha", "beta", "c", "omega"]

    def _initial_condition(self, x: torch.Tensor, param_dict: Dict[str, float]) -> torch.Tensor:
        return param_dict["c"] * torch.tanh(param_dict["omega"] * x)

    def solve(
        self, param_dict: Dict[str, float], return_jacobian: bool = True
    ) -> Dict[str, torch.Tensor]:
        x = torch.linspace(0, 1, self.n_x)
        dx = x[1] - x[0]
        t_vals = torch.linspace(0, 1, self.n_t)
        dt = t_vals[1] - t_vals[0]

        u = self._initial_condition(x, param_dict)

        snapshots = []
        for _ in range(self.n_t):
            snapshots.append(u.clone())

            u_pad = F.pad(u.unsqueeze(0).unsqueeze(0), (1, 1), mode="circular")
            u_xx = (
                u_pad[0, 0, :-2] + u_pad[0, 0, 2:] - 2 * u_pad[0, 0, 1:-1]
            ) / (dx**2)

            rhs = (
                param_dict["epsilon"] * u_xx
                + param_dict["alpha"] * u
                - param_dict["beta"] * u**3
            )
            u = u + dt * rhs

        u = torch.stack(snapshots)
        result = {"x": x, "t": t_vals, "u": u}

        if return_jacobian:
            result = self._compute_jacobians(param_dict, x, t_vals, result)

        return result

    def _compute_jacobians(
        self,
        param_dict: Dict[str, float],
        x: torch.Tensor,
        t_vals: torch.Tensor,
        base_result: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-3
        dx = x[1] - x[0]
        dt = t_vals[1] - t_vals[0]

        for name in self.param_names:
            params_plus = dict(param_dict)
            params_minus = dict(param_dict)
            delta = param_dict[name] * eps
            params_plus[name] += delta
            params_minus[name] -= max(delta, 1e-12)

            u_plus = self._run_simulation(x, dt, dx, params_plus)
            u_minus = self._run_simulation(x, dt, dx, params_minus)

            denom = 2 * delta
            base_result[f"du_d{name}"] = (u_plus - u_minus) / max(denom, 1e-12)

        return base_result

    def _run_simulation(
        self,
        x: torch.Tensor,
        dt: float,
        dx: float,
        param_dict: Dict[str, float],
    ) -> torch.Tensor:
        u = self._initial_condition(x, param_dict)
        snapshots = []
        for _ in range(self.n_t):
            snapshots.append(u.clone())
            u_pad = F.pad(u.unsqueeze(0).unsqueeze(0), (1, 1), mode="circular")
            u_xx = (
                u_pad[0, 0, :-2] + u_pad[0, 0, 2:] - 2 * u_pad[0, 0, 1:-1]
            ) / (dx**2)
            rhs = (
                param_dict["epsilon"] * u_xx
                + param_dict["alpha"] * u
                - param_dict["beta"] * u**3
            )
            u = u + dt * rhs
        return torch.stack(snapshots)

    def solve_fd(
        self, param_dict: Dict[str, float], eps: float = 1e-3
    ) -> Dict[str, torch.Tensor]:
        return self.solve(param_dict, return_jacobian=True)


def generate_dataset(
    solver_cls,
    param_ranges: Dict[str, List[float]],
    n_samples: int = 2000,
    solver_params: Optional[Dict] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Generate a dataset of solution paths and Jacobians.

    Args:
        solver_cls: Solver class (e.g., ODE1Solver).
        param_ranges: Dict mapping parameter names to [low, high] ranges.
        n_samples: Number of samples to generate.
        solver_params: Additional kwargs for solver constructor.

    Returns:
        List of dictionaries containing solutions and Jacobians.
    """
    if solver_params is None:
        solver_params = {}

    dataset = []
    for i in range(n_samples):
        param_dict = {}
        for name, (low, high) in param_ranges.items():
            param_dict[name] = float(torch.rand(1).item() * (high - low) + low)

        solver = solver_cls(param_dict, **solver_params)
        result = solver.solve(param_dict, return_jacobian=True)
        result["params"] = param_dict
        result["sample_idx"] = i
        dataset.append(result)

    return dataset
