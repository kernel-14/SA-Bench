## data_generation/ode_solver.py
import torch
import torchdiffeq
import numpy as np
from typing import Callable, Dict, Tuple, Any, Optional
import copy

# Ensure utils.py functions are available without circular import issues
# In a real project, these would be separate modules, and imports would be direct.
# For the purpose of this single-file generation, we assume utils functions are directly accessible.
# Or, if this file were part of the larger structure, `from utils import get_ode_equation_function` etc.
# For now, importing from a "virtual" utils to satisfy type hints.
# The `get_ode_equation_function` is defined in `utils.py` according to the plan.
# I'll re-implement the `get_ode_equation_function` here to make this file self-contained for testing,
# but in the actual project, it would be imported from `utils.py`.

# --- Start re-implementation of get_ode_equation_function from utils.py for self-containment ---
# This block should ideally be replaced by `from utils import get_ode_equation_function`
# when integrated into the full project structure.
import math

def get_ode_equation_function(ode_id: str) -> Callable[..., Any]:
    """
    Returns the Python function representing the RHS of the ODE, parameterized for `torchdiffeq`.
    The returned function computes d(state)/dt given the current time, state, and parameters.
    It expects `params` to be a dict of torch.Tensor.

    Args:
        ode_id: Identifier for the ODE (e.g., "ODE1", "ODE2").

    Returns:
        A callable function `f(t: torch.Tensor, u_state: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor`
        that computes the RHS of the ODE.
    """
    if ode_id == "ODE1":
        # ODE1: Composite Harmonic Oscillator
        # du/dt = alpha sin(alpha pi t) + beta cos(beta pi t)
        # Note: gamma only affects initial condition u(0) = sin(gamma pi), not the dynamics.
        def rhs_ode1(t: torch.Tensor, u_state: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
            alpha = params['alpha']
            beta = params['beta']
            
            # u_state is the current value of u. For this ODE, du/dt does not depend on u.
            # It's good practice to keep u_state in the signature as odeint expects it.
            # Ensure t is a scalar for the calculation even if it comes in as a tensor for odeint.
            t_scalar = t.item() if t.numel() == 1 else t
            
            du_dt = alpha * torch.sin(alpha * math.pi * t_scalar) + beta * torch.cos(beta * math.pi * t_scalar)
            # Unsqueeze to match expected output shape for odeint (e.g., (1,) if state is scalar)
            return du_dt.unsqueeze(-1) if u_state.ndim == 1 and du_dt.ndim == 0 else du_dt


        return rhs_ode1

    elif ode_id == "ODE2":
        # ODE2: Duffing Oscillator Equation
        # d^2x/dt^2 + delta dx/dt + alpha x + beta x^3 = gamma cos(omega t)
        # Convert to 1st order system:
        # u1 = x, u2 = dx/dt
        # du1/dt = u2
        # du2/dt = -delta u2 - alpha u1 - beta u1^3 + gamma cos(omega t)
        def rhs_ode2(t: torch.Tensor, u_state: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
            # u_state is a tensor of shape (batch_size, 2) or (2,) for [x, dx/dt]
            # When odeint passes u_state, it's (state_dim,) or (batch_size, state_dim) if u0 was batched.
            # We assume u_state is (state_dim,) here, so u_state[0] is x, u_state[1] is dx/dt.
            x_val = u_state[0]
            dx_dt_val = u_state[1]

            delta_param = params['delta']
            alpha_param = params['alpha']
            beta_param = params['beta']
            gamma_param = params['gamma']
            omega_param = params['omega']

            # Ensure t is a scalar for the calculation even if it comes in as a tensor for odeint.
            t_scalar = t.item() if t.numel() == 1 else t

            du1_dt = dx_dt_val
            du2_dt = -delta_param * dx_dt_val - alpha_param * x_val - beta_param * (x_val**3) + gamma_param * torch.cos(omega_param * t_scalar)
            
            return torch.stack([du1_dt, du2_dt], dim=-1)

        return rhs_ode2

    else:
        raise ValueError(f"Unknown ODE ID: {ode_id}")
# --- End re-implementation of get_ode_equation_function ---


class ODESolver:
    """
    Provides a differentiable ODE solver and finite difference gradient computation.
    """

    def __init__(self) -> None:
        """
        Initializes the ODESolver.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ode_func(self, t: torch.Tensor, u_state: torch.Tensor, params_and_eq_fn: Tuple[Dict[str, torch.Tensor], Callable]) -> torch.Tensor:
        """
        Helper function wrapping the ODE RHS in a format compatible with torchdiffeq.odeint.

        Args:
            t: Current time, a scalar tensor.
            u_state: Current state vector of the ODE, a tensor of shape (state_dim,).
            params_and_eq_fn: A tuple containing (params_dict, equation_fn).
                              params_dict: A dictionary of ODE parameters.
                              equation_fn: The user-provided ODE right-hand side function.

        Returns:
            The derivative of the state vector (du/dt), a tensor of shape (state_dim,).
        """
        params, equation_fn = params_and_eq_fn
        # Ensure params are on the same device as u_state and t
        params_on_device = {k: v.to(u_state.device) for k, v in params.items()}
        return equation_fn(t, u_state, params_on_device)

    def solve(
        self,
        equation_fn: Callable,
        u0: torch.Tensor,
        t_span: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Solves an ODE over a specified time span using torchdiffeq's built-in solver.

        Args:
            equation_fn: The ODE's right-hand side function.
                         Expected signature: f(t, u_state, params) -> du_dt.
            u0: The initial state of the ODE, a tensor of shape (state_dim,).
            t_span: A 1D tensor of shape (num_time_steps,) containing the time points
                    at which to evaluate the solution.
            params: A dictionary of ODE parameters, where each value is a scalar torch.Tensor.
                    If parameters need gradients, their requires_grad attribute should be set to True.

        Returns:
            A torch.Tensor of shape (num_time_steps, state_dim), representing the
            solution u(t) at each time point in t_span.
        """
        # Ensure u0 and t_span are on the correct device
        u0 = u0.to(self.device)
        t_span = t_span.to(self.device)
        params_on_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device) for k, v in params.items()}


        # Use odeint to solve the ODE
        # The _ode_func wrapper is needed to pass additional args (params and equation_fn)
        solution = torchdiffeq.odeint(
            func=self._ode_func,
            y0=u0,
            t=t_span,
            method='rk4', # Default method, can be configured if needed
            options={'step_size': t_span[1] - t_span[0]} if len(t_span) > 1 else {},
            args=(params_on_device, equation_fn),
        )
        # solution shape: (num_time_steps, state_dim)
        return solution

    def solve_and_derive_gradients(
        self,
        equation_fn: Callable,
        u0: torch.Tensor,
        t_span: torch.Tensor,
        params: Dict[str, torch.Tensor],
        grad_method: str = 'AD', # Default to 'AD'
        fd_epsilon: float = 1e-4, # Default from config.yaml, can be overridden
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solves the ODE and computes the gradients of the solution with respect to its parameters.

        Args:
            equation_fn: The ODE's right-hand side function.
            u0: The initial state of the ODE, a tensor of shape (state_dim,).
            t_span: A 1D tensor of shape (num_time_steps,) containing the time points.
            params: A dictionary of ODE parameters. For AD, these tensors *must* have
                    requires_grad=True by cloning and setting. For FD, they can be regular tensors.
            grad_method: The method for gradient computation ('AD' or 'FD').
            fd_epsilon: The perturbation size for finite difference calculations.

        Returns:
            A tuple (u_solution, du_dp_true).
            u_solution: The ODE solution, shape (num_time_steps, state_dim).
            du_dp_true: The true Jacobian, shape (num_time_steps, state_dim, num_parameters).

        Raises:
            ValueError: If an unsupported grad_method is provided.
        """
        u0 = u0.to(self.device)
        t_span = t_span.to(self.device)

        # Convert all parameters to tensors and move to device if they aren't already
        params_tensors = {k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device) for k, v in params.items()}
        p_names = list(params_tensors.keys())
        num_parameters = len(p_names)

        if grad_method == 'AD':
            # 1. Parameter Preparation for AD
            # Clone parameters and ensure requires_grad=True for Jacobian computation
            p_tensors_ad = [p_val.clone().detach().requires_grad_(True) for p_val in params_tensors.values()]
            params_ad_dict = {p_name: p_tensors_ad[i] for i, p_name in enumerate(p_names)}

            # Define a wrapper function for Jacobian computation
            # This function takes a single tensor of parameters, reconstructs the dict,
            # and calls the ODE solver.
            def _compute_u_for_jacobian(p_vector_input: torch.Tensor) -> torch.Tensor:
                current_params_dict = {}
                offset = 0
                for i, p_name in enumerate(p_names):
                    # Assuming parameters are scalar (single value in p_vector_input)
                    current_params_dict[p_name] = p_vector_input[offset]
                    offset += 1
                return self.solve(equation_fn, u0, t_span, current_params_dict)
            
            # Form the input parameter vector for jacobian
            p_vector_input = torch.cat(p_tensors_ad)
            
            # Compute the solution u_solution using the AD-enabled parameters
            u_solution = _compute_u_for_jacobian(p_vector_input)

            # Compute Jacobian using torch.autograd.functional.jacobian
            # The output shape will be (num_time_steps, state_dim, num_parameters) directly
            # if vectorize=True and the output of _compute_u_for_jacobian is as expected.
            # create_graph=False as we don't need to differentiate through the Jacobian itself.
            du_dp_true = torch.autograd.functional.jacobian(
                func=_compute_u_for_jacobian,
                inputs=p_vector_input,
                create_graph=False,
                vectorize=True # Can sometimes accelerate by batching computations
            )

            # du_dp_true's shape is (num_time_steps, state_dim, num_parameters)
            # if p_vector_input has len num_parameters.
            # If state_dim is 1, solution is (num_time_steps, 1). Jacobain output is (num_time_steps, 1, num_parameters)
            # If state_dim > 1, solution is (num_time_steps, state_dim). Jacobian output is (num_time_steps, state_dim, num_parameters)
            
            return u_solution, du_dp_true

        elif grad_method == 'FD':
            # 1. Base Solution
            u_solution = self.solve(equation_fn, u0, t_span, params_tensors)

            # 2. Initialize Jacobian Storage
            jacobian_columns = []

            # 3. Iterate and Perturb for each parameter
            for p_name in p_names:
                p_val_original = params_tensors[p_name]

                # Positive Perturbation
                params_plus_dict = copy.deepcopy(params_tensors)
                params_plus_dict[p_name] = p_val_original + fd_epsilon
                u_plus = self.solve(equation_fn, u0, t_span, params_plus_dict)

                # Negative Perturbation
                params_minus_dict = copy.deepcopy(params_tensors)
                params_minus_dict[p_name] = p_val_original - fd_epsilon
                u_minus = self.solve(equation_fn, u0, t_span, params_minus_dict)

                # Compute Central Finite Difference for this parameter
                du_dp_i = (u_plus - u_minus) / (2 * fd_epsilon)
                jacobian_columns.append(du_dp_i)

            # 4. Assemble Jacobian: stack (num_time_steps, state_dim) tensors
            # into (num_time_steps, state_dim, num_parameters)
            if jacobian_columns:
                du_dp_true = torch.stack(jacobian_columns, dim=-1)
            else:
                # If no parameters, Jacobian is empty or zero-sized
                # Assume (num_time_steps, state_dim, 0)
                du_dp_true = torch.empty(u_solution.shape[0], u_solution.shape[1], 0, device=self.device)
            
            return u_solution, du_dp_true

        else:
            raise ValueError(f"Unsupported gradient method: {grad_method}. Choose 'AD' or 'FD'.")

