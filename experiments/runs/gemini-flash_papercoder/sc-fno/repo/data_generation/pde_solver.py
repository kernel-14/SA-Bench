## data_generation/pde_solver.py

import torch
import torchdiffeq
import numpy as np
from typing import Callable, Dict, Tuple, Optional, List, Any
import functools

# Assuming Config is available from config.py and get_pde_equation_function, get_device
# and spatial derivative helpers are available from utils.py
from config import Config
from utils import get_pde_equation_function, get_device, _compute_dx_1d, _compute_dxx_1d, \
    _compute_dx_2d, _compute_dy_2d, _compute_dxx_2d, _compute_dyy_2d

class PDESolver:
    """
    Provides methods to solve Partial Differential Equations (PDEs) numerically
    and to compute their gradients with respect to parameters.
    Supports both Automatic Differentiation (AD) for differentiable solvers
    and Finite Difference (FD) for generic solvers.
    Also includes functionality to compute PDE residuals for Physics-Informed Neural Networks (PINNs).
    """

    def __init__(self, config: Config):
        """
        Initializes the PDESolver with configuration parameters.

        Args:
            config (Config): The configuration object containing PDE details and solver settings.
        """
        self.config = config
        self.device = get_device()
        self.fd_epsilon = self.config.get("dataset_generation.fd_epsilon", 1e-4)

    def _solve_poisson_fft(self, omega: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        """
        Solves the 2D Poisson equation (nabla^2 psi = -omega) using FFT for periodic boundary conditions.
        Assumes omega is of shape (S_x, S_y).

        Args:
            omega (torch.Tensor): Vorticity field.
            dx (float): Spatial step size in x-direction.
            dy (float): Spatial step size in y-direction.

        Returns:
            torch.Tensor: Stream function psi.
        """
        S_x, S_y = omega.shape
        kx = 2 * np.pi * torch.fft.fftfreq(S_x, d=dx).to(self.device)
        ky = 2 * np.pi * torch.fft.fftfreq(S_y, d=dy).to(self.device)

        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
        
        K_squared = Kx**2 + Ky**2
        # Avoid division by zero for the DC component (k=0,0) and handle it
        K_squared_no_zero = K_squared.clone()
        K_squared_no_zero[0, 0] = 1.0 

        omega_hat = torch.fft.fft2(omega)
        
        psi_hat = -omega_hat / K_squared_no_zero
        psi_hat[0, 0] = 0.0 # Enforce zero mean for psi, as it's arbitrary

        psi = torch.fft.ifft2(psi_hat).real
        return psi

    def _pde_rhs_func(self, t: torch.Tensor, u_flat: torch.Tensor, func_args: Tuple[Callable, torch.Tensor, Optional[torch.Tensor], List[str], Dict[str, Any]]) -> torch.Tensor:
        """
        Helper function for torchdiffeq.odeint, computing the RHS of the PDE.
        This function handles spatial discretization and passes required derivatives to equation_fn.

        Args:
            t (torch.Tensor): Current time.
            u_flat (torch.Tensor): Flattened state vector at current time t.
            func_args (Tuple): A tuple containing:
                - equation_fn (Callable): The function defining the PDE's right-hand side (du/dt).
                - params_tensor (torch.Tensor): Concatenated differentiable PDE parameters.
                - spatial_grid_x (torch.Tensor): 1D tensor of x-coordinates.
                - spatial_grid_y (Optional[torch.Tensor]): 1D tensor of y-coordinates for 2D problems.
                - param_names (List[str]): Names of the parameters corresponding to params_tensor.
                - equation_config (Dict[str, Any]): Configuration dict for the specific PDE.

        Returns:
            torch.Tensor: Flattened time derivative (du/dt) of the state.
        """
        equation_fn, params_tensor, spatial_grid_x, spatial_grid_y, param_names, equation_config = func_args

        spatial_dim = 1 if spatial_grid_y is None else 2
        pde_id = equation_config.get('type')
        
        S_x = equation_config.spatial_discretization_S_x
        S_y = equation_config.spatial_discretization_S_y if spatial_dim == 2 and equation_config.spatial_discretization_S_y is not None else 0

        # Reconstruct params dictionary from params_tensor
        params_dict = {}
        offset = 0
        for name in param_names:
            param_config = equation_config.parameters.get(name)
            if isinstance(param_config, list): # Scalar parameter
                params_dict[name] = params_tensor[offset]
                offset += 1
            elif isinstance(param_config, dict) and 'zone_range' in param_config: # Zoned parameter (e.g., alpha_zone_range)
                num_zones = equation_config.get('num_zones', S_x)
                params_dict[name] = params_tensor[offset : offset + num_zones]
                offset += num_zones
            else: # Fixed param or other complex structure
                raise NotImplementedError(f"Parameter type for {name} not handled in PDESolver._pde_rhs_func")
        
        # Add fixed parameters from config if any (e.g., Re for Navier-Stokes)
        if equation_config.get('fixed_params'):
            params_dict.update({k: torch.tensor(v, device=self.device) for k, v in equation_config.fixed_params.items()})

        # Calculate dx and dy
        dx = (x_span_end - x_span_start) / (S_x - 1) if S_x > 1 else 1.0 # Avoid division by zero if S_x=1
        if spatial_grid_x.numel() > 1:
            dx = spatial_grid_x[1] - spatial_grid_x[0]
        dy = None
        if spatial_grid_y is not None and spatial_grid_y.numel() > 1:
            dy = spatial_grid_y[1] - spatial_grid_y[0]

        # Prepare state for equation_fn
        if pde_id == "PDE1": # State: [u, du/dt]
            u_state = u_flat.view(2, S_x)
            u_val = u_state[0]
            u_t_val = u_state[1]
            pde_args = {
                "u_val": u_val,
                "u_t_val": u_t_val,
                "u_xx": _compute_dxx_1d(u_val, dx),
                "t": t,
                **params_dict
            }
        elif pde_id == "PDE2" or pde_id == "PDE4" or pde_id == "PDE2_Zoned": # State: [u]
            u_val = u_flat.view(S_x)
            pde_args = {
                "u_val": u_val,
                "u_x": _compute_dx_1d(u_val, dx),
                "u_xx": _compute_dxx_1d(u_val, dx),
                "t": t,
                **params_dict
            }
        elif pde_id == "PDE3": # State: [omega]
            omega = u_flat.view(S_x, S_y)
            psi = self._solve_poisson_fft(omega, dx, dy)
            psi_derivs = {
                "psi_x": _compute_dx_2d(psi, dx),
                "psi_y": _compute_dy_2d(psi, dy)
            }
            omega_derivs = {
                "omega_x": _compute_dx_2d(omega, dx),
                "omega_y": _compute_dy_2d(omega, dy),
                "omega_xx": _compute_dxx_2d(omega, dx),
                "omega_yy": _compute_dyy_2d(omega, dy)
            }
            pde_args = {
                "omega": omega,
                "t": t,
                **psi_derivs,
                **omega_derivs,
                **params_dict
            }
        else:
            raise ValueError(f"Unknown PDE ID for RHS function: {pde_id}")

        du_dt = equation_fn(**pde_args)
        return du_dt.view(-1) # Flatten back


    def solve(self, equation_fn: Callable[..., torch.Tensor], u0: torch.Tensor, x_span: torch.Tensor,
              t_span: torch.Tensor, params: Dict[str, torch.Tensor], equation_id: str) -> torch.Tensor:
        """
        Solves a given PDE numerically using a method-of-lines approach with torchdiffeq.

        Args:
            equation_fn (Callable): The Python function representing the PDE's right-hand side (du/dt).
            u0 (torch.Tensor): Initial condition (spatial grid), shape (S_x,) or (S_x, S_y) or (2, S_x) for PDE1.
            x_span (torch.Tensor): 1D tensor of spatial x-coordinates.
            t_span (torch.Tensor): 1D tensor of time points where the solution is desired.
            params (Dict[str, torch.Tensor]): Dictionary of PDE parameters.
            equation_id (str): String identifier for the PDE (e.g., "PDE1", "PDE2").

        Returns:
            torch.Tensor: The computed solution u(t,x) or u(t,x,y), shape (N_t, S_x) or (N_t, S_x, S_y).
        """
        # Ensure tensors are on the correct device
        u0 = u0.to(self.device)
        x_span = x_span.to(self.device)
        t_span = t_span.to(self.device)
        params_on_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device) for k, v in params.items()}

        equation_config = self.config.get(f"equations.{equation_id}")
        if equation_config is None:
            raise ValueError(f"Equation configuration not found for ID: {equation_id}")
        
        # Flatten initial condition
        u0_flat = u0.view(-1)

        # Prepare params_tensor and param_names for _pde_rhs_func
        # This needs to handle both scalar and zoned parameters
        param_names = sorted(params_on_device.keys())
        params_list_flat = []
        for name in param_names:
            param_val = params_on_device[name]
            params_list_flat.append(param_val.view(-1))
        params_tensor = torch.cat(params_list_flat)

        # Determine if 2D spatial grid is needed
        spatial_grid_y = None
        if equation_config.get('spatial_discretization_S_y') is not None:
            S_y = equation_config.spatial_discretization_S_y
            y_span = torch.linspace(x_span[0], x_span[-1], S_y, device=self.device) # Assuming y_span range same as x_span
            spatial_grid_y = y_span

        # Define func_args for _pde_rhs_func
        func_args = (equation_fn, params_tensor, x_span, spatial_grid_y, param_names, equation_config)

        # Solve the ODE system
        u_sol_flat = torchdiffeq.odeint(self._pde_rhs_func, u0_flat, t_span, method='dopri5', args=(func_args,))

        # Reshape solution to (N_t, S_x) or (N_t, 2, S_x) for PDE1 or (N_t, S_x, S_y)
        output_shape = [len(t_span)]
        if equation_id == "PDE1":
            output_shape.extend([2, equation_config.spatial_discretization_S_x])
        elif spatial_grid_y is None:
            output_shape.append(equation_config.spatial_discretization_S_x)
        else:
            output_shape.extend([equation_config.spatial_discretization_S_x, equation_config.spatial_discretization_S_y])
        
        u_sol = u_sol_flat.view(output_shape)
        
        # Special handling for PDE3: only return the last time step solution (vorticity omega)
        if equation_id == "PDE3":
            # PDE3 maps initial condition (omega at t=0) to omega at t=3s.
            # u_sol will be (N_t, S_x, S_y). We want the last time step.
            return u_sol[-1:] # Returns shape (1, S_x, S_y)
        
        return u_sol


    def solve_and_derive_gradients(self, equation_fn: Callable[..., torch.Tensor], u0: torch.Tensor, x_span: torch.Tensor,
                                   t_span: torch.Tensor, params: Dict[str, torch.Tensor],
                                   grad_method: str, equation_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solves the PDE and computes its gradients (Jacobian) with respect to parameters.

        Args:
            equation_fn (Callable): The function defining the PDE's right-hand side.
            u0 (torch.Tensor): Initial condition.
            x_span (torch.Tensor): 1D tensor of spatial x-coordinates.
            t_span (torch.Tensor): 1D tensor of time points.
            params (Dict[str, torch.Tensor]): Dictionary of PDE parameters.
            grad_method (str): Method for gradient computation ("AD" for Automatic Differentiation, "FD" for Finite Difference).
            equation_id (str): Identifier for the PDE.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing
                - u_sol (torch.Tensor): The computed solution.
                - jacobian (torch.Tensor): The Jacobian of u_sol w.r.t. parameters.
        """
        # Ensure all params are on the correct device for processing
        params_on_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device) for k, v in params.items()}
        
        if grad_method == "AD":
            # Prepare params_tensor for AD: Concatenate all parameter values into a single tensor
            # and ensure it requires gradients.
            param_names = sorted(params_on_device.keys())
            params_list_flat = []
            param_original_shapes = {} # Store original shapes to reconstruct
            for name in param_names:
                param_val = params_on_device[name]
                param_original_shapes[name] = param_val.shape
                params_list_flat.append(param_val.view(-1))
            params_tensor = torch.cat(params_list_flat).requires_grad_(True)

            # Create a wrapper function for self.solve that takes a single params_tensor
            def solve_wrapper(p_tensor: torch.Tensor) -> torch.Tensor:
                # Reconstruct params dict from p_tensor and param_names
                current_params_dict = {}
                offset = 0
                for name in param_names:
                    original_shape = param_original_shapes[name]
                    param_len = original_shape.numel()
                    current_params_dict[name] = p_tensor[offset : offset + param_len].view(original_shape)
                    offset += param_len
                
                return self.solve(equation_fn, u0, x_span, t_span, current_params_dict, equation_id)

            # Compute solution and its Jacobian
            u_sol = solve_wrapper(params_tensor)
            
            # The Jacobian function requires a single input tensor, which is p_tensor.
            # The output of solve_wrapper is u_sol.
            # `torch.autograd.functional.jacobian` computes df/dx where f is output, x is input.
            jacobian_raw = torch.autograd.functional.jacobian(solve_wrapper, params_tensor, create_graph=False)
            
            # Reshape Jacobian to (output_shape, input_param_elements)
            # If u_sol is (N_t, S_x) and params_tensor is (N_param_elements,), 
            # jacobian_raw will be (N_t, S_x, N_param_elements). This is often the desired format.
            # If u_sol is (N_t, 2, S_x) (for PDE1), then jacobian_raw will be (N_t, 2, S_x, N_param_elements)
            
            # Reconstruct Jacobian shape to match desired (..., N_param_elements)
            # The `jacobian_raw` usually has shape (output_flat_dim, input_flat_dim).
            # If `vectorize=True`, it can directly provide (output_shape, input_shape) but depends on torchdiffeq behavior.
            
            # The `jacobian_raw` from `torch.autograd.functional.jacobian` when `inputs` is 1D tensor
            # and `func` returns a multi-dimensional tensor will have shape:
            # (func_output_shape[0], ..., func_output_shape[-1], inputs_shape[0])
            # So it's already in the desired (u_sol.shape, num_param_elements) form.
            
            return u_sol.detach(), jacobian_raw.detach()

        elif grad_method == "FD":
            u_sol = self.solve(equation_fn, u0, x_span, t_span, params_on_device, equation_id)
            
            param_names = sorted(params_on_device.keys())
            num_param_elements = 0
            # Calculate total number of elements across all parameters for Jacobian
            for name in param_names:
                num_param_elements += params_on_device[name].numel()

            jacobian_shape = u_sol.shape + (num_param_elements,)
            jacobian = torch.zeros(jacobian_shape, device=self.device, dtype=u_sol.dtype)
            
            param_element_idx = 0
            for p_name in param_names:
                original_value = params_on_device[p_name]
                
                # Iterate through elements if a parameter is a tensor (e.g., zoned PDE2)
                for i in range(original_value.numel()):
                    # Perturb parameter element: p_i + epsilon
                    params_plus = params_on_device.copy()
                    perturbed_val_plus = original_value.clone()
                    perturbed_val_plus.view(-1)[i] += self.fd_epsilon
                    params_plus[p_name] = perturbed_val_plus
                    u_plus = self.solve(equation_fn, u0, x_span, t_span, params_plus, equation_id)

                    # Perturb parameter element: p_i - epsilon
                    params_minus = params_on_device.copy()
                    perturbed_val_minus = original_value.clone()
                    perturbed_val_minus.view(-1)[i] -= self.fd_epsilon
                    params_minus[p_name] = perturbed_val_minus
                    u_minus = self.solve(equation_fn, u0, x_span, t_span, params_minus, equation_id)

                    # Compute finite difference for this specific parameter element
                    dudi = (u_plus - u_minus) / (2 * self.fd_epsilon)
                    jacobian[..., param_element_idx] = dudi
                    param_element_idx += 1

            return u_sol.detach(), jacobian.detach()

        else:
            raise ValueError(f"Unknown gradient method: {grad_method}. Must be 'AD' or 'FD'.")

    def _get_pde_residual_evaluator(self, equation_id: str) -> Callable[..., torch.Tensor]:
        """
        Internal helper to get the specific PDE residual evaluator function based on equation_id.
        This function defines N[u] = 0 for each PDE.
        """
        if equation_id == "PDE1":
            # PDE1: d2u/dt2 - (c^2 d2u/dx2 + alpha du/dt + beta u + gamma sin(omega u)) = 0
            def residual_pde1(u_pred: torch.Tensor, t_coords: torch.Tensor, x_coords: torch.Tensor,
                              dudt: torch.Tensor, d2udt2: torch.Tensor, dudx: torch.Tensor, d2udx2: torch.Tensor,
                              params: Dict[str, torch.Tensor]) -> torch.Tensor:
                c = params['c']
                alpha_param = params['alpha']
                beta_param = params['beta']
                gamma_param = params['gamma']
                omega_param = params['omega']

                return d2udt2 - (c**2 * d2udx2 + alpha_param * dudt + beta_param * u_pred + gamma_param * torch.sin(omega_param * u_pred))
            return residual_pde1
        
        elif equation_id == "PDE2" or equation_id == "PDE2_Zoned":
            # PDE2: (1/pi) du/dt + alpha u du/dx - gamma d2u/dx2 - delta sin(omega t) = 0
            def residual_pde2(u_pred: torch.Tensor, t_coords: torch.Tensor, x_coords: torch.Tensor,
                              dudt: torch.Tensor, dudx: torch.Tensor, d2udx2: torch.Tensor,
                              params: Dict[str, torch.Tensor]) -> torch.Tensor:
                # Handle zoned parameters (params are already prepared by compute_pde_residual)
                alpha_val = params.get('alpha', params.get('alpha_zones')) # Use 'alpha' or 'alpha_zones'
                gamma_val = params.get('gamma', params.get('global_gamma'))
                delta_val = params.get('delta', params.get('delta_zones')) # Use 'delta' or 'delta_zones'
                omega_val = params.get('omega', params.get('global_omega'))

                return (1/np.pi) * dudt + alpha_val * u_pred * dudx - gamma_val * d2udx2 - delta_val * torch.sin(omega_val * t_coords)
            return residual_pde2

        elif equation_id == "PDE4":
            # PDE4: du/dt - (epsilon d2u/dx2 + alpha u - beta u^3) = 0
            def residual_pde4(u_pred: torch.Tensor, t_coords: torch.Tensor, x_coords: torch.Tensor,
                              dudt: torch.Tensor, dudx: torch.Tensor, d2udx2: torch.Tensor,
                              params: Dict[str, torch.Tensor]) -> torch.Tensor:
                epsilon = params['epsilon']
                alpha_param = params['alpha']
                beta_param = params['beta']
                return dudt - (epsilon * d2udx2 + alpha_param * u_pred - beta_param * (u_pred**3))
            return residual_pde4
        
        else:
            raise NotImplementedError(f"PINN residual evaluator not implemented for PDE ID: {equation_id}")

    def compute_pde_residual(self, u_model: torch.Tensor, x_coords: torch.Tensor, t_coords: torch.Tensor,
                             p_params: Dict[str, torch.Tensor], equation_fn: Callable, equation_id: str) -> torch.Tensor:
        """
        Computes the residual of the PDE (N[u]) at specific collocation points for PINN loss.

        Args:
            u_model (torch.Tensor): The FNO model's prediction u(x,t) at sampled collocation points.
                                    Must have requires_grad=True and create_graph=True in its computation.
            x_coords (torch.Tensor): 1D tensor of x-coordinates for sampled collocation points.
                                     Must have requires_grad=True.
            t_coords (torch.Tensor): 1D tensor of t-coordinates for sampled collocation points.
                                     Must have requires_grad=True.
            p_params (Dict[str, torch.Tensor]): Dictionary of PDE parameters for these sampled points.
            equation_fn (Callable): The function representing the PDE's RHS. (Unused for core residual calculation,
                                    but needed for signature compatibility and can provide parameter context).
            equation_id (str): Identifier for the PDE.

        Returns:
            torch.Tensor: A tensor representing the PDE residual at the collocation points.
        """
        # Ensure inputs are on the correct device
        u_model = u_model.to(self.device)
        x_coords = x_coords.to(self.device)
        t_coords = t_coords.to(self.device)
        p_params_on_device = {k: v.to(self.device) for k, v in p_params.items()}

        equation_config = self.config.get(f"equations.{equation_id}")
        if equation_config is None:
            raise ValueError(f"Equation configuration not found for ID: {equation_id}")

        # Compute derivatives using torch.autograd.grad
        # This will be `(N_colloc_pts,)` for each derivative
        
        # du/dt
        dudt = torch.autograd.grad(
            u_model, t_coords,
            grad_outputs=torch.ones_like(u_model, device=self.device),
            create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        if dudt is None: dudt = torch.zeros_like(u_model) # Handle cases where u_model doesn't depend on t_coords

        # du/dx
        dudx = torch.autograd.grad(
            u_model, x_coords,
            grad_outputs=torch.ones_like(u_model, device=self.device),
            create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        if dudx is None: dudx = torch.zeros_like(u_model)

        # d^2u/dx^2
        d2udx2 = torch.autograd.grad(
            dudx, x_coords,
            grad_outputs=torch.ones_like(dudx, device=self.device),
            create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        if d2udx2 is None: d2udx2 = torch.zeros_like(u_model)

        # d^2u/dt^2 (needed for PDE1)
        d2udt2 = None
        if equation_id == "PDE1":
            d2udt2 = torch.autograd.grad(
                dudt, t_coords,
                grad_outputs=torch.ones_like(dudt, device=self.device),
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            if d2udt2 is None: d2udt2 = torch.zeros_like(u_model)

        # Prepare parameters for the residual evaluator.
        # This is where zoned parameters need to be handled carefully:
        # p_params are for collocation points, so if alpha_zones was (S_x), it means alpha_i for x_i.
        # But here p_params means the parameter value *at that collocation point*.
        # So for zoned PDEs, we need to ensure p_params is correctly interpolated/matched to the collocation points.
        # Assuming p_params directly maps to collocation points for zoned case (e.g. p_params['alpha_zones'] is (N_colloc_pts,))
        # For non-zoned, it's (N_colloc_pts,) repeated scalar.
        
        # Get the specific residual evaluator function
        residual_evaluator = self._get_pde_residual_evaluator(equation_id)

        # Evaluate the residual
        pde_residual_args = {
            "u_pred": u_model,
            "t_coords": t_coords,
            "x_coords": x_coords,
            "dudt": dudt,
            "dudx": dudx,
            "d2udx2": d2udx2,
            "params": p_params_on_device # Pass the dictionary directly
        }
        if d2udt2 is not None:
            pde_residual_args["d2udt2"] = d2udt2
        
        # PDE3 (Navier-Stokes) is not currently supported for PINN residual calculation based on the paper,
        # as it typically involves a system of equations and solving for psi.
        # The provided config does not have PINN losses for PDE3.
        # If it were, it would need similar logic to _pde_rhs_func for calculating psi and its derivatives.
        if equation_id == "PDE3":
            raise NotImplementedError("PINN residual for PDE3 (Navier-Stokes) not implemented/supported by paper's description for L_Eq.")


        residual = residual_evaluator(**pde_residual_args)
        
        # The PINN loss function L_Eq uses |N[u]|^2.
        return (residual**2).mean()

