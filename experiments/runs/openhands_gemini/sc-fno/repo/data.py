
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Dict, List, Tuple, Union

# Assuming torchdiffeq is available for numerical integration if needed.
# from torchdiffeq import odeint

from .differential_equations.ode1 import ODE1
from .differential_equations.ode2 import ODE2
from .differential_equations.pde1 import PDE1
from .differential_equations.pde2 import PDE2
from .differential_equations.pde3 import PDE3
from .differential_equations.pde4 import PDE4

class EquationDataset(Dataset):
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataGenerator:
    def __init__(self, equation_name: str, config):
        self.equation_name = equation_name
        self.config = config
        self.equation = self._get_equation_instance()
        self.param_ranges = config.current_equation_params
        self.time_steps = config.current_equation_time_steps
        self.spatial_x = config.current_equation_spatial_x
        self.spatial_y = config.current_equation_spatial_y
        self.M = config.current_equation_M # Initial M time steps

        # Prepare time and spatial grids
        self.t_domain = torch.linspace(0, 1, self.time_steps) # Default, override per equation
        if equation_name == "PDE2":
            self.t_domain = torch.linspace(0, torch.pi, self.time_steps)
        elif equation_name == "PDE3":
            self.t_domain = torch.linspace(0, 3, self.time_steps)

        if self.spatial_x is not None:
            self.x_domain = torch.linspace(0, 1, self.spatial_x)
            self.dx = self.x_domain[1] - self.x_domain[0]
        if self.spatial_y is not None:
            self.y_domain = torch.linspace(0, 1, self.spatial_y)


    def _get_equation_instance(self):
        if self.equation_name == "ODE1":
            return ODE1()
        elif self.equation_name == "ODE2":
            return ODE2()
        elif self.equation_name == "PDE1":
            return PDE1()
        elif self.equation_name == "PDE2":
            return PDE2()
        elif self.equation_name == "PDE3":
            return PDE3()
        elif self.equation_name == "PDE4":
            return PDE4()
        else:
            raise ValueError(f"Unknown equation: {self.equation_name}")

    def _sample_parameters(self) -> Dict[str, torch.Tensor]:
        sampled_params = {}
        for param_name, bounds in self.param_ranges.items():
            low, high = bounds
            # Ensure parameters are tensors and require gradients for AD
            sampled_params[param_name] = (low + (high - low) * torch.rand(1)).to(self.config.device)
            sampled_params[param_name].requires_grad_(True)
        return sampled_params

    def _numerical_solver(self, initial_state: torch.Tensor, sampled_params: Dict[str, torch.Tensor]):
        """
        Generic numerical solver placeholder. In a full implementation, this would
        use torchdiffeq.odeint or a custom PDE solver.
        For now, it returns dummy data or calls specific analytical solutions.
        """
        t_span = self.t_domain.to(self.config.device)

        if self.equation_name == "ODE1":
            alpha = sampled_params["alpha"]
            beta = sampled_params["beta"]
            gamma = sampled_params["gamma"]
            u_true = self.equation.solution(t_span, alpha, beta, gamma)
            
            # Calculate analytical sensitivities for ODE1
            sensitivities = self.equation.get_sensitivities(t_span, {k: v.detach() for k,v in sampled_params.items()})
            
            # Stack sensitivities into a single tensor for consistency
            # Shape will be (time_steps, num_params)
            sens_tensors = [sensitivities[p_name] for p_name in self.param_ranges.keys()]
            du_dp_true = torch.stack(sens_tensors, dim=-1) # (time_steps, num_params)
            
            return u_true.unsqueeze(-1), du_dp_true # u_true shape (time_steps, 1)

        elif self.equation_name == "PDE3": # Navier-Stokes, output is vorticity at final time
            # For PDE3, u is (spatial_x, spatial_y), t is [0, 3]
            # initial_state here would be the initial vorticity field
            # This is a complex simulation, will return placeholder
            initial_vorticity_params = {
                "alpha": sampled_params["alpha"],
                "beta": sampled_params["beta"]
            }
            u_true_initial = self.equation.initial_condition(self.x_domain, self.y_domain,
                                                             initial_vorticity_params["alpha"].detach(),
                                                             initial_vorticity_params["beta"].detach())
            # Simplistic placeholder for final vorticity (assuming it's just the initial for now)
            u_true_final = u_true_initial # (spatial_x, spatial_y)

            # Need to reshape for FNO input if 2D spatial
            # For PDE3, paper says "maps ... to the solution of vorticity at the final time step t = 3s"
            # So, output is (spatial_x, spatial_y, 1) at final time.
            # Initial conditions will be (spatial_x, spatial_y, 1) at t=0
            # For the purpose of training, `u_true` should match the FNO output format.
            # Here, we expect a single time point output for PDE3.
            u_true = u_true_final.unsqueeze(0).unsqueeze(-1) # (1, spatial_x, spatial_y, 1)
            
            # Calculate sensitivities with respect to initial condition parameters
            # This is the tricky part without a full differentiable PDE solver.
            # For demonstration, we'll use AD on initial condition directly.
            # In a real scenario, this would involve AD through the numerical solver.
            param_names = list(self.param_ranges.keys())
            du_dp_true_list = []
            
            for p_name in param_names:
                p_val = sampled_params[p_name]
                if p_val.grad is not None:
                    p_val.grad.zero_()
                
                # Recompute initial condition with gradient tracking
                _alpha = sampled_params["alpha"] if p_name != "alpha" else p_val
                _beta = sampled_params["beta"] if p_name != "beta" else p_val

                u_initial_with_grad = self.equation.initial_condition(self.x_domain, self.y_domain, _alpha, _beta)
                
                # Assume final state is related to initial state for sensitivity calculation placeholder
                # This is a strong simplification for the sake of getting sensitivities to compile
                grad_output = torch.autograd.grad(outputs=u_initial_with_grad,
                                                  inputs=p_val,
                                                  grad_outputs=torch.ones_like(u_initial_with_grad),
                                                  create_graph=True,
                                                  allow_unused=True)
                du_dp_true_list.append(grad_output[0].unsqueeze(0).unsqueeze(-1)) # (1, spatial_x, spatial_y, 1)
            
            # du_dp_true: (1, spatial_x, spatial_y, num_params)
            du_dp_true = torch.cat(du_dp_true_list, dim=-1)
            
            return u_true, du_dp_true

        else:
            # Placeholder for other equations
            # For ODEs/1D PDEs: (time_steps, output_dim)
            # For 2D PDEs: (time_steps, spatial_x, spatial_y, output_dim)
            if self.spatial_x is None: # ODEs or 1D spatial
                output_dim = 1 if self.equation_name not in ["ODE2", "PDE1"] else 2 # ODE2 returns [x, x_dot], PDE1 returns [u, v]
                u_true = torch.randn(self.time_steps, output_dim)
                du_dp_true = torch.randn(self.time_steps, output_dim, len(self.param_ranges))
            else: # 2D spatial PDEs like PDE1, PDE2, PDE4
                output_dim = 1 if self.equation_name not in ["PDE1"] else 2
                u_true = torch.randn(self.time_steps, self.spatial_x, output_dim) if self.spatial_y is None \
                            else torch.randn(self.time_steps, self.spatial_x, self.spatial_y, output_dim)
                du_dp_true = torch.randn(u_true.shape + (len(self.param_ranges),))

            print(f"Warning: Using dummy data for {self.equation_name} numerical solver. "
                  "Implement `torchdiffeq.odeint` or similar for full functionality.")
            
            # If using AD for sensitivities, we'd need to run a differentiable solver
            # and then use torch.autograd.grad
            # For now, return dummy sensitivities
            # du_dp_true = self._compute_sensitivities_ad(u_true, sampled_params)
            return u_true, du_dp_true


    def _compute_sensitivities_ad(self, u_true: torch.Tensor, sampled_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute sensitivities using Automatic Differentiation.
        This function assumes that u_true was generated by a differentiable process
        that tracked gradients w.r.t. sampled_params.
        """
        param_names = list(self.param_ranges.keys())
        sensitivities = []
        
        # Ensure outputs are scalars for torch.autograd.grad if necessary,
        # or handle vector-Jacobian product.
        # For simplicity, we assume u_true is differentiable w.r.t. each param.
        
        # We need to iterate over each element of u_true (solution path)
        # and compute its gradient w.r.t. each parameter.
        # This can be computationally intensive.
        # A more efficient way is to compute Jacobian-vector products (JVP)
        # or vector-Jacobian products (VJP).
        
        # Here's a simplified approach that might not be efficient for large outputs:
        # It computes (d(u_i)/d(p_j)) for each output element u_i and each parameter p_j
        
        # Flatten u_true for easier gradient computation
        u_true_flat = u_true.reshape(-1)
        
        du_dp_list_per_param = [] # list of (time_steps * spatial_dims * output_dim) for each param

        for p_name in param_names:
            param = sampled_params[p_name]
            # Ensure param requires grad
            if not param.requires_grad:
                raise ValueError(f"Parameter {p_name} does not require gradients. Set requires_grad_(True).")
            
            # Compute gradient of each element of u_true with respect to the current parameter
            # This is essentially computing one column of the Jacobian for each param.
            # This will result in (u_true.numel(),) for each param.
            
            # To get du/dp as a tensor with similar shape to u, we need to iterate
            # through parameters and compute grad.
            
            grads_for_param = []
            
            # If u_true is the direct output of the differentiable solver
            # We can directly compute grad_outputs=torch.ones_like(u_true)
            # This will sum gradients over the output elements,
            # which is useful for loss computation, but not for direct Jacobians.

            # To get full Jacobian, we need to do vjp/jvp or iterate through outputs
            # Paper mentions "partial_u_partial_p is the Jacobian of the predicted outputs
            # with respect to the input parameters".
            
            # Let's assume u_true is (N_samples, T, X, Y, C)
            # and params is (N_params,)
            # We want Jacobian (N_samples, T, X, Y, C, N_params)
            
            # Simplest for now: if u_true was generated by a single forward pass
            # and params were torch.autograd.Variable
            
            if u_true.grad_fn is None:
                # If u_true was obtained directly from an analytical solution or a non-differentiable solver,
                # its grad_fn will be None. In this case, AD won't work.
                # For ODE1, we have analytical sensitivities.
                if self.equation_name == "ODE1":
                    return self.equation.get_sensitivities(self.t_domain.to(u_true.device), {k:v.detach() for k,v in sampled_params.items()})
                else:
                    print(f"Warning: Cannot compute AD sensitivities for {self.equation_name}. Returning dummy.")
                    # Return dummy sensitivities with the correct shape
                    num_params = len(param_names)
                    if self.spatial_x is None: # ODEs or 1D spatial
                         output_dim = 1 if self.equation_name not in ["ODE2", "PDE1"] else 2
                         return torch.randn(self.time_steps, output_dim, num_params).to(self.config.device)
                    else: # 2D spatial PDEs like PDE1, PDE2, PDE4
                         output_dim = 1 if self.equation_name not in ["PDE1"] else 2
                         if self.spatial_y is None: # 1D spatial PDE
                             return torch.randn(self.time_steps, self.spatial_x, output_dim, num_params).to(self.config.device)
                         else: # 2D spatial PDE
                             # For PDE3, u_true is (1, spatial_x, spatial_y, 1)
                             if self.equation_name == "PDE3":
                                 return torch.randn(1, self.spatial_x, self.spatial_y, output_dim, num_params).to(self.config.device)
                             else:
                                 return torch.randn(self.time_steps, self.spatial_x, self.spatial_y, output_dim, num_params).to(self.config.device)

            grad_output = torch.autograd.grad(outputs=u_true,
                                              inputs=param,
                                              grad_outputs=torch.ones_like(u_true),
                                              create_graph=True,
                                              allow_unused=True) # allow_unused=True important if param not directly used
            if grad_output[0] is not None:
                du_dp_list_per_param.append(grad_output[0].unsqueeze(-1)) # Add a param dimension
            else:
                # If parameter was unused, its gradient is None. Fill with zeros.
                # Need to determine the expected shape of du_dp for this parameter
                dummy_grad = torch.zeros_like(u_true).unsqueeze(-1)
                du_dp_list_per_param.append(dummy_grad)
        
        # Stack all per-parameter gradients: (T, X, Y, C, N_params)
        # Note: This assumes u_true already has the batch dimension removed if needed,
        # or that this function is called for a single sample.
        # The paper implies sensitivities are (du/dp) for a single (u_j, p_j) pair.
        
        # For ODE1, output shape was (time_steps, 1, num_params)
        # For PDE3, output shape is (1, spatial_x, spatial_y, 1, num_params)
        # So we expect the last dimension to be num_params.
        
        if self.equation_name == "ODE1": # Analytical sensitivities are preferred for ODE1
             return self.equation.get_sensitivities(self.t_domain.to(u_true.device), {k:v.detach() for k,v in sampled_params.items()})
        elif du_dp_list_per_param:
            return torch.cat(du_dp_list_per_param, dim=-1)
        else:
            return None # Fallback

    def generate_data(self, num_samples: int) -> List[Dict[str, torch.Tensor]]:
        data = []
        for _ in range(num_samples):
            sampled_params = self._sample_parameters()
            
            # Initial state for ODE/PDE solver.
            # This needs to be parameterized or generated based on the problem.
            initial_state = None 
            if self.equation_name == "ODE1":
                # u(0) = sin(gamma * pi)
                initial_state = torch.sin(sampled_params["gamma"] * torch.pi).to(self.config.device)
            elif self.equation_name == "ODE2":
                # x(0) = epsilon, x_dot(0) = zeta
                initial_state = torch.tensor([sampled_params["epsilon"].item(), sampled_params["zeta"].item()], device=self.config.device)
            elif self.equation_name == "PDE1":
                # u(x,0) = u0, du/dt(x,0) = u'0
                # For simplicity, let's assume u0 is a random spatial distribution and u'0 = 0
                u0_x = torch.randn(self.spatial_x, device=self.config.device)
                du0dt_x = torch.zeros(self.spatial_x, device=self.config.device)
                initial_state = torch.stack([u0_x, du0dt_x], dim=-1).to(self.config.device) # (spatial_x, 2)
            elif self.equation_name == "PDE2":
                initial_state = self.equation.initial_condition(self.x_domain.to(self.config.device)).to(self.config.device)
            elif self.equation_name == "PDE3":
                # Initial vorticity field
                initial_state_params = {p: sampled_params[p] for p in ["alpha", "beta"]}
                initial_state = self.equation.initial_condition(self.x_domain.to(self.config.device), 
                                                                self.y_domain.to(self.config.device),
                                                                initial_state_params["alpha"].detach(), # Detach for initial state
                                                                initial_state_params["beta"].detach()).to(self.config.device)
            elif self.equation_name == "PDE4":
                initial_state_params = {p: sampled_params[p] for p in ["c", "omega"]}
                initial_state = self.equation.initial_condition(self.x_domain.to(self.config.device),
                                                                initial_state_params["c"].detach(),
                                                                initial_state_params["omega"].detach()).to(self.config.device)

            u_true, du_dp_true = self._numerical_solver(initial_state, sampled_params)

            # Need to convert sampled_params to a single tensor for FNO input
            # The order of parameters must be consistent.
            param_tensor = torch.stack([sampled_params[p] for p in self.param_ranges.keys()]).squeeze(1)
            
            # Store the initial conditions separately as well
            # FNO input format is critical here.
            # "Neural Operators are neural networks whose inputs are initial conditions and physical parameters,
            # and whose output is a function u"
            # The schematic (Fig A.7) shows Parameters (P), Spatial/Temporal (X:[x,y,t]), and function a(x) (initial conditions).
            # The paper states for ODE/PDE1,2,4, M initial time steps of u are used. For PDE3, initial conditions at t=0.

            initial_condition_for_fno = None
            if self.equation_name == "ODE1":
                # FNO input needs to include the initial condition of the ODE (u(0))
                # For ODE1, initial_state is u(0) which is a scalar.
                # FNO takes (batch, time_steps, input_channels).
                # The paper specifies: "u : [0 : M] U p -> u : [M : N]"
                # So, initial_condition_for_fno should be u(t) for t=0...M-1
                initial_condition_for_fno = u_true[:self.M].detach() # (M, 1)
            elif self.equation_name == "ODE2":
                # Similar to ODE1, input is first M time steps of [x, x_dot]
                initial_condition_for_fno = u_true[:self.M].detach() # (M, 2) for x and x_dot
            elif self.equation_name == "PDE1":
                # For PDE1, input is first M time steps of [u, du/dt] across spatial_x
                initial_condition_for_fno = u_true[:self.M].detach() # (M, spatial_x, 2)
            elif self.equation_name in ["PDE2", "PDE4"]:
                # For PDE2/4, input is first M time steps of u(x,t) across spatial_x
                initial_condition_for_fno = u_true[:self.M].detach() # (M, spatial_x, output_dim)
            elif self.equation_name == "PDE3":
                # For PDE3, input is the initial vorticity distribution at t=0.
                initial_condition_for_fno = initial_state.unsqueeze(0).detach() # (1, spatial_x, spatial_y)



            data.append({
                "params": param_tensor.float(),
                "initial_conditions": initial_condition_for_fno.float(),
                "u_true": u_true.float(), # (time_steps, ...) or (spatial_x, spatial_y, ...) for PDE3 final state
                "du_dp_true": du_dp_true.float() # (time_steps, ..., num_params)
            })
        return data

    def get_dataloaders(self, num_samples: int):
        all_data = self.generate_data(num_samples)
        
        # Split data into train, validation, and test sets
        train_ratio, val_ratio, test_ratio = self.config.train_test_split
        assert train_ratio + val_ratio + test_ratio == 1.0

        train_size = int(train_ratio * num_samples)
        val_size = int(val_ratio * num_samples)
        test_size = num_samples - train_size - val_size

        train_data = all_data[:train_size]
        val_data = all_data[train_size : train_size + val_size]
        test_data = all_data[train_size + val_size :]

        train_dataset = EquationDataset(train_data)
        val_dataset = EquationDataset(val_data)
        test_dataset = EquationDataset(test_data)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

