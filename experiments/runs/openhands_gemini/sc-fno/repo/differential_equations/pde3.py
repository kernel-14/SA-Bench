
import torch
import torch.fft
import torch.nn.functional as F

class PDE3:
    """
    PDE3: Stream Function-Vorticity Formulation of the Navier-Stokes Equations
    Equations:
        d(omega)/dt + psi_y * d(omega)/dx - psi_x * d(omega)/dy = (1/Re) * (d^2(omega)/dx^2 + d^2(omega)/dy^2)
        d^2(psi)/dx^2 + d^2(psi)/dy^2 = -omega
    Initial condition: omega(x, y, 0) = f(x, y; alpha, beta)
    Parameters: alpha, beta (for initial condition)
    Domain: spatial x, y in [0, 1], temporal t in [0, 3]
    Reynolds Number Re = 1000
    """
    def __init__(self, Re: float = 1000.0):
        self.Re = Re

    def initial_condition(self, x: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Initial vorticity distribution.
        f(x, y; alpha, beta) = sin(alpha*x)*cos(beta*y) + cos(alpha*y)*sin(beta*x) + sin(alpha*x + beta*y)*cos(alpha*y - beta*x)
        x, y are 1D arrays, need to meshgrid them for 2D.
        """
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        term1 = torch.sin(alpha * X) * torch.cos(beta * Y)
        term2 = torch.cos(alpha * Y) * torch.sin(beta * X)
        term3 = torch.sin(alpha * X + beta * Y) * torch.cos(alpha * Y - beta * X)
        
        return term1 + term2 + term3

    def _get_k_space(self, N: int, L: float = 1.0) -> torch.Tensor:
        """Helper to get wavenumber array for FFT."""
        k = torch.cat((torch.arange(0, N/2), torch.arange(-N/2, 0))).to(dtype=torch.float32) * (2 * torch.pi / L)
        return k

    def rhs(self, t: torch.Tensor, omega: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
            spatial_x_res: int, spatial_y_res: int, L: float = 1.0) -> torch.Tensor:
        """
        Right-hand side of the PDE for numerical solvers.
        omega shape: (batch_size, spatial_x_res, spatial_y_res)
        alpha, beta are parameters for the initial condition, typically fixed during time evolution.
        Here we assume omega is the state variable.
        This implementation uses spectral methods for spatial derivatives.
        """
        # Ensure omega is float for complex conversion
        omega = omega.to(dtype=torch.float32)
        
        batch_size = omega.shape[0]
        
        # 1. Fourier transform of omega
        omega_hat = torch.fft.fft2(omega)

        # Get wavenumbers
        kx = self._get_k_space(spatial_x_res, L).to(omega.device)
        ky = self._get_k_space(spatial_y_res, L).to(omega.device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')

        # 2. Solve for psi in Fourier space: d^2(psi)/dx^2 + d^2(psi)/dy^2 = -omega
        # This means -(kx^2 + ky^2) * psi_hat = -omega_hat
        # So, psi_hat = omega_hat / (kx^2 + ky^2)
        
        # Avoid division by zero for the (0,0) mode
        denominator = -(KX**2 + KY**2)
        denominator[0, 0] = 1.0 # Set to 1 to avoid NaN, will be zeroed out later
        
        psi_hat = -omega_hat / denominator
        psi_hat[:, 0, 0] = 0.0 # Zero out the (0,0) mode, as psi has zero mean (no net flow)

        # 3. Compute derivatives of psi in Fourier space
        psi_x_hat = 1j * KX * psi_hat
        psi_y_hat = 1j * KY * psi_hat

        # 4. Inverse Fourier transform to get psi, psi_x, psi_y in real space
        psi_x = torch.real(torch.fft.ifft2(psi_x_hat))
        psi_y = torch.real(torch.fft.ifft2(psi_y_hat))

        # 5. Compute second derivatives of omega in Fourier space
        laplacian_omega_hat = -(KX**2 + KY**2) * omega_hat
        laplacian_omega = torch.real(torch.fft.ifft2(laplacian_omega_hat))

        # 6. Combine terms to get d(omega)/dt
        # Advection term: psi_y * d(omega)/dx - psi_x * d(omega)/dy
        # Compute d(omega)/dx and d(omega)/dy in real space from omega_hat
        omega_x_hat = 1j * KX * omega_hat
        omega_y_hat = 1j * KY * omega_hat
        
        omega_x = torch.real(torch.fft.ifft2(omega_x_hat))
        omega_y = torch.real(torch.fft.ifft2(omega_y_hat))

        advection_term = psi_y * omega_x - psi_x * omega_y
        diffusion_term = (1 / self.Re) * laplacian_omega

        domedt = -advection_term + diffusion_term
        
        return domedt

    def solution(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, initial_condition_params: dict):
        """
        No analytical solution provided in the paper. This would typically be solved numerically.
        Placeholder for consistency.
        """
        raise NotImplementedError("Analytical solution for PDE3 not provided in the paper.")

    def get_sensitivities(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, initial_condition_params: dict):
        """
        No analytical sensitivities provided in the paper. This would typically be computed via AD.
        Placeholder for consistency.
        """
        raise NotImplementedError("Analytical sensitivities for PDE3 not provided in the paper.")

