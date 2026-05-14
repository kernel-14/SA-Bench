"""Dataset generation for PDE problems used in experiments:
Burgers' equation, Gray–Scott reaction–diffusion, Navier–Stokes,
Heat equation with convection, Advection equation, and
reaction–diffusion with advection (PDEBench-based).

Each dataset produces (input_functions, output_solution) pairs.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.fft
from scipy.integrate import solve_ivp


# ============================================================================
#  Helper: 1D Spectral solver
# ============================================================================

def spectral_grad_1d(u, dx, order=1):
    """Compute spatial derivative via FFT (1D)."""
    N = u.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    u_hat = np.fft.fft(u, axis=-1)
    if order == 1:
        du = np.fft.ifft(1j * k * u_hat, axis=-1).real
    elif order == 2:
        du = np.fft.ifft(-k ** 2 * u_hat, axis=-1).real
    else:
        raise ValueError
    return du


def spectral_lap_2d(u, dx):
    """Compute Laplacian on 2D grid via FFT."""
    N = u.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(k, k)
    K2 = KX**2 + KY**2
    u_hat = np.fft.fft2(u)
    lap = np.fft.ifft2(-K2 * u_hat).real
    return lap


# ============================================================================
#  Burgers' equation  –  Section 4
# ============================================================================

def generate_burgers_data(config):
    """
    Generate 1D Burgers' equation data: du/dt + u * du/dx = nu * d2u/dx2.

    Inputs: initial condition u0(x), viscosity nu (scalar)
    Output: u(x, T)
    """
    N = config.get('nx', 256)
    L = config.get('L', 1.0)
    T = config.get('T', 1.0)
    n_samples = config.get('n_samples', 1000)
    nu_min = config.get('nu_min', 0.001)
    nu_max = config.get('nu_max', 0.1)
    dt = config.get('dt', 0.001)

    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    n_steps = int(T / dt)

    inputs = []
    outputs = []

    for _ in range(n_samples):
        nu = np.random.uniform(nu_min, nu_max)
        # Random smooth initial condition
        n_modes = np.random.randint(3, 8)
        u0 = np.zeros(N)
        for m in range(1, n_modes + 1):
            amp = np.random.randn() * 0.5 / m
            phase = np.random.rand() * 2 * np.pi
            u0 += amp * np.sin(2 * np.pi * m * x / L + phase)

        u = u0.copy()
        for _ in range(n_steps):
            ux = spectral_grad_1d(u, dx)
            uxx = spectral_grad_1d(u, dx, order=2)
            rhs = -u * ux + nu * uxx
            u = u + dt * rhs

        # Input: concatenate u0 with nu at each spatial point
        nu_arr = np.full((N,), nu)
        inp = np.stack([u0, nu_arr], axis=0)  # (2, N)
        out = u[np.newaxis, :]                 # (1, N)

        inputs.append(inp)
        outputs.append(out)

    inputs = np.array(inputs, dtype=np.float32)  # (n_samples, 2, N)
    outputs = np.array(outputs, dtype=np.float32)  # (n_samples, 1, N)
    return inputs, outputs


# ============================================================================
#  Gray–Scott reaction–diffusion  –  Section 4
# ============================================================================

def generate_grayscott_data(config):
    """
    2D Gray–Scott model:
        du/dt = Du * lap(u) - u * v^2 + F * (1 - u)
        dv/dt = Dv * lap(v) + u * v^2 - (F + k) * v

    Inputs: initial conditions (u0, v0), parameters (Du, Dv, F, k)
    Output: (u(x,T), v(x,T))
    """
    N = config.get('nx', 64)
    L = config.get('L', 2.5)
    T = config.get('T', 5.0)
    n_samples = config.get('n_samples', 500)
    dt = config.get('dt', 0.1)

    Du = config.get('Du', 0.16)
    Dv = config.get('Dv', 0.08)
    F = config.get('F', 0.035)
    k = config.get('k', 0.065)

    dx = L / N
    n_steps = int(T / dt)
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)

    inputs = []
    outputs = []

    for _ in range(n_samples):
        # Vary F and k slightly
        F_val = F + np.random.uniform(-0.005, 0.005)
        k_val = k + np.random.uniform(-0.005, 0.005)

        # Random initial condition: small perturbation in center
        u0 = np.ones((N, N)) * 0.5
        v0 = np.zeros((N, N))
        cx, cy = N // 2, N // 2
        r = 5
        u0[cx - r:cx + r, cy - r:cy + r] = 0.5 + np.random.uniform(-0.1, 0.1, (2 * r, 2 * r))
        v0[cx - r:cx + r, cy - r:cy + r] = 0.25 + np.random.uniform(-0.05, 0.05, (2 * r, 2 * r))

        u, v = u0.copy(), v0.copy()
        for _ in range(n_steps):
            lap_u = spectral_lap_2d(u, dx)
            lap_v = spectral_lap_2d(v, dx)

            uvv = u * v * v
            u = u + dt * (Du * lap_u - uvv + F_val * (1 - u))
            v = v + dt * (Dv * lap_v + uvv - (F_val + k_val) * v)

        # Input: u0, v0, Du, Dv, F, k
        # Stack parameters as constant fields
        Du_arr = np.full((N, N), Du, dtype=np.float32)
        Dv_arr = np.full((N, N), Dv, dtype=np.float32)
        F_arr = np.full((N, N), F_val, dtype=np.float32)
        k_arr = np.full((N, N), k_val, dtype=np.float32)

        inp = np.stack([u0, v0, Du_arr, Dv_arr, F_arr, k_arr], axis=0)
        out = np.stack([u, v], axis=0)

        inputs.append(inp)
        outputs.append(out)

    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)
    return inputs, outputs


# ============================================================================
#  Navier–Stokes (2D, incompressible)  –  Section 4
# ============================================================================

def generate_navierstokes_data(config):
    """
    2D Navier–Stokes with vorticity-stream function formulation.
    d(w)/dt + u·grad(w) = nu * lap(w) + f

    Inputs: initial vorticity w0, viscosity nu
    Output: vorticity w(x,y,T)
    """
    N = config.get('nx', 64)
    L = config.get('L', 2 * np.pi)
    T = config.get('T', 1.0)
    n_samples = config.get('n_samples', 500)
    nu_min = config.get('nu_min', 1e-4)
    nu_max = config.get('nu_max', 1e-3)
    dt = config.get('dt', 0.01)

    dx = L / N
    n_steps = int(T / dt)
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k)
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    lap_op = -4 * np.pi ** 2 * K2 / L ** 2

    inputs = []
    outputs = []

    for _ in range(n_samples):
        nu = np.random.uniform(nu_min, nu_max)

        # Random initial vorticity
        w0 = np.random.randn(N, N)
        w0 = scipy.fft.fft2(w0)
        w0[K2 > (N // 3) ** 2] = 0  # keep low modes
        w0 = scipy.fft.ifft2(w0).real
        w0 = w0 * 0.1 / np.std(w0)

        w = w0.copy()
        forcing = np.random.randn(N, N) * 0.001
        forcing = scipy.fft.fft2(forcing)
        forcing[K2 > 4] = 0
        forcing = scipy.fft.ifft2(forcing).real

        for _ in range(n_steps):
            w_hat = scipy.fft.fft2(w)
            psi_hat = -w_hat / (K2 + 1e-10)
            psi_hat[0, 0] = 0
            psi = scipy.fft.ifft2(psi_hat).real

            u = np.real(scipy.fft.ifft2(1j * KY * psi_hat))
            v = -np.real(scipy.fft.ifft2(1j * KX * psi_hat))

            wx = np.real(scipy.fft.ifft2(1j * KX * w_hat))
            wy = np.real(scipy.fft.ifft2(1j * KY * w_hat))

            rhs = - (u * wx + v * wy) + nu * np.real(scipy.fft.ifft2(lap_op * w_hat)) + forcing
            w = w + dt * rhs

        nu_arr = np.full((N, N), nu, dtype=np.float32)
        inp = np.stack([w0, nu_arr], axis=0)
        out = w[np.newaxis, :, :]

        inputs.append(inp)
        outputs.append(out)

    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)
    return inputs, outputs


# ============================================================================
#  Heat equation (base) & Heat + Convection (extension)  –  Section 4
# ============================================================================

def generate_heat_data(config):
    """
    Heat equation: du/dt = alpha * lap(u).
    For extension: add convection term beta * grad(u).
    """
    N = config.get('nx', 64)
    L = config.get('L', 1.0)
    T = config.get('T', 0.5)
    n_samples = config.get('n_samples', 500)
    with_convection = config.get('with_convection', False)
    dt = config.get('dt', 0.001)

    dx = L / N
    n_steps = int(T / dt)

    inputs = []
    outputs = []

    for _ in range(n_samples):
        alpha = np.random.uniform(0.01, 0.1)
        beta = np.random.uniform(0.1, 1.0) if with_convection else 0.0

        u0 = np.random.randn(N, N) * 0.1
        u0 = scipy.fft.fft2(u0)
        k = np.fft.fftfreq(N) * N
        KX, KY = np.meshgrid(k, k)
        K2 = KX ** 2 + KY ** 2
        u0[K2 > (N // 4) ** 2] = 0
        u0 = scipy.fft.ifft2(u0).real

        u = u0.copy()
        alpha_arr = np.full((N, N), alpha, dtype=np.float32)

        for _ in range(n_steps):
            u_hat = scipy.fft.fft2(u)
            lap = -4 * np.pi ** 2 * K2 / L ** 2
            dlap = alpha * np.real(scipy.fft.ifft2(lap * u_hat))
            dconv = 0.0
            if with_convection:
                ux = np.real(scipy.fft.ifft2(1j * KX * u_hat))
                uy = np.real(scipy.fft.ifft2(1j * KY * u_hat))
                dconv = -beta * (ux + uy)
            u = u + dt * (dlap + dconv)

        beta_arr = np.full((N, N), beta, dtype=np.float32)
        in_channels = [u0, alpha_arr]
        if with_convection:
            in_channels.append(beta_arr)
        inp = np.stack(in_channels, axis=0)
        out = u[np.newaxis, :, :]

        inputs.append(inp)
        outputs.append(out)

    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)
    return inputs, outputs


# ============================================================================
#  Reaction–Diffusion with Advection  –  Section 4
# ============================================================================

def generate_rd_advection_data(config):
    """
    Reaction–diffusion equation with advection term:
        du/dt = Du * lap(u) + R(u) - beta * grad(u)

    Base: FitzHugh–Nagumo type reaction–diffusion
    Extension: add advection velocity beta.
    """
    N = config.get('nx', 64)
    L = config.get('L', 1.0)
    T = config.get('T', 2.0)
    n_samples = config.get('n_samples', 500)
    with_advection = config.get('with_advection', False)
    dt = config.get('dt', 0.005)

    dx = L / N
    n_steps = int(T / dt)
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k)

    inputs = []
    outputs = []

    for _ in range(n_samples):
        Du = np.random.uniform(0.01, 0.05)
        Dv = np.random.uniform(0.005, 0.02)
        beta = np.random.uniform(0.1, 0.5) if with_advection else 0.0
        a = np.random.uniform(-0.1, 0.1)
        b = np.random.uniform(0.01, 0.05)

        u0 = np.random.rand(N, N) * 0.1
        v0 = np.random.rand(N, N) * 0.1 + 0.5

        u, v = u0.copy(), v0.copy()

        for _ in range(n_steps):
            u_hat = scipy.fft.fft2(u)
            v_hat = scipy.fft.fft2(v)
            lu = np.real(scipy.fft.ifft2(-4 * np.pi ** 2 * (KX ** 2 + KY ** 2) / L ** 2 * u_hat))
            lv = np.real(scipy.fft.ifft2(-4 * np.pi ** 2 * (KX ** 2 + KY ** 2) / L ** 2 * v_hat))

            Rn_u = u - u ** 3 - v + a
            Rn_v = b * (u - v)

            du = Du * lu + Rn_u
            dv = Dv * lv + Rn_v

            if with_advection:
                ux = np.real(scipy.fft.ifft2(1j * KX * u_hat))
                uy = np.real(scipy.fft.ifft2(1j * KY * u_hat))
                vx = np.real(scipy.fft.ifft2(1j * KX * v_hat))
                vy = np.real(scipy.fft.ifft2(1j * KY * v_hat))
                du -= beta * (ux + uy)
                dv -= beta * (vx + vy)

            u = u + dt * du
            v = v + dt * dv

        Du_arr = np.full((N, N), Du, dtype=np.float32)
        Dv_arr = np.full((N, N), Dv, dtype=np.float32)
        beta_arr = np.full((N, N), beta, dtype=np.float32)
        a_arr = np.full((N, N), a, dtype=np.float32)
        b_arr = np.full((N, N), b, dtype=np.float32)

        in_ch = [u0, v0, Du_arr, Dv_arr]
        if with_advection:
            in_ch.append(beta_arr)
        inp = np.stack(in_ch, axis=0)
        out = np.stack([u, v], axis=0)

        inputs.append(inp)
        outputs.append(out)

    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)
    return inputs, outputs


# ============================================================================
#  PDEBench-style Advection equation  –  Section 4
# ============================================================================

def generate_advection_data(config):
    """
    Advection equation: du/dt + beta * grad(u) = 0
    """
    N = config.get('nx', 256)
    L = config.get('L', 1.0)
    T = config.get('T', 1.0)
    n_samples = config.get('n_samples', 1000)
    dt = config.get('dt', 0.001)
    dx = L / N
    n_steps = int(T / dt)

    inputs = []
    outputs = []

    for _ in range(n_samples):
        beta = np.random.uniform(0.5, 2.0)
        x = np.linspace(0, L, N, endpoint=False)
        u0 = np.exp(-((x - L / 2) ** 2) / (2 * 0.05 ** 2)) * np.random.uniform(0.8, 1.2)

        u = u0.copy()
        for _ in range(n_steps):
            ux = spectral_grad_1d(u, dx)
            u = u - dt * beta * ux

        beta_arr = np.full((N,), beta, dtype=np.float32)
        inp = np.stack([u0, beta_arr], axis=0)
        out = u[np.newaxis, :]

        inputs.append(inp)
        outputs.append(out)

    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)
    return inputs, outputs


# ============================================================================
#  PDE Benchmark dataset wrapper
# ============================================================================

class PDEDataset(Dataset):
    """Generic PDE dataset wrapper."""

    def __init__(self, inputs, outputs, device='cpu'):
        self.inputs = torch.from_numpy(inputs)
        self.outputs = torch.from_numpy(outputs)
        self.device = device

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    def to(self, device):
        self.inputs = self.inputs.to(device)
        self.outputs = self.outputs.to(device)
        self.device = device
        return self


class MultiPhysicsDataset(Dataset):
    """
    Dataset for multi-physics training with problem labels.
    Each sample is (input, output, problem_name).
    """

    def __init__(self, datasets_dict):
        """
        Args:
            datasets_dict: {problem_name: (inputs_np, outputs_np)}
        """
        self.samples = []
        for name, (inp, out) in datasets_dict.items():
            for i in range(len(inp)):
                self.samples.append((torch.from_numpy(inp[i]),
                                     torch.from_numpy(out[i]), name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
#  Data generation entry point
# ============================================================================

DATASET_GENERATORS = {
    'burgers': generate_burgers_data,
    'grayscott': generate_grayscott_data,
    'navierstokes': generate_navierstokes_data,
    'heat': generate_heat_data,
    'heat_convection': generate_heat_data,
    'rd': generate_rd_advection_data,
    'rd_advection': generate_rd_advection_data,
    'advection': generate_advection_data,
}


def load_pde_data(config, split='train'):
    """
    Load or generate PDE dataset.

    Args:
        config: dict with keys:
            - problem: str, one of the DATASET_GENERATORS keys
            - data_dir: optional path for cached data
            - train_samples, test_samples: int
        split: 'train' or 'test'

    Returns: PDEDataset
    """
    problem = config['problem']
    n_train = config.get('train_samples', 1000)
    n_test = config.get('test_samples', 200)

    gen_config = dict(config)
    gen_config['n_samples'] = n_train if split == 'train' else n_test

    if 'with_convection' in config:
        gen_config['with_convection'] = config['with_convection']
    if 'with_advection' in config:
        gen_config['with_advection'] = config['with_advection']

    # Map problem name to generator
    if problem in DATASET_GENERATORS:
        generator = DATASET_GENERATORS[problem]
    elif problem.startswith('heat'):
        generator = generate_heat_data
    elif problem.startswith('rd') or problem.startswith('grayscott'):
        generator = generate_rd_advection_data
    elif problem.startswith('navierstokes') or problem.startswith('ns_'):
        generator = generate_navierstokes_data
    elif problem.startswith('burgers'):
        generator = generate_burgers_data
    else:
        generator = DATASET_GENERATORS.get(problem, generate_advection_data)

    inputs, outputs = generator(gen_config)
    return PDEDataset(inputs, outputs)


def load_multiphysics_data(configs_dict, split='train'):
    """
    Load multiple PDE datasets for multi-physics pretraining.

    Args:
        configs_dict: {problem_name: config_dict}
        split: 'train' or 'test'

    Returns: MultiPhysicsDataset
    """
    datasets = {}
    for name, cfg in configs_dict.items():
        ds = load_pde_data(cfg, split=split)
        datasets[name] = (np.array([t[0].numpy() for t in ds]),
                          np.array([t[1].numpy() for t in ds]))
    return MultiPhysicsDataset(datasets)


def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
