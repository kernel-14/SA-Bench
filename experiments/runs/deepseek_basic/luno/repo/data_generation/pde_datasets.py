"""PDE dataset generation following Section D.1.

Generates data for:
- Burgers' equation (1D)
- Hyper Diffusion equation (1D)
- Kuramoto-Sivashinsky (conservative) equation (1D)
- Advection-Diffusion-Reaction equation (2D) with OOD variants

Dataset specifications from Table 3:
- Spatial resolution: 256 (1D), 100×100 (2D)
- Temporal resolution: 59 time steps
- Training trajectories: 25 (low data) or 1000 (OOD)
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class PDEDataset:
    """Container for PDE trajectory data."""
    name: str
    train_inputs: jnp.ndarray   # (n_train, n_t, n_x, d_fields)
    train_targets: jnp.ndarray  # (n_train, n_t, n_x, d_fields)
    val_inputs: jnp.ndarray
    val_targets: jnp.ndarray
    test_inputs: jnp.ndarray
    test_targets: jnp.ndarray
    dx: float
    dt: float
    n_x: int
    n_t: int


def generate_burgers_data(
    n_train: int = 25,
    n_val: int = 250,
    n_test: int = 250,
    n_x: int = 256,
    n_t: int = 59,
    key: jax.random.PRNGKey = None,
) -> PDEDataset:
    """Generate Burgers' equation data.
    
    Burgers' equation: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
    
    Following APEBench (Koehler et al., 2024).
    
    Args:
        n_train: Number of training trajectories (25 in paper)
        n_val: Number of validation trajectories
        n_test: Number of test trajectories
        n_x: Spatial resolution (256 in paper)
        n_t: Temporal resolution (59 in paper)
        key: JAX random key
    
    Returns:
        PDEDataset with train/val/test splits
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    keys = jax.random.split(key, 3)
    
    total_trajs = n_train + n_val + n_test
    
    # Generate initial conditions and solve PDE
    train_data = _solve_burgers_batch(keys[0], n_train, n_x, n_t)
    val_data = _solve_burgers_batch(keys[1], n_val, n_x, n_t)
    test_data = _solve_burgers_batch(keys[2], n_test, n_x, n_t)
    
    # Format: FNO takes 10 input time steps, predicts next step
    n_input_steps = 10
    
    def format_sequences(data):
        """Convert trajectories to (input_window, next_step) pairs."""
        inputs = []
        targets = []
        for traj in range(data.shape[0]):
            for t in range(n_input_steps, n_t):
                inp = data[traj, t-n_input_steps:t]  # 10 input steps
                tgt = data[traj, t]                   # next step
                inputs.append(inp)
                targets.append(tgt)
        return jnp.stack(inputs), jnp.stack(targets)
    
    train_inp, train_tgt = format_sequences(train_data)
    val_inp, val_tgt = format_sequences(val_data)
    test_inp, test_tgt = format_sequences(test_data)
    
    return PDEDataset(
        name='burgers',
        train_inputs=train_inp,
        train_targets=train_tgt,
        val_inputs=val_inp,
        val_targets=val_tgt,
        test_inputs=test_inp,
        test_targets=test_tgt,
        dx=1.0 / n_x,
        dt=0.01,
        n_x=n_x,
        n_t=n_t,
    )


def _solve_burgers_batch(
    key: jax.random.PRNGKey,
    n_trajs: int,
    n_x: int,
    n_t: int,
    viscosity: float = 0.01,
) -> jnp.ndarray:
    """Solve Burgers' equation for multiple trajectories.
    
    Uses spectral method with dealiasing.
    
    Returns:
        Array of shape (n_trajs, n_t, n_x)
    """
    x = jnp.linspace(0, 1, n_x)
    dx = 1.0 / n_x
    dt = 0.01
    
    def solve_single(rng_key):
        # Random initial condition: superposition of sine waves
        n_modes = 5
        coeffs = jax.random.normal(rng_key, (n_modes,)) * 0.5
        phases = jax.random.uniform(rng_key, (n_modes,)) * 2 * jnp.pi
        
        u = jnp.zeros(n_x)
        for k in range(n_modes):
            u += coeffs[k] * jnp.sin(2 * jnp.pi * (k + 1) * x + phases[k])
        
        # Time stepping using spectral method
        u_traj = [u]
        k_vals = 2 * jnp.pi * jnp.fft.rfftfreq(n_x, d=dx)
        
        for _ in range(n_t - 1):
            u = u_traj[-1]
            u_hat = jnp.fft.rfft(u)
            
            # Burgers: ∂u/∂t = -u ∂u/∂x + ν ∂²u/∂x²
            # In spectral space: du_hat/dt = -ik/2 * (u²)_hat - ν k² u_hat
            ux = jnp.fft.irfft(1j * k_vals * u_hat, n=n_x)
            nonlinear = -u * ux
            nonlinear_hat = jnp.fft.rfft(nonlinear)
            
            # RK4 step
            def rhs(u_hat_current):
                u_current = jnp.fft.irfft(u_hat_current, n=n_x)
                ux_current = jnp.fft.irfft(1j * k_vals * u_hat_current, n=n_x)
                nonlinear_current = -u_current * ux_current
                nonlinear_hat_current = jnp.fft.rfft(nonlinear_current)
                return nonlinear_hat_current - viscosity * k_vals**2 * u_hat_current
            
            # RK4
            k1 = dt * rhs(u_hat)
            k2 = dt * rhs(u_hat + 0.5 * k1)
            k3 = dt * rhs(u_hat + 0.5 * k2)
            k4 = dt * rhs(u_hat + k3)
            
            u_hat_new = u_hat + (k1 + 2*k2 + 2*k3 + k4) / 6
            u_new = jnp.fft.irfft(u_hat_new, n=n_x)
            u_traj.append(u_new)
        
        return jnp.stack(u_traj)
    
    keys = jax.random.split(key, n_trajs)
    trajectories = jax.vmap(solve_single)(keys)
    return trajectories


def generate_advection_diffusion_ood_data(
    key: jax.random.PRNGKey = None,
) -> dict:
    """Generate out-of-distribution datasets for the Advection-Diffusion equation.
    
    Following Section D.1.2:
    
    Five OOD variants:
    1. Base: Gaussian blobs + constant velocity
    2. Flip: Base + velocity reversed at center
    3. Pos: Base + triangular heat source
    4. Pos-Neg: Base + heat source + heat sink
    5. Pos-Neg-Flip: All combined
    
    Spatial resolution: 100 × 100
    Temporal resolution: Δt = 5e-10 (fine), subsampled to 59 steps
    α = 0.026 (diffusion coefficient)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    keys = jax.random.split(key, 5)
    
    datasets = {}
    
    # Common parameters
    n_x = 100
    n_y = 100
    n_t = 59
    alpha = 0.026
    
    # Generate base dataset (1000 training trajectories)
    datasets['base'] = _generate_adr_dataset(
        keys[0], 1000, n_x, n_y, n_t, alpha,
        flip_velocity=False, add_source=False, add_sink=False,
    )
    
    # Generate OOD variants (250 test trajectories each)
    datasets['flip'] = _generate_adr_dataset(
        keys[1], 250, n_x, n_y, n_t, alpha,
        flip_velocity=True, add_source=False, add_sink=False,
    )
    
    datasets['pos'] = _generate_adr_dataset(
        keys[2], 250, n_x, n_y, n_t, alpha,
        flip_velocity=False, add_source=True, add_sink=False,
    )
    
    datasets['pos_neg'] = _generate_adr_dataset(
        keys[3], 250, n_x, n_y, n_t, alpha,
        flip_velocity=False, add_source=True, add_sink=True,
    )
    
    datasets['pos_neg_flip'] = _generate_adr_dataset(
        keys[4], 250, n_x, n_y, n_t, alpha,
        flip_velocity=True, add_source=True, add_sink=True,
    )
    
    return datasets


def _generate_adr_dataset(
    key: jax.random.PRNGKey,
    n_trajs: int,
    n_x: int,
    n_y: int,
    n_t: int,
    alpha: float,
    flip_velocity: bool = False,
    add_source: bool = False,
    add_sink: bool = False,
) -> jnp.ndarray:
    """Generate Advection-Diffusion-Reaction equation trajectories.
    
    ∂u/∂t + v · ∇u = α ∇²u + R
    
    Solved using 9-point stencil with RK4.
    
    Returns:
        Array of shape (n_trajs, n_t, n_x, n_y)
    """
    keys = jax.random.split(key, n_trajs)
    
    def generate_single(rng_key):
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)
        
        # Generate initial condition: random number (1-10) of Gaussian blobs
        n_blobs = jax.random.randint(k1, (), 1, 11)
        
        x = jnp.linspace(0, 1, n_x)
        y = jnp.linspace(0, 1, n_y)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        
        u = jnp.zeros((n_x, n_y))
        
        # Place Gaussian blobs
        for b in range(n_blobs):
            cx = jax.random.uniform(k2, ()) * 0.8 + 0.1
            cy = jax.random.uniform(k2, ()) * 0.8 + 0.1
            sigma = jax.random.uniform(k2, ()) * 0.05 + 0.02
            amplitude = jax.random.uniform(k2, ()) * 0.5 + 0.5
            
            u += amplitude * jnp.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        
        # Velocity field: constant random
        vx = jax.random.uniform(k3, ()) * 2 - 1
        vy = jax.random.uniform(k3, ()) * 2 - 1
        
        if flip_velocity:
            # Reverse velocity at center
            vx_field = vx * jnp.where(X < 0.5, 1.0, -1.0)
            vy_field = vy * jnp.where(Y < 0.5, 1.0, -1.0)
        else:
            vx_field = vx * jnp.ones_like(X)
            vy_field = vy * jnp.ones_like(Y)
        
        # Reaction term
        R = jnp.zeros((n_x, n_y))
        
        if add_source:
            # Randomly placed triangular heat source
            sx = jax.random.uniform(k4, ()) * 0.6 + 0.2
            sy = jax.random.uniform(k4, ()) * 0.6 + 0.2
            source_strength = 10.0
            # Triangular shape
            dist = jnp.sqrt((X - sx)**2 + (Y - sy)**2)
            R += source_strength * jnp.maximum(0.0, 1.0 - dist / 0.1)
        
        if add_sink:
            # Cloud-shaped heat sink
            sink_x = jax.random.uniform(k4, ()) * 0.6 + 0.2
            sink_y = jax.random.uniform(k4, ()) * 0.6 + 0.2
            sink_strength = -5.0
            dist = jnp.sqrt((X - sink_x)**2 + (Y - sink_y)**2)
            # Cloud shape: smoother than triangle
            R += sink_strength * jnp.exp(-dist**2 / 0.005)
        
        # Time stepping (9-point stencil, RK4)
        dx = 1.0 / (n_x - 1)
        dy = 1.0 / (n_y - 1)
        dt = 5e-10 * 200  # Subsampled: 200 fine steps per saved step
        dt_fine = 5e-10
        
        u_traj = [u]
        u_current = u
        
        for step in range(n_t - 1):
            # 200 fine steps per saved step
            for _ in range(200):
                # RK4 with 9-point stencil
                def adr_rhs(u_state):
                    # Compute spatial derivatives using 9-point stencil
                    # Laplacian (9-point)
                    laplacian = (
                        -20 * u_state
                        + 4 * (jnp.roll(u_state, 1, axis=0) + jnp.roll(u_state, -1, axis=0)
                              + jnp.roll(u_state, 1, axis=1) + jnp.roll(u_state, -1, axis=1))
                        + (jnp.roll(jnp.roll(u_state, 1, axis=0), 1, axis=1)
                           + jnp.roll(jnp.roll(u_state, 1, axis=0), -1, axis=1)
                           + jnp.roll(jnp.roll(u_state, -1, axis=0), 1, axis=1)
                           + jnp.roll(jnp.roll(u_state, -1, axis=0), -1, axis=1))
                    ) / (6 * dx * dy)
                    
                    # Gradient (central difference)
                    dudx = (jnp.roll(u_state, -1, axis=0) - jnp.roll(u_state, 1, axis=0)) / (2 * dx)
                    dudy = (jnp.roll(u_state, -1, axis=1) - jnp.roll(u_state, 1, axis=1)) / (2 * dy)
                    
                    # Advection term
                    advection = vx_field * dudx + vy_field * dudy
                    
                    # Full RHS
                    return -advection + alpha * laplacian + R
                
                # RK4
                k1 = dt_fine * adr_rhs(u_current)
                k2 = dt_fine * adr_rhs(u_current + 0.5 * k1)
                k3 = dt_fine * adr_rhs(u_current + 0.5 * k2)
                k4 = dt_fine * adr_rhs(u_current + k3)
                
                u_current = u_current + (k1 + 2*k2 + 2*k3 + k4) / 6
            
            u_traj.append(u_current)
        
        return jnp.stack(u_traj)
    
    trajectories = []
    for i in range(n_trajs):
        traj = generate_single(keys[i])
        trajectories.append(traj)
    
    return jnp.stack(trajectories)


def generate_hyper_diffusion_data(
    key: jax.random.PRNGKey = None,
    n_train: int = 25,
    n_val: int = 250,
    n_test: int = 250,
    n_x: int = 256,
    n_t: int = 59,
) -> PDEDataset:
    """Generate Hyper-Diffusion equation data.
    
    ∂u/∂t = -ν ∂⁴u/∂x⁴
    
    Higher-order diffusion, smooth solutions.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Placeholder: return synthetic data matching expected dimensions
    total = n_train + n_val + n_test
    n_input_steps = 10
    
    keys = jax.random.split(key, 3)
    
    def generate(n, rng):
        trajs = []
        for i in range(n):
            # Smooth random initial condition
            x = jnp.linspace(0, 1, n_x)
            coeffs = jax.random.normal(rng, (3,)) * 0.3
            u0 = jnp.zeros(n_x)
            for k, c in enumerate(coeffs):
                u0 += c * jnp.sin(2 * jnp.pi * (k + 1) * x)
            
            # Simple hyper-diffusion evolution
            traj = [u0]
            k_vals = 2 * jnp.pi * jnp.fft.rfftfreq(n_x, d=1.0/n_x)
            u_hat = jnp.fft.rfft(u0)
            
            dt = 0.001
            for _ in range(n_t - 1):
                u_hat = u_hat * jnp.exp(-dt * k_vals**4)
                u = jnp.fft.irfft(u_hat, n=n_x)
                traj.append(u)
            
            trajs.append(jnp.stack(traj))
        return jnp.stack(trajs)
    
    train_trajs = generate(n_train, keys[0])
    val_trajs = generate(n_val, keys[1])
    test_trajs = generate(n_test, keys[2])
    
    def format_sequences(data):
        inputs, targets = [], []
        for traj in range(data.shape[0]):
            for t in range(n_input_steps, n_t):
                inputs.append(data[traj, t-n_input_steps:t])
                targets.append(data[traj, t])
        return jnp.stack(inputs), jnp.stack(targets)
    
    return PDEDataset(
        name='hyper_diffusion',
        train_inputs=format_sequences(train_trajs)[0],
        train_targets=format_sequences(train_trajs)[1],
        val_inputs=format_sequences(val_trajs)[0],
        val_targets=format_sequences(val_trajs)[1],
        test_inputs=format_sequences(test_trajs)[0],
        test_targets=format_sequences(test_trajs)[1],
        dx=1.0/n_x, dt=0.001, n_x=n_x, n_t=n_t,
    )


def generate_kuramoto_sivashinsky_data(
    key: jax.random.PRNGKey = None,
    n_train: int = 25,
    n_val: int = 250,
    n_test: int = 250,
    n_x: int = 256,
    n_t: int = 59,
) -> PDEDataset:
    """Generate Kuramoto-Sivashinsky (conservative) equation data.
    
    ∂u/∂t + u ∂u/∂x + ∂²u/∂x² + ν ∂⁴u/∂x⁴ = 0
    
    Known for spatiotemporal chaos.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    n_input_steps = 10
    keys = jax.random.split(key, 3)
    
    def generate(n, rng):
        trajs = []
        for _ in range(n):
            x = jnp.linspace(0, 2*jnp.pi, n_x)
            u0 = jax.random.normal(rng, (n_x,)) * 0.1
            
            traj = [u0]
            k_vals = 2 * jnp.pi * jnp.fft.rfftfreq(n_x, d=2*jnp.pi/n_x)
            u_hat = jnp.fft.rfft(u0)
            
            dt = 0.01
            nu = 0.01
            
            for _ in range(n_t - 1):
                u = jnp.fft.irfft(u_hat, n=n_x)
                # Nonlinear term: u * u_x
                ux = jnp.fft.irfft(1j * k_vals * u_hat, n=n_x)
                nonlinear = -u * ux
                nonlinear_hat = jnp.fft.rfft(nonlinear)
                
                # Linear terms: -k² + ν k⁴
                linear = -k_vals**2 - nu * k_vals**4
                
                # ETD or simple RK4
                u_hat = u_hat + dt * (nonlinear_hat + linear * u_hat)
                u = jnp.fft.irfft(u_hat, n=n_x)
                traj.append(u)
            
            trajs.append(jnp.stack(traj))
        return jnp.stack(trajs)
    
    train_trajs = generate(n_train, keys[0])
    val_trajs = generate(n_val, keys[1])
    test_trajs = generate(n_test, keys[2])
    
    def format_sequences(data):
        inputs, targets = [], []
        for traj in range(data.shape[0]):
            for t in range(n_input_steps, n_t):
                inputs.append(data[traj, t-n_input_steps:t])
                targets.append(data[traj, t])
        return jnp.stack(inputs), jnp.stack(targets)
    
    return PDEDataset(
        name='kuramoto_sivashinsky',
        train_inputs=format_sequences(train_trajs)[0],
        train_targets=format_sequences(train_trajs)[1],
        val_inputs=format_sequences(val_trajs)[0],
        val_targets=format_sequences(val_trajs)[1],
        test_inputs=format_sequences(test_trajs)[0],
        test_targets=format_sequences(test_trajs)[1],
        dx=2*jnp.pi/n_x, dt=0.01, n_x=n_x, n_t=n_t,
    )
