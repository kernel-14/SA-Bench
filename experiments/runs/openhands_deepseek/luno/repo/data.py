"""PDE data generation following APEBench and the LUNO paper specifications.

Provides solvers for:
- Burgers' equation (1D)
- Hyper Diffusion equation (1D)
- Kuramoto-Sivashinsky (conservative, 1D)
- Advection-Diffusion-Reaction equation (2D) with OOD variants
"""

from typing import Tuple, Optional, Dict
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# 1D PDE solvers
# ---------------------------------------------------------------------------

def burgers_equation(
    n_trajectories: int,
    n_steps: int,
    spatial_resolution: int,
    domain_size: float = 1.0,
    nu: float = 0.01,
    dt: float = 1e-3,
    key: jax.Array = None,
) -> jnp.ndarray:
    """Generate Burgers' equation trajectories.

    u_t + u * u_x = nu * u_xx

    Uses spectral method (Fourier-Galerkin) as in APEBench.
    Returns trajectories of shape (n_trajectories, n_steps, spatial_resolution).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    dx = domain_size / spatial_resolution
    # Wavenumbers for spectral method
    k = 2 * jnp.pi * jnp.fft.rfftfreq(spatial_resolution, d=dx)

    def solve_one(key_i):
        # Initial condition: random superposition of sinusoids
        key_ic, _ = jax.random.split(key_i)
        n_modes_init = 10
        coeffs = jax.random.normal(key_ic, (2, n_modes_init)) * 0.5 / jnp.arange(1, n_modes_init + 1)
        x = jnp.linspace(0, domain_size, spatial_resolution)
        u0 = jnp.zeros(spatial_resolution)
        for i in range(n_modes_init):
            u0 += coeffs[0, i] * jnp.sin(2 * jnp.pi * (i + 1) * x / domain_size)
            u0 += coeffs[1, i] * jnp.cos(2 * jnp.pi * (i + 1) * x / domain_size)

        def step(u, _):
            u_hat = jnp.fft.rfft(u)
            # Nonlinear term in physical space, then FFT
            u_x = jnp.fft.irfft(1j * k * u_hat)
            nonlinear = u * u_x
            n_hat = jnp.fft.rfft(nonlinear)
            # Spectral viscosity
            u_hat_new = (u_hat - dt * n_hat) / (1 + dt * nu * k ** 2)
            u_new = jnp.fft.irfft(u_hat_new)
            return u_new, u_new

        _, trajectory = jax.lax.scan(step, u0, jnp.arange(n_steps - 1))
        trajectory = jnp.concatenate([u0[None, :], trajectory], axis=0)
        return trajectory

    keys = jax.random.split(key, n_trajectories)
    trajectories = jax.vmap(solve_one)(keys[:, None])
    return jnp.squeeze(trajectories)


def hyper_diffusion(
    n_trajectories: int,
    n_steps: int,
    spatial_resolution: int,
    domain_size: float = 1.0,
    nu: float = 0.01,
    dt: float = 1e-4,
    key: jax.Array = None,
) -> jnp.ndarray:
    """Generate Hyper Diffusion equation trajectories.

    u_t = -u_xxxx (fourth-order diffusion)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    dx = domain_size / spatial_resolution
    k = 2 * jnp.pi * jnp.fft.rfftfreq(spatial_resolution, d=dx)

    def solve_one(key_i):
        key_ic, _ = jax.random.split(key_i)
        n_modes_init = 10
        coeffs = jax.random.normal(key_ic, (2, n_modes_init)) * 0.5 / jnp.arange(1, n_modes_init + 1)
        x = jnp.linspace(0, domain_size, spatial_resolution)
        u0 = jnp.zeros(spatial_resolution)
        for i in range(n_modes_init):
            u0 += coeffs[0, i] * jnp.sin(2 * jnp.pi * (i + 1) * x / domain_size)
            u0 += coeffs[1, i] * jnp.cos(2 * jnp.pi * (i + 1) * x / domain_size)

        def step(u, _):
            u_hat = jnp.fft.rfft(u)
            u_hat_new = u_hat / (1 + dt * nu * k ** 4)
            u_new = jnp.fft.irfft(u_hat_new)
            return u_new, u_new

        _, trajectory = jax.lax.scan(step, u0, jnp.arange(n_steps - 1))
        trajectory = jnp.concatenate([u0[None, :], trajectory], axis=0)
        return trajectory

    keys = jax.random.split(key, n_trajectories)
    trajectories = jax.vmap(solve_one)(keys[:, None])
    return jnp.squeeze(trajectories)


def kuramoto_sivashinsky(
    n_trajectories: int,
    n_steps: int,
    spatial_resolution: int,
    domain_size: float = 1.0,
    dt: float = 1e-4,
    key: jax.Array = None,
) -> jnp.ndarray:
    """Generate Kuramoto-Sivashinsky (conservative) trajectories.

    u_t = -u * u_x - u_xx - u_xxxx

    Uses spectral method (ETDRK4-like integration).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    dx = domain_size / spatial_resolution
    k = 2 * jnp.pi * jnp.fft.rfftfreq(spatial_resolution, d=dx)
    L = k ** 2 - k ** 4  # Linear operator in Fourier space

    def solve_one(key_i):
        key_ic, _ = jax.random.split(key_i)
        n_modes_init = 4
        coeffs = jax.random.normal(key_ic, (2, n_modes_init)) * 0.3 / jnp.arange(1, n_modes_init + 1)
        x = jnp.linspace(0, domain_size, spatial_resolution)
        u0 = jnp.zeros(spatial_resolution)
        for i in range(n_modes_init):
            u0 += coeffs[0, i] * jnp.sin(2 * jnp.pi * (i + 1) * x / domain_size)
            u0 += coeffs[1, i] * jnp.cos(2 * jnp.pi * (i + 1) * x / domain_size)

        def step(u, _):
            u_hat = jnp.fft.rfft(u)
            # Nonlinear term
            u_x = jnp.fft.irfft(1j * k * u_hat)
            nonlinear = -0.5 * (u ** 2)
            n_hat = jnp.fft.rfft(jnp.fft.irfft(1j * k * jnp.fft.rfft(nonlinear)))  # -u * u_x
            # ETD1: u_{n+1} = e^{L*dt} * u_n + (e^{L*dt}-1)/L * N(u_n)
            eLdt = jnp.exp(L * dt)
            factor = jnp.where(jnp.abs(L) > 1e-8, (eLdt - 1) / L, dt)
            u_hat_new = eLdt * u_hat + factor * n_hat
            u_new = jnp.fft.irfft(u_hat_new)
            return u_new, u_new

        _, trajectory = jax.lax.scan(step, u0, jnp.arange(n_steps - 1))
        trajectory = jnp.concatenate([u0[None, :], trajectory], axis=0)
        return trajectory

    keys = jax.random.split(key, n_trajectories)
    trajectories = jax.vmap(solve_one)(keys[:, None])
    return jnp.squeeze(trajectories)


# ---------------------------------------------------------------------------
# 2D Advection-Diffusion-Reaction PDE solver
# ---------------------------------------------------------------------------

def advection_diffusion_reaction(
    n_trajectories: int,
    n_steps: int,
    spatial_resolution: int,
    domain_size: float = 1.0,
    diffusion_coefficient: float = 0.026,
    dt: float = 5e-10,
    ode_steps: int = 200,
    variant: str = "base",
    key: jax.Array = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Generate Advection-Diffusion-Reaction equation trajectories.

    du/dt + v * grad u = alpha * laplace u + R

    OOD variants:
    - base: Gaussian blobs, constant velocity, no reaction
    - flip: velocity reversed at center
    - pos: adds triangular heat source
    - pos_neg: adds triangular heat source and cloud-shaped heat sink
    - pos_neg_flip: pos_neg with flipped velocity

    Returns trajectories (n_trajectories, n_steps, H, W) and auxiliary fields.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    h = spatial_resolution
    dx = domain_size / h
    alpha = diffusion_coefficient
    ode_dt = dt
    subsample = ode_steps // n_steps

    def solve_one(key_i):
        keys = jax.random.split(key_i, 6)

        # Initial condition: random Gaussian blobs
        n_blobs = jax.random.randint(keys[0], (), 1, 11)
        blob_positions = jax.random.uniform(keys[1], (n_blobs, 2), minval=0.15, maxval=0.85) * domain_size
        blob_sigmas = jax.random.uniform(keys[2], (n_blobs,), minval=0.03, maxval=0.1) * domain_size
        blob_amps = jax.random.uniform(keys[3], (n_blobs,), minval=0.5, maxval=2.0)

        x = jnp.linspace(0, domain_size, h)
        y = jnp.linspace(0, domain_size, h)
        X, Y = jnp.meshgrid(x, y)
        u0 = jnp.zeros((h, h))
        for i in range(n_blobs):
            rx = (X - blob_positions[i, 0]) / blob_sigmas[i]
            ry = (Y - blob_positions[i, 1]) / blob_sigmas[i]
            u0 += blob_amps[i] * jnp.exp(-0.5 * (rx ** 2 + ry ** 2))

        # Velocity field
        vx = jax.random.uniform(keys[4], ()) * 0.5 + 0.25
        vy = jax.random.uniform(keys[5], ()) * 0.5 + 0.25

        # Flip: reverse velocity at center
        if "flip" in variant:
            vx_field = jnp.where(X < domain_size / 2, vx, -vx)
            vy_field = jnp.where(Y < domain_size / 2, vy, -vy)
        else:
            vx_field = jnp.full_like(X, vx)
            vy_field = jnp.full_like(Y, vy)

        # Reaction term
        if "pos" in variant:
            # Triangular heat source at random location
            heat_x, heat_y = domain_size * 0.5, domain_size * 0.5
            R = jnp.maximum(0.0, 0.5 - jnp.sqrt((X - heat_x) ** 2 + (Y - heat_y) ** 2) / domain_size) * 0.3
        else:
            R = jnp.zeros((h, h))

        if "neg" in variant:
            # Cloud-shaped heat sink
            sink_x, sink_y = domain_size * 0.3, domain_size * 0.7
            sink = -0.2 * jnp.exp(-0.5 * ((X - sink_x) ** 2 + (Y - sink_y) ** 2) / (0.05 * domain_size) ** 2)
            R += sink

        # Finite difference coefficients (9-point stencil)
        def laplacian(u):
            u_padded = jnp.pad(u[None, None, :, :], 1, mode='wrap')
            kernel = jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=jnp.float32) / dx ** 2
            lap = jax.lax.conv_general_dilated(
                u_padded[:, None, :, :],
                kernel[None, None, :, :],
                window_strides=(1, 1),
                padding='VALID',
            )
            return lap[0, 0]

        def grad_x(u):
            u_padded = jnp.pad(u[None, None, :, :], 1, mode='wrap')
            kernel = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32) / (6 * dx)
            gx = jax.lax.conv_general_dilated(
                u_padded[:, None, :, :],
                kernel[None, None, :, :],
                window_strides=(1, 1),
                padding='VALID',
            )
            return gx[0, 0]

        def grad_y(u):
            u_padded = jnp.pad(u[None, None, :, :], 1, mode='wrap')
            kernel = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=jnp.float32) / (6 * dx)
            gy = jax.lax.conv_general_dilated(
                u_padded[:, None, :, :],
                kernel[None, None, :, :],
                window_strides=(1, 1),
                padding='VALID',
            )
            return gy[0, 0]

        # RK4 step
        def rk4_step(u):
            def rhs(uu):
                return -vx_field * grad_x(uu) - vy_field * grad_y(uu) + alpha * laplacian(uu) + R

            k1 = rhs(u)
            k2 = rhs(u + 0.5 * ode_dt * k1)
            k3 = rhs(u + 0.5 * ode_dt * k2)
            k4 = rhs(u + ode_dt * k3)
            return u + (ode_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        def scan_fun(u, _):
            u_new = u
            for _ in range(subsample):
                u_new = rk4_step(u_new)
            return u_new, u_new

        _, trajectory = jax.lax.scan(scan_fun, u0, jnp.arange(n_steps - 1))
        trajectory = jnp.concatenate([u0[None, :, :], trajectory], axis=0)

        aux = {
            "vx_field": vx_field,
            "vy_field": vy_field,
            "R": R,
        }
        return trajectory, aux

    keys = jax.random.split(key, n_trajectories)
    out = jax.vmap(solve_one)(keys[:, None])
    trajectories, aux = out[0], out[1]
    trajectories = jnp.squeeze(trajectories, axis=1)
    aux = {k: jnp.squeeze(v, axis=1) for k, v in aux.items()}
    return trajectories, aux


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def generate_dataset(
    pde_name: str,
    n_trajectories: int,
    n_steps: int,
    spatial_resolution: int,
    domain_size: float = 1.0,
    variant: str = "base",
    key: jax.Array = None,
) -> Tuple[jnp.ndarray, Optional[Dict]]:
    """Generate a dataset for a specific PDE.

    Returns trajectories and (optionally) auxiliary fields.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    key_gen, key_noise = jax.random.split(key)

    if pde_name == "burgers":
        data = burgers_equation(n_trajectories, n_steps, spatial_resolution, domain_size, key=key_gen)
        return data, None
    elif pde_name == "hyper_diffusion":
        data = hyper_diffusion(n_trajectories, n_steps, spatial_resolution, domain_size, key=key_gen)
        return data, None
    elif pde_name == "kuramoto_sivashinsky":
        data = kuramoto_sivashinsky(n_trajectories, n_steps, spatial_resolution, domain_size, key=key_gen)
        return data, None
    elif pde_name == "advection_diffusion":
        return advection_diffusion_reaction(
            n_trajectories, n_steps, spatial_resolution, domain_size,
            variant=variant, key=key_gen,
        )
    else:
        raise ValueError(f"Unknown PDE name: {pde_name}")


def prepare_fno_data(
    trajectories: jnp.ndarray,
    n_input_steps: int = 10,
    n_output_steps: int = 1,
    spatial_dim: int = 1,
    aux_fields: Optional[Dict] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert full trajectories into (input, target) pairs for FNO training.

    For 1D: input shape (n_pairs, n_input_steps, spatial_resolution) -> (n_pairs, n_input_steps, spatial_resolution)
    For 2D: input shape (n_pairs, n_input_steps, H, W) -> (n_pairs, H, W, n_input_steps + channels)

    The paper uses 10 input time steps to predict the next time step.
    Velocity and reaction terms are concatenated as additional channels.
    """
    n_trajectories = trajectories.shape[0]
    n_total_steps = trajectories.shape[1]

    pairs_input = []
    pairs_target = []

    for t in range(n_total_steps - n_input_steps - n_output_steps + 1):
        inp = trajectories[:, t:t + n_input_steps]
        tgt = trajectories[:, t + n_input_steps:t + n_input_steps + n_output_steps]

        if spatial_dim == 1:
            # (n_traj, n_input_steps, N) -> (n_traj, N, n_input_steps)
            inp = jnp.transpose(inp, (0, 2, 1))
            tgt = jnp.transpose(tgt, (0, 2, 1))
        elif spatial_dim == 2:
            inp = jnp.transpose(inp, (0, 2, 3, 1))
            tgt = jnp.transpose(tgt, (0, 2, 3, 1))

        pairs_input.append(inp)
        pairs_target.append(tgt)

    X = jnp.concatenate(pairs_input, axis=0)
    y = jnp.concatenate(pairs_target, axis=0)

    # Add auxiliary fields as extra channels (for 2D advection-diffusion)
    if aux_fields is not None and spatial_dim == 2:
        batch_size = X.shape[0]
        vx = aux_fields["vx_field"][:, None, :, :]  # add step dim
        vy = aux_fields["vy_field"][:, None, :, :]
        R = aux_fields["R"][:, None, :, :]
        # Tile to match X shape
        vx_rep = jnp.tile(vx[:, 0:1], (1, X.shape[1], 1, 1))
        vy_rep = jnp.tile(vy[:, 0:1], (1, X.shape[1], 1, 1))
        R_rep = jnp.tile(R[:, 0:1], (1, X.shape[1], 1, 1))
        # Repeat across trajectories to match pairs
        n_pairs_per_traj = X.shape[0] // n_trajectories
        vx_expanded = jnp.repeat(jnp.transpose(vx[:, 0], (1, 2, 0)), n_pairs_per_traj, axis=0)
        vy_expanded = jnp.repeat(jnp.transpose(vy[:, 0], (1, 2, 0)), n_pairs_per_traj, axis=0)
        R_expanded = jnp.repeat(jnp.transpose(R[:, 0], (1, 2, 0)), n_pairs_per_traj, axis=0)

        X = jnp.concatenate([X, vx_expanded[..., None], vy_expanded[..., None], R_expanded[..., None]], axis=-1)

    return X, y


def split_train_val_test(
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    key: jax.Array = None,
):
    """Shuffle and split data into train/val/test sets."""
    if key is None:
        key = jax.random.PRNGKey(0)

    n_total = X.shape[0]
    idx = jax.random.permutation(key, n_total)
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:n_train + n_val + n_test]

    return (
        X[idx_train], y[idx_train],
        X[idx_val], y[idx_val],
        X[idx_test], y[idx_test],
    )
