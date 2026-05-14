## linearization.py

from typing import Callable, Dict, Any
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor
from jax import random
import numpy as np


class Linearization:
    """
    Linearization class for computing Jacobian tensors and constructing
    function-valued Gaussian Processes (GPs) using the LUNO framework.
    """

    def __init__(self, model: Any):
        """
        Initialize the Linearization class with a trained model instance.

        Args:
            model (Any): Instance of the FourierNeuralOperator model.
        """
        self.model = model
        self.jacobian = None  # Cache Jacobian after computation
        self.gp_mean = None  # GP mean function
        self.gp_covariance = None  # GP covariance matrix

    def compute_jacobian(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the Jacobian matrix of model outputs with respect to the trainable weights.

        Args:
            inputs (jnp.ndarray): Input tensor containing function samples.

        Returns:
            jnp.ndarray: Jacobian tensor representing ∂w f(a, x, w).
        """
        # Flatten model weights for Jacobian computation
        weights = self.model.trainable_weights()

        def model_output_fn(flat_weights, x_batch):
            """Reshape and compute model output for given weights and inputs."""
            reshaped_weights = jax.tree_unflatten(
                self.model.variables["params"].tree_strucutre, flat_weights
            )
            return self.model.apply({"params": reshaped_weights}, x_batch)

        # Compute Jacobian: ∂w f(a, x, w*)
        batched_fn = jax.vmap(
            lambda x: jax.jacobian(model_output_fn, argnums=0)(weights, x),
            in_axes=0
        )
        jacobian = batched_fn(inputs)

        # Cache and return
        self.jacobian = jacobian
        return jacobian

    def construct_gp(self, mean: jnp.ndarray, covariance: jnp.ndarray) -> Callable:
        """
        Construct a function-valued Gaussian Process (GP) leveraging the linearized
        neural operator.

        Args:
            mean (jnp.ndarray): Mean of the GP posterior, typically f(a, x, w*).
            covariance (jnp.ndarray): Weight-space covariance matrix Σ.

        Returns:
            Callable: A function that provides GP interface, including sampling.
        """
        if self.jacobian is None:
            raise ValueError("Jacobian not yet computed. Call compute_jacobian() first.")

        # Compute predictive covariance: J @ Σ @ J.T
        jacobian_flat = self.jacobian.reshape(self.jacobian.shape[0], -1)
        predictive_covariance = jacobian_flat @ covariance @ jacobian_flat.T

        # Cache GP properties
        self.gp_mean = mean
        self.gp_covariance = predictive_covariance

        # Return a callable GP posterior
        def gp_posterior(inputs: jnp.ndarray) -> Dict[str, Any]:
            """
            GP posterior providing mean, variance, and sampling.

            Args:
                inputs (jnp.ndarray): Grid points or input samples for evaluation.

            Returns:
                Dict[str, Any]: Output dictionary with keys:
                    - "mean": Marginal mean of the function-valued GP.
                    - "variance": Point-wise variance across the input domain.
                    - "samples": Lazy sample functions from the GP.
            """
            # Mean and variance computation
            variance = jnp.diag(self.gp_covariance)

            def sample_gp(key: random.PRNGKey, num_samples: int):
                """
                Generate samples from the Gaussian Process.

                Args:
                    key (random.PRNGKey): JAX random key for sampling.
                    num_samples (int): Number of samples to draw.

                Returns:
                    jnp.ndarray: Function samples from the GP.
                """
                chol_factor = cho_factor(self.gp_covariance)
                normal_samples = random.normal(key, shape=(num_samples, mean.shape[0]))
                samples = self.gp_mean + normal_samples @ chol_factor[0]
                return samples

            return {
                "mean": self.gp_mean,
                "variance": variance,
                "sample_gp": sample_gp
            }

        return gp_posterior

    def sample_gp(self, inputs: jnp.ndarray, num_samples: int = 200) -> jnp.ndarray:
        """
        Generate samples from the GP at specified inputs.

        Args:
            inputs (jnp.ndarray): Input tensor for evaluation.
            num_samples (int, optional): Number of functional samples. Defaults to 200.

        Returns:
            jnp.ndarray: Sampled functions evaluated at the given inputs.
        """
        if self.gp_mean is None or self.gp_covariance is None:
            raise ValueError("Call construct_gp() to define the GP first.")

        key = random.PRNGKey(42)  # Ensure reproducibility
        chol_factor = cho_factor(self.gp_covariance)
        normal_samples = random.normal(key, shape=(num_samples, self.gp_mean.shape[0]))
        samples = self.gp_mean + normal_samples @ chol_factor[0]
        return samples
