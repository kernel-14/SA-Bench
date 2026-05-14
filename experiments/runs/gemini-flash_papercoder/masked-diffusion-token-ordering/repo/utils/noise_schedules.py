import math
import torch
from typing import Callable, Union

class NoiseScheduler:
    """
    Manages different noise schedules for the Masked Diffusion Model's forward process.
    Provides alpha_t, its derivative alpha_t_prime, and the masking probability
    (1 - alpha_t) for any given continuous time t between 0 and 1.
    """

    def __init__(self, schedule_type: str, num_steps: int) -> None:
        """
        Initializes the NoiseScheduler with a specific schedule type and number of steps.

        Args:
            schedule_type (str): The type of noise schedule ('linear' or 'cosine').
            num_steps (int): The total number of discrete steps in the diffusion process.

        Raises:
            ValueError: If an unsupported schedule_type is provided.
        """
        if schedule_type not in ["linear", "cosine"]:
            raise ValueError(f"Unsupported schedule type: {schedule_type}. "
                             "Available types are 'linear', 'cosine'.")

        self._schedule_type: str = schedule_type
        self._num_steps: int = num_steps

        # Select the appropriate alpha and alpha_prime functions based on schedule type
        if self._schedule_type == "linear":
            self._alpha_fn: Callable[[float], float] = self._linear_alpha
            self._alpha_prime_fn: Callable[[float], float] = self._linear_alpha_prime
        elif self._schedule_type == "cosine":
            self._alpha_fn: Callable[[float], float] = self._cosine_alpha
            self._alpha_prime_fn: Callable[[float], float] = self._cosine_alpha_prime

    def _linear_alpha(self, t: float) -> float:
        """
        Implements a linear noise schedule for alpha_t.
        alpha_t = 1 - t
        Ensures alpha_0 = 1 (no masking) and alpha_1 = 0 (full masking).

        Args:
            t (float): Continuous time value between 0.0 and 1.0.

        Returns:
            float: The value of alpha_t.
        """
        return 1.0 - t

    def _linear_alpha_prime(self, t: float) -> float:
        """
        Calculates the derivative of the linear alpha_t schedule with respect to t.
        alpha_t' = -1

        Args:
            t (float): Continuous time value between 0.0 and 1.0.

        Returns:
            float: The value of alpha_t_prime.
        """
        # For a linear function, the derivative is constant.
        # We return -1.0 as the derivative of (1 - t) is -1.
        return -1.0

    def _cosine_alpha(self, t: float) -> float:
        """
        Implements a cosine noise schedule for alpha_t.
        alpha_t = cos(pi/2 * t)
        Ensures alpha_0 = 1 and alpha_1 = 0.

        Args:
            t (float): Continuous time value between 0.0 and 1.0.

        Returns:
            float: The value of alpha_t.
        """
        return math.cos(0.5 * math.pi * t)

    def _cosine_alpha_prime(self, t: float) -> float:
        """
        Calculates the derivative of the cosine alpha_t schedule with respect to t.
        alpha_t' = -pi/2 * sin(pi/2 * t)

        Args:
            t (float): Continuous time value between 0.0 and 1.0.

        Returns:
            float: The value of alpha_t_prime.
        """
        return -0.5 * math.pi * math.sin(0.5 * math.pi * t)

    def get_alpha(self, t: float) -> float:
        """
        Retrieves the alpha_t value for a given continuous time t.

        Args:
            t (float): A continuous time value between 0.0 and 1.0.

        Returns:
            float: The computed alpha_t value.
        """
        # Ensure t is within [0, 1] bounds
        t_clipped: float = max(0.0, min(1.0, t))
        return self._alpha_fn(t_clipped)

    def get_alpha_prime(self, t: float) -> float:
        """
        Retrieves the derivative of alpha_t (alpha_t_prime) for a given continuous time t.

        Args:
            t (float): A continuous time value between 0.0 and 1.0.

        Returns:
            float: The computed alpha_t_prime value.
        """
        # Ensure t is within [0, 1] bounds
        t_clipped: float = max(0.0, min(1.0, t))
        return self._alpha_prime_fn(t_clipped)

    def get_mask_prob(self, t: float) -> float:
        """
        Calculates the masking probability (1 - alpha_t) for a given continuous time t.
        This represents the probability that a token is masked in the forward process.

        Args:
            t (float): A continuous time value between 0.0 and 1.0.

        Returns:
            float: The masking probability (1 - alpha_t).
        """
        return 1.0 - self.get_alpha(t)

    @property
    def discrete_timesteps(self) -> torch.Tensor:
        """
        Generates a torch.Tensor of evenly spaced continuous time values
        from 0.0 to 1.0, representing the discrete steps of the diffusion process.

        Returns:
            torch.Tensor: A tensor of discrete time values.
        """
        # num_steps + 1 points to include both 0.0 and 1.0
        return torch.linspace(0.0, 1.0, self._num_steps + 1)

# Example Usage (for testing purposes, remove in final integration if not needed)
if __name__ == '__main__':
    print("--- Testing Linear Noise Schedule ---")
    linear_scheduler = NoiseScheduler(schedule_type="linear", num_steps=1000)

    # Test alpha_t
    print(f"Linear alpha_t at t=0.0: {linear_scheduler.get_alpha(0.0)}")  # Expected: ~1.0
    print(f"Linear alpha_t at t=0.5: {linear_scheduler.get_alpha(0.5)}")  # Expected: ~0.5
    print(f"Linear alpha_t at t=1.0: {linear_scheduler.get_alpha(1.0)}")  # Expected: ~0.0

    # Test alpha_t_prime
    print(f"Linear alpha_t_prime at t=0.0: {linear_scheduler.get_alpha_prime(0.0)}") # Expected: -1.0
    print(f"Linear alpha_t_prime at t=0.5: {linear_scheduler.get_alpha_prime(0.5)}") # Expected: -1.0

    # Test mask_prob
    print(f"Linear mask_prob at t=0.0: {linear_scheduler.get_mask_prob(0.0)}") # Expected: ~0.0
    print(f"Linear mask_prob at t=0.5: {linear_scheduler.get_mask_prob(0.5)}") # Expected: ~0.5

    # Test discrete_timesteps
    print(f"Linear discrete_timesteps (first 5): {linear_scheduler.discrete_timesteps[:5]}")
    print(f"Linear discrete_timesteps (last 5): {linear_scheduler.discrete_timesteps[-5:]}")
    assert linear_scheduler.discrete_timesteps.shape[0] == 1001

    print("\n--- Testing Cosine Noise Schedule ---")
    cosine_scheduler = NoiseScheduler(schedule_type="cosine", num_steps=500)

    # Test alpha_t
    print(f"Cosine alpha_t at t=0.0: {cosine_scheduler.get_alpha(0.0)}")  # Expected: ~1.0
    print(f"Cosine alpha_t at t=0.5: {cosine_scheduler.get_alpha(0.5)}")  # Expected: ~0.707 (cos(pi/4))
    print(f"Cosine alpha_t at t=1.0: {cosine_scheduler.get_alpha(1.0)}")  # Expected: ~0.0

    # Test alpha_t_prime
    print(f"Cosine alpha_t_prime at t=0.0: {cosine_scheduler.get_alpha_prime(0.0)}") # Expected: ~0.0
    print(f"Cosine alpha_t_prime at t=0.5: {cosine_scheduler.get_alpha_prime(0.5)}") # Expected: ~-1.11 (-(pi/2)*sin(pi/4))
    print(f"Cosine alpha_t_prime at t=1.0: {cosine_scheduler.get_alpha_prime(1.0)}") # Expected: ~-1.57 (-(pi/2)*sin(pi/2))

    # Test mask_prob
    print(f"Cosine mask_prob at t=0.0: {cosine_scheduler.get_mask_prob(0.0)}") # Expected: ~0.0
    print(f"Cosine mask_prob at t=0.5: {cosine_scheduler.get_mask_prob(0.5)}") # Expected: ~0.293

    # Test discrete_timesteps
    print(f"Cosine discrete_timesteps (first 5): {cosine_scheduler.discrete_timesteps[:5]}")
    print(f"Cosine discrete_timesteps (last 5): {cosine_scheduler.discrete_timesteps[-5:]}")
    assert cosine_scheduler.discrete_timesteps.shape[0] == 501

    print("\n--- Testing Edge Cases / Clipping ---")
    print(f"Linear alpha_t at t=-0.1: {linear_scheduler.get_alpha(-0.1)}") # Should clip to 0.0 -> 1.0
    print(f"Linear alpha_t at t=1.1: {linear_scheduler.get_alpha(1.1)}")   # Should clip to 1.0 -> 0.0

    print("\n--- Testing Invalid Schedule Type ---")
    try:
        invalid_scheduler = NoiseScheduler(schedule_type="unknown", num_steps=100)
    except ValueError as e:
        print(f"Caught expected error: {e}")
