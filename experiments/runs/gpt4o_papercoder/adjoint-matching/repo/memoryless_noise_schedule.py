# memoryless_noise_schedule.py

import numpy as np
from typing import List

class NoiseSchedule:
    """
    NoiseSchedule class responsible for computing memoryless noise schedules 
    and providing them to Trainer or Adjoint Matching modules during fine-tuning.

    Attributes:
        timesteps (int): Number of discrete timesteps for generating the schedule.
        offset (float): Numerical stability offset for noise formula.
        t_values (np.ndarray): Array of discrete timesteps evenly spaced between 0 and 1.
        noise_schedule (List[float]): Precomputed values of memoryless noise schedule.
    """

    def __init__(self, config: dict):
        """
        Initializes the NoiseSchedule using parameters from `config.yaml`.

        Args:
            config (dict): Configuration dictionary containing `training.timesteps` and 
                           `training.noise_schedule_offset`.
        """
        # Load timesteps and offset from the config
        self.timesteps: int = config['training'].get('timesteps', 40)
        self.offset: float = config['training'].get('noise_schedule_offset', 0.025)

        # Generate evenly spaced timesteps t_values
        self.t_values: np.ndarray = np.linspace(0, 1, self.timesteps, endpoint=False)

        # Precompute the noise schedule values
        self.noise_schedule: List[float] = self.generate_schedule()

    def generate_schedule(self) -> List[float]:
        """
        Generates a memoryless noise schedule σ(t) for all discrete timesteps t_k.

        Formula Used:
        σ(t) = sqrt(2 * (1 - t + offset) / (t + offset))

        Handles edge cases numerically near t ~ 0 to prevent σ(t) → ∞.

        Returns:
            List[float]: Noise schedule values for each timestep in `t_values`.
        """
        noise_schedule = []
        for t in self.t_values:
            try:
                # Compute the stabilized memoryless noise schedule formula
                sigma_t = np.sqrt(2 * (1 - t + self.offset) / (t + self.offset))
                noise_schedule.append(sigma_t)
            except ZeroDivisionError:
                # Handle division by zero, although offset prevents this by design
                noise_schedule.append(float('inf'))
        return noise_schedule

    def get_sigma(self, t: float) -> float:
        """
        Retrieves the value of σ(t) at a specific arbitrary timestep t. 
        If t is not a precomputed timestep in `t_values`, interpolates its value.

        Args:
            t (float): The specific time at which to retrieve/interpolate σ(t).

        Returns:
            float: Noise schedule value at time t.
        """
        if t in self.t_values:
            index = np.where(self.t_values == t)[0][0]
            return self.noise_schedule[index]
        else:
            return float(np.interp(t, self.t_values, self.noise_schedule))

    def get_schedule(self) -> List[float]:
        """
        Returns the precomputed noise schedule σ(t) for all timesteps.

        Returns:
            List[float]: Precomputed values of σ(t).
        """
        return self.noise_schedule
