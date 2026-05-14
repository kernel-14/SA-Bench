import numpy as np
import torch
from tensorboardX import SummaryWriter
from typing import Dict, Any, Union

# Score Normalizer Constants from Appendix B.3
# These scores are for the -v4 versions of Gym and -v5 versions of Atari.
_GYM_SCORES: Dict[str, Dict[str, float]] = {
    'Ant-v4': {'random': -70.288, 'td3': 3942.0},
    'HalfCheetah-v4': {'random': -289.415, 'td3': 10574.0},
    'Hopper-v4': {'random': 18.791, 'td3': 3226.0},
    'Humanoid-v4': {'random': 120.423, 'td3': 5165.0},
    'Walker2d-v4': {'random': 2.791, 'td3': 3946.0},
}

_ATARI_SCORES: Dict[str, Dict[str, float]] = {
    'Alien': {'random': 227.8, 'human': 7127.7},
    'Amidar': {'random': 5.8, 'human': 1719.5},
    'Assault': {'random': 222.4, 'human': 742.0},
    'Asterix': {'random': 210.0, 'human': 8503.3},
    'Asteroids': {'random': 719.1, 'human': 47388.7},
    'Atlantis': {'random': 12850.0, 'human': 29028.1},
    'BankHeist': {'random': 14.2, 'human': 753.1},
    'BattleZone': {'random': 2360.0, 'human': 37187.5},
    'BeamRider': {'random': 363.9, 'human': 16926.5},
    'Berzerk': {'random': 123.7, 'human': 2630.4},
    'Bowling': {'random': 23.1, 'human': 160.7},
    'Boxing': {'random': 0.1, 'human': 12.1},
    'Breakout': {'random': 1.7, 'human': 30.5},
    'Centipede': {'random': 2090.9, 'human': 12017.0},
    'ChopperCommand': {'random': 811.0, 'human': 7387.8},
    'CrazyClimber': {'random': 10780.5, 'human': 35829.4},
    # Defender and Surround are noted as 'not used' in Appendix B.3,
    # so they are omitted from this lookup table.
    'DemonAttack': {'random': 152.1, 'human': 1971.0},
    'DoubleDunk': {'random': -18.6, 'human': -16.4},
    'Enduro': {'random': 0.0, 'human': 860.5},
    'FishingDerby': {'random': -91.7, 'human': -38.7},
    'Freeway': {'random': 0.0, 'human': 29.6},
    'Frostbite': {'random': 65.2, 'human': 4334.7},
    'Gopher': {'random': 257.6, 'human': 2412.5},
    'Gravitar': {'random': 173.0, 'human': 3351.4},
    'Hero': {'random': 1027.0, 'human': 30826.4},
    'IceHockey': {'random': -11.2, 'human': 0.9},
    'Jamesbond': {'random': 29.0, 'human': 302.8},
    'Kangaroo': {'random': 52.0, 'human': 3035.0},
    'Krull': {'random': 1598.0, 'human': 2665.5},
    'KungFuMaster': {'random': 258.5, 'human': 22736.3},
    'MontezumaRevenge': {'random': 0.0, 'human': 4753.3},
    'MsPacman': {'random': 307.3, 'human': 6951.6},
    'NameThisGame': {'random': 2292.3, 'human': 8049.0},
    'Phoenix': {'random': 761.4, 'human': 7242.6},
    'Pitfall': {'random': -229.4, 'human': 6463.7},
    'Pong': {'random': -20.7, 'human': 14.6},
    'PrivateEye': {'random': 24.9, 'human': 69571.3},
    'Qbert': {'random': 163.9, 'human': 13455.0},
    'Riverraid': {'random': 1338.5, 'human': 17118.0},
    'RoadRunner': {'random': 11.5, 'human': 7845.0},
    'Robotank': {'random': 2.2, 'human': 11.9},
    'Seaquest': {'random': 68.4, 'human': 42054.7},
    'Skiing': {'random': -17098.1, 'human': -4336.9},
    'Solaris': {'random': 1236.3, 'human': 12326.7},
    'SpaceInvaders': {'random': 148.0, 'human': 1668.7},
    'StarGunner': {'random': 664.0, 'human': 10250.0},
    'Tennis': {'random': -23.8, 'human': -8.3},
    'TimePilot': {'random': 3568.0, 'human': 5229.2},
    'Tutankham': {'random': 11.4, 'human': 167.6},
    'UpNDown': {'random': 533.4, 'human': 11693.2},
    'Venture': {'random': 0.0, 'human': 1187.5},
    'VideoPinball': {'random': 16256.9, 'human': 17667.9},
    'WizardOfWor': {'random': 563.5, 'human': 4756.5},
    'YarsRevenge': {'random': 3092.9, 'human': 54576.9},
    'Zaxxon': {'random': 32.5, 'human': 9173.3},
}


class Logger:
    """A simple wrapper for TensorBoardX SummaryWriter to log training and evaluation metrics."""

    def __init__(self, log_dir: str):
        """Initializes the Logger.

        Args:
            log_dir: The directory where TensorBoard log files will be saved.
        """
        self.writer = SummaryWriter(log_dir)

    def log(self, metrics: Dict[str, Union[float, int, np.ndarray, torch.Tensor]], step: int):
        """Logs a dictionary of metrics to TensorBoard.

        Scalars are logged with add_scalar, and array-like objects with multiple
        elements are logged as histograms with add_histogram.

        Args:
            metrics: A dictionary where keys are metric names (e.g., 'train/loss')
                     and values are the metric's numerical value (scalar or array-like).
            step: The global training step at which to log these metrics.
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, global_step=step)
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                if value.ndim == 0 or (value.ndim == 1 and value.shape[0] == 1):
                    # Scalar tensor/numpy array (e.g., a loss or single TD error)
                    self.writer.add_scalar(key, value.item(), global_step=step)
                else:
                    # Array-like tensor/numpy array (e.g., a batch of TD errors)
                    # Convert to numpy for add_histogram if it's a torch.Tensor
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    self.writer.add_histogram(key, value, global_step=step)
            else:
                print(f"Warning: Metric '{key}' has unsupported type '{type(value)}'. Skipping logging.")

    def close(self):
        """Closes the TensorBoardX SummaryWriter."""
        self.writer.close()


class RewardNormalizer:
    """Tracks the mean of absolute rewards for value target normalization."""

    def __init__(self):
        """Initializes the RewardNormalizer.

        Initializes counters and sum for calculating the current mean,
        and sets the initial target mean to 1.0 as a baseline before
        any rewards are observed, as implied by the paper's "No reward scaling"
        ablation where r = r' = 1.
        """
        self._current_reward_sum_abs: float = 0.0
        self._current_reward_count: int = 0
        self._target_reward_mean_abs: float = 1.0

    def update_mean(self, reward: float):
        """Updates the running sum and count of absolute rewards.

        Args:
            reward: The scalar reward received from an environment step.
        """
        self._current_reward_sum_abs += abs(reward)
        self._current_reward_count += 1

    def get_mean(self) -> float:
        """Calculates and returns the current average absolute reward.

        Returns:
            The current mean of absolute rewards, or 1.0 if no rewards have been recorded yet
            to avoid division by zero and maintain consistency with initial `_target_reward_mean_abs`.
        """
        if self._current_reward_count == 0:
            return 1.0
        return self._current_reward_sum_abs / self._current_reward_count

    def get_target_mean(self) -> float:
        """Returns the stored target average absolute reward.

        This value lags behind the `_current_reward_mean_abs` and is updated
        periodically by `update_target_mean()` to provide a stable scaling factor
        for value targets.

        Returns:
            The target mean of absolute rewards.
        """
        return self._target_reward_mean_abs

    def update_target_mean(self):
        """Updates the target average absolute reward to the current average.

        This method should be called by the `Trainer` periodically, synchronized
        with the target network updates, as specified in the paper (Section 4.2.2).
        """
        self._target_reward_mean_abs = self.get_mean()


class ScoreNormalizer:
    """Provides utilities for normalizing scores across different RL benchmarks.

    Uses static reference scores (random, TD3, human) as provided in Appendix B.3
    of the paper to compute normalized scores.
    """

    def __init__(self, benchmark: str, task_name: str):
        """Initializes the ScoreNormalizer.

        Args:
            benchmark: The name of the benchmark (e.g., "gym", "dmc", "atari").
                       Case-insensitive, converted to lowercase for internal consistency.
            task_name: The specific name of the environment task (e.g., "Ant-v4", "Alien").
                       For Atari, the "ALE/" prefix is removed for lookup.

        Raises:
            ValueError: If the provided benchmark or task_name is not recognized
                        or does not have corresponding reference scores.
        """
        self.benchmark: str = benchmark.lower()
        self.task_name: str = task_name
        self.random_score: float = 0.0  # Default, overridden by benchmark-specific values
        self.ref_score: float = 1.0     # Default, overridden by benchmark-specific values

        if self.benchmark == "gym":
            if self.task_name not in _GYM_SCORES:
                raise ValueError(f"Gym task '{task_name}' not found in reference scores.")
            self.random_score = _GYM_SCORES[self.task_name]['random']
            self.ref_score = _GYM_SCORES[self.task_name]['td3']
        elif self.benchmark == "atari":
            # Atari task names in the lookup table do not contain "ALE/".
            # Strip "ALE/" prefix if present from gymnasium env_name.
            clean_task_name: str = self.task_name.split('/')[-1]
            if clean_task_name not in _ATARI_SCORES:
                raise ValueError(f"Atari task '{clean_task_name}' not found in reference scores.")
            self.random_score = _ATARI_SCORES[clean_task_name]['random']
            self.ref_score = _ATARI_SCORES[clean_task_name]['human']
        elif self.benchmark == "dmc":
            # For DMC, the paper specifies using raw default reward, no specific
            # normalization values are needed here. `random_score` and `ref_score`
            # defaults will not be used in the `normalize_score` method for DMC.
            pass
        else:
            raise ValueError(f"Unsupported benchmark for score normalization: '{benchmark}'")

    def normalize_score(self, raw_score: float) -> float:
        """Normalizes a raw score based on the initialized benchmark and task.

        For Gym, it computes TD3-Normalized score: (raw_score - random) / (TD3 - random).
        For Atari, it computes Human-Normalized score: (raw_score - random) / (Human - random).
        For DMC, it returns the raw score as per the paper's evaluation protocol.

        Args:
            raw_score: The raw score obtained from an evaluation episode.

        Returns:
            The normalized score as a float.
        """
        if self.benchmark == "dmc":
            # For DMC, the paper states that default reward is used,
            # implying no further normalization is applied.
            return raw_score
        
        denominator: float = self.ref_score - self.random_score
        # Handle cases where the denominator is zero or very close to zero
        if abs(denominator) < 1e-6:
            # If reference and random scores are identical, implies no learnable range.
            # A common approach is to return 0.0 (no improvement over random)
            # or the raw_score itself. Sticking with 0.0 for consistency if no
            # meaningful scaling range exists.
            return 0.0

        normalized_score: float = (raw_score - self.random_score) / denominator
        return normalized_score

