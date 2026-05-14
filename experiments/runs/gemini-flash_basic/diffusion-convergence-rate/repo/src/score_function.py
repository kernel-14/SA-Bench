from abc import ABC, abstractmethod

class ScoreFunction(ABC):
    """Abstract base class for score functions."""

    def __init__(self, d: int):
        self.d = d  # Dimension of the data

    @abstractmethod
    def __call__(self, x, t):
        """
        Evaluates the score function at a given point x and time t.

        Args:
            x (np.ndarray): The input data point.
            t (float): The time step.

        Returns:
            np.ndarray: The score at x and t.
        """
        pass

    @abstractmethod
    def get_true_score(self, x, t):
        """
        Evaluates the true (or target) score function at a given point x and time t.
        This method is primarily for theoretical comparison and might not be used in practice.

        Args:
            x (np.ndarray): The input data point.
            t (float): The time step.

        Returns:
            np.ndarray: The true score at x and t.
        """
        pass
