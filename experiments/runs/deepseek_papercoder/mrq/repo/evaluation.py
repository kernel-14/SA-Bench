# evaluation.py

"""
Evaluation module for the MR.Q algorithm.

Implements the `Evaluation` class, which collects undiscounted episode returns
using a deterministic policy (no exploration noise) and returns summary statistics
(mean, median, standard deviation, raw returns).

Used by the `Trainer` class during periodic evaluations and at the end of training.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from gymnasium import Env

from config import Config
from agent import MRQAgent


class Evaluation:
    """
    Evaluation runner for MR.Q.

    Given an already pre‑processed environment and a trained agent, this class
    executes a fixed number of episodes with deterministic actions.
    All agent modules are set to evaluation mode and gradient computation is
    temporarily disabled.

    Parameters
    ----------
    cfg : Config
        Global configuration (used to obtain the default number of episodes
        from `cfg.evaluation.num_episodes`).
    env : Env
        A Gymnasium environment that has been fully wrapped with the necessary
        preprocessing (action repeat, frame stacking, sticky actions, etc.).
        This environment should be separate from the training environment and
        should **not** end episodes on life loss for Atari.
    agent : MRQAgent
        The MR.Q agent whose policy will be evaluated. Its networks are
        temporarily switched to evaluation mode and are restored afterwards.
    """

    def __init__(self, cfg: Config, env: Env, agent: MRQAgent) -> None:
        """
        Initialise the evaluation runner.

        Parameters
        ----------
        cfg : Config
            Configuration containing evaluation settings.
        env : Env
            Evaluation environment.
        agent : MRQAgent
            Agent to evaluate.
        """
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.default_episodes = cfg.evaluation.num_episodes

    def run(self, num_episodes: Optional[int] = None) -> Dict[str, object]:
        """
        Run evaluation episodes and return performance statistics.

        Parameters
        ----------
        num_episodes : int, optional
            Number of episodes to run. If ``None``, the value from
            ``cfg.evaluation.num_episodes`` is used.

        Returns
        -------
        dict
            A dictionary with the following keys:
                - ``"mean"``   : float, average undiscounted return.
                - ``"median"`` : float, median undiscounted return.
                - ``"std"``    : float, standard deviation of the returns.
                - ``"returns"``: list of float, raw return of each episode.
        """
        if num_episodes is None:
            num_episodes = self.default_episodes

        # Set all agent modules to evaluation mode and disable gradient computation
        with torch.no_grad():
            self.agent.encoder.eval()
            self.agent.q1.eval()
            self.agent.q2.eval()
            self.agent.policy.eval()

            returns: List[float] = []
            for _ in range(num_episodes):
                obs, info = self.env.reset()
                obs = np.asarray(obs, dtype=np.float32)
                done = False
                episode_return = 0.0

                while not done:
                    # Deterministic action from the policy (no exploration noise)
                    action = self.agent.select_action(obs, explore=False, step=0)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    obs = np.asarray(obs, dtype=np.float32)
                    episode_return += reward
                    done = terminated or truncated

                returns.append(episode_return)

            # Restore training mode
            self.agent.encoder.train()
            self.agent.q1.train()
            self.agent.q2.train()
            self.agent.policy.train()

        # Compute simple statistics
        returns_arr = np.asarray(returns, dtype=np.float32)
        mean_ret = float(np.mean(returns_arr))
        median_ret = float(np.median(returns_arr))
        std_ret = float(np.std(returns_arr))

        return {
            "mean": mean_ret,
            "median": median_ret,
            "std": std_ret,
            "returns": returns,
        }
