"""
Prioritized Generative Replay (PGR)

A framework for scalable, guidable generative replay in online RL.
Uses conditional diffusion models with classifier-free guidance to generate
high-relevance synthetic transitions for policy training.

Paper: "Prioritized Generative Replay" (Wang et al., 2024)
"""

from .diffusion import ConditionalDiffusion, TransitionDenoiser
from .relevance import CuriosityRelevance, ReturnRelevance, TDErrorRelevance, RewardRelevance
from .replay_buffer import ReplayBuffer, NormalizedReplayBuffer
from .redq import REDQAgent, GaussianActor, QNetwork
from .pgr_trainer import PGRTrainer

__all__ = [
    "ConditionalDiffusion",
    "TransitionDenoiser",
    "CuriosityRelevance",
    "ReturnRelevance",
    "TDErrorRelevance",
    "RewardRelevance",
    "ReplayBuffer",
    "NormalizedReplayBuffer",
    "REDQAgent",
    "GaussianActor",
    "QNetwork",
    "PGRTrainer",
]
