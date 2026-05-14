# Policy Training using Prioritized Generative Replay

import torch
from src.pgr_framework import PrioritizedGenerativeReplay
from src.relevance_functions import curiosity_relevance, return_based_relevance, td_error_relevance

class PolicyTrainer:
    def __init__(self, policy, optimizer):
        self.policy = policy
        self.optimizer = optimizer
        self.pgr = PrioritizedGenerativeReplay()

    def train(self, real_data, relevance_function, guidance_scale):
        # Add real transitions to buffer
        self.pgr.add_to_real_buffer(real_data)

        # Generate synthetic transitions
        synthetic_data = []
        for transition in real_data:
            condition = relevance_function(*transition)
            self.pgr.conditional_generate(condition, guidance_scale)

        # Mix real and synthetic data
        mixed_data = self.pgr.mix_buffers(ratio=0.5)

        # Optimize policy using mixed data
        loss = self.policy.compute_loss(mixed_data)
        self.optimizer.step(loss)


