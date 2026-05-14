# Prioritized Generative Replay Framework

import torch
import numpy as np
from models.diffusion import ConditionalDiffusionModel

class PrioritizedGenerativeReplay:
    def __init__(self, real_buffer_size=1000000, synthetic_buffer_size=1000000):
        # Real and synthetic replay buffers
        self.D_real = []
        self.D_synthetic = []
        self.real_buffer_size = real_buffer_size
        self.synthetic_buffer_size = synthetic_buffer_size

        # Conditional generative model
        self.G = ConditionalDiffusionModel()

    def add_to_real_buffer(self, transitions):
        if len(self.D_real) + len(transitions) > self.real_buffer_size:
            self.D_real = self.D_real[len(transitions):]  # Remove oldest samples
        self.D_real.extend(transitions)

    def add_to_synthetic_buffer(self, transitions):
        if len(self.D_synthetic) + len(transitions) > self.synthetic_buffer_size:
            self.D_synthetic = self.D_synthetic[len(transitions):]
        self.D_synthetic.extend(transitions)

    def conditional_generate(self, relevance_condition, guidance_scale=1.0):
        generated = self.G.sample(condition=relevance_condition, guidance_scale=guidance_scale)
        self.add_to_synthetic_buffer(generated)

    def mix_buffers(self, ratio=0.5):
        real_samples = np.random.choice(self.D_real, int(len(self.D_real) * ratio))
        synthetic_samples = np.random.choice(self.D_synthetic, int(len(self.D_synthetic) * (1 - ratio)))
        return np.concatenate([real_samples, synthetic_samples])

    def train_policy(self, policy, optimizer):
        mixed_data = self.mix_buffers()
        loss = policy.compute_loss(mixed_data)
        optimizer.step(loss)


