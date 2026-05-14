from src.noise_schedule import memoryless_noise_schedule

class FlowMatchingModel(nn.Module):
    def __init__(self, velocity_fn, beta_fn):
        super(FlowMatchingModel, self).__init__()
        self.velocity_fn = velocity_fn
        self.beta_fn = beta_fn

    def forward(self, x, t):
        noise = memoryless_noise_schedule(t, self.beta_fn)
        # Simulate process using noise
        return self.velocity_fn(x, t) + noise

class DenoisingDiffusionModel(nn.Module):
    def __init__(self, noise_predictor, beta_fn):
        super(DenoisingDiffusionModel, self).__init__()
        self.noise_predictor = noise_predictor
        self.beta_fn = beta_fn

    def forward(self, x, t):
        noise = memoryless_noise_schedule(t, self.beta_fn)
        epsilon = self.noise_predictor(x, t)
        return epsilon + noise

