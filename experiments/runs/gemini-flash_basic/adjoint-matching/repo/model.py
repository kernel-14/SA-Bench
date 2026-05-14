import math

class BaseModel:
    def __init__(self):
        pass

    def alpha_t(self, t: float) -> float:
        raise NotImplementedError

    def beta_t(self, t: float) -> float:
        raise NotImplementedError

    def dot_alpha_t(self, t: float) -> float:
        raise NotImplementedError

    def dot_beta_t(self, t: float) -> float:
        raise NotImplementedError
    
    def kappa_t(self, t: float) -> float:
        dot_alpha = self.dot_alpha_t(t)
        alpha = self.alpha_t(t)
        if alpha == 0:
            # Handle potential division by zero at t=0 for FM.
            # As per context in paper (e.g. line 231), sigma(t) is infinite at t=0,
            # this implies related terms might also be singular.
            # We'll return a large number or handle it contextually if it appears in actual use.
            # For now, let's return a very large float as a placeholder.
            return float('inf') if dot_alpha != 0 else 0.0
        return dot_alpha / alpha

    def eta_t(self, t: float) -> float:
        dot_alpha = self.dot_alpha_t(t)
        alpha = self.alpha_t(t)
        beta = self.beta_t(t)
        dot_beta = self.dot_beta_t(t)
        
        # Handle division by zero for alpha_t if it occurs
        if alpha == 0:
            return float('inf') # Or some other appropriate handling
        
        return beta * ((dot_alpha / alpha) * beta - dot_beta)

    def memoryless_sigma_t(self, t: float) -> float:
        eta = self.eta_t(t)
        # Sigma(t) is sqrt(2 * eta_t)
        # Need to ensure eta is non-negative for sqrt.
        # If eta becomes negative, it indicates a problem with the definition or context.
        if eta < 0:
            raise ValueError(f"eta_t is negative ({eta}) at time {t}, cannot compute real sigma_t.")
        return math.sqrt(2 * eta)

class FlowMatchingModel(BaseModel):
    def alpha_t(self, t: float) -> float:
        return t

    def beta_t(self, t: float) -> float:
        return 1.0 - t

    def dot_alpha_t(self, t: float) -> float:
        return 1.0

    def dot_beta_t(self, t: float) -> float:
        return -1.0
    
    def kappa_t(self, t: float) -> float:
        # From Table 1: alpha_t / alpha_t = 1/t
        if t == 0:
            return float('inf')
        return 1.0 / t

    def eta_t(self, t: float) -> float:
        # From Table 1: (1-t)/t
        if t == 0:
            return float('inf')
        return (1.0 - t) / t

class DDPMModel(BaseModel):
    # For DDPM, we need a schedule alpha_bar_t
    # The paper uses alpha_bar_t as a base for calculations in Table 1
    # We will assume a default linear schedule for alpha_bar_t for now, 
    # as the paper doesn't specify it in the main text for DDPM directly, 
    # but uses it as a placeholder.
    # In practice, alpha_bar_t is often a predefined schedule.
    # For this exercise, we will assume a simple linear schedule for alpha_bar_t
    # as alpha_bar_K = 1, alpha_bar_0 = 0 from DDIM description (line 73)
    # and "uniform discretization of time, i.e. t = k/K" (line 82) implies alpha_bar_t is proportional to t.
    # A common simple schedule is alpha_bar_t = t.
    # However, this leads to issues when taking derivative, specifically dot_alpha_bar_t / alpha_bar_t.
    # Let's consider a schedule like alpha_bar_t = exp(-f(t)) or a cosine schedule from literature.
    # Since the paper doesn't specify, I will make a reasonable assumption.
    # Let's assume a function `alpha_bar_schedule(t)` is provided externally or to be defined later.
    # For now, I will define a placeholder for alpha_bar_t and its derivative.
    
    # Placeholder for the alpha_bar_t schedule
    def alpha_bar_t_schedule(self, t: float) -> float:
        # This is a placeholder. A real implementation would use specific alpha_bar schedules.
        # Common schedules are often sigmoidal or cosine-based.
        # For simplicity and to avoid division by zero issues at t=0 when dot_alpha_bar_t / alpha_bar_t is computed,
        # let's assume a simple schedule like t itself, but with a small epsilon.
        # Or more accurately, often a cosine schedule is used.
        # Since the paper doesn't specify, I'll use a placeholder that can be easily changed.
        # A common practice is to define it such that alpha_bar_0 is close to 0 and alpha_bar_1 is 1.
        # Let's assume a linear increase from a small epsilon to 1.
        # Or, a more common schedule is a cosine schedule. Since there is no concrete definition,
        # I will leave this as an abstract method and highlight this assumption in the README.
        raise NotImplementedError("alpha_bar_t_schedule must be implemented by a concrete DDPM model.")

    def dot_alpha_bar_t_schedule(self, t: float) -> float:
        raise NotImplementedError("dot_alpha_bar_t_schedule must be implemented by a concrete DDPM model.")

    def alpha_t(self, t: float) -> float:
        # From Line 85, for DDIM: alpha_t = sqrt(alpha_bar_t_schedule)
        return math.sqrt(self.alpha_bar_t_schedule(t))

    def beta_t(self, t: float) -> float:
        # From Line 85, for DDIM: beta_t = sqrt(1 - alpha_bar_t_schedule)
        return math.sqrt(1.0 - self.alpha_bar_t_schedule(t))
    
    def kappa_t(self, t: float) -> float:
        # From Table 1: dot_alpha_bar_t / alpha_bar_t
        alpha_bar = self.alpha_bar_t_schedule(t)
        if alpha_bar == 0:
            return float('inf')
        return self.dot_alpha_bar_t_schedule(t) / alpha_bar

    def eta_t(self, t: float) -> float:
        # From Table 1: dot_alpha_bar_t / (2 * alpha_bar_t)
        alpha_bar = self.alpha_bar_t_schedule(t)
        if alpha_bar == 0:
            return float('inf')
        return self.dot_alpha_bar_t_schedule(t) / (2.0 * alpha_bar)

    # For DDPM, memoryless_sigma_t is sqrt(dot_alpha_bar_t / alpha_bar_t)
    # which is sqrt(2 * eta_t) as eta_t is dot_alpha_bar_t / (2 * alpha_bar_t)
    # So the base class implementation of memoryless_sigma_t will work.

# Example of a concrete DDPM model with a simple linear alpha_bar_t schedule for demonstration
# In a real scenario, this would be based on the specific DDPM variant.
class LinearAlphaBarDDPMModel(DDPMModel):
    def alpha_bar_t_schedule(self, t: float) -> float:
        # Very simple linear schedule. Often schedules have a small non-zero start.
        # This schedule might cause issues for kappa_t and eta_t at t=0 as alpha_bar_t would be 0.
        # A more robust schedule might be needed for actual training.
        return t

    def dot_alpha_bar_t_schedule(self, t: float) -> float:
        return 1.0

# Another common schedule for alpha_bar_t is a cosine schedule, which avoids alpha_bar_t=0 at t=0.
# Example of a cosine schedule, adapted from typical diffusion models.
# alpha_bar_t = cos((t + s) / (1 + s) * pi / 2)^2
# where s is a small offset.
class CosineAlphaBarDDPMModel(DDPMModel):
    def __init__(self, s: float = 0.008):
        super().__init__()
        self.s = s
        self.max_beta = math.cos(self.s / (1.0 + self.s) * math.pi / 2.0)**2
        
    def alpha_bar_t_schedule(self, t: float) -> float:
        term = (t + self.s) / (1.0 + self.s) * math.pi / 2.0
        return (math.cos(term)**2) / self.max_beta

    def dot_alpha_bar_t_schedule(self, t: float) -> float:
        term = (t + self.s) / (1.0 + self.s) * math.pi / 2.0
        # Derivative of cos(u)^2 is 2*cos(u)*(-sin(u))*du/dt = -sin(2u)*du/dt
        du_dt = 1.0 / (1.0 + self.s) * math.pi / 2.0
        return (-math.sin(2.0 * term) * du_dt) / self.max_beta

