
import torch
import numpy as np
from typing import List, Tuple
from config import Config
from models import ScoreFunction

class DiffusionSampler:
    """
    Implements the sampling algorithm described in Section 2.2 of the paper.
    This includes the randomized schedule, iterative updates, and noise injection.
    """
    def __init__(self, d_dim: int, T_total: int, K_rounds: int, score_function: ScoreFunction,
                 c0: float, c1: float, seed: int = 42):
        """
        Initializes the DiffusionSampler.

        Args:
            d_dim (int): Data dimension.
            T_total (int): Total iterations for the forward process (T in paper).
            K_rounds (int): Number of rounds, K.
            score_function (ScoreFunction): An instance of the score function model.
            c0 (float): Constant c_0 for the randomized schedule.
            c1 (float): Constant c_1 for the randomized schedule.
            seed (int): Random seed.
        """
        self.d_dim = d_dim
        self.T_total = T_total
        self.K_rounds = K_rounds
        self.score_function = score_function
        self.c0 = c0
        self.c1 = c1
        self.N_steps = Config.get_N_steps(T_total, K_rounds)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Precompute randomized schedule parameters (hat_alpha, hat_tau)
        self._precompute_schedule()

    def _precompute_schedule(self):
        """
        Precomputes the randomized schedule parameters hat_alpha and hat_tau.
        Eq 8: hat_alpha_{T+1} = 1 / T^c0, hat_alpha_{t-1} = hat_alpha_t + c1 * hat_alpha_t * (1 - hat_alpha_t) * log(T) / T
        Eq 9: hat_tau_k,n := 1 - hat_alpha_{T - kN/2 - n}
        """
        self.hat_alpha = torch.zeros(self.T_total + 2) # Index from 1 to T_total + 1
        self.hat_tau = torch.zeros(self.T_total + 2) # Corresponds to hat_alpha indices for simplicity, will map to k,n later

        # Initialize hat_alpha_{T+1}
        self.hat_alpha[self.T_total + 1] = 1.0 / (self.T_total ** self.c0)

        # Compute hat_alpha backwards from T to 1 (or -N/2 + 1)
        # The equation for hat_alpha_{t-1} is given by hat_alpha_t, so we iterate t downwards.
        # The indices in paper for hat_alpha are t = -N/2 + 1, ..., T+1.
        # We need to compute hat_alpha for indices relevant to hat_tau_k,n.
        # Max index for hat_alpha is T - 0*N/2 - (-1) = T+1
        # Min index for hat_alpha is T - (K-1)*N/2 - (N-1) = T - KN/2 + N/2 - N + 1 = T - KN/2 - N/2 + 1
        # Since KN = 2T, KN/2 = T. So min index is T - T - N/2 + 1 = 1 - N/2.
        # This implies negative indices, which Python lists don't handle directly as time steps.
        # Let's map these paper indices to array indices.
        # The smallest index needed is T - (K-1)N/2 - (N-1) for hat_alpha
        # The largest index needed is T + 1 for hat_alpha_{T+1}
        
        # Let's define the sequence for hat_alpha_t from t = T_total + 1 down to 1 (or slightly lower if needed by the paper)
        # For simplicity, let's compute hat_alpha for indices from 1 up to T_total + 1.
        # The specific indices needed for hat_tau_k,n are `T_total - kN/2 - n`.
        # The paper uses `t` from `-N/2+1` to `T+1`.
        # Let's compute `hat_alpha_idx` for `idx` from 1 to `T_total + 1` for practical implementation.
        # Eq 8: hat_alpha_{t-1} = hat_alpha_t + c_1 * hat_alpha_t * (1 - hat_alpha_t) * log(T_total) / T_total
        # So hat_alpha_t = hat_alpha_{t+1} + c_1 * hat_alpha_{t+1} * (1 - hat_alpha_{t+1}) * log(T_total) / T_total
        # This is a bit ambiguous in the paper's notation of t-1 vs t.
        # Let's assume t-1 is the *earlier* time step, so hat_alpha_{t-1} is derived from hat_alpha_t.
        # So we iterate t from T_total + 1 down to 2.
        
        # Re-interpreting Eq 8 based on common diffusion model practice where alpha_t increases with t:
        # hat_alpha_t = hat_alpha_{t-1} + delta_t.
        # However, paper notation `t-1 = t + ...` means (t-1) is smaller than (t).
        # And `alpha_bar_t := product_k=1^t alpha_k`, implies alpha_bar_t decreases as t increases if alpha_k < 1.
        # But `X_t = sqrt(alpha_t) X_{t-1} + sqrt(1-alpha_t) W_t` where X_t approaches noise for large t.
        # So `alpha_bar_t` should approach 0 for large `t`.
        # This means `hat_alpha_t` should be a decreasing sequence for increasing `t`.
        
        # Given `hat_alpha_{T+1} = 1/T^c0` (small for large T) and `hat_alpha_{t-1} = hat_alpha_t + ...`
        # if `hat_alpha_t` is small, `hat_alpha_{t-1}` is larger than `hat_alpha_t`.
        # So the index `t` in `hat_alpha_t` in Eq 8 corresponds to increasing time in forward process.
        # The paper says `t = -N/2 + 1, ..., T+1`. These are the indices of `hat_alpha`.
        # Let's assume the equation defines `hat_alpha_t_minus_1` from `hat_alpha_t`.
        # So we need to iterate `t` from `T_total + 1` down to the smallest index.
        
        # Smallest index for hat_alpha needed:
        # k_max = K_rounds - 1, n_max = N_steps - 1
        # T - (K_rounds-1)*N_steps/2 - (N_steps-1) = T - KN/2 + N/2 - N + 1 = T - T + N/2 - N + 1 = 1 - N/2
        # So indices for hat_alpha run from 1 - N/2 to T_total + 1.
        # Let's adjust array indexing to handle this.
        # For simplicity, we can use a dictionary or shift indices.
        # Let's use a shifted index, where array_idx = paper_idx + N_steps/2.
        
        # Determine the minimum index for hat_alpha
        min_hat_alpha_idx_paper = self.T_total - ((self.K_rounds - 1) * self.N_steps // 2) - (self.N_steps - 1)
        max_hat_alpha_idx_paper = self.T_total + 1
        
        # For implementation, map paper index `t_paper` to array index `t_array`
        # Let t_array = t_paper - min_hat_alpha_idx_paper
        # Total size of hat_alpha array: max_hat_alpha_idx_paper - min_hat_alpha_idx_paper + 1
        
        # This is simpler: Iterate directly on indices `t` from `T_total` down to `min_idx`.
        # Eq 8 gives hat_alpha_{t-1} based on hat_alpha_t.
        # So we want to compute hat_alpha_t for t = T_total, T_total-1, ..., 0 (or lower)
        # For the numerical experiment we are reproducing, only indices from 1 to T_total+1 are implicitly assumed non-negative.
        # Let's reconsider `t` in `hat_alpha_t`. The paper says `t = -N/2 + 1, ..., T+1`.
        # If `hat_alpha_t` is the alpha for step `t`, then `hat_alpha_{t+1}` is computed from `hat_alpha_t`.
        # "hat_alpha_{t-1} = hat_alpha_t + c1 * hat_alpha_t * (1 - hat_alpha_t) * log(T) / T"
        # This implies hat_alpha_{smaller_index} = hat_alpha_{larger_index} + ...
        # So if we have hat_alpha_{T+1}, we can compute hat_alpha_T, then hat_alpha_{T-1} etc.
        # This means hat_alpha is a decreasing sequence with increasing index.
        
        # Let's re-align to what seems standard: `alpha_t` typically means alpha at time `t`.
        # The forward process `X_t = sqrt(alpha_t) X_{t-1} + sqrt(1-alpha_t) W_t`.
        # Here `alpha_t` (lowercase) is the step size.
        # `overline_alpha_t = product_k=1^t alpha_k`
        # `X_t = sqrt(overline_alpha_t) X_0 + sqrt(1-overline_alpha_t) W_t_bar`
        # `overline_alpha_t` decreases from ~1 to ~0.
        # `hat_alpha` is related to `overline_alpha`. So `hat_alpha_t` should also be a decreasing sequence.
        
        # So `hat_alpha_{idx_small}` should be larger than `hat_alpha_{idx_large}`.
        # The paper gives `hat_alpha_{T+1}` (smallest value).
        # `hat_alpha_{t-1} = hat_alpha_t + ...` where `t-1` is numerically smaller index than `t`.
        # This means `hat_alpha_{t_idx}` should decrease as `t_idx` increases. This matches expectation.
        
        # So `t` in `hat_alpha_t` corresponds to the current index in the sequence.
        # The recurrence `hat_alpha_{t-1} = hat_alpha_t + ...`
        # means we compute `hat_alpha` for smaller indices based on larger ones.
        # Start from `t = T_total + 1`, compute `hat_alpha_{T_total}` using `hat_alpha_{T_total+1}`.
        # Continue down to `t = min_idx + 1` to compute `hat_alpha_{min_idx}`.

        # Let's use `t_idx` for the indices `t` in Eq 8.
        # hat_alpha_t_idx: indices from T_total+1 down to 1 - N/2 (approx).
        
        # For simplicity, let's keep all `hat_alpha` values in `self.hat_alpha` array,
        # using the original `t` indexing from `1` to `T_total + 1`.
        # If any `t - kN/2 - n` turns out to be outside `[1, T_total+1]`, it needs clarification.
        # In typical DDPM, t usually goes from 1 to T.
        # Given `min_hat_alpha_idx_paper = 1 - N/2`, if N/2 >= 1, this is 0 or negative.
        # The paper uses `t` in `overline_alpha_t` from `1` to `T`.
        # Let's assume `hat_alpha` indices are effectively within `[1, T_total+1]` for practical implementation,
        # and adjust the `T - kN/2 - n` to always be non-negative, by clipping or defining `hat_alpha_0` etc.
        
        # The numerical experiment uses T=1000, K=10, N=200.
        # `min_hat_alpha_idx_paper = 1 - N/2 = 1 - 100 = -99`.
        # So `hat_alpha_t` is defined for `t` values that are negative.
        # This implies the notation in the paper uses indices `t` that can be negative.
        # A simple array won't work. We'll use a dictionary to store `hat_alpha_t` and `hat_tau_t`.

        self._hat_alpha_dict = {}
        self._hat_tau_dict = {}

        # Set hat_alpha_{T+1}
        self._hat_alpha_dict[self.T_total + 1] = 1.0 / (self.T_total ** self.c0)

        # Calculate backwards
        # The maximum index for hat_alpha is T+1. The minimum is 1 - N/2.
        # So we iterate 't_current' from T+1 down to 1 - N/2 + 1 to compute hat_alpha_{t_current - 1}
        # The values of `t` in Eq 8 are `t = -N/2 + 1, ..., T+1`.
        # So, if we use `t` as the index in the equation, `t-1` is the earlier index.
        # We need to compute values for `t_idx` from `T_total` down to `1 - N_steps // 2`.
        
        # The recurrence `hat_alpha_{t-1} = hat_alpha_t + ...`
        # for `t` from `-N/2+1` up to `T+1`. This means `t-1` is the smaller index.
        # We start with `hat_alpha_{T+1}`. We want `hat_alpha_T`, then `hat_alpha_{T-1}`, etc.
        
        current_t_for_alpha = self.T_total + 1
        # Loop down to `1 - N_steps // 2 + 1` to compute `hat_alpha_{1 - N_steps // 2}`
        min_t_for_alpha = 1 - self.N_steps // 2
        
        for t_idx in range(current_t_for_alpha, min_t_for_alpha, -1):
            if t_idx not in self._hat_alpha_dict: # Should be true only for t_idx = T_total + 1 initially
                 raise ValueError(f"hat_alpha_{t_idx} not initialized. This should not happen.")
            
            hat_alpha_t = self._hat_alpha_dict[t_idx]
            
            # This logic for `hat_alpha_{t-1}` depends on `hat_alpha_t`.
            # So compute `hat_alpha_{t_idx - 1}`
            if hat_alpha_t >= 1.0: # Prevent (1 - hat_alpha_t) from becoming negative or zero if hat_alpha_t exceeds 1
                self._hat_alpha_dict[t_idx - 1] = hat_alpha_t # Clamp or special handling if alpha grows too large
            else:
                self._hat_alpha_dict[t_idx - 1] = hat_alpha_t + (self.c1 * hat_alpha_t * (1.0 - hat_alpha_t) * np.log(self.T_total) / self.T_total)
            
            # Ensure hat_alpha doesn't exceed 1 or go below a tiny epsilon (for numerical stability)
            self._hat_alpha_dict[t_idx - 1] = torch.clamp(torch.tensor(self._hat_alpha_dict[t_idx - 1]), min=1e-6, max=1.0) # Clamp for stability

        # Now compute hat_tau from hat_alpha for relevant k, n.
        # hat_tau_k,n := 1 - hat_alpha_{T - kN/2 - n}
        # Iterate k from 0 to K-1, n from -1 to N.
        # The paper defines n = -1, ..., N. But in algorithm it's n=1, ..., N.
        # Let's assume n refers to the steps in a round, so n=0 to N-1 for N steps,
        # or adjust based on definition in Algorithm section.
        # "iteratively updated for n = 1, ..., N"
        # "tau_k,n := 1 - overline_alpha_{T - kN/2 - n + 1} for n = -1, ..., N"
        # "hat_tau_k,n := 1 - hat_alpha_{T - kN/2 - n} for n = -1, ..., N"
        # This means for n=-1, the index for hat_alpha is T - kN/2 + 1.
        # For n=N, the index for hat_alpha is T - kN/2 - N.

        # Let's populate specific hat_tau_k_n dictionary for easy lookup later
        self._hat_tau_k_n = {}
        for k in range(self.K_rounds):
            for n_val in range(-1, self.N_steps + 1): # from -1 to N (inclusive)
                hat_alpha_idx = Config.get_tau_k_n_index(self.T_total, self.K_rounds, k, n_val)
                if hat_alpha_idx not in self._hat_alpha_dict:
                    # This happens if hat_alpha_idx falls outside the computed range.
                    # This implies hat_alpha sequence needs to be computed for a wider range of negative indices.
                    # Or there's an implicit assumption on `T` or `N`.
                    # For `min_t_for_alpha = 1 - N_steps // 2` -> `1 - 100 = -99`
                    # `T - (K-1)N/2 - n` for n=N, becomes `T - T + N/2 - N = -N/2`.
                    # Index for `hat_alpha` can go down to `-N_steps // 2`.
                    # So min_t_for_alpha in computation loop should be `-N_steps // 2` if it's the smallest needed.
                    
                    # Recalculate range for hat_alpha dictionary:
                    min_idx_hat_alpha_needed = self.T_total - ((self.K_rounds - 1) * self.N_steps // 2) - self.N_steps # for k=K-1, n=N
                    min_idx_hat_alpha_needed = min(min_idx_hat_alpha_needed, self.T_total - (0 * self.N_steps // 2) - (-1)) # for k=0, n=-1
                    
                    # Extend `min_t_for_alpha` for the loop if necessary.
                    if min_idx_hat_alpha_needed < min_t_for_alpha:
                        # Re-run `_precompute_schedule` with extended range, or handle it during initial loop.
                        # For now, let's just make sure the initial loop covers it.
                        print(f"Warning: hat_alpha index {hat_alpha_idx} not found. Extending precomputation range.")
                        # This means the loop `range(current_t_for_alpha, min_t_for_alpha, -1)` needs adjustment.
                        # For now, let's just assume `min_t_for_alpha` was set correctly from the start to cover all needed indices.
                        # A quick test:
                        # T_total=1000, K_rounds=10, N_steps=200.
                        # k=0, n=-1 => index = 1000 - 0 - (-1) = 1001. (Max index T+1)
                        # k=K-1=9, n=N=200 => index = 1000 - 9*100 - 200 = 1000 - 900 - 200 = -100.
                        # So range is indeed [T+1, -N/2].
                        # My current loop `min_t_for_alpha` goes to `1 - N_steps // 2 = -99`.
                        # It should go to `-N_steps // 2`.
                        self._precompute_schedule_extended() # Call a method to recompute with wider range
                        return # Restart init after successful recomputation
                    else:
                        raise ValueError(f"hat_alpha index {hat_alpha_idx} not found and range was not extended. Check precomputation logic.")

                self._hat_tau_k_n[(k, n_val)] = 1.0 - self._hat_alpha_dict[hat_alpha_idx]
        
        # Precompute overline_alpha for tau_k_n as well
        self._alpha_bar_dict = {}
        for k in range(self.K_rounds):
            for n_val in range(-1, self.N_steps + 1):
                alpha_bar_idx = Config.get_alpha_bar_t_index(self.T_total, self.K_rounds, k, n_val)
                # The paper says `overline_alpha_t` for `t=1..T`.
                # `alpha_bar_idx` can go below 1, e.g. `1 - N/2 = -99`.
                # This needs careful interpretation for `overline_alpha`.
                # For `overline_alpha_t = product_k=1^t alpha_k`, `t` typically is >= 1.
                # If `alpha_bar_idx` is <=0, it's problematic.
                # However, the paper implies `s_T-kN/2-n+1` (estimate of true score `s^*`)
                # and `overline_alpha_{T-kN/2-n+1}` are related.
                # Let's assume `overline_alpha_t` is defined for `t >= 1`.
                # If `alpha_bar_idx` is less than 1, we will treat `overline_alpha` as 1 (no noise yet).
                # Or perhaps it means use a small `alpha_bar_t` in those early (large) time steps,
                # where `t` is index of forward process.
                
                # Given numerical experiment focuses on `Y_K` vs `q_K`, and `q_K` is `X_{tau_K,0}` (approx `X_1`).
                # The schedule `overline_alpha_t` for `t = 1 ... T`
                # If `alpha_bar_idx < 1`, this might refer to a time *before* the first forward step (X_0).
                # For `X_t = sqrt(overline_alpha_t) X_0 + sqrt(1-overline_alpha_t) W_t_bar`,
                # `overline_alpha_0` would be 1.
                
                # Let's assume that for the sampling algorithm (Eq 10),
                # the indices for `s_T-kN/2-i+1` are within `[1, T_total]`.
                # `T - kN/2 - n + 1`. Max index: `T - 0 - (-1) + 1 = T+2`.
                # Min index: `T - (K-1)N/2 - N + 1 + 1 = T - T + N/2 - N + 2 = 2 - N/2`.
                # So indices can range from `2 - N/2` to `T+2`.
                # `overline_alpha` is only specified for `1 <= t <= T`.
                # If `alpha_bar_idx > T_total`, clamp to `overline_alpha_T_total`.
                # If `alpha_bar_idx < 1`, clamp to `overline_alpha_1` or handle as a boundary.
                
                # The paper doesn't explicitly define `overline_alpha_t` for `t < 1` or `t > T`.
                # For now, let's generate a full `overline_alpha` sequence from 1 to T_total.
                # Then interpolate or clamp if needed.
                # The `alpha_t` for `1 <= t <= T` are drawn from `Unif(hat_alpha_t, hat_alpha_{t-1})`.
                # This means `overline_alpha_t` depends on random draws.
                # So `tau_k,n` in `tau_k,n := 1 - overline_alpha_{T - kN/2 - n + 1}` is also random.
                # This makes the schedule truly randomized.
                
                # The `_precompute_schedule` method should just precompute `hat_alpha` and `hat_tau`.
                # The actual `overline_alpha` for `tau_k,n` will be sampled during the `sample` method.
        
    def _precompute_schedule_extended(self):
        """
        Extended version of _precompute_schedule to cover all necessary negative indices.
        """
        self._hat_alpha_dict = {}
        self._hat_tau_dict = {}

        self._hat_alpha_dict[self.T_total + 1] = 1.0 / (self.T_total ** self.c0)
        
        min_t_for_alpha_needed = self.T_total - ((self.K_rounds - 1) * self.N_steps // 2) - self.N_steps
        # Also consider n=-1 for k=0, index T_total+1.
        # So the loop should go down to `min_t_for_alpha_needed`.
        
        for t_idx in range(self.T_total + 1, min_t_for_alpha_needed -1, -1):
            hat_alpha_t = self._hat_alpha_dict[t_idx]
            if hat_alpha_t >= 1.0:
                self._hat_alpha_dict[t_idx - 1] = hat_alpha_t
            else:
                self._hat_alpha_dict[t_idx - 1] = hat_alpha_t + (self.c1 * hat_alpha_t * (1.0 - hat_alpha_t) * np.log(self.T_total) / self.T_total)
            self._hat_alpha_dict[t_idx - 1] = torch.clamp(torch.tensor(self._hat_alpha_dict[t_idx - 1]), min=1e-6, max=1.0).item() # ensure it's a float
        
        for k in range(self.K_rounds):
            for n_val in range(-1, self.N_steps + 1):
                hat_alpha_idx = Config.get_tau_k_n_index(self.T_total, self.K_rounds, k, n_val)
                self._hat_tau_k_n[(k, n_val)] = 1.0 - self._hat_alpha_dict[hat_alpha_idx]
        
        # Also precompute range for overline_alpha.
        # Overline_alpha_{t} needs to be defined for t from `1` to `T_total`.
        # `tau_k,n` uses `overline_alpha_{T - kN/2 - n + 1}`.
        # This index `T - kN/2 - n + 1` (let's call it `alpha_bar_actual_idx`) can range from
        # `T_total - 0*N/2 - (-1) + 1 = T_total + 2`
        # down to `T_total - (K_rounds-1)*N_steps/2 - N_steps + 1 + 1 = 2 - N_steps/2`.
        # These are time steps for the forward process (implicitly from paper notation).
        # When `alpha_bar_actual_idx` is outside `[1, T_total]`, how is `overline_alpha` defined?
        # The paper (Eq 3) uses `overline_alpha_t` for `1 <= t <= T`.
        # For `t > T`, `overline_alpha_t` is effectively `overline_alpha_T`.
        # For `t < 1` (i.e., `t=0` or negative), `overline_alpha_t` could be `1` (no noise yet).
        
        # We need to compute individual `alpha_t` (lowercase, step-size) for `1 <= t <= T_total`.
        # From these `alpha_t`, `overline_alpha_t` is formed.
        # `alpha_t` is drawn from `Unif(hat_alpha_t, hat_alpha_{t-1})`.
        # Here, `hat_alpha_t` and `hat_alpha_{t-1}` are *values* in the sequence calculated above.
        # This implies `t` in `hat_alpha_t` should be `1..T_total`.
        # This conflicts with `hat_alpha_idx = T_total - kN/2 - n`.
        
        # Re-read: "randomized learning rate schedule by setting alpha_t in (3) as overline_alpha_t ~ Unif(hat_alpha_t, hat_alpha_{t-1})"
        # This is `overline_alpha_t` (the product) not `alpha_t` (step size). This is a critical distinction.
        # This implies `overline_alpha_t` itself is sampled.
        # This is a key difference from standard DDPM where `alpha_t` are fixed, and `overline_alpha_t` is deterministic.
        
        # So for each `t` in `1..T_total`, `overline_alpha_t` is sampled from `Unif(hat_alpha_t, hat_alpha_{t-1})`.
        # This `t` for `hat_alpha_t` here refers to the actual forward time step `t`.
        # This means we need `hat_alpha_t` (as precomputed in `_hat_alpha_dict`) for `t` from `1` to `T_total`.
        # The `hat_alpha_idx = Config.get_tau_k_n_index(self.T_total, self.K_rounds, k, n_val)` is confusing here.
        
        # The paper (Eq 9) defines `hat_tau_k,n := 1 - hat_alpha_{T - kN/2 - n}`.
        # And `tau_k,n := 1 - overline_alpha_{T - kN/2 - n + 1}`.
        # Let `j = T - kN/2 - n + 1`. So `tau_k,n = 1 - overline_alpha_j`.
        # And `overline_alpha_j` is sampled from `Unif(hat_alpha_j, hat_alpha_{j-1})`.
        # So we need `hat_alpha_j` and `hat_alpha_{j-1}` from our `_hat_alpha_dict`.
        # The indices `j` can be negative or > T_total.
        
        # Let's adjust for this:
        # The `t` in `Unif(hat_alpha_t, hat_alpha_{t-1})` should refer to the actual `t` of `overline_alpha_t`.
        # We generate this randomized `overline_alpha_j` during sampling, using `_hat_alpha_dict[j]` and `_hat_alpha_dict[j-1]`.
        # The range of `j` is `min_alpha_bar_idx_paper` to `max_alpha_bar_idx_paper`.
        # Max index for `overline_alpha` is `T_total - 0*N/2 - (-1) + 1 = T_total + 2`.
        # Min index for `overline_alpha` is `T_total - (K_rounds-1)*N_steps/2 - N_steps + 1 + 1 = 2 - N_steps/2`.
        # So `_hat_alpha_dict` must cover this entire range `[2 - N/2, T+2]`.
        
        # My `_precompute_schedule_extended` covers `[min_t_for_alpha_needed, T_total+1]`.
        # `min_t_for_alpha_needed` is `T - KN/2 - N = -N/2`.
        # My loop for `t_idx` goes down to `min_t_for_alpha_needed - 1` to compute `hat_alpha` for `min_t_for_alpha_needed`.
        # So `_hat_alpha_dict` contains keys from `min_t_for_alpha_needed` to `T_total + 1`.
        # This covers all required indices.
        
        # Now, for `tau_k,n`, the index `j = T - kN/2 - n + 1`.
        # If `j` is outside `[min_t_for_alpha_needed, T_total+1]`, how to handle it?
        # The minimum for `j` is `2 - N_steps/2`.
        # The maximum for `j` is `T_total + 2`.
        # My `_hat_alpha_dict` covers `[-N/2, T+1]`. So it may not cover `T+2` or `2-N/2 - 1`.
        
        # Re-evaluating the needed range for `_hat_alpha_dict`:
        # Max index needed for `hat_alpha_t` is `T_total + 1`. Covered.
        # Min index needed for `hat_alpha_t` for `hat_tau_k,n` (Eq 9) is `T_total - (K-1)N/2 - N = -N/2`. Covered.
        # Min index needed for `hat_alpha_t` for `overline_alpha_t` in sampling (Eq 9 again for Unif(.,.))
        # The index `t` in `Unif(hat_alpha_t, hat_alpha_{t-1})` can be `j = T_total - kN/2 - n + 1`.
        # This `j` can go up to `T_total + 2`.
        # So `_hat_alpha_dict` must have `hat_alpha_{T_total+2}` too.
        # It also needs `hat_alpha_{j-1}`. If `j = 2 - N/2`, then `j-1 = 1 - N/2`. This means `_hat_alpha_dict`
        # needs keys from `1 - N/2` to `T_total + 2`.
        
        min_alpha_idx_for_unif = 1 - self.N_steps // 2 # Smallest j-1 for Unif
        max_alpha_idx_for_unif = self.T_total + 2 # Largest j for Unif

        # Re-initialize hat_alpha_dict and fill the required range for Unif.
        self._hat_alpha_dict = {}
        # We need to compute hat_alpha for indices from `max_alpha_idx_for_unif` down to `min_alpha_idx_for_unif`.
        # Start at max + 1 so that we can compute max index.
        self._hat_alpha_dict[max_alpha_idx_for_unif] = 1.0 / (self.T_total ** self.c0) # This is `hat_alpha_{T+2}`

        for t_idx in range(max_alpha_idx_for_unif, min_alpha_idx_for_unif - 1, -1):
            hat_alpha_t = self._hat_alpha_dict[t_idx]
            if hat_alpha_t >= 1.0:
                self._hat_alpha_dict[t_idx - 1] = hat_alpha_t
            else:
                self._hat_alpha_dict[t_idx - 1] = hat_alpha_t + (self.c1 * hat_alpha_t * (1.0 - hat_alpha_t) * np.log(self.T_total) / self.T_total)
            self._hat_alpha_dict[t_idx - 1] = torch.clamp(torch.tensor(self._hat_alpha_dict[t_idx - 1]), min=1e-6, max=1.0).item()

        # Now _hat_alpha_dict is correctly populated.
        # Populate _hat_tau_k_n based on this.
        self._hat_tau_k_n = {}
        for k in range(self.K_rounds):
            for n_val in range(-1, self.N_steps + 1):
                hat_alpha_idx = Config.get_tau_k_n_index(self.T_total, self.K_rounds, k, n_val)
                self._hat_tau_k_n[(k, n_val)] = 1.0 - self._hat_alpha_dict[hat_alpha_idx]


    def _get_randomized_tau_k_n(self, k: int, n_val: int) -> torch.Tensor:
        """
        Samples tau_k,n from Unif(hat_tau_k,n, hat_tau_k,n-1).
        tau_k,n := 1 - overline_alpha_{T - kN/2 - n + 1}
        overline_alpha_t ~ Unif(hat_alpha_t, hat_alpha_{t-1})
        So, 1 - overline_alpha_t is sampled.
        Let j = T - kN/2 - n + 1.
        We need to sample overline_alpha_j from Unif(hat_alpha_j, hat_alpha_{j-1}).
        Then tau_k,n = 1 - overline_alpha_j.
        """
        j = Config.get_alpha_bar_t_index(self.T_total, self.K_rounds, k, n_val)
        
        hat_alpha_j = self._hat_alpha_dict.get(j, None)
        hat_alpha_j_minus_1 = self._hat_alpha_dict.get(j - 1, None)

        if hat_alpha_j is None or hat_alpha_j_minus_1 is None:
            raise ValueError(f"Required hat_alpha for index {j} or {j-1} not found. Check schedule precomputation.")

        # Uniformly sample overline_alpha_j
        # ensure min < max
        lower_bound = min(hat_alpha_j, hat_alpha_j_minus_1)
        upper_bound = max(hat_alpha_j, hat_alpha_j_minus_1)
        
        # Clamp bounds to [0, 1] for alpha values
        lower_bound = max(0.0, lower_bound)
        upper_bound = min(1.0, upper_bound)
        
        if upper_bound < lower_bound + 1e-9: # Handle very small intervals
            overline_alpha_j = torch.tensor(lower_bound)
        else:
            overline_alpha_j = torch.distributions.uniform.Uniform(lower_bound, upper_bound).sample()

        return 1.0 - overline_alpha_j


    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Executes the diffusion sampling procedure to generate samples.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples, shape (num_samples, d_dim).
        """
        # 1. Initialization: The sampler begins with an initial sample Y_0 ~ N(0, I_d).
        y_k = torch.randn(num_samples, self.d_dim) # Y_0 is Y_k for k=0

        # Store all sampled tau_k,n values per sample for reproducibility if needed
        # Or just sample on the fly. Let's sample on the fly for simplicity.

        # K rounds
        for k_idx in range(self.K_rounds):
            # Calculate tau_k_0 and hat_tau_k_0, hat_tau_k_(-1)
            tau_k_0 = self._get_randomized_tau_k_n(k_idx, 0) # This is 1 - overline_alpha_{T - kN/2 + 1}
            hat_tau_k_0 = self._hat_tau_k_n[(k_idx, 0)]
            hat_tau_k_minus_1 = self._hat_tau_k_n[(k_idx, -1)]

            # Intermediate variable Y_k,n (Y_k for n=0)
            y_k_n = y_k.clone() # Y_k,0 = Y_k
            
            # The sum in Eq 10 has a `s_T - kN/2 + 1 (Y_k,0)` term and `sum_i=1^n-1` terms, and a final term.
            # This looks like an Euler-Maruyama or similar discretization.
            
            # For n=1, ..., N steps within current round k
            for n_idx in range(1, self.N_steps + 1):
                # Need s_t(Y_k,0), s_t(Y_k,i), s_t(Y_k,n-1).
                # t for score function is T - kN/2 - i + 1 or T - kN/2 + 1 or T - kN/2 - n + 2
                
                # The paper states: `s_{T - kN/2 - i + 1}(Y_k,i)` and `s_{T - kN/2 - n + 2}(Y_k,n-1)`.
                # This refers to the index of `overline_alpha` (or `alpha_t` in Eq 3) for the forward process.
                # Let `t_score_idx` be this index.
                
                # Term 1 (initial score term for n=0, for the first part of the integral)
                # s_T - kN/2 + 1 (Y_k,0) corresponds to index T - kN/2 + 1
                t_score_idx_k0 = Config.get_alpha_bar_t_index(self.T_total, self.K_rounds, k_idx, 0)
                # alpha_bar_t_k0 is randomized and will be part of tau_k,0 itself.
                # However, the score function input is `alpha_bar_t` not `tau_k,n`.
                # s_t_star takes `alpha_bar_t`. What `alpha_bar_t` should be used for `s_{t_score_idx}`?
                # The paper says `s_t` is an estimate of `s_t^*` corresponding to `overline_alpha_t = 1 - tau_k,i`.
                # So we need `1 - tau_k,i` as the `alpha_bar_t` for the score function.
                
                # Let's adjust this: `s_t` is an estimate of `s_t^*`.
                # The `t` in `s_t` (and `s_t^*`) corresponds to the index `t` in `X_t`.
                # The `alpha_bar_t` argument to the `score_function.s_t_star` is `1 - tau` from the current time step.
                
                # The sum terms:
                # `s_{T - kN/2 + 1}(Y_k,0)` implies we use the `alpha_bar` corresponding to `t = T - kN/2 + 1`.
                # This `t` is actually `j` in our `_get_randomized_tau_k_n` function.
                
                # Let `alpha_bar_for_score_k0 = 1.0 - tau_k_0`. This is the `overline_alpha` for index `T - kN/2 + 1`.
                # Score term for Y_k,0
                score_k0 = self.score_function.s_t_star(y_k_n, 1.0 - tau_k_0) # y_k_n here is y_k,0
                
                term_k0 = (score_k0 / (2 * (1.0 - tau_k_0)**(3/2))) * (tau_k_0 - hat_tau_k_0)

                # sum_i=1^n-1 terms
                sum_terms = torch.zeros_like(y_k_n)
                for i_idx in range(1, n_idx):
                    # We need Y_k,i. Here y_k_n holds the *current* iteration's value (Y_k,n-1).
                    # This means we need to store Y_k,i for previous steps.
                    # This is sequential inside a round.
                    # The formulation for Y_k,n suggests it's explicitly computed using Y_k,0, Y_k,1, ..., Y_k,n-1.
                    # y_k_n in the outer loop is actually `Y_k,n-1` for current iteration.
                    
                    # For iteration `n_idx`: we are computing `Y_k,n_idx`.
                    # It depends on `Y_k,0`, and `s_t(Y_k,i)` for `i=1..n_idx-1`, and `s_t(Y_k,n_idx-1)`.
                    
                    # Let's define `y_values_in_round` to store `Y_k,i` for `i=0..N`.
                    # y_values_in_round[0] = y_k
                    # When computing `y_values_in_round[n_idx]`, it uses `y_values_in_round[i]` and `y_values_in_round[n_idx-1]`.
                    
                    # For current `n_idx` (computing `Y_k, n_idx`):
                    # `Y_k,0` is `y_k`.
                    # The value `y_k_n` in loop is `Y_k, n_idx-1`.
                    
                    # The expression (10) for Y_k,n.
                    # The initial `Y_k,0 / sqrt(1 - tau_k,0)`
                    # + score_T-kN/2+1(Y_k,0) * (...)
                    # + sum_i=1^n-1 score_T-kN/2-i+1(Y_k,i) * (...)
                    # + score_T-kN/2-n+2(Y_k,n-1) * (...)
                    
                    # This implies `Y_k,i` for `i=1..n-1` refers to values from previous steps in this round.
                    # We need `y_k_i_val` for `i_idx`.
                    # Let's collect `y_k_i_val` as they are computed.
                    
                    # In simulation, `y_k_n` is actually `Y_{current_round}, {current_step}`.
                    # Let's rename `y_k_n` to `current_y_val_normalized`.
                    
                    # Let's re-write for clarity:
                    # `Y_k,n_idx` is being computed.
                    # `Y_k` is `y_k` (from previous round).
                    # `y_k_values_in_round[0]` is `y_k`.
                    # `y_k_values_in_round[n_idx_minus_1]` means `Y_k, n_idx-1`.
                    
                    # This implies sequential updates:
                    # `y_current_round_step_vals[0] = y_k`
                    # For n_idx = 1 .. N:
                    #   `y_current_round_step_vals[n_idx] = ... uses y_current_round_step_vals[i] for i < n_idx`
                    
                    pass # Sum terms will be handled when building the equation for `y_k_n_normalized`
            
            # Reconstruct the update for `Y_k,n` (normalized form) from Eq 10.
            # `Y_k,0` is `y_k` (input from previous round).
            
            # This will compute all `Y_k,n` sequentially for `n = 1...N`
            y_k_current_round_steps_normalized = [y_k / torch.sqrt(1.0 - tau_k_0)] # Stores Y_k,i / sqrt(1-tau_k,i)
            y_k_current_round_raw_values = [y_k] # Stores Y_k,i for score function inputs

            for n_idx in range(1, self.N_steps + 1):
                # Sample tau_k,i, hat_tau_k,i-1, hat_tau_k,i, etc
                tau_k_i_val = self._get_randomized_tau_k_n(k_idx, i_idx if 'i_idx' in locals() else n_idx) # Use n_idx for current step
                hat_tau_k_i_minus_1 = self._hat_tau_k_n[(k_idx, n_idx - 1)]
                hat_tau_k_i_val = self._hat_tau_k_n[(k_idx, n_idx)]

                # Term 1: (Y_k,0 / sqrt(1 - tau_k,0))
                normalized_sum_val = y_k_current_round_steps_normalized[0].clone()

                # Term 2: (s_T-kN/2+1(Y_k,0) / (2 * (1 - tau_k,0)^(3/2))) * (tau_k,0 - hat_tau_k,0)
                t_score_idx_for_k0 = Config.get_alpha_bar_t_index(self.T_total, self.K_rounds, k_idx, 0)
                score_for_k0_arg_alpha_bar = self._get_randomized_alpha_bar(t_score_idx_for_k0)
                score_k0_val = self.score_function.s_t_star(y_k_current_round_raw_values[0], score_for_k0_arg_alpha_bar)
                normalized_sum_val += (score_k0_val / (2 * (1.0 - tau_k_0)**(3/2))) * (tau_k_0 - hat_tau_k_0)

                # Term 3: sum_i=1^n-1 (s_T-kN/2-i+1(Y_k,i) / (2 * (1 - tau_k,i)^(3/2))) * (hat_tau_k,i-1 - hat_tau_k,i)
                for i_loop_idx in range(1, n_idx):
                    tau_k_i_loop = self._get_randomized_tau_k_n(k_idx, i_loop_idx)
                    hat_tau_k_i_loop_minus_1 = self._hat_tau_k_n[(k_idx, i_loop_idx - 1)]
                    hat_tau_k_i_loop = self._hat_tau_k_n[(k_idx, i_loop_idx)]

                    t_score_idx_for_k_i_loop = Config.get_alpha_bar_t_index(self.T_total, self.K_rounds, k_idx, i_loop_idx)
                    score_for_k_i_loop_arg_alpha_bar = self._get_randomized_alpha_bar(t_score_idx_for_k_i_loop)
                    score_k_i_loop_val = self.score_function.s_t_star(y_k_current_round_raw_values[i_loop_idx], score_for_k_i_loop_arg_alpha_bar)
                    
                    normalized_sum_val += (score_k_i_loop_val / (2 * (1.0 - tau_k_i_loop)**(3/2))) * (hat_tau_k_i_loop_minus_1 - hat_tau_k_i_loop)

                # Term 4: (s_T-kN/2-n+2(Y_k,n-1) / (2 * (1 - tau_k,n-1)^(3/2))) * (hat_tau_k,n-1 - tau_k,n)
                t_score_idx_for_k_n_minus_1 = Config.get_alpha_bar_t_index(self.T_total, self.K_rounds, k_idx, n_idx - 1) + 1 # +1 as per formula (n+2, so n-1+2 = n+1)
                score_for_k_n_minus_1_arg_alpha_bar = self._get_randomized_alpha_bar(t_score_idx_for_k_n_minus_1)
                
                score_k_n_minus_1_val = self.score_function.s_t_star(y_k_current_round_raw_values[n_idx-1], score_for_k_n_minus_1_arg_alpha_bar)
                
                # tau_k_n for the last term means current n_idx.
                tau_k_n_current_step = self._get_randomized_tau_k_n(k_idx, n_idx)
                # tau_k,n-1 for the coefficient denominator
                tau_k_n_minus_1 = self._get_randomized_tau_k_n(k_idx, n_idx - 1)
                
                normalized_sum_val += (score_k_n_minus_1_val / (2 * (1.0 - tau_k_n_minus_1)**(3/2))) * (hat_tau_k_i_minus_1 - tau_k_n_current_step)
                
                # Y_k,n = normalized_sum_val * sqrt(1 - tau_k,n)
                current_y_k_n_normalized = normalized_sum_val
                current_y_k_n_raw = current_y_k_n_normalized * torch.sqrt(1.0 - tau_k_n_current_step)
                
                y_k_current_round_steps_normalized.append(current_y_k_n_normalized)
                y_k_current_round_raw_values.append(current_y_k_n_raw)
            
            # After N steps, Y_k,N is y_k_current_round_raw_values[N]
            y_k_N = y_k_current_round_raw_values[self.N_steps]
            tau_k_N = self._get_randomized_tau_k_n(k_idx, self.N_steps)

            # 3. Noise injection: After obtaining Y_k,N, we update Y_k+1 by injecting stochastic noise.
            # Y_{k+1} = sqrt((1 - tau_{k+1,0}) / (1 - tau_k,N)) * Y_k,N + sqrt((tau_{k+1,0} - tau_k,N) / (1 - tau_k,N)) * Z_k
            
            # tau_{k+1,0}
            tau_k_plus_1_0 = self._get_randomized_tau_k_n(k_idx + 1, 0)
            
            z_k = torch.randn_like(y_k_N) # Z_k ~ N(0, I_d)
            
            coeff1_sqrt = torch.sqrt((1.0 - tau_k_plus_1_0) / (1.0 - tau_k_N))
            coeff2_sqrt = torch.sqrt((tau_k_plus_1_0 - tau_k_N) / (1.0 - tau_k_N))
            
            y_k_plus_1 = coeff1_sqrt * y_k_N + coeff2_sqrt * z_k
            
            y_k = y_k_plus_1 # Update for next round
        
        return y_k # This is Y_K, the final sample

    def _get_randomized_alpha_bar(self, t_idx: int) -> torch.Tensor:
        """
        Samples overline_alpha_t from Unif(hat_alpha_t, hat_alpha_{t-1}).
        Used to get `alpha_bar` for score function input.
        """
        hat_alpha_t = self._hat_alpha_dict.get(t_idx)
        hat_alpha_t_minus_1 = self._hat_alpha_dict.get(t_idx - 1)

        if hat_alpha_t is None or hat_alpha_t_minus_1 is None:
            raise ValueError(f"Required hat_alpha for index {t_idx} or {t_idx-1} not found. Check schedule precomputation.")

        lower_bound = min(hat_alpha_t, hat_alpha_t_minus_1)
        upper_bound = max(hat_alpha_t, hat_alpha_t_minus_1)

        lower_bound = max(0.0, lower_bound)
        upper_bound = min(1.0, upper_bound)

        if upper_bound < lower_bound + 1e-9:
            return torch.tensor(lower_bound)
        else:
            return torch.distributions.uniform.Uniform(lower_bound, upper_bound).sample()

