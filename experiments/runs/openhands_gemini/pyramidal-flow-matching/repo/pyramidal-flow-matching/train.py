
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

from config import get_config
from model import PyramidalFlowMatchingModel
from data import get_dataloader # Assuming get_dataloader is defined in data.py
import math

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = PyramidalFlowMatchingModel(config)
        
        # Initialize optimizer based on config
        if config.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=config.learning_rate_stage12, 
                betas=(config.beta1, config.beta2_stage1), # Stage 1 default
                eps=config.eps, 
                weight_decay=config.weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer {config.optimizer} not supported.")

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() and config.numerical_precision == 'bfloat16' else None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Timestep schedules for pyramidal flow matching
        # Paper implies K=3 stages. For a uniform partitioning, these could be [0, 1/3], [1/3, 2/3], [2/3, 1]
        # But the paper's specific formulas (e.g., in inference) suggest more complex relationships.
        # For training, we'll use a simplified uniform sampling across stages.
        self.k_stages = config.num_pyramid_stages # K from paper

    def _get_pyramid_stage_timesteps(self, k_idx):
        # This function would return s_k and e_k for a given pyramid stage k.
        # The paper defines these implicitly in the formulas.
        # For simplicity, let's assume k_idx goes from 0 to K-1.
        # The paper: "k-th time window [s_k, e_k]" and "only the final stage operates at full resolution"
        # "Under a uniform stage partitioning, the idea of spatial pyramid reduces the computational cost to a factor of nearly 1/K."
        # This suggests something like:
        # Stage 0: [0, 1/K] (lowest resolution / most noisy)
        # Stage 1: [1/K, 2/K]
        # ...
        # Stage K-1: [(K-1)/K, 1] (highest resolution / least noisy)

        # Let's map k_idx from 0 to K-1 to actual stages.
        # The paper's k starts from highest resolution for Down(x, 2^k)
        # Here k_idx=0 means highest compression, k_idx=K-1 means lowest compression
        # This needs careful alignment with the paper's notation.
        
        # In 3.2.1 Unified Training: Down(x_1, 2^k) and Down(x_1, 2^(k+1))
        # k ranges from 0 (highest resolution) to K-1 (lowest resolution)
        # So for k=0, it's Down(x_1, 1) -> full resolution
        # For k=K-1, it's Down(x_1, 2^(K-1))
        
        # Let's define the stages by the downsampling factor applied to x_1.
        # Stage `p`: corresponds to `k` in Down(x_1, 2^k), where `p` goes from 0 to K-1
        # `p=0` is highest resolution (Downsample factor 2^0 = 1)
        # `p=K-1` is lowest resolution (Downsample factor 2^(K-1))

        # This method is simplified. The actual s_k, e_k are internal to the flow matching process.
        # For the training objective, we primarily need x_0 (noise), x_1 (data), and t.
        # The `s_k` and `e_k` in equations (8) and (9) are for endpoints of probability paths.
        # For the loss, we need: current latent x_t, and the target vector field u_t = x_e_k - x_s_k

        # For the purpose of implementing the training loss, we need to choose a `k` (pyramid stage)
        # and sample `t` within its range.
        
        # The core idea is that for each training step, we randomly select a pyramid stage `k`
        # and then sample `t` within its corresponding time window.
        
        # Let's assume uniform distribution over `k` stages for sampling during training.
        # The actual values of s_k and e_k for the piecewise flow need to be carefully defined.
        
        # Paper says: "For the k-th time window [s_k, e_k], let t' = (t - s_k) / (e_k - s_k) denote the rescaled timestep"
        # and "only the last stage is performed at full resolution"

        # Let's define uniform time windows:
        segment_length = 1.0 / self.k_stages
        s_k = k_idx * segment_length
        e_k = (k_idx + 1) * segment_length
        
        return s_k, e_k
    
    def _downsample_latent(self, latent, factor):
        # Placeholder for VAE's internal downsampling, or just torch.nn.functional.interpolate
        if factor == 1:
            return latent
        # Assuming latent is (C, T, H, W)
        # Resize H and W
        new_H = latent.shape[-2] // factor
        new_W = latent.shape[-1] // factor
        return F.interpolate(latent, size=(latent.shape[-3], new_H, new_W), mode='nearest') # Or bilinear

    def _upsample_latent(self, latent, factor):
        # Placeholder for VAE's internal upsampling
        if factor == 1:
            return latent
        # Assuming latent is (C, T, H, W)
        # Resize H and W
        new_H = latent.shape[-2] * factor
        new_W = latent.shape[-1] * factor
        return F.interpolate(latent, size=(latent.shape[-3], new_H, new_W), mode='nearest') # Or bilinear

    def train_step(self, batch, current_stage_idx):
        self.optimizer.zero_grad()

        x_1 = batch['x_1'].to(self.device) # Clean latent
        text_embeddings = batch['text_embeddings'].to(self.device)

        # Randomly sample a pyramid stage `p` from 0 to K-1 (K=config.num_pyramid_stages)
        # `p=0` corresponds to the smallest downsampling factor (highest resolution stage, k=0 in paper's Down(x, 2^k))
        # `p=K-1` corresponds to the largest downsampling factor (lowest resolution stage, k=K-1 in paper's Down(x, 2^(K-1)))
        
        # We need to choose a `k` for the Down functions as in (8) and (9).
        # k=0 means Down(x, 2^0) = Down(x, 1) -> full resolution
        # k=K-1 means Down(x, 2^(K-1)) -> lowest resolution
        
        # Let's sample `k_down_factor_exponent` which is `k` in Down(x_1, 2^k) from 0 to K-1
        k_down_factor_exponent = random.randint(0, self.k_stages - 1)
        
        # Sample noise n ~ N(0, I)
        n_noise = torch.randn_like(x_1).to(self.device)

        # Compute endpoints for the current pyramid stage (Eqs. 9 & 10)
        # End: x_e_k = e_k * Down(x_1, 2^k) + (1 - e_k) * n
        # Start: x_s_k = s_k * Up(Down(x_1, 2^(k+1))) + (1 - s_k) * n
        
        # We need s_k and e_k for the specific pyramid stage (k_down_factor_exponent)
        # Let's adapt the paper's implicit stage definition for s_k and e_k.
        # The paper reinterprets the original denoising trajectory into K stages.
        # Each stage interpolates between a pixelated/noisier latent and a pixelate-free/cleaner latent.
        
        # For simplicity in training, we'll pick a 't' and a 'k_down_factor_exponent'
        # The model's loss term (Eq. 11) is based on the difference (x_e_k - x_s_k)
        
        # Let's redefine 't' to be the single timestep in [0, 1] we sample for flow matching.
        # The 'k' from the paper (Down(x_1, 2^k)) defines the resolution level of the "End" point.
        # The 'k+1' defines the resolution level of the "Start" point (Down(x_1, 2^(k+1))).
        
        # So, for each training sample, we:
        # 1. Sample `t` uniformly from [0, 1]. This `t` is the `t` in x_t.
        # 2. Sample a pyramid level `k` (from 0 to K-1, where k=0 is highest resolution, k=K-1 lowest).
        
        # Generate x_s_k and x_e_k according to sampled `k` and `t`.
        # The `t` used for the loss calculation is actually `t'` in the piecewise flow (rescaled timestep).
        # The paper's overall loss (Eq. 11) averages over `k` and `t`.
        
        t_val = batch['t'].to(self.device) # Sampled uniformly from [0, 1] for the whole process.
                                            # We then use this to pick s_k, e_k or define effective t'
        
        # Let's align with the overall flow matching objective from 3.1:
        # E_t, q(x_1), p_t(x_t | x_1) || v_t(x_t) - u_t(x_t | x_1) ||^2
        # where u_t(x_t | x_1) = x_1 - x_0 (if x_0 is noise)
        # In our case, the u_t is (x_e_k - x_s_k) and x_t is interpolated between them.
        
        # To compute x_s_k and x_e_k, we need a specific 'k' (downsampling exponent) for the current sample.
        # The paper implicitly samples k and t for each batch.
        
        # Sample k for Down(x_1, 2^k)
        # k ranges from 0 to self.config.num_pyramid_stages - 1
        current_k_exponent = random.randint(0, self.config.num_pyramid_stages - 1)

        # Downsample x_1 for current_k and current_k+1 resolution levels
        x_1_at_k = self._downsample_latent(x_1, 2**current_k_exponent)
        
        # For Start point, it's Up(Down(x_1, 2^(k+1)))
        # Handle k+1 for the last stage: if current_k_exponent is K-1, then k+1 would be K.
        # This means Down(x_1, 2^K) would be the lowest resolution possible.
        # For the Start point, it comes from a more pixelated (lower resolution) representation.
        
        # The original paper's definition of stages in Section 3.2.1 seems to imply:
        # End point resolution = 2^k
        # Start point resolution = 2^(k+1) (which is lower res, then upsampled)
        # So, larger k means lower resolution.
        
        # Let's pick an `s_k_val` and `e_k_val` that correspond to the actual time values for the interpolated path.
        # These are not the `t_val` itself, but the specific start/end times of the current segment.
        # The paper's piecewise flow formulation is a bit ambiguous without explicit s_k, e_k values.
        
        # Let's use the definition: x_t = t' * End + (1 - t') * Start
        # where t' is rescaled to [0,1] within the current (s_k, e_k) window.
        
        # Since we sample a global `t_val` in [0,1] and a `k_down_factor_exponent`
        # Let's map this `t_val` to an effective `s_k` and `e_k` for the loss calculation.
        
        # Simplified interpretation: for each (x_1, t) pair, we also sample a `k`
        # The goal is to predict the vector field between a slightly noised, lower-res version of x_1
        # and a slightly noised, higher-res version of x_1.
        
        # Compute x_e_k (End point)
        # This is at resolution 2^current_k_exponent
        x_e_k = t_val * x_1_at_k + (1 - t_val) * self._downsample_latent(n_noise, 2**current_k_exponent)

        # Compute x_s_k (Start point)
        # This is at resolution 2^(current_k_exponent + 1) then upsampled
        # Ensure that current_k_exponent + 1 does not exceed K. If it does, we can cap it or adjust.
        # The paper says: Up(Down(x_1, 2^(k+1))), so for the lowest resolution END point, the START point
        # would be even lower, perhaps even from pure noise.
        # For simplicity, if current_k_exponent + 1 >= K, we can just use the lowest available resolution.
        
        k_plus_1_exponent = min(current_k_exponent + 1, self.config.num_pyramid_stages - 1)
        x_1_at_k_plus_1 = self._downsample_latent(x_1, 2**k_plus_1_exponent)
        
        x_s_k_orig_res = self._upsample_latent(x_1_at_k_plus_1, 2**(k_plus_1_exponent - current_k_exponent)) # Upsample to current_k resolution
        
        x_s_k = t_val * x_s_k_orig_res + (1 - t_val) * self._downsample_latent(n_noise, 2**current_k_exponent)

        # Interpolate x_t for the current stage (needs resolution adjustment)
        # x_t should be at the resolution of x_e_k and x_s_k (i.e., 2^current_k_exponent)
        x_t_interpolated = t_val * x_e_k + (1 - t_val) * x_s_k # This isn't quite right, x_t is from noise, not interpolated endpoints
        
        # The paper states: x_t = t * x_1 + (1-t) * x_0 (Section 3.1)
        # For Pyramidal Flow Matching, it's: x_t = t' * Down(x_e_k, res) + (1-t') * Up(Down(x_s_k, res))
        # This requires `x_t` to be generated using a specific `k` and a local `t_prime`
        
        # Let's simplify and use the main Flow Matching objective (Eq. 11) with the specific (x_e_k - x_s_k) as target.
        # The model v_t predicts the vector field.
        # x_t is the input to the model, which is noisy.
        
        # The input x_t to the model is one of the interpolated states, not x_e_k or x_s_k.
        # It's generated by:
        # x_t = t' * Down(x_1, 2^k) + (1 - t') * noise (where noise is at current resolution)
        # This makes it similar to a diffusion process input.
        
        # Let's use the definition from Eq. (7) and the overall flow matching logic.
        # The velocity field target is `x_e_k - x_s_k` as per Eq. (11).
        
        # We need to compute x_t (noisy input to the model) based on the sampled `t_val` and `k`.
        # This x_t should be at the resolution corresponding to `k`.
        # Let's define it as `x_t = t_val * x_1_at_k + (1 - t_val) * self._downsample_latent(n_noise, 2**current_k_exponent)`
        
        # This requires the `n_noise` to be consistent. The paper uses a single `n` for coupling.
        
        # Let's re-align with (9) and (10) for endpoints.
        # These endpoints are `x_e_k` and `x_s_k` as defined in the paper using a shared `n`.
        # The actual `t_prime` (rescaled timestep) should be sampled from [0, 1].
        # We will use the original `t_val` from batch as this `t_prime`.
        
        # For simplicity, assume that `s_k` and `e_k` are fixed to 0 and 1, respectively,
        # for a given pyramid stage. The stages are implicitly defined by the resolution of x_1 and x_0.
        
        # Let's consider `k_down_factor_exponent` as the current spatial resolution level (0=full, 1=half, etc.)
        # The input `x_t` to the model should be at this resolution.
        
        # Define x_0 at current resolution:
        x_0_at_k = self._downsample_latent(n_noise, 2**current_k_exponent)
        
        # Define x_1 at current resolution:
        x_1_at_k = self._downsample_latent(x_1, 2**current_k_exponent)
        
        # Generate x_t for the model input
        x_t_input = t_val * x_1_at_k + (1 - t_val) * x_0_at_k
        
        # Compute the target vector field u_t = x_1_at_k - x_0_at_k for vanilla flow matching.
        # For pyramidal, it's (x_e_k - x_s_k) as per Eq. (11).
        
        # The paper's x_e_k and x_s_k from Eqs. 9 and 10 are defined with global `e_k` and `s_k` which are time values.
        # This is where the time windows from `_get_pyramid_stage_timesteps` become relevant for `s_k` and `e_k`.
        
        # Let `p` be the current pyramid stage (0 to K-1).
        # When sampling for stage `p`, we use `s_p` and `e_p`.
        
        s_p, e_p = self._get_pyramid_stage_timesteps(current_k_exponent) # Using current_k_exponent as p for now

        # Use t_val to linearly interpolate between s_p and e_p to get the effective t for (9) and (10)
        # This might be t_prime, or the t in N(t*Down(x1), (1-t)^2 * I)
        
        # Let's follow the core of Eq. 11 for the loss calculation.
        # The target vector is (x_e_k - x_s_k)
        # And the input x_t to the model is sampled along the path between x_s_k and x_e_k.
        
        # Re-evaluating Eq. (9) and (10):
        # x_e_k = e_k * Down(x_1, 2^k) + (1 - e_k) * n
        # x_s_k = s_k * Up(Down(x_1, 2^(k+1))) + (1 - s_k) * n
        
        # The 'k' in these equations is the pyramid level of the 'End' point.
        # It means `Down(x_1, 2^k)` refers to a downsampling factor of `2^k`.
        # So `k` goes from 0 (full res) to `K-1` (lowest res).
        
        # We need to sample a `k` (current_k_exponent) and a `t` (t_val from batch).
        # We need to decide what `s_k` and `e_k` correspond to.
        
        # The paper describes "a piecewise flow that divides [0,1] into K time windows"
        # Let's say we have global time `T_global` in [0,1].
        # For a given stage `k_idx` (0 to K-1), its time window is `[k_idx/K, (k_idx+1)/K]`.
        # Let `T_global` be the `t` from the batch.
        # We determine which `k_idx` it falls into: `current_k_idx = floor(T_global * K)`
        
        # Then, `s_k = current_k_idx / K` and `e_k = (current_k_idx + 1) / K`.
        # And the `t_prime = (T_global - s_k) / (e_k - s_k)`. This `t_prime` is for the interpolation.
        
        # The `k` in Down(x_1, 2^k) for x_e_k, and Down(x_1, 2^(k+1)) for x_s_k,
        # is the resolution exponent. Let's call it `res_k_exp`.
        
        # Let's assume that `current_k_idx` (derived from `T_global`) also determines `res_k_exp`.
        # So, `res_k_exp = current_k_idx`.
        
        # This means:
        # x_e_k uses resolution `2^current_k_idx`
        # x_s_k uses resolution `2^(current_k_idx+1)` (then upsampled)
        
        # Let's re-calculate x_s_k and x_e_k using this interpretation.
        
        current_k_idx = torch.floor(t_val * self.k_stages).long() # Which stage the global t falls into
        s_k_val = current_k_idx.float() / self.k_stages
        e_k_val = (current_k_idx.float() + 1.0) / self.k_stages
        
        # Ensure that s_k_val and e_k_val are tensors with batch dimension
        s_k_val = s_k_val.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1) # Make it broadcastable
        e_k_val = e_k_val.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1) # Make it broadcastable

        # The resolution exponent for the End point:
        res_k_exp = current_k_idx
        
        # Handle the edge case for k+1 if res_k_exp is the last stage
        res_k_plus_1_exp = torch.min(res_k_exp + 1, torch.tensor(self.k_stages - 1, device=self.device))
        
        x_1_at_res_k = self._downsample_latent(x_1, 2**res_k_exp)
        x_1_at_res_k_plus_1 = self._downsample_latent(x_1, 2**res_k_plus_1_exp)
        
        # x_e_k = e_k * Down(x_1, 2^k) + (1 - e_k) * n (Eq. 9)
        x_e_k = e_k_val * x_1_at_res_k + (1 - e_k_val) * self._downsample_latent(n_noise, 2**res_k_exp)
        
        # x_s_k = s_k * Up(Down(x_1, 2^(k+1))) + (1 - s_k) * n (Eq. 10)
        # Up(Down(x_1, 2^(k+1))) must match the resolution of Down(x_1, 2^k) for element-wise subtraction.
        # This implies the Up function scales it up from 2^(k+1) resolution to 2^k resolution.
        upsampled_x_1_from_k_plus_1 = self._upsample_latent(x_1_at_res_k_plus_1, 2**(res_k_plus_1_exp - res_k_exp))
        
        x_s_k = s_k_val * upsampled_x_1_from_k_plus_1 + (1 - s_k_val) * self._downsample_latent(n_noise, 2**res_k_exp)
        
        # The target vector field u_t (Eq. 11)
        target_velocity_field = x_e_k - x_s_k
        
        # The input to the model `v_t(x_t)` where `x_t` is between `x_s_k` and `x_e_k`
        # `x_t = t_prime * x_e_k + (1 - t_prime) * x_s_k` (from 3.2, after rescaling `t`)
        
        # Rescale `t_val` to `t_prime` within the current stage's window.
        t_prime = (t_val - s_k_val) / (e_k_val - s_k_val + 1e-6) # Add epsilon for stability
        
        x_t_model_input = t_prime * x_e_k + (1 - t_prime) * x_s_k
        
        # Reshape x_t_model_input for the DiT (B, N_tokens, latent_dim)
        # This requires flattening the (C, T, H, W) latent into a sequence of tokens.
        # Each "token" here is a C-dimensional vector.
        # Flatten (C, T, H, W) to (T*H*W, C) and then add batch dimension.
        
        # Assuming x_t_model_input shape is (B, C, T, H, W)
        B, C, T, H, W = x_t_model_input.shape
        x_t_model_input_flattened = rearrange(x_t_model_input, 'b c t h w -> b (t h w) c')
        
        # Now, `C` is the feature dimension, and `T*H*W` is the sequence length (N_tokens)
        
        # The model expects (batch, sequence_length, embedding_dim)
        # Ensure the `latent_dim` in `model.py` is consistent with `C` here.
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.scaler else torch.no_grad():
            predicted_velocity_field = self.model(x_t_model_input_flattened, t_val.squeeze(), text_embeddings)
            
            # The predicted velocity field also needs to be reshaped back to (B, C, T, H, W)
            # or the target_velocity_field needs to be flattened. Let's flatten the target.
            target_velocity_field_flattened = rearrange(target_velocity_field, 'b c t h w -> b (t h w) c')
            
            loss = F.mse_loss(predicted_velocity_field, target_velocity_field_flattened)

        if self.scaler:
            self.scaler.scale(loss).backward()
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            self.optimizer.step()

        return loss.item()

    def train(self):
        # Stage 1: Image Training
        print("\n" + "="*20 + " Stage 1: Image Training " + "="*20)
        self._set_optimizer_params_for_stage(1)
        image_dataloader = get_dataloader(self.config, is_image_data=True, current_stage=1, is_training=True)
        for epoch in range(self.config.stage1_epochs):
            for i, batch in enumerate(tqdm(image_dataloader, desc=f"Stage 1 Epoch {epoch+1}/{self.config.stage1_epochs}")):
                loss = self.train_step(batch, current_stage_idx=0) # For image, assume it's always the highest resolution stage
                if i % 100 == 0:
                    print(f"Stage 1 - Epoch {epoch+1}, Step {i}, Loss: {loss:.4f}")

        # Stage 2: Low-Resolution Video Training (2s then 5s)
        print("\n" + "="*20 + " Stage 2: Low-Resolution Video Training " + "="*20)
        self._set_optimizer_params_for_stage(2)
        video_dataloader_2s = get_dataloader(self.config, is_image_data=False, current_stage=2, is_training=True)
        for epoch in range(self.config.stage2_epochs_2s):
            for i, batch in enumerate(tqdm(video_dataloader_2s, desc=f"Stage 2 (2s) Epoch {epoch+1}/{self.config.stage2_epochs_2s}")):
                loss = self.train_step(batch, current_stage_idx=0) # Need to consider how current_stage_idx relates to k_down_factor_exponent
                if i % 100 == 0:
                    print(f"Stage 2 (2s) - Epoch {epoch+1}, Step {i}, Loss: {loss:.4f}")

        # Continue Stage 2 with 5s videos
        video_dataloader_5s = get_dataloader(self.config, is_image_data=False, current_stage=2, is_training=True) # Potentially different data loader for 5s videos
        for epoch in range(self.config.stage2_epochs_5s):
            for i, batch in enumerate(tqdm(video_dataloader_5s, desc=f"Stage 2 (5s) Epoch {epoch+1}/{self.config.stage2_epochs_5s}")):
                loss = self.train_step(batch, current_stage_idx=0)
                if i % 100 == 0:
                    print(f"Stage 2 (5s) - Epoch {epoch+1}, Step {i}, Loss: {loss:.4f}")

        # Stage 3: High-Resolution Video Training
        print("\n" + "="*20 + " Stage 3: High-Resolution Video Training " + "="*20)
        self._set_optimizer_params_for_stage(3)
        video_dataloader_high_res = get_dataloader(self.config, is_image_data=False, current_stage=3, is_training=True)
        for epoch in range(self.config.stage3_epochs):
            for i, batch in enumerate(tqdm(video_dataloader_high_res, desc=f"Stage 3 Epoch {epoch+1}/{self.config.stage3_epochs}")):
                loss = self.train_step(batch, current_stage_idx=0)
                if i % 100 == 0:
                    print(f"Stage 3 - Epoch {epoch+1}, Step {i}, Loss: {loss:.4f}")

        print("\nTraining complete!")


    def _set_optimizer_params_for_stage(self, stage):
        if stage == 1:
            self.optimizer.param_groups[0]['lr'] = self.config.learning_rate_stage12
            self.optimizer.param_groups[0]['betas'] = (self.config.beta1, self.config.beta2_stage1)
        elif stage == 2:
            self.optimizer.param_groups[0]['lr'] = self.config.learning_rate_stage12
            self.optimizer.param_groups[0]['betas'] = (self.config.beta1, self.config.beta2_stages23)
        elif stage == 3:
            self.optimizer.param_groups[0]['lr'] = self.config.learning_rate_stage3
            self.optimizer.param_groups[0]['betas'] = (self.config.beta1, self.config.beta2_stages23)
        print(f"Optimizer parameters set for Stage {stage}: LR={self.optimizer.param_groups[0]['lr']}, Betas={self.optimizer.param_groups[0]['betas']}")

if __name__ == '__main__':
    config = get_config()
    
    # Dummy overrides for testing purposes if running directly
    # In a real setup, these would come from command line or a proper config management
    config.stage1_epochs = 1
    config.stage2_epochs_2s = 1
    config.stage2_epochs_5s = 1
    config.stage3_epochs = 1
    config.global_batch_size_stage1 = 2
    config.global_batch_size_stage2 = 2
    config.global_batch_size_stage3 = 2

    trainer = Trainer(config)
    trainer.train()
