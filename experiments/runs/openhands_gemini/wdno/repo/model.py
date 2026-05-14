
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_wavelets import DWT, IDWT # For 1D and 2D DWT/IDWT
from pytorch_wavelets.dwt.dwt_3d import DWT3D, IDWT3D # For 3D DWT/IDWT
import math
from tqdm.auto import tqdm

from config import Config
from modules import Unet, GaussianDiffusion

# Helper functions (can be moved to utils.py later if needed)
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class WDNO(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.is_3d = False
        if config.pde_type in ["2d_fluid", "era5"]:
            self.is_3d = True

        # Wavelet Transform initialization
        if self.is_3d: # 2D fluid, ERA5 -> 3D data (time, spatial_x, spatial_y)
            self.dwt = DWT3D(J=1, mode=config.wavelet_mode_2d, wave=config.wavelet_type_2d)
            self.idwt = IDWT3D(J=1, mode=config.wavelet_mode_2d, wave=config.wavelet_type_2d)
            # For 3D DWT with J=1, yl is (B, C, D/2, H/2, W/2) and yh is 7 bands, each (B, C, D/2, H/2, W/2)
            self.factor_wavelet_channels = 8
        else: # 1D Burgers, Advection, Navier-Stokes -> 2D data (time, space)
            self.dwt = DWT(wave=config.wavelet_type_1d, mode=config.wavelet_mode_1d)
            self.idwt = IDWT(wave=config.wavelet_type_1d, mode=config.wavelet_mode_1d)
            # For 2D DWT with J=1, yl is (B, C, H/2, W/2) and yh is 3 bands, each (B, C, H/2, W/2)
            self.factor_wavelet_channels = 4

        # Determine input/output channels for Unet
        # These are the channel counts *after* wavelet transform for the respective inputs
        unet_channels_x_wt = self.config.raw_input_channels_x * self.factor_wavelet_channels # Channels of the data being diffused (wavelet coeffs of x_input)
        unet_channels_cond_wt = self.config.raw_input_channels_cond * self.factor_wavelet_channels # Channels for W_a (wavelet coeffs of condition_input)

        if self.config.enable_super_resolution_training:
            # For super-resolution training, add the wavelet channels of the low-res input
            unet_channels_cond_wt += self.config.raw_input_channels_x * self.factor_wavelet_channels

        self.noise_predictor = Unet(
            dim=config.unet_initial_dim,
            channels=unet_channels_x_wt,
            cond_channels=unet_channels_cond_wt,
            dim_mults=config.unet_dim_mults,
            resnet_block_groups=config.unet_resnet_block_groups,
            attn_heads=config.unet_attention_heads,
            attn_dim_head=config.unet_attention_hidden_dim,
            is_3d=self.is_3d,
            conv_kernel_size=config.fluid_kernel_size_conv3d if self.is_3d else config.unet_conv_kernel_size,
            conv_padding=config.fluid_padding_conv3d if self.is_3d else math.ceil((config.unet_conv_kernel_size - 1) / 2),
            conv_stride=config.fluid_stride_conv3d if self.is_3d else 1,
            downsample_kernel=config.fluid_kernel_size_downsampling if self.is_3d else None,
            downsample_stride=config.fluid_stride_downsampling if self.is_3d else None,
            downsample_padding=config.fluid_padding_downsampling if self.is_3d else None,
        )

        # For Gaussian Diffusion, the model passed should be the noise predictor
        self.diffusion = GaussianDiffusion(
            model=self.noise_predictor,
            image_size=(config.data_res_h, config.data_res_w) if not self.is_3d else (config.data_res_h, config.data_res_w, config.data_res_d),
            timesteps=config.num_diffusion_steps,
            beta_schedule=config.beta_schedule,
            channels=unet_channels_x_wt, # This should be the channels of the wavelet coeffs
            loss_type='l2' # Paper uses MSE
        )

    def _apply_dwt(self, data, raw_channels):
        # Applies DWT and concatenates coefficients for Unet input
        # data: (B, ..., C) or (B, C, ...) as per pytorch_wavelets DWT input
        # raw_channels: number of channels in the raw data
        if self.is_3d:
            # DWT3D expects (N, C, D, H, W)
            # data comes in (B, D, H, W, C) for 2D fluid/ERA5, needs permute
            if data.dim() == 5: # Assuming (B, D, H, W, C)
                data_in = data.permute(0, 4, 1, 2, 3) # (B, C, D, H, W)
            else: # Assuming data is (B, D, H, W) and C=1 or (B, D, H, W, C) already
                if data.dim() == 4: # (B, D, H, W)
                    data_in = data.unsqueeze(1) # (B, 1, D, H, W)
                else:
                    data_in = data # Assume it's already (B, C, D, H, W) compatible
            
            coeffs = self.dwt(data_in)
            yl, yh = coeffs[0], coeffs[1][0] # Assuming J=1
            yh_reshaped = rearrange(yh, 'b c s d h w -> b (c s) d h w')
            return torch.cat([yl, yh_reshaped], dim=1)
        else:
            # DWT expects (N, C, H, W)
            # data comes in (B, T, S, C) for 1D PDE (time, spatial, channels), needs permute
            if data.dim() == 4: # Assuming (B, T, S, C)
                data_in = data.permute(0, 3, 1, 2) # (B, C, T, S)
            else: # Assuming data is (B, T, S) and C=1 or (B, T, S, C) already
                if data.dim() == 3: # (B, T, S)
                    data_in = data.unsqueeze(1) # (B, 1, T, S)
                else:
                    data_in = data # Assume it's already (B, C, T, S) compatible
            
            coeffs = self.dwt(data_in)
            yl, yh = coeffs[0], coeffs[1][0] # Assuming J=1
            yh_reshaped = rearrange(yh, 'b c s h w -> b (c s) h w')
            return torch.cat([yl, yh_reshaped], dim=1)

    def _apply_idwt(self, wavelet_coeffs, raw_channels_original_input):
        # Applies IDWT to concatenated wavelet coefficients
        # wavelet_coeffs: (B, C_total, ...)
        # raw_channels_original_input: number of channels expected in the original (pre-DWT) data
        
        if self.is_3d:
            # C_total = raw_channels_original_input * 8
            yl = wavelet_coeffs[:, :raw_channels_original_input, ...]
            yh_reshaped = wavelet_coeffs[:, raw_channels_original_input:, ...]
            yh = rearrange(yh_reshaped, 'b (c s) d h w -> b c s d h w', c=raw_channels_original_input, s=7).unsqueeze(0) # Put into list for J=1
            
            reconstructed_data = self.idwt((yl, yh))
            # Reshape back to (B, D, H, W, C)
            return reconstructed_data.permute(0, 2, 3, 4, 1) if reconstructed_data.shape[1] > 1 else reconstructed_data.squeeze(1)
        else:
            # C_total = raw_channels_original_input * 4
            yl = wavelet_coeffs[:, :raw_channels_original_input, ...]
            yh_reshaped = wavelet_coeffs[:, raw_channels_original_input:, ...]
            yh = rearrange(yh_reshaped, 'b (c s) h w -> b c s h w', c=raw_channels_original_input, s=3).unsqueeze(0) # Put into list for J=1
            
            reconstructed_data = self.idwt((yl, yh))
            # Reshape back to (B, H, W, C)
            return reconstructed_data.permute(0, 2, 3, 1) if reconstructed_data.shape[1] > 1 else reconstructed_data.squeeze(1)


    def forward(self, x_data, condition_data, low_res_data=None):
        # x_data: original data (u or f trajectory) - (B, T, S, C) or (B, D, H, W, C)
        # condition_data: a (initial condition or parameters) - (B, T, S, C) or (B, D, H, W, C)
        # low_res_data: for super-resolution training (W_low_res) - (B, T_low, S_low, C) or (B, D_low, H_low, W_low, C)

        wavelet_input_x = self._apply_dwt(x_data, self.config.raw_input_channels_x)
        wavelet_input_cond = self._apply_dwt(condition_data, self.config.raw_input_channels_cond)

        if low_res_data is not None:
            wavelet_input_low_res = self._apply_dwt(low_res_data, self.config.raw_input_channels_x) # Assuming low_res has same channels as x_data
            wavelet_input_cond = torch.cat([wavelet_input_cond, wavelet_input_low_res], dim=1)

        loss = self.diffusion(wavelet_input_x, cond=wavelet_input_cond)
        return loss

    @torch.no_grad()
    def sample(self, shape, condition_data, low_res_data=None, guidance_func=None, guidance_weight=0.):
        # shape: (batch_size, num_wavelet_channels, H_out, W_out) or (batch_size, num_wavelet_channels, D_out, H_out, W_out)
        # condition_data: original space condition (a) - (B, T, S, C) or (B, D, H, W, C)
        # low_res_data: original space low-resolution data for SR - (B, T_low, S_low, C) or (B, D_low, H_low, W_low, C)
        # guidance_func: for control tasks, takes W_f_hat (wavelet coeffs) as input and returns objective I(scalar)

        batch_size = shape[0]
        device = self.config.device

        # Prepare conditioning input
        wavelet_input_cond = self._apply_dwt(condition_data, self.config.raw_input_channels_cond)
        if low_res_data is not None:
            wavelet_input_low_res = self._apply_dwt(low_res_data, self.config.raw_input_channels_x) # Assuming low_res has same channels as x_data
            wavelet_input_cond = torch.cat([wavelet_input_cond, wavelet_input_low_res], dim=1)

        # Initialize with Gaussian noise in wavelet domain
        img = torch.randn(shape, device=device)

        ddim_timesteps = torch.arange(0, self.diffusion.num_timesteps, self.diffusion.num_timesteps // self.config.ddim_sampling_iterations).long()
        ddim_timesteps = list(ddim_timesteps.flip(0).cpu().numpy())
        ddim_timesteps = ddim_timesteps[:-1] + [0] # Add t=0

        for i, t in enumerate(tqdm(ddim_timesteps[:-1], desc='sampling loop time step')):
            t_next = ddim_timesteps[i+1]
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            alpha_prod = self.diffusion.extract(self.diffusion.alphas_cumprod, t_tensor, img.shape)
            alpha_prod_next = self.diffusion.extract(self.diffusion.alphas_cumprod, torch.tensor([t_next], device=device), img.shape)
            
            # Predict noise for conditional (current state of W_u^(k) and W_a)
            model_output_cond = self.noise_predictor(img, t_tensor, cond=wavelet_input_cond)
            pred_noise = model_output_cond

            # Calculate x_start_pred (W_f_hat^(k))
            x_start_pred = self.diffusion.predict_xstart_from_noise(img, t_tensor, pred_noise)
            x_start_pred.clamp_(-1., 1.) # Clamp to typical diffusion range [-1, 1]

            # Apply guidance for control tasks
            if guidance_func is not None and guidance_weight > 0:
                x_start_pred.requires_grad_(True)
                objective_I = guidance_func(x_start_pred) # guidance_func takes wavelet coeffs
                grad_I = torch.autograd.grad(objective_I, x_start_pred)[0]
                
                # Add guidance to the noise prediction
                pred_noise = pred_noise + guidance_weight * grad_I
                x_start_pred.requires_grad_(False) # Turn off grad after use

            # Re-calculate x_start_pred with potentially guided noise for DDIM update
            x_start_pred_final = self.diffusion.predict_xstart_from_noise(img, t_tensor, pred_noise)
            x_start_pred_final.clamp_(-1., 1.)

            # Calculate sigma for DDIM
            sigma = self.config.ddim_eta * torch.sqrt((1 - alpha_prod_next) / (1 - alpha_prod) * (1 - alpha_prod / alpha_prod_next))
            if t_next == 0:
                sigma = 0

            # DDIM update step
            mean_pred = x_start_pred_final * torch.sqrt(alpha_prod_next) + torch.sqrt(1 - alpha_prod_next - sigma**2) * pred_noise
            
            noise_for_update = torch.randn_like(img) if t_next != 0 else torch.zeros_like(img)
            img = mean_pred + sigma * noise_for_update
            
        final_wavelet_coeffs = img

        # Apply Inverse Wavelet Transform
        # For sample, the output is the reconstructed data, whose raw_channels are config.raw_input_channels_x
        reconstructed_data = self._apply_idwt(final_wavelet_coeffs, self.config.raw_input_channels_x)

        return reconstructed_data

        self.noise_predictor = Unet(
            dim=config.unet_initial_dim,
            channels=unet_channels,
            cond_channels=unet_cond_channels,
            dim_mults=config.unet_dim_mults,
            resnet_block_groups=config.unet_resnet_block_groups,
            attn_heads=config.unet_attention_heads,
            attn_dim_head=config.unet_attention_hidden_dim,
            is_3d=self.is_3d,
            conv_kernel_size=config.fluid_kernel_size_conv3d if self.is_3d else config.unet_conv_kernel_size,
            conv_padding=config.fluid_padding_conv3d if self.is_3d else math.ceil((config.unet_conv_kernel_size - 1) / 2),
            conv_stride=config.fluid_stride_conv3d if self.is_3d else 1,
            downsample_kernel=config.fluid_kernel_size_downsampling if self.is_3d else None,
            downsample_stride=config.fluid_stride_downsampling if self.is_3d else None,
            downsample_padding=config.fluid_padding_downsampling if self.is_3d else None,
        )

        # For Gaussian Diffusion, the model passed should be the noise predictor
        self.diffusion = GaussianDiffusion(
            model=self.noise_predictor,
            image_size=(config.data_res_h, config.data_res_w) if not self.is_3d else (config.data_res_h, config.data_res_w, config.data_res_d),
            timesteps=config.num_diffusion_steps,
            beta_schedule=config.beta_schedule,
            channels=unet_channels, # This should be the channels of the wavelet coeffs
            loss_type='l2' # Paper uses MSE
        )

    def forward(self, x_data, condition_data, low_res_data=None):
        # x_data: original data (u or f trajectory)
        # condition_data: a (initial condition or parameters)
        # low_res_data: for super-resolution training (W_low_res)

        # 1. Apply Wavelet Transform
        # DWT returns (yl, yh), yl is low-pass, yh is list of high-pass bands
        # yl: (batch, channels, H_out, W_out)
        # yh: list of (batch, channels, 3, H_out, W_out) tuples for each level (for 2D DWT)
        # For 3D DWT, yh is list of (batch, channels, 7, D_out, H_out, W_out) tuples
        if self.is_3d:
            # input to DWT3D is (N, C, D, H, W)
            # Our data is (Batch, Time, Spatial_X, Spatial_Y, Channels) or similar for PDE
            # Need to reshape data to (Batch, Channels, Time, Spatial_X, Spatial_Y)
            # Assuming input x_data has shape (B, D, H, W, C)
            x_data = x_data.permute(0, 4, 1, 2, 3) if x_data.dim() == 5 else x_data.unsqueeze(1) # (B, C, D, H, W)
            condition_data = condition_data.permute(0, 4, 1, 2, 3) if condition_data.dim() == 5 else condition_data.unsqueeze(1)
            # Perform DWT
            # The wavelet transform for 2D fluid problem (Table 3): "data of size 32x64x64 are transformed into eight sets of wavelet coefficients, each sized 18x34x34."
            # This suggests J=1 (one level of decomposition).
            # pytorch_wavelets DWT3D (wave, mode, J=1)
            # If input is (N,C,D,H,W), then yl is (N,C,D/2,H/2,W/2) and yh has 7 bands (LH, HL, HH, LLH, LHL, HLL, HHH) each (N,C,D/2,H/2,W/2)
            # The paper says "eight sets of wavelet coefficients", which implies LL, LH, HL, HH for 2D or LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH for 3D.
            # DWT3D for J=1 returns (yl, [yh_l, ..., yh_7]) where each yh_i is a band.
            # Concatenate yl and all yh bands along the channel dimension.
            # Let's assume for simplicity we stack them all as input to UNet.
            coeffs_x = self.dwt(x_data)
            coeffs_x_yl, coeffs_x_yh = coeffs_x[0], coeffs_x[1][0] # Assuming J=1
            # coeffs_x_yh has shape (batch, channels, 7, D_out, H_out, W_out)
            # Reshape coeffs_x_yh to (batch, channels*7, D_out, H_out, W_out) and concatenate with yl
            coeffs_x_yh_reshaped = rearrange(coeffs_x_yh, 'b c s d h w -> b (c s) d h w')
            wavelet_input_x = torch.cat([coeffs_x_yl, coeffs_x_yh_reshaped], dim=1) # (B, C_total, D_out, H_out, W_out)

            coeffs_cond = self.dwt(condition_data)
            coeffs_cond_yl, coeffs_cond_yh = coeffs_cond[0], coeffs_cond[1][0]
            coeffs_cond_yh_reshaped = rearrange(coeffs_cond_yh, 'b c s d h w -> b (c s) d h w')
            wavelet_input_cond = torch.cat([coeffs_cond_yl, coeffs_cond_yh_reshaped], dim=1)

            if low_res_data is not None:
                low_res_data = low_res_data.permute(0, 4, 1, 2, 3) if low_res_data.dim() == 5 else low_res_data.unsqueeze(1)
                coeffs_low_res = self.dwt(low_res_data)
                coeffs_low_res_yl, coeffs_low_res_yh = coeffs_low_res[0], coeffs_low_res[1][0]
                coeffs_low_res_yh_reshaped = rearrange(coeffs_low_res_yh, 'b c s d h w -> b (c s) d h w')
                wavelet_input_low_res = torch.cat([coeffs_low_res_yl, coeffs_low_res_yh_reshaped], dim=1)
                # Concatenate with condition_data
                wavelet_input_cond = torch.cat([wavelet_input_cond, wavelet_input_low_res], dim=1)

        else: # 2D DWT for 1D PDE data
            # input to DWT is (N, C, H, W)
            # Our data is (Batch, Time, Spatial, Channels) or (Batch, H, W, C)
            # Reshape to (Batch, Channels, Time, Spatial)
            x_data = x_data.permute(0, 3, 1, 2) if x_data.dim() == 4 else x_data.unsqueeze(1) # (B, C, H, W)
            condition_data = condition_data.permute(0, 3, 1, 2) if condition_data.dim() == 4 else condition_data.unsqueeze(1)

            coeffs_x = self.dwt(x_data)
            coeffs_x_yl, coeffs_x_yh = coeffs_x[0], coeffs_x[1][0] # Assuming J=1
            # coeffs_x_yh has shape (batch, channels, 3, H_out, W_out) (LH, HL, HH bands)
            # Reshape coeffs_x_yh to (batch, channels*3, H_out, W_out) and concatenate with yl
            coeffs_x_yh_reshaped = rearrange(coeffs_x_yh, 'b c s h w -> b (c s) h w')
            wavelet_input_x = torch.cat([coeffs_x_yl, coeffs_x_yh_reshaped], dim=1)

            coeffs_cond = self.dwt(condition_data)
            coeffs_cond_yl, coeffs_cond_yh = coeffs_cond[0], coeffs_cond[1][0]
            coeffs_cond_yh_reshaped = rearrange(coeffs_cond_yh, 'b c s h w -> b (c s) h w')
            wavelet_input_cond = torch.cat([coeffs_cond_yl, coeffs_cond_yh_reshaped], dim=1)

            if low_res_data is not None:
                low_res_data = low_res_data.permute(0, 3, 1, 2) if low_res_data.dim() == 4 else low_res_data.unsqueeze(1)
                coeffs_low_res = self.dwt(low_res_data)
                coeffs_low_res_yl, coeffs_low_res_yh = coeffs_low_res[0], coeffs_low_res[1][0]
                coeffs_low_res_yh_reshaped = rearrange(coeffs_low_res_yh, 'b c s h w -> b (c s) h w')
                wavelet_input_low_res = torch.cat([coeffs_low_res_yl, coeffs_low_res_yh_reshaped], dim=1)
                # Concatenate with condition_data
                wavelet_input_cond = torch.cat([wavelet_input_cond, wavelet_input_low_res], dim=1)

        # Pass wavelet coefficients to the diffusion model for training
        # The Unet in GaussianDiffusion expects `x` (noisy data) and `cond` (conditioning data)
        # However, the current GaussianDiffusion `p_losses` expects only `x_start` and `t`.
        # The `model` (noise_predictor) within GaussianDiffusion is called as `self.model(x_noisy, t)`
        # This means we need to modify GaussianDiffusion or the noise_predictor to accept `cond`.
        # For now, let's assume `self.noise_predictor`'s forward method directly takes `cond` as an argument.
        loss = self.diffusion(wavelet_input_x, cond=wavelet_input_cond)
        return loss

    @torch.no_grad()
    def sample(self, shape, condition_data, low_res_data=None, guidance_func=None, guidance_weight=0.):
        # condition_data: W_a
        # low_res_data: W_low_res for super-resolution
        # guidance_func: for control tasks, takes W_f_hat as input and returns gradient w.r.t. objective I

        batch, *image_size = shape
        device = self.config.device

        # Prepare conditioning input
        if self.is_3d:
            condition_data_wt = condition_data.permute(0, 4, 1, 2, 3) if condition_data.dim() == 5 else condition_data.unsqueeze(1)
            coeffs_cond = self.dwt(condition_data_wt)
            coeffs_cond_yl, coeffs_cond_yh = coeffs_cond[0], coeffs_cond[1][0]
            coeffs_cond_yh_reshaped = rearrange(coeffs_cond_yh, 'b c s d h w -> b (c s) d h w')
            wavelet_input_cond = torch.cat([coeffs_cond_yl, coeffs_cond_yh_reshaped], dim=1)

            if low_res_data is not None:
                low_res_data_wt = low_res_data.permute(0, 4, 1, 2, 3) if low_res_data.dim() == 5 else low_res_data.unsqueeze(1)
                coeffs_low_res = self.dwt(low_res_data_wt)
                coeffs_low_res_yl, coeffs_low_res_yh = coeffs_low_res[0], coeffs_low_res[1][0]
                coeffs_low_res_yh_reshaped = rearrange(coeffs_low_res_yh, 'b c s d h w -> b (c s) d h w')
                wavelet_input_low_res = torch.cat([coeffs_low_res_yl, coeffs_low_res_yh_reshaped], dim=1)
                wavelet_input_cond = torch.cat([wavelet_input_cond, wavelet_input_low_res], dim=1)
        else:
            condition_data_wt = condition_data.permute(0, 3, 1, 2) if condition_data.dim() == 4 else condition_data.unsqueeze(1)
            coeffs_cond = self.dwt(condition_data_wt)
            coeffs_cond_yl, coeffs_cond_yh = coeffs_cond[0], coeffs_cond[1][0]
            coeffs_cond_yh_reshaped = rearrange(coeffs_cond_yh, 'b c s h w -> b (c s) h w')
            wavelet_input_cond = torch.cat([coeffs_cond_yl, coeffs_cond_yh_reshaped], dim=1)

            if low_res_data is not None:
                low_res_data_wt = low_res_data.permute(0, 3, 1, 2) if low_res_data.dim() == 4 else low_res_data.unsqueeze(1)
                coeffs_low_res = self.dwt(low_res_data_wt)
                coeffs_low_res_yl, coeffs_low_res_yh = coeffs_low_res[0], coeffs_low_res[1][0]
                coeffs_low_res_yh_reshaped = rearrange(coeffs_low_res_yh, 'b c s h w -> b (c s) h w')
                wavelet_input_low_res = torch.cat([coeffs_low_res_yl, coeffs_low_res_yh_reshaped], dim=1)
                wavelet_input_cond = torch.cat([wavelet_input_cond, wavelet_input_low_res], dim=1)

        # Initialize with Gaussian noise
        img = torch.randn(shape, device=device)
        x_start = None

        ddim_timesteps = torch.arange(0, self.diffusion.num_timesteps, self.diffusion.num_timesteps // self.config.ddim_sampling_iterations).long()
        ddim_timesteps = list(ddim_timesteps.flip(0).cpu().numpy())
        ddim_timesteps = ddim_timesteps[:-1] + [0] # Add t=0

        for i, t in enumerate(tqdm(ddim_timesteps[:-1], desc='sampling loop time step')):
            t_next = ddim_timesteps[i+1]
            t_tensor = torch.full((batch,), t, device=device, dtype=torch.long)

            alpha_prod = self.diffusion.extract(self.diffusion.alphas_cumprod, t_tensor, img.shape)
            alpha_prod_next = self.diffusion.extract(self.diffusion.alphas_cumprod, torch.tensor([t_next], device=device), img.shape)
            # beta_prod = 1 - alpha_prod # Not used in DDIM sampling formula

            # Classifier-free guidance style for conditional generation
            # Predict noise for both conditional and unconditional (if guidance_weight > 0)
            model_output_cond = self.noise_predictor(img, t_tensor, cond=wavelet_input_cond)

            pred_noise = model_output_cond
            x_start_pred = self.diffusion.predict_xstart_from_noise(img, t_tensor, pred_noise)

            # Apply guidance for control tasks
            if guidance_func is not None and guidance_weight > 0:
                # Calculate W_f_hat (approximate noise-free W_f)
                # Equation (5) from paper: W_f_hat = (W_f^(k) - sqrt(1 - alpha_k_bar) * epsilon_theta) / sqrt(alpha_k_bar)
                # Here, img is W_f^(k), pred_noise is epsilon_theta, alpha_prod is alpha_k_bar
                # Note: `img` here is the noisy wavelet coefficients. The paper uses W_f^(k) for control.
                # So `x_start_pred` is equivalent to W_f_hat.
                
                # We need to make x_start_pred differentiable for gradient calculation
                x_start_pred.requires_grad_(True)
                
                # Convert W_f_hat back to original space to compute I(u,f)
                # This requires inverse wavelet transform of x_start_pred and then simulation
                # For now, let's assume guidance_func takes wavelet coefficients directly
                # If guidance_func works on original space, we would need to IDWT x_start_pred,
                # compute I, then DWT the gradient. This is complex and might be specific to implementation.
                # The paper states: "We calculate I in Eq. 4 based on W_f_hat instead of directly using W_f^(k)"
                # "lambda * nabla_W_f_hat I(W_f_hat)"
                
                # Compute objective I from x_start_pred
                objective_I = guidance_func(x_start_pred)
                
                # Calculate gradient
                grad_I = torch.autograd.grad(objective_I, x_start_pred)[0]
                
                # Apply guidance (equation 4 from paper)
                # epsilon_theta_guided = epsilon_theta + lambda * nabla_W_f_hat I
                # We need epsilon_theta corresponding to x_start_pred, not pred_noise directly.
                # pred_noise is epsilon_theta(W_f^(k), W_a, k)
                # To apply guidance, we modify epsilon_theta directly.
                
                # Convert grad_I to noise space (epsilon)
                # This part is a bit tricky. The paper's equation 4 has lambda * nabla_W_f_hat I(W_f_hat) directly added to epsilon_theta.
                # This implies grad_I is already in the 'noise-like' direction or scaled appropriately.
                # Let's assume the gradient 'grad_I' is directly added to the predicted noise for simplicity, as per Equation (4)
                # Or, more precisely, if we follow the spirit of guidance in diffusion models (classifier guidance)
                # the gradient is usually applied to the predicted x_start, then converted back to noise.
                # However, the paper explicitly says `epsilon_theta + lambda * nabla_W_f_hat I`.
                # So we simply add `lambda * grad_I` to `pred_noise`.
                
                # A crucial detail: The gradient is w.r.t. W_f_hat, but applied to epsilon_theta.
                # This implies a transformation from x_start_space gradient to noise_space gradient.
                # d(x_start) = (x_t - sqrt(1-alpha)*epsilon) / sqrt(alpha) -> d(epsilon) = -(sqrt(alpha)/(sqrt(1-alpha))) * d(x_start)
                # This is a simplification, but follows standard classifier guidance literature where gradient wrt x_0 is used.
                
                # For now, let's assume the paper's formula implies direct addition after calculating the gradient.
                # This is a common simplification in some guidance implementations.
                
                # pred_noise_guided = pred_noise + guidance_weight * grad_I
                
                # If we consider pred_noise as epsilon_theta(W_f^(k), W_a, k), then
                # (epsilon_theta_guided - epsilon_theta) = lambda * grad_I
                # This means we would add guidance_weight * grad_I to the predicted noise.
                # However, grad_I is wrt x_start_pred, not noise.
                
                # Let's re-evaluate Equation (4):
                # W_f^(k-1) = W_f^(k) - eta * (epsilon_theta(...) + lambda * nabla_W_f_hat I(W_f_hat^(k))) + xi
                # This implies the gradient is directly added to the noise prediction term before being scaled by eta.
                # This interpretation simplifies things but is unconventional compared to typical classifier guidance.
                # Let's use this for now, interpreting `nabla_W_f_hat I` as a term that contributes directly to the effective noise.
                
                # If `grad_I` is a gradient with respect to `x_start_pred`, then `grad_I` has the same shape as `x_start_pred`.
                # We need to transform this gradient into the "noise space" to add it to `pred_noise`.
                # The transformation from x_start to noise is: noise = (x_t - sqrt(alpha_bar)*x_0) / sqrt(1-alpha_bar)
                # d(noise)/d(x_0) = -sqrt(alpha_bar) / sqrt(1-alpha_bar)
                # So, grad_epsilon = grad_I * (-sqrt(alpha_bar) / sqrt(1-alpha_bar))
                # Let's approximate it by scaling the gradient term.
                
                # This is crucial for control. Let's follow the paper's formula (Eq 4) more literally
                # It means the `nabla_W_f_hat I` is directly added to `epsilon_theta`.
                # This implies the `guidance_func` should ideally return a gradient that is already in the 'noise-like' magnitude/space
                # or we need to carefully scale it.
                
                # For now, we will interpret `lambda * grad_I` as a direct additive term to the noise `pred_noise`.
                # This means `grad_I` needs to have the same shape as `pred_noise`.
                
                # Let's re-examine: `epsilon_theta(W_f^(k), W_a, k)` is the *predicted noise*.
                # `nabla_W_f_hat I(W_f_hat^(k))` is the *gradient of the objective w.r.t. the denoised estimate*.
                # These two quantities are in different "spaces" (noise space vs data space).
                # A common way to reconcile this in classifier guidance is to compute `epsilon_uncond`, `epsilon_cond`,
                # and then `epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)`.
                # Or, for classifier guidance: `epsilon_guided = epsilon - w * sqrt(1-alpha_bar) * grad_x_0(log P(y|x_0))`.
                # The paper's equation is unusual in direct addition.
                
                # Let's follow the general formula: epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)
                # In control, the goal is to make I smaller. So we steer away from higher I.
                # A more standard approach for objective guidance is:
                # pred_noise = pred_noise - guidance_weight * grad_I_transformed_to_noise_space
                
                # However, the paper's Eq 4 is: (epsilon_theta + lambda * nabla_W_f_hat I).
                # This suggests directly adding the gradient term to the predicted noise.
                # This means grad_I should have the same dimensions as the noise and x_start_pred should require grad.
                # Let's assume grad_I is calculated with respect to x_start_pred and it is of the same shape as x_start_pred.
                
                # Convert grad_I to be compatible with pred_noise.
                # A direct addition would mean grad_I is an "additional noise-like component".
                # Let's stick to the paper's formulation for now and assume `guidance_func` provides the gradient `nabla_W_f_hat I`.
                
                # The paper also states "we can only model and train p(W_f | W_a) as represented in the training set, where f is typically not optimal.
                # To address this issue, we view the control problem from an energy optimization perspective, and thus during inference,
                # we enhance the denoising process with guidance I to steer the generation of f towards a smaller I."
                # This implies that the guidance term is meant to *modify* the predicted noise from the learned distribution.
                
                # The term added is `lambda * nabla_W_f_hat I(W_f_hat)`.
                # So we must compute this `nabla_W_f_hat I(W_f_hat)`.
                
                # The prediction `x_start_pred` (which is W_f_hat) needs gradients to be computed.
                # If x_start_pred has shape (B, C, D, H, W) or (B, C, H, W) and gradient_func
                # returns a scalar, then torch.autograd.grad(scalar, tensor) works.
                
                grad_I = torch.autograd.grad(objective_I, x_start_pred, retain_graph=False)[0] # Calculate gradient
                
                # Now, how to add grad_I to pred_noise?
                # Option 1: Direct addition as implied by paper's notation:
                # pred_noise = pred_noise + guidance_weight * grad_I (if grad_I is already noise-scaled or needs to be)
                
                # Option 2: Modify x_start_pred then convert back to noise, consistent with classifier guidance style.
                # x_start_guided = x_start_pred - guidance_weight * grad_I (negative sign to minimize I)
                # pred_noise_guided = self.diffusion.predict_noise_from_xstart(img, t_tensor, x_start_guided)
                # This seems more mathematically sound for typical guidance.
                
                # Let's try Option 2, as it's more aligned with common diffusion guidance principles,
                # even if the paper's equation seems to suggest direct addition to epsilon.
                # The paper's formulation (epsilon + lambda * grad_I) is usually meant when grad_I is "grad_epsilon_log P(y|x)".
                # If grad_I is "grad_x_0 log P(y|x_0)", it needs scaling.
                # A more robust interpretation from similar papers (e.g., guided diffusion) is that you modify x_start,
                # then re-derive the noise prediction.
                
                x_start_guided = x_start_pred - guidance_weight * grad_I
                x_start_guided.clamp_(-1., 1.) # Clamp after modification
                
                # Now predict noise from the guided x_start
                pred_noise_guided = self.diffusion.predict_noise_from_xstart(img, t_tensor, x_start_guided)
                pred_noise = pred_noise_guided # Use this guided noise for the update
            
            x_start_pred.requires_grad_(False) # Turn off grad after use

            # DDIM step calculation
            sigma = self.config.ddim_eta * torch.sqrt((1 - alpha_prod_next) / (1 - alpha_prod) * (1 - alpha_prod / alpha_prod_next))
            if t_next == 0:
                sigma = 0

            mean_pred = x_start_pred * torch.sqrt(alpha_prod_next) + torch.sqrt(1 - alpha_prod_next - sigma**2) * pred_noise
            
            # For t_next = 0, no noise is added.
            # Else, add noise from N(0, I) scaled by sigma.
            noise_for_update = torch.randn_like(img) if t_next != 0 else torch.zeros_like(img)
            img = mean_pred + sigma * noise_for_update
            
        final_wavelet_coeffs = img

        # 2. Apply Inverse Wavelet Transform
        # Inverse wavelet transform: Input yl (low-pass) and yh (list of high-pass bands)
        # We need to reverse the concatenation done during DWT.
        if self.is_3d:
            # wavelet_input_x was (B, C_total, D_out, H_out, W_out)
            # C_total = C_orig + C_orig * 7 = C_orig * 8
            # Split back into yl and yh
            num_original_channels = self.config.data_channels
            yl = final_wavelet_coeffs[:, :num_original_channels, ...]
            yh_reshaped = final_wavelet_coeffs[:, num_original_channels:, ...]
            # Reshape yh_reshaped back to (batch, channels, 7, D_out, H_out, W_out)
            yh = rearrange(yh_reshaped, 'b (c s) d h w -> b c s d h w', c=num_original_channels, s=7).unsqueeze(0) # Put into list for J=1
            
            reconstructed_data = self.idwt((yl, yh))
            reconstructed_data = reconstructed_data.permute(0, 2, 3, 4, 1) if reconstructed_data.shape[1] > 1 else reconstructed_data.squeeze(1) # (B, D, H, W, C)
        else:
            # wavelet_input_x was (B, C_total, H_out, W_out)
            # C_total = C_orig + C_orig * 3 = C_orig * 4
            num_original_channels = self.config.data_channels
            yl = final_wavelet_coeffs[:, :num_original_channels, ...]
            yh_reshaped = final_wavelet_coeffs[:, num_original_channels:, ...]
            # Reshape yh_reshaped back to (batch, channels, 3, H_out, W_out)
            yh = rearrange(yh_reshaped, 'b (c s) h w -> b c s h w', c=num_original_channels, s=3).unsqueeze(0) # Put into list for J=1
            
            reconstructed_data = self.idwt((yl, yh))
            reconstructed_data = reconstructed_data.permute(0, 2, 3, 1) if reconstructed_data.shape[1] > 1 else reconstructed_data.squeeze(1) # (B, H, W, C)

        return reconstructed_data

