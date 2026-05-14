import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .modules import UNet2D, UNet3D
from .diffusion import Diffusion
from .wavelet_utils import (
    WaveletTransform1D, WaveletTransform2D, WaveletTransform3D,
    duplicate_low_res_to_high_res, pad_to_match
)


class WDNOBase(nn.Module):
    """Base WDNO model.
    
    Performs diffusion-based generative modeling in the wavelet domain.
    Handles simulation and control tasks with optional super-resolution.
    """
    
    def __init__(self, config, experiment_type='1d', task='simulation'):
        super().__init__()
        self.experiment_type = experiment_type
        self.task = task
        
        # Wavelet transforms
        if experiment_type == '1d':
            self.wavelet_2d = WaveletTransform2D(
                wavelet=config['wavelet']['type_1d'],
                mode=config['wavelet']['mode_1d']
            )
            self.wavelet_1d = WaveletTransform1D(
                wavelet=config['wavelet']['type_1d'],
                mode=config['wavelet']['mode_1d']
            )
        elif experiment_type == '2d':
            self.wavelet_3d = WaveletTransform3D(
                wavelet=config['wavelet']['type_2d'],
                mode=config['wavelet']['mode_2d']
            )
            self.wavelet_2d = WaveletTransform2D(
                wavelet=config['wavelet']['type_2d'],
                mode=config['wavelet']['mode_2d']
            )
            self.wavelet_1d = WaveletTransform1D(
                wavelet=config['wavelet']['type_2d'],
                mode='periodization'
            )
        
        # Diffusion process
        self.diffusion = Diffusion(
            timesteps=config['diffusion']['num_timesteps'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
            schedule=config['diffusion']['schedule']
        )
        
        # DDIM parameters
        self.ddim_steps = config['ddim']['sampling_steps']
        self.ddim_eta = config['ddim']['eta']
        
        # Control parameters
        self.guidance_weight = config.get('control', {}).get('guidance_weight', 120000)
        self.guidance_scheduler_type = config.get('control', {}).get('guidance_scheduler', 'cosine')
    
    def get_noise_prediction_net(self):
        raise NotImplementedError
    
    def forward_diffusion_loss(self, x_start, cond, cond_extra=None):
        """Compute DDPM training loss in wavelet domain."""
        batch_size = x_start.shape[0]
        t = torch.randint(1, self.diffusion.timesteps, (batch_size,), device=x_start.device)
        
        noise_pred_net = self.get_noise_prediction_net()
        loss = self.diffusion.training_loss(
            noise_pred_net, x_start, t, cond=cond
        )
        return loss
    
    @torch.no_grad()
    def sample(self, shape, cond, guidance_fn=None):
        """Sample from the diffusion model using DDIM."""
        noise_pred_net = self.get_noise_prediction_net()
        
        # Build guidance schedule if needed
        guidance_schedule = None
        if guidance_fn is not None and self.guidance_scheduler_type == 'cosine':
            def cosine_schedule(t):
                progress = 1 - t / self.diffusion.timesteps
                return self.guidance_weight * (np.cos(progress * np.pi) + 1) / 2
            guidance_schedule = cosine_schedule
        
        sample = self.diffusion.ddim_sample(
            noise_pred_net, shape, cond=cond,
            sampling_steps=self.ddim_steps, eta=self.ddim_eta,
            guidance_fn=guidance_fn,
            guidance_weight=self.guidance_weight,
            guidance_schedule=guidance_schedule
        )
        return sample


class WDNO1D(WDNOBase):
    """WDNO for 1D PDE problems (Burgers, Advection, Navier-Stokes).
    
    Uses 2D wavelet transform (time x space domain).
    Base-Resolution Model: predicts entire trajectory from initial condition and force.
    
    Architecture from Table 18: Two U-Nets
    - phi(f): encodes condition (control/force)
    - theta(u, f): denoising network that takes noisy state + encoded condition
    """
    
    def __init__(self, config, task='simulation', data_channels=1, cond_channels=2):
        super().__init__(config, experiment_type='1d', task=task)
        self.data_channels = data_channels
        self.cond_channels = cond_channels
        
        unet_cfg = config['unet_1d']
        
        # phi(f): condition encoder (UNet phi from Table 18)
        self.cond_encoder = UNet2D(
            in_channels=cond_channels * 4,
            out_channels=cond_channels * 4,
            cond_channels=0,
            init_dim=unet_cfg['init_dim'],
            dim_mult=unet_cfg['dim_mult_phi'],
            down_up_layers=unet_cfg['down_up_layers'],
            kernel_size=unet_cfg['kernel_size'],
            resnet_groups=unet_cfg['resnet_groups'],
            attn_heads=unet_cfg['attn_heads'],
            attn_hidden_dim=unet_cfg['attn_hidden_dim'],
        )
        
        # theta(u, f): denoising network (UNet theta from Table 18)
        # Takes noisy data (4x channels) + encoded condition (cond_channels * 4)
        self.noise_pred_net = UNet2D(
            in_channels=data_channels * 4,
            out_channels=data_channels * 4,
            cond_channels=cond_channels * 4,
            init_dim=unet_cfg['init_dim'],
            dim_mult=unet_cfg['dim_mult_theta'],
            down_up_layers=unet_cfg['down_up_layers'],
            kernel_size=unet_cfg['kernel_size'],
            resnet_groups=unet_cfg['resnet_groups'],
            attn_heads=unet_cfg['attn_heads'],
            attn_hidden_dim=unet_cfg['attn_hidden_dim'],
        )
    
    def get_noise_prediction_net(self):
        return self.noise_pred_net
    
    def encode_condition_with_phi(self, w_cond):
        """Process condition through phi network (Table 18)."""
        B = w_cond.shape[0]
        t_zero = torch.zeros(B, dtype=torch.long, device=w_cond.device)
        return self.cond_encoder(w_cond, t_zero, cond=None)
    
    def encode(self, x):
        """Encode data to wavelet domain.
        
        Args:
            x: (B, C, T, X) raw data tensor
        Returns:
            w: (B, 4*C, T//2, X//2) wavelet coefficients
        """
        return self.wavelet_2d.decompose(x)
    
    def decode(self, w):
        """Decode wavelet coefficients back to data domain."""
        return self.wavelet_2d.reconstruct(w)
    
    def encode_cond(self, cond):
        """Encode conditioning to wavelet domain.
        
        For 1D conditions (initial condition, target state), do 1D wavelet
        and repeat to match 2D wavelet coefficient shape.
        
        Args:
            cond: dict with keys like 'u0', 'f', 'uT'
        Returns:
            w_cond: concatenated wavelet coefficients
        """
        w_conds = []
        
        if 'u0' in cond:
            u0 = cond['u0']  # (B, X)
            u0 = u0.unsqueeze(1)  # (B, 1, X)
            w_u0 = self.wavelet_1d.decompose(u0)  # cA: (B, C, X//2), cD: (B, C, X//2)
            w_u0_cA = w_u0['cA'].unsqueeze(-1).repeat(1, 1, 1, cond['target_shape'][1])
            w_u0_cD = w_u0['cD'].unsqueeze(-1).repeat(1, 1, 1, cond['target_shape'][1])
            w_conds.extend([w_u0_cA, w_u0_cD])
        
        if 'f' in cond:
            f = cond['f']  # (B, T, X)
            f = f.unsqueeze(1)  # (B, 1, T, X)
            w_f = self.wavelet_2d.decompose(f)  # (B, 4, T//2, X//2)
            w_conds.append(w_f)
        
        if 'uT' in cond:
            uT = cond['uT']  # (B, X)
            uT = uT.unsqueeze(1)
            w_uT = self.wavelet_1d.decompose(uT)
            target_shape = w_conds[0].shape[-2:] if w_conds else cond.get('target_shape', (40, 60))
            w_uT_cA = w_uT['cA'].unsqueeze(-1).repeat(1, 1, 1, target_shape[1])
            w_uT_cD = w_uT['cD'].unsqueeze(-1).repeat(1, 1, 1, target_shape[1])
            w_conds.extend([w_uT_cA, w_uT_cD])
        
        if not w_conds:
            raise ValueError("No conditioning provided")
        
        return torch.cat(w_conds, dim=1)
    
    def training_step(self, batch):
        """Single training step for BRM."""
        x = batch['data']  # (B, C, T, X) - full trajectory
        cond = batch['cond']  # dict with conditioning
        
        # Encode to wavelet domain
        w_x = self.encode(x)
        w_cond = self.encode_cond(cond)
        
        loss = self.forward_diffusion_loss(w_x, w_cond)
        return loss
    
    @torch.no_grad()
    def sample_simulation(self, cond, shape_hint=None):
        """Sample simulation trajectory.
        
        Args:
            cond: dict with 'u0' and 'f' for simulation
            shape_hint: optional (T, X) shape for output
        Returns:
            pred: (B, C, T, X) predicted trajectory
        """
        if shape_hint is None:
            shape_hint = cond.get('target_shape', (80, 120))
        
        # Determine wavelet coefficient shape
        w_shape = (
            cond['u0'].shape[0],  # batch
            self.data_channels * 4,  # 4x wavelet channels
            (shape_hint[0] + 1) // 2,  # wavelet T
            (shape_hint[1] + 1) // 2,  # wavelet X
        )
        
        w_cond = self.encode_cond(cond)
        
        w_pred = self.sample(w_shape, w_cond)
        pred = self.decode(w_pred)
        
        # Crop to original shape if needed
        if pred.shape[2] > shape_hint[0]:
            pred = pred[:, :, :shape_hint[0], :]
        if pred.shape[3] > shape_hint[1]:
            pred = pred[:, :, :, :shape_hint[1]]
        
        return pred
    
    @torch.no_grad()
    def sample_control(self, cond, guidance_fn=None):
        """Sample control sequence.
        
        Args:
            cond: dict with 'u0', 'uT' for control tasks
            guidance_fn: function that computes guidance gradient
        Returns:
            f_pred: (B, T, X) predicted control force
        """
        shape_hint = cond.get('target_shape', (80, 120))
        
        w_shape = (
            cond['u0'].shape[0],
            self.cond_channels * 4,  # force has same wavelet structure
            (shape_hint[0] + 1) // 2,
            (shape_hint[1] + 1) // 2,
        )
        
        w_cond = self.encode_cond(cond)
        w_pred = self.sample(w_shape, w_cond, guidance_fn=guidance_fn)
        
        f_pred = self.decode(w_pred)
        if f_pred.shape[2] > shape_hint[0]:
            f_pred = f_pred[:, :, :shape_hint[0], :]
        if f_pred.shape[3] > shape_hint[1]:
            f_pred = f_pred[:, :, :, :shape_hint[1]]
        
        return f_pred


class WDNO2D(WDNOBase):
    """WDNO for 2D PDE problems (incompressible fluid, ERA5).
    
    Uses 3D wavelet transform (time x space_x x space_y).
    """
    
    def __init__(self, config, task='simulation', data_channels=1, cond_channels=1):
        super().__init__(config, experiment_type='2d', task=task)
        self.data_channels = data_channels
        self.cond_channels = cond_channels
        
        unet_cfg = config['unet_3d']
        self.noise_pred_net = UNet3D(
            in_channels=data_channels * 8,  # 3D wavelet gives 8x channels
            out_channels=data_channels * 8,
            cond_channels=cond_channels * 8,
            init_dim=unet_cfg['init_dim'],
            attn_heads=unet_cfg['attn_heads'],
        )
    
    def get_noise_prediction_net(self):
        return self.noise_pred_net
    
    def encode(self, x):
        """Encode data to 3D wavelet domain.
        
        Args:
            x: (B, C, T, H, W)
        Returns:
            w: (B, 8*C, T//2, H//2, W//2)
        """
        return self.wavelet_3d.decompose(x)
    
    def decode(self, w):
        """Decode 3D wavelet coefficients."""
        return self.wavelet_3d.reconstruct(w)
    
    def encode_cond(self, cond):
        """Encode conditioning to wavelet domain."""
        w_conds = []
        
        if 'density0' in cond:
            d0 = cond['density0']  # (B, H, W)
            d0 = d0.unsqueeze(1)  # (B, 1, H, W)
            w_d0 = self.wavelet_2d.decompose(d0)  # (B, 4, H//2, W//2)
            # Repeat across time dimension
            target_T = cond.get('target_shape', [32, 64, 64])[0] // 2
            w_d0 = w_d0.unsqueeze(2).repeat(1, 1, target_T, 1, 1)
            w_conds.append(w_d0)
        
        if 'control' in cond:
            ctrl = cond['control']  # (B, C_ctrl, T, H, W)
            w_ctrl = self.wavelet_3d.decompose(ctrl)
            w_conds.append(w_ctrl)
        
        if 'velocity0' in cond:
            v0 = cond['velocity0']  # (B, C_vel, H, W)
            v0 = v0.unsqueeze(3) if v0.dim() == 3 else v0
            if v0.dim() == 3:
                v0 = v0.unsqueeze(1)
            w_v0 = self.wavelet_2d.decompose(v0)
            target_T = cond.get('target_shape', [32, 64, 64])[0] // 2
            w_v0 = w_v0.unsqueeze(2).repeat(1, 1, target_T, 1, 1)
            w_conds.extend([w_v0])
        
        if not w_conds:
            raise ValueError("No conditioning provided")
        
        return torch.cat(w_conds, dim=1)
    
    def training_step(self, batch):
        """Single training step for BRM."""
        x = batch['data']
        cond = batch['cond']
        
        w_x = self.encode(x)
        w_cond = self.encode_cond(cond)
        
        loss = self.forward_diffusion_loss(w_x, w_cond)
        return loss
    
    @torch.no_grad()
    def sample_simulation(self, cond, shape_hint=None):
        """Sample 2D simulation trajectory."""
        if shape_hint is None:
            shape_hint = cond.get('target_shape', (32, 64, 64))
        
        T, H, W = shape_hint
        
        w_shape = (
            cond['density0'].shape[0],
            self.data_channels * 8,
            (T + 1) // 2,
            (H + 1) // 2,
            (W + 1) // 2,
        )
        
        w_cond = self.encode_cond(cond)
        w_pred = self.sample(w_shape, w_cond)
        pred = self.decode(w_pred)
        
        # Crop
        if pred.shape[2] > T:
            pred = pred[:, :, :T, :, :]
        for d in [3, 4]:
            if pred.shape[d] > shape_hint[d - 2]:
                pred = pred.narrow(d, 0, shape_hint[d - 2])
        
        return pred


class SuperResolutionModel(nn.Module):
    """Super-Resolution Model (SRM) for multi-resolution training.
    
    Conditions on low-resolution wavelet coefficients and equation parameters
    to generate high-resolution wavelet coefficients.
    
    Learns p(W_h | W_l, W_{a_h}) where h = high-res, l = low-res.
    """
    
    def __init__(self, config, experiment_type='1d', data_channels=1, cond_channels=2):
        super().__init__()
        self.experiment_type = experiment_type
        
        if experiment_type == '1d':
            self.wavelet_2d = WaveletTransform2D(
                wavelet=config['wavelet']['type_1d'],
                mode=config['wavelet']['mode_1d']
            )
            # U-Net: input = noisy high-res + low-res duplicated + high-res cond
            unet_cfg = config['unet_1d']
            total_in = data_channels * 4  # noisy high-res
            total_in += data_channels * 4  # low-res duplicated
            total_in += cond_channels * 4  # high-res condition
            self.noise_pred_net = UNet2D(
                in_channels=total_in,
                out_channels=data_channels * 4,
                cond_channels=0,  # already included in input
                init_dim=unet_cfg['init_dim'],
                dim_mult=unet_cfg['dim_mult_theta'],
                down_up_layers=unet_cfg['down_up_layers'],
                kernel_size=unet_cfg['kernel_size'],
                resnet_groups=unet_cfg['resnet_groups'],
                attn_heads=unet_cfg['attn_heads'],
                attn_hidden_dim=unet_cfg['attn_hidden_dim'],
            )
        elif experiment_type == '2d':
            self.wavelet_3d = WaveletTransform3D(
                wavelet=config['wavelet']['type_2d'],
                mode=config['wavelet']['mode_2d']
            )
            unet_cfg = config['unet_3d']
            total_in = data_channels * 8  # noisy high-res
            total_in += data_channels * 8  # low-res duplicated
            total_in += cond_channels * 8  # high-res condition
            self.noise_pred_net = UNet3D(
                in_channels=total_in,
                out_channels=data_channels * 8,
                cond_channels=0,
                init_dim=unet_cfg['init_dim'],
                attn_heads=unet_cfg['attn_heads'],
            )
        
        self.diffusion = Diffusion(
            timesteps=config['diffusion']['num_timesteps'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
            schedule=config['diffusion']['schedule']
        )
        self.ddim_steps = config['ddim']['sampling_steps']
        self.ddim_eta = config['ddim']['eta']
    
    def forward_diffusion_loss(self, x_high, x_low, cond):
        """Compute training loss for SRM.
        
        Args:
            x_high: high-resolution wavelet coefficients
            x_low: low-resolution wavelet coefficients (duplicated to high-res size)
            cond: high-resolution conditioning wavelet coefficients
        """
        batch_size = x_high.shape[0]
        t = torch.randint(1, self.diffusion.timesteps, (batch_size,), device=x_high.device)
        
        x_input = torch.cat([x_high, x_low, cond], dim=1)
        
        noise = torch.randn_like(x_high)
        x_t = self.diffusion.q_sample(x_high, t, noise=noise)
        x_t = torch.cat([x_t, x_low, cond], dim=1)
        
        predicted_noise = self.noise_pred_net(x_t, t, cond=None)
        return (noise - predicted_noise).pow(2).mean()
    
    @torch.no_grad()
    def super_resolve(self, w_low, w_cond_high, target_shape, num_sr_steps=1):
        """Perform zero-shot super-resolution.
        
        Args:
            w_low: low-resolution wavelet coefficients (B, C, *spatial_low)
            w_cond_high: high-resolution conditioning wavelet coefficients
            target_shape: target spatial shape after super-resolution
            num_sr_steps: number of super-resolution steps
        
        Returns:
            w_high: high-resolution wavelet coefficients
        """
        current = w_low
        current_shape = w_low.shape[2:]
        
        for step in range(num_sr_steps):
            # Calculate intermediate target shape
            new_shape = tuple(s * 2 for s in current_shape)
            
            # Duplicate low-res to match target shape
            w_low_dup = duplicate_low_res_to_high_res(current, new_shape)
            
            # Prepare conditioning at current resolution level
            if w_cond_high.shape[2:] != new_shape:
                w_cond_current = F.interpolate(
                    w_cond_high, size=new_shape, mode='bilinear' if self.experiment_type == '1d' else 'trilinear',
                    align_corners=False
                )
            else:
                w_cond_current = w_cond_high
            
            # Sample from SRM
            w_shape = (current.shape[0], current.shape[1], *new_shape)
            w_noisy = torch.randn(w_shape).to(current.device)
            
            x_input = torch.cat([w_noisy, w_low_dup, w_cond_current], dim=1)
            
            w_new = self.diffusion.ddim_sample(
                self.noise_pred_net, w_shape,
                cond=None,
                sampling_steps=self.ddim_steps,
                eta=self.ddim_eta,
            )
            
            current = w_new
            current_shape = new_shape
        
        # Final crop/pad to match exact target shape
        if current_shape != target_shape:
            for d in range(len(target_shape)):
                if current.shape[d + 2] != target_shape[d]:
                    current = current.narrow(d + 2, 0, min(current.shape[d + 2], target_shape[d]))
        
        return current
