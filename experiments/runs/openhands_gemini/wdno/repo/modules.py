
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from einops import rearrange, reduce
from einops.layers.torch import EinMix

from layers import ResnetBlock, SinusoidalPosEmb, Attention, Downsample, Upsample, Block3D, Conv3DBlock, Downsample3D, Upsample3D

# Helper functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3, # Channels for the noisy input (W_u)
        cond_channels=0, # Channels for the conditioning input (W_a, W_low_res)
        with_time_emb=True,
        resnet_block_groups=8,
        attn_heads=4,
        attn_dim_head=32,
        is_3d=False,
        conv_kernel_size=(3,3,3),
        conv_padding=(1,1,1),
        conv_stride=(1,1,1),
        downsample_kernel=(1,4,4),
        downsample_stride=(1,2,2),
        downsample_padding=(0,1,1)
    ):
        super().__init__()

        self.channels = channels
        self.cond_channels = cond_channels
        self.is_3d = is_3d

        init_dim = default(init_dim, dim)
        input_channels = channels + cond_channels # Total channels after concatenation

        if is_3d:
            self.init_conv = nn.Conv3d(input_channels, init_dim, conv_kernel_size, padding=conv_padding)
        else:
            self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        block3d_klass = partial(Block3D, groups=resnet_block_groups, kernel_size=conv_kernel_size, padding=conv_padding, stride=conv_stride)

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block3d_klass(dim_in, dim_out) if is_3d else block_klass(dim_in, dim_out),
                block3d_klass(dim_out, dim_out) if is_3d else block_klass(dim_out, dim_out),
                Attention(dim_out, heads=attn_heads, dim_head=attn_dim_head) if not is_3d else nn.Identity(), # 3D Attention is complex, use Identity for now
                Downsample3D(dim_out, kernel=downsample_kernel, stride=downsample_stride, padding=downsample_padding) if is_3d and not is_last else (Downsample(dim_out) if not is_last else nn.Identity())
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block3d_klass(mid_dim, mid_dim) if is_3d else block_klass(mid_dim, mid_dim)
        self.mid_attn = Attention(mid_dim, heads=attn_heads, dim_head=attn_dim_head) if not is_3d else nn.Identity()
        self.mid_block2 = block3d_klass(mid_dim, mid_dim) if is_3d else block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block3d_klass(dim_out * 2, dim_in) if is_3d else block_klass(dim_out * 2, dim_in),
                block3d_klass(dim_in, dim_in) if is_3d else block_klass(dim_in, dim_in),
                Attention(dim_in, heads=attn_heads, dim_head=attn_dim_head) if not is_3d else nn.Identity(),
                Upsample3D(dim_in, kernel=downsample_kernel, stride=downsample_stride, padding=downsample_padding) if is_3d and not is_last else (Upsample(dim_in) if not is_last else nn.Identity())
            ]))

        # Output convolution should produce 'channels' number of channels, as it predicts noise for 'x'
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block3d_klass(init_dim, init_dim) if is_3d else block_klass(init_dim, init_dim),
            nn.Conv3d(init_dim, out_dim, 1) if is_3d else nn.Conv2d(init_dim, out_dim, 1)
        )

    def forward(self, x, time, cond=None):
        # x: noisy input (W_u^(k))
        # time: time embedding
        # cond: conditioning input (W_a, W_low_res)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        if cond is not None:
            # Assume cond has compatible spatial dimensions
            x = torch.cat([x, cond], dim=1) # Concatenate along channel dimension

        x = self.init_conv(x)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)
            x = block2(x)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


# Gaussian Diffusion utilities
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        beta_schedule='linear',
        objective='pred_noise',
        auto_normalize=True,
        channels=1,
        loss_type='l2'
    ):
        super().__init__()
        self.model = model
        self.channels = channels
        self.image_size = image_size # This could be (time, H, W) for 3D or (H, W) for 2D

        self.objective = objective
        self.loss_type = loss_type

        if beta_schedule == 'linear':
            betas = self.linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer provides a way to store state that shouldn't be considered a model parameter
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_0) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation for the posterior variance running in the denoising process
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

    def predict_noise_from_xstart(self, x_t, t, x_start):
        return (
            (x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_start) /
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        )

    def predict_xstart_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        noise = self.model(x, t, x_self_cond)
        x_start = self.predict_xstart_from_noise(x, t, noise)

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_self_cond=x_self_cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        device = self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((shape[0],), t, device=device, dtype=torch.long))
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, timesteps, ddim_eta):
        batch, device = shape[0], self.betas.device
        total_timesteps = self.num_timesteps
        ddim_timesteps = torch.arange(0, total_timesteps, total_timesteps // timesteps).long()
        ddim_timesteps = list(ddim_timesteps.flip(0).cpu().numpy())
        ddim_timesteps = ddim_timesteps[:-1] + [0] # Add t=0

        img = torch.randn(shape, device=device)
        x_start = None

        for i, t in enumerate(ddim_timesteps[:-1]):
            t_next = ddim_timesteps[i+1]
            alpha_prod = extract(self.alphas_cumprod, torch.tensor([t], device=device), img.shape)
            alpha_prod_next = extract(self.alphas_cumprod, torch.tensor([t_next], device=device), img.shape)
            beta_prod = 1 - alpha_prod

            if x_start is None: # Initial prediction
                model_output = self.model(img, torch.full((batch,), t, device=device, dtype=torch.long))
                pred_noise = model_output
                x_start = self.predict_xstart_from_noise(img, torch.full((batch,), t, device=device, dtype=torch.long), pred_noise)
                x_start.clamp_(-1., 1.)
            else: # Use previous x_start for re-estimation
                pred_noise = self.predict_noise_from_xstart(img, torch.full((batch,), t, device=device, dtype=torch.long), x_start)


            sigma = ddim_eta * torch.sqrt((1 - alpha_prod_next) / (1 - alpha_prod) * (1 - alpha_prod / alpha_prod_next))
            if t_next == 0:
                sigma = 0

            mean_pred = x_start * torch.sqrt(alpha_prod_next) + torch.sqrt(1 - alpha_prod_next - sigma**2) * pred_noise
            img = mean_pred + sigma * torch.randn_like(img)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Sample x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if self.loss_type == 'l1':
            loss = F.l1_loss(model_out, target)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(model_out, target)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(model_out, target)
        else:
            raise ValueError(f'unknown loss type {self.loss_type}')

        return loss

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, t, *args, **kwargs)

