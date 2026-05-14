import torch
import torch.nn as nn
import torch.nn.functional as F

from nfig.utils.frequency_utils import generate_frequency_masks

class FrequencyGuidedDecomposer(nn.Module):
    def __init__(self, num_frequency_bands: int = 10, scaling_factors: list[int] = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]):
        super().__init__()
        self.num_frequency_bands = num_frequency_bands
        self.scaling_factors = scaling_factors

    def forward(self, f: torch.Tensor) -> list[torch.Tensor]:
        # f: B x C x H' x W'
        batch_size, channels, H_prime, W_prime = f.shape
        device = f.device

        # Generate frequency masks dynamically based on current feature map size
        frequency_masks = generate_frequency_masks(H_prime, W_prime, self.scaling_factors, device)

        # Perform 2D FFT
        fft_f = torch.fft.fft2(f, dim=(-2, -1))
        fft_f_shifted = torch.fft.fftshift(fft_f, dim=(-2, -1)) # Shift zero-frequency component to center

        hat_f_i_list = []
        for i in range(self.num_frequency_bands):
            # Apply the i-th frequency mask (M_i) to the shifted FFT spectrum
            # Mask needs to be broadcastable: (1, 1, H', W') multiplied by (B, C, H', W')
            masked_fft_f_shifted = fft_f_shifted * frequency_masks[i]
            
            # Shift back and inverse FFT
            masked_fft_f = torch.fft.ifftshift(masked_fft_f_shifted, dim=(-2, -1))
            hat_f_i = torch.fft.ifft2(masked_fft_f, dim=(-2, -1)).real # Take real part of complex output
            hat_f_i_list.append(hat_f_i)
            
        return hat_f_i_list

class FrequencyGuidedComposer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hat_f_i_list: list[torch.Tensor], target_h: int, target_w: int) -> torch.Tensor:
        if not hat_f_i_list:
            # Handle case where list is empty, though typically should not happen if num_frequency_bands > 0
            return torch.zeros(1, 1, target_h, target_w, device=hat_f_i_list[0].device) # Return a zero tensor with appropriate dimensions

        # Initialize tilde_f with zeros, matching the expected output shape.
        batch_size, channels, _, _ = hat_f_i_list[0].shape
        tilde_f = torch.zeros(batch_size, channels, target_h, target_w, device=hat_f_i_list[0].device)

        for hat_f_i in hat_f_i_list:
            # Interpolate to target size if necessary
            if hat_f_i.shape[-2:] != (target_h, target_w):
                interpolated_f_i = F.interpolate(hat_f_i, size=(target_h, target_w), mode='bilinear', align_corners=False)
            else:
                interpolated_f_i = hat_f_i
            tilde_f += interpolated_f_i
        return tilde_f

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_x = x.permute(0, 2, 3, 1).contiguous() # B x H x W x C
        original_shape = flat_x.shape
        flat_x = flat_x.view(-1, self.embedding_dim) # (B*H*W) x C

        distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1) # (B*H*W)

        quantized = self.embedding(encoding_indices) # (B*H*W) x C
        
        quantized = quantized.view(original_shape).permute(0, 3, 1, 2).contiguous()
        
        quantized_st = x + (quantized - x).detach()

        token_indices = encoding_indices.view(x.shape[0], x.shape[2], x.shape[3])

        return quantized_st, token_indices

class FrequencyGuidedResidualQuantization(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int, scaling_factors: list[int]):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.scaling_factors = scaling_factors
        self.num_frequency_bands = len(scaling_factors)

        self.quantizer = VectorQuantizer(codebook_size, embedding_dim)

    def forward(self, hat_f_i_list: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        quantized_v_i_list = [] 
        token_indices_list = [] 

        if not hat_f_i_list:
            return [], []
        
        H_prime, W_prime = hat_f_i_list[0].shape[-2:]
        batch_size, channels = hat_f_i_list[0].shape[:2]

        current_residual = torch.zeros(batch_size, channels, H_prime, W_prime, device=hat_f_i_list[0].device)

        for i in range(self.num_frequency_bands):
            current_hat_f_i = hat_f_i_list[i] 
            
            if i == 0:
                signal_to_quantize_for_vi = current_hat_f_i
            else:
                signal_to_quantize_for_vi = current_residual + current_hat_f_i

            s = self.scaling_factors[i]
            h_i, w_i = H_prime // s, W_prime // s

            v_i_input = F.interpolate(signal_to_quantize_for_vi, size=(h_i, w_i), mode='bilinear', align_corners=False)
            
            quantized_v_i, token_indices_i = self.quantizer(v_i_input) 

            upsampled_quantized_v_i = F.interpolate(quantized_v_i, size=(H_prime, W_prime), mode='bilinear', align_corners=False)
            
            current_residual = signal_to_quantize_for_vi - upsampled_quantized_v_i
            
            quantized_v_i_list.append(quantized_v_i) 
            token_indices_list.append(token_indices_i) 

        return quantized_v_i_list, token_indices_list

class Encoder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, num_res_blocks: int, ch_mult: list[int]):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        current_channels = embedding_dim
        for i, mult in enumerate(ch_mult):
            out_channels = embedding_dim * mult
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.AvgPool2d(2) # Downsample
            ))
            current_channels = out_channels
        self.final_conv = nn.Conv2d(current_channels, embedding_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        f = self.final_conv(h) 
        return f

class Decoder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, num_res_blocks: int, ch_mult: list[int]):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, embedding_dim * ch_mult[-1] // ch_mult[-1] if ch_mult else embedding_dim, kernel_size=3, padding=1) # Adjusted initial conv to handle potentially empty ch_mult or first mult
        
        self.up_blocks = nn.ModuleList()
        current_channels = embedding_dim * (ch_mult[-1] if ch_mult else 1) # If ch_mult is empty, assume multiplier of 1
        for i, mult in enumerate(reversed(ch_mult)):
            out_channels = embedding_dim * mult
            self.up_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'), 
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ))
            current_channels = out_channels
        
        self.final_conv = nn.Conv2d(current_channels, embedding_dim, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(embedding_dim, 3, kernel_size=3, padding=1) 

    def forward(self, f_hat: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(f_hat)
        for block in self.up_blocks:
            h = block(h)
        h = self.final_conv(h)
        x_recon = self.conv_out(h) 
        return x_recon

class FRVAE(nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 embedding_dim: int = 256, 
                 codebook_size: int = 4096, 
                 encoder_ch_mult: list[int] = [1, 1, 2, 2, 4], 
                 decoder_ch_mult: list[int] = [4, 2, 2, 1, 1], 
                 num_res_blocks: int = 2, 
                 scaling_factors: list[int] = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]):
        super().__init__()
        self.encoder = Encoder(in_channels, embedding_dim, num_res_blocks, encoder_ch_mult)
        self.decomposer = FrequencyGuidedDecomposer(num_frequency_bands=len(scaling_factors), scaling_factors=scaling_factors) 
        self.residual_quantization = FrequencyGuidedResidualQuantization(codebook_size, embedding_dim, scaling_factors)
        self.composer = FrequencyGuidedComposer()
        self.decoder = Decoder(embedding_dim, embedding_dim, num_res_blocks, decoder_ch_mult) 

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        f = self.encoder(x) 

        hat_f_i_list = self.decomposer(f)

        quantized_v_i_list, token_indices_list = self.residual_quantization(hat_f_i_list)
        
        if not quantized_v_i_list: 
            return torch.zeros_like(x), []

        H_prime, W_prime = f.shape[-2:]
        
        composed_quantized_f = self.composer(quantized_v_i_list, H_prime, W_prime)
        
        x_recon = self.decoder(composed_quantized_f)

        return x_recon, token_indices_list
