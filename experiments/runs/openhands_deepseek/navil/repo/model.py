"""NaViL: Native Multimodal Large Language Model.

End-to-end model combining:
- Visual Encoder (bidirectional, 2D-RoPE)
- Connector (PixelShuffle + MLP projection)
- MoE-extended LLM (modality-specific attention + FFN experts)

Supports visual multi-scale packing for any-resolution inference.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NaViLConfig, TrainingConfig, VisualEncoderConfig, LLMConfig, ConnectorConfig
from modules import (
    Connector,
    MoEDecoder,
    VisualEncoder,
)


class NaViL(nn.Module):
    """NaViL: Native MLLM with modality-specific MoE.

    Architecture (Fig. 8):
    1. Visual Encoder: extracts visual tokens with bidirectional attention
    2. Connector: downsamples via PixelShuffle + MLP projection
    3. MoE LLM: processes concatenated visual and text tokens with
       modality-specific attention and FFN experts

    Special tokens:
    - <begin_of_image>, <end_of_image>: mark image token boundaries
    - <end_of_line>: inserted at end of each row of image patches
    - <end_of_scale>: inserted after each scale's image tokens
    """

    def __init__(self, config: NaViLConfig):
        super().__init__()
        self.config = config

        self.visual_encoder = VisualEncoder(
            depth=config.visual_encoder.depth,
            width=config.visual_encoder.width,
            mlp_width=config.visual_encoder.mlp_width,
            n_heads=config.visual_encoder.n_heads,
            patch_size=config.visual_encoder.patch_size,
            max_image_size=config.visual_encoder.max_image_size,
            dropout=config.visual_encoder.dropout,
        )

        self.connector = Connector(
            visual_dim=config.visual_encoder.width,
            llm_dim=config.llm.dim,
            pixel_shuffle_scale=config.connector.pixel_shuffle_scale,
            mlp_hidden_mult=config.connector.mlp_hidden_mult,
        )

        self.llm = MoEDecoder(
            depth=config.llm.depth,
            dim=config.llm.dim,
            n_heads=config.llm.n_heads,
            mlp_dim=config.llm.mlp_dim,
            vocab_size=config.llm.vocab_size,
            max_seq_len=config.llm.max_seq_len,
            dropout=config.llm.dropout,
        )

        self.special_tokens = config.special_tokens
        self.use_moe = config.use_moe

    def encode_images(
        self,
        images: torch.Tensor,
        return_spatial: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Encode images through visual encoder + connector.

        Args:
            images: [batch, 3, H, W] (pre-padded to multiples of patch_size)
            return_spatial: if True, return (h, w) of encoded feature map

        Returns:
            visual_tokens: [batch, num_patches, llm_dim]
        """
        tokens, h, w = self.visual_encoder(images)
        projected = self.connector(tokens, h, w)
        if return_spatial:
            return projected, h, w
        return projected

    def encode_images_multiscale(
        self,
        image: torch.Tensor,
        downsample_rate: float,
        min_area: int = 256,
    ) -> List[torch.Tensor]:
        """Encode images with visual multi-scale packing.

        Continuously downsamples image until area < min_area,
        encodes each scale independently.

        Args:
            image: [3, H, W] single image
            downsample_rate: tau, e.g., sqrt(2)/2
            min_area: threshold to stop downsampling

        Returns:
            list of [num_patches_i, llm_dim] for each scale
        """
        scales = [image]
        h, w = image.shape[-2:]

        while h * w * (downsample_rate ** 2) >= min_area:
            h = int(h * downsample_rate)
            w = int(w * downsample_rate)
            if h < 1 or w < 1:
                break
            downscaled = F.interpolate(
                image.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False,
            ).squeeze(0)
            scales.append(downscaled)

        encoded = []
        for scale_img in scales:
            if scale_img.dim() == 3:
                scale_img = scale_img.unsqueeze(0)
            tokens = self.encode_images(scale_img)
            encoded.append(tokens.squeeze(0))
        return encoded

    def _get_special_token_ids(self, tokenizer) -> Dict[str, int]:
        """Get token IDs for special tokens."""
        ids = {}
        for name, token in self.special_tokens.items():
            if hasattr(tokenizer, "encode"):
                ids[name] = tokenizer.encode(token)[0]
            elif hasattr(tokenizer, "token_to_id"):
                ids[name] = tokenizer.token_to_id(token)
            else:
                raise ValueError(f"Cannot get token id for {token}")
        return ids

    def build_input_sequence(
        self,
        input_ids: torch.Tensor,
        visual_features: torch.Tensor,
        special_token_ids: Dict[str, int],
        image_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build multimodal input by replacing image placeholder tokens with visual features.

        Args:
            input_ids: [batch, text_seq_len] tokenized text with image placeholders
            visual_features: [batch, num_vis_tokens, llm_dim] encoded image features
            special_token_ids: mapping of special token names to IDs
            image_token_id: token ID used as image placeholder

        Returns:
            embeddings: [batch, total_seq_len, llm_dim]
            modality_mask: [batch, total_seq_len] True=visual, False=text
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        dim = self.config.llm.dim

        text_emb = self.llm.token_embedding(input_ids)
        all_embeddings = []
        all_masks = []

        begin_id = special_token_ids.get("begin_of_image", -1)
        end_id = special_token_ids.get("end_of_image", -1)
        eol_id = special_token_ids.get("end_of_line", -1)

        for b in range(batch_size):
            ids = input_ids[b]
            emb_batch = []
            mask_batch = []

            i = 0
            while i < len(ids):
                tok_id = ids[i].item()

                if tok_id == begin_id:
                    emb_batch.append(text_emb[b, i:i + 1])
                    mask_batch.append(torch.zeros(1, dtype=torch.bool, device=device))
                    i += 1

                    num_vis = visual_features[b].shape[0]
                    emb_batch.append(visual_features[b])
                    mask_batch.append(torch.ones(num_vis, dtype=torch.bool, device=device))

                elif tok_id == end_id or tok_id == eol_id:
                    emb_batch.append(text_emb[b, i:i + 1])
                    mask_batch.append(torch.zeros(1, dtype=torch.bool, device=device))
                    i += 1
                else:
                    emb_batch.append(text_emb[b, i:i + 1])
                    mask_batch.append(torch.zeros(1, dtype=torch.bool, device=device))
                    i += 1

            all_embeddings.append(torch.cat(emb_batch, dim=0))
            all_masks.append(torch.cat(mask_batch, dim=0))

        max_len = max(e.shape[0] for e in all_embeddings)
        padded_emb = torch.zeros(batch_size, max_len, dim, device=device, dtype=all_embeddings[0].dtype)
        padded_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

        for b in range(batch_size):
            length = all_embeddings[b].shape[0]
            padded_emb[b, :length] = all_embeddings[b]
            padded_mask[b, :length] = all_masks[b]

        return padded_emb, padded_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        special_token_ids: Dict[str, int],
        labels: Optional[torch.Tensor] = None,
        use_multiscale: bool = False,
        downsample_rate: float = 0.7071067811865476,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass with next-token prediction.

        Args:
            input_ids: [batch, text_seq_len] text with image placeholders
            images: [batch, 3, H, W] padded input images
            special_token_ids: mapping of special token names to IDs
            labels: [batch, text_seq_len] for computing NTP loss
            use_multiscale: enable visual multi-scale packing
            downsample_rate: tau for multi-scale

        Returns:
            dict with keys: logits, loss (if labels provided)
        """
        if use_multiscale:
            all_vis_features = []
            for b in range(images.shape[0]):
                scales = self.encode_images_multiscale(
                    images[b], downsample_rate,
                )
                all_vis_features.append(torch.cat(scales, dim=0))
            batch_vis_features = torch.stack(all_vis_features)
        else:
            batch_vis_features = self.encode_images(images)

        embeddings, modality_mask = self.build_input_sequence(
            input_ids, batch_vis_features, special_token_ids,
        )

        batch, seq_len, _ = embeddings.shape
        freqs_cis = self.llm._freqs_cis.to(embeddings.device)

        from layers import create_causal_mask
        attn_mask = create_causal_mask(seq_len, embeddings.device, embeddings.dtype)

        x = embeddings
        for layer in self.llm.layers:
            x = layer(x, modality_mask, freqs_cis, attn_mask)

        x = self.llm.final_norm(x)
        logits = self.llm.lm_head(x)

        result = {"logits": logits}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        special_token_ids: Dict[str, int],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_multiscale: bool = True,
        downsample_rate: float = 0.7071067811865476,
    ) -> torch.Tensor:
        """Autoregressive generation for inference."""
        if use_multiscale:
            all_vis_features = []
            for b in range(images.shape[0]):
                scales = self.encode_images_multiscale(images[b], downsample_rate)
                all_vis_features.append(torch.cat(scales, dim=0))
            batch_vis_features = torch.stack(all_vis_features)
        else:
            batch_vis_features = self.encode_images(images)

        embeddings, modality_mask = self.build_input_sequence(
            input_ids, batch_vis_features, special_token_ids,
        )

        x = embeddings
        generated_ids = input_ids.clone()
        device = input_ids.device

        for _ in range(max_new_tokens):
            seq_len = x.shape[1]
            freqs_cis = self.llm._freqs_cis.to(device)
            from layers import create_causal_mask
            attn_mask = create_causal_mask(seq_len, device, x.dtype)

            for layer in self.llm.layers:
                x = layer(x, modality_mask, freqs_cis, attn_mask)

            x_normed = self.llm.final_norm(x)
            logits = self.llm.lm_head(x_normed[:, -1:, :])

            if temperature > 0:
                logits = logits / temperature
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")
                if top_k is not None:
                    top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
                    min_vals = top_k_vals[..., -1:]
                    logits[logits < min_vals] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs.squeeze(0), num_samples=1)

            next_embed = self.llm.token_embedding(next_token)
            x = torch.cat([x, next_embed], dim=1)

            new_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
            modality_mask = torch.cat([modality_mask, new_mask], dim=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        return generated_ids
