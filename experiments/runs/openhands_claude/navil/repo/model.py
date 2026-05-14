"""
NaViL: Native Multimodal Large Language Model

Full model combining:
  - VisualEncoder (bidirectional transformer + 2D-RoPE)
  - PixelShuffleConnector (pixel shuffle + MLP)
  - MoEExtendedLLM (causal transformer with modality-specific MoE)
  - Visual Multi-scale Packing (Sec. 4.1)
  - Special token handling: <begin_of_image>, <end_of_image>,
                            <end_of_line>, <end_of_scale>
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import (
    MoEConfig,
    NaViLConfig,
    get_navil_2b_config,
    get_navil_9b_config,
)
from layers import PixelShuffleConnector
from modules import MoEExtendedLLM, VisualEncoder


# ── Modality IDs ──────────────────────────────────────────────────────────────
VISUAL_MODALITY     = 0
LINGUISTIC_MODALITY = 1


# ── NaViL Model ───────────────────────────────────────────────────────────────

class NaViL(nn.Module):
    """
    Native Multimodal Large Language Model.

    Architecture (Fig. 8):
      1. Images are encoded by VisualEncoder -> visual token embeddings
      2. PixelShuffleConnector downsamples and projects to LLM dim
      3. Special tokens are inserted around image token sequences
      4. Visual and text tokens are concatenated and fed to MoEExtendedLLM
      5. Next-token prediction loss over the full sequence

    Visual Multi-scale Packing (Sec. 4.1):
      - Original image is repeatedly downsampled by τ = √2/2
      - Each scale is encoded separately
      - Scale embeddings are concatenated with <end_of_scale> separators
    """

    def __init__(self, cfg: NaViLConfig):
        super().__init__()
        self.cfg = cfg

        self.visual_encoder = VisualEncoder(cfg.visual_encoder)
        self.connector = PixelShuffleConnector(
            visual_dim=cfg.visual_encoder.width,
            llm_dim=cfg.llm.width,
            downsample_factor=cfg.connector.pixel_shuffle_factor,
        )
        self.llm = MoEExtendedLLM(cfg.llm, cfg.moe)

    # ── Special token embedding helpers ───────────────────────────────────────

    def _get_special_token_embed(self, token_id: int) -> torch.Tensor:
        """Return embedding for a special token. Shape: (1, 1, D)."""
        tid = torch.tensor([[token_id]], device=next(self.parameters()).device)
        return self.llm.embed_tokens(tid)

    # ── Visual Multi-scale Packing ─────────────────────────────────────────────

    def _build_multiscale_sequence(
        self,
        image: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Given a single image tensor (3, H, W), produce a list of downsampled
        images following the multi-scale packing strategy.

        τ = √2/2, stop when area < min_area.
        Returns list of image tensors at decreasing resolutions.
        """
        tau = self.cfg.multiscale_downsample_rate
        min_area = self.cfg.multiscale_min_area
        patch_size = self.cfg.visual_encoder.patch_size

        scales = [image]
        _, H, W = image.shape

        while True:
            new_H = int(H * tau)
            new_W = int(W * tau)
            # Snap to nearest multiple of patch_size
            new_H = max(patch_size, (new_H // patch_size) * patch_size)
            new_W = max(patch_size, (new_W // patch_size) * patch_size)
            if new_H * new_W < min_area:
                break
            downsampled = F.interpolate(
                image.unsqueeze(0).float(),
                size=(new_H, new_W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).to(image.dtype)
            scales.append(downsampled)
            H, W = new_H, new_W

        return scales

    # ── Encode a single image (all scales) ────────────────────────────────────

    def encode_image(
        self,
        image: torch.Tensor,
        use_multiscale: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode one image into a sequence of visual token embeddings.

        Returns:
          visual_embeds: (N_vis, D_llm)  — projected visual tokens
          row_end_mask:  (N_vis,) bool   — True at positions where <end_of_line>
                                           should be inserted
        """
        cfg = self.cfg
        patch_size = cfg.visual_encoder.patch_size
        r = cfg.connector.pixel_shuffle_factor

        if use_multiscale:
            scales = self._build_multiscale_sequence(image)
        else:
            scales = [image]

        all_embeds = []
        all_row_ends = []

        for scale_img in scales:
            _, H, W = scale_img.shape
            # Pad to multiples of patch_size
            pad_h = (patch_size - H % patch_size) % patch_size
            pad_w = (patch_size - W % patch_size) % patch_size
            if pad_h > 0 or pad_w > 0:
                scale_img = F.pad(scale_img, (0, pad_w, 0, pad_h))

            _, H_pad, W_pad = scale_img.shape
            num_h = H_pad // patch_size
            num_w = W_pad // patch_size

            # Encode through visual encoder
            vis_tokens, enc_h, enc_w = self.visual_encoder(
                scale_img.unsqueeze(0)
            )   # (1, enc_h*enc_w, vis_dim)

            # Connector: pixel shuffle + MLP
            proj_tokens, proj_h, proj_w = self.connector(
                vis_tokens, enc_h, enc_w
            )   # (1, proj_h*proj_w, llm_dim)

            proj_tokens = proj_tokens.squeeze(0)   # (proj_h*proj_w, llm_dim)

            # Build row-end mask: True at the last token of each row
            row_end = torch.zeros(proj_h * proj_w, dtype=torch.bool)
            for row in range(proj_h):
                row_end[(row + 1) * proj_w - 1] = True

            all_embeds.append(proj_tokens)
            all_row_ends.append(row_end)

        # Concatenate scales; <end_of_scale> tokens are inserted in build_input_embeds
        visual_embeds = torch.cat(all_embeds, dim=0)
        row_end_mask  = torch.cat(all_row_ends, dim=0)

        return visual_embeds, row_end_mask, [e.shape[0] for e in all_embeds]

    # ── Build full multimodal input embeddings ─────────────────────────────────

    def build_input_embeds(
        self,
        input_ids: torch.Tensor,
        images: Optional[List[torch.Tensor]],
        use_multiscale: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Replace image placeholder tokens in input_ids with actual visual embeddings.

        Conventions:
          - input_ids contains cfg.image_patch_token_id as placeholder for each
            visual token position
          - <begin_of_image> and <end_of_image> bracket each image's tokens
          - <end_of_line> is inserted after each row of image tokens
          - <end_of_scale> is inserted after each scale's tokens

        Returns:
          inputs_embeds: (1, T_total, D)
          modality_mask: (1, T_total)  — 0=visual, 1=linguistic
          attention_mask: (1, T_total) — all ones (no padding in this path)
        """
        device = input_ids.device
        embed_fn = self.llm.embed_tokens

        # Embed all text tokens first
        text_embeds = embed_fn(input_ids)   # (1, T, D)

        if images is None or len(images) == 0:
            modality_mask = torch.ones(
                input_ids.shape, dtype=torch.long, device=device
            )
            attn_mask = torch.ones_like(modality_mask)
            return text_embeds, modality_mask, attn_mask

        cfg = self.cfg
        boi_id  = cfg.begin_of_image_token_id
        eoi_id  = cfg.end_of_image_token_id
        eol_id  = cfg.end_of_line_token_id
        eos_id  = cfg.end_of_scale_token_id

        boi_emb = embed_fn(torch.tensor([[boi_id]], device=device))   # (1,1,D)
        eoi_emb = embed_fn(torch.tensor([[eoi_id]], device=device))
        eol_emb = embed_fn(torch.tensor([[eol_id]], device=device))
        eos_emb = embed_fn(torch.tensor([[eos_id]], device=device))

        # Build the full sequence token by token
        seq_embeds = []
        seq_modality = []

        # We process batch size 1 for simplicity; batched version pads
        ids = input_ids[0]   # (T,)
        txt = text_embeds[0] # (T, D)

        img_idx = 0
        t = 0
        while t < len(ids):
            tok = ids[t].item()
            if tok == boi_id and img_idx < len(images):
                # Insert <begin_of_image>
                seq_embeds.append(boi_emb[0])           # (1, D)
                seq_modality.append(torch.tensor([LINGUISTIC_MODALITY]))

                # Encode image with multi-scale packing
                vis_embeds, row_end_mask, scale_lengths = self.encode_image(
                    images[img_idx], use_multiscale=use_multiscale
                )
                vis_embeds = vis_embeds.to(device)
                row_end_mask = row_end_mask.to(device)

                # Insert visual tokens with <end_of_line> and <end_of_scale>
                pos = 0
                for s_idx, s_len in enumerate(scale_lengths):
                    for i in range(s_len):
                        seq_embeds.append(vis_embeds[pos].unsqueeze(0))
                        seq_modality.append(torch.tensor([VISUAL_MODALITY]))
                        if row_end_mask[pos]:
                            seq_embeds.append(eol_emb[0])
                            seq_modality.append(torch.tensor([LINGUISTIC_MODALITY]))
                        pos += 1
                    # <end_of_scale> after each scale
                    seq_embeds.append(eos_emb[0])
                    seq_modality.append(torch.tensor([LINGUISTIC_MODALITY]))

                # Insert <end_of_image>
                seq_embeds.append(eoi_emb[0])
                seq_modality.append(torch.tensor([LINGUISTIC_MODALITY]))

                img_idx += 1
                # Skip placeholder tokens in input_ids that correspond to this image
                # (caller is responsible for placing exactly the right number of
                #  image_patch_token_id placeholders, or we skip to eoi_id)
                t += 1
                while t < len(ids) and ids[t].item() != eoi_id:
                    t += 1
                t += 1  # skip eoi_id in input_ids (already inserted above)
            else:
                seq_embeds.append(txt[t].unsqueeze(0))
                seq_modality.append(torch.tensor([LINGUISTIC_MODALITY]))
                t += 1

        inputs_embeds = torch.cat(seq_embeds, dim=0).unsqueeze(0)   # (1, T_total, D)
        modality_mask = torch.cat(seq_modality, dim=0).unsqueeze(0) # (1, T_total)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=device
        )

        return inputs_embeds, modality_mask, attention_mask

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        images: Optional[List[torch.Tensor]] = None,
        modality_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
        use_multiscale: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with next-token prediction loss.

        When images are provided alongside input_ids, build_input_embeds is
        called to splice visual tokens into the sequence.
        """
        if inputs_embeds is None and images is not None:
            inputs_embeds, modality_mask, attention_mask = self.build_input_embeds(
                input_ids, images, use_multiscale=use_multiscale
            )
            input_ids = None

        llm_out = self.llm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            modality_mask=modality_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_logits=True,
        )

        output = {
            "logits": llm_out["logits"],
            "hidden_states": llm_out["hidden_states"],
        }
        if use_cache:
            output["past_key_values"] = llm_out["past_key_values"]

        if labels is not None:
            logits = llm_out["logits"]
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output

    # ── Inference helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[List[torch.Tensor]] = None,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        use_multiscale: bool = True,
    ) -> torch.Tensor:
        """Greedy / top-p sampling generation."""
        device = input_ids.device

        # Build initial embeddings
        if images is not None:
            inputs_embeds, modality_mask, attention_mask = self.build_input_embeds(
                input_ids, images, use_multiscale=use_multiscale
            )
        else:
            inputs_embeds = self.llm.embed_tokens(input_ids)
            modality_mask = torch.ones(
                input_ids.shape, dtype=torch.long, device=device
            )
            attention_mask = torch.ones_like(modality_mask)

        past_key_values = None
        generated = []

        # Prefill
        out = self.llm(
            inputs_embeds=inputs_embeds,
            modality_mask=modality_mask,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_logits=True,
        )
        past_key_values = out["past_key_values"]
        next_token_logits = out["logits"][:, -1, :]

        for _ in range(max_new_tokens):
            next_token = _sample(next_token_logits, temperature, top_p)
            generated.append(next_token)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Decode step
            next_embeds = self.llm.embed_tokens(next_token)
            next_modality = torch.ones(next_token.shape, dtype=torch.long, device=device)

            out = self.llm(
                inputs_embeds=next_embeds,
                modality_mask=next_modality,
                past_key_values=past_key_values,
                use_cache=True,
                return_logits=True,
            )
            past_key_values = out["past_key_values"]
            next_token_logits = out["logits"][:, -1, :]

        return torch.cat(generated, dim=1)

    # ── Parameter group helpers (for staged training) ─────────────────────────

    def get_visual_params(self) -> List[nn.Parameter]:
        """Visual encoder parameters."""
        return list(self.visual_encoder.parameters())

    def get_connector_params(self) -> List[nn.Parameter]:
        """Connector (pixel shuffle + MLP) parameters."""
        return list(self.connector.parameters())

    def get_moe_visual_params(self) -> List[nn.Parameter]:
        """Visual MoE expert parameters (expert_id=0)."""
        params = []
        for layer in self.llm.layers:
            for proj in [layer.attn.q_projs[0], layer.attn.k_projs[0],
                         layer.attn.v_projs[0], layer.attn.o_projs[0]]:
                params.extend(proj.parameters())
            for proj in [layer.ffn.gate_projs[0], layer.ffn.up_projs[0],
                         layer.ffn.down_projs[0]]:
                params.extend(proj.parameters())
        return params

    def get_moe_linguistic_params(self) -> List[nn.Parameter]:
        """Linguistic MoE expert parameters (expert_id=1)."""
        params = []
        for layer in self.llm.layers:
            for proj in [layer.attn.q_projs[1], layer.attn.k_projs[1],
                         layer.attn.v_projs[1], layer.attn.o_projs[1]]:
                params.extend(proj.parameters())
            for proj in [layer.ffn.gate_projs[1], layer.ffn.up_projs[1],
                         layer.ffn.down_projs[1]]:
                params.extend(proj.parameters())
        return params

    def get_llm_text_non_attn_params(self) -> List[nn.Parameter]:
        """LLM text parameters excluding attention projections (embed, norm, lm_head)."""
        params = []
        params.extend(self.llm.embed_tokens.parameters())
        params.extend(self.llm.norm.parameters())
        params.extend(self.llm.lm_head.parameters())
        for layer in self.llm.layers:
            params.extend(layer.norm1.parameters())
            params.extend(layer.norm2.parameters())
        return params

    def get_llm_attn_text_params(self) -> List[nn.Parameter]:
        """LLM attention text parameters (linguistic expert projections)."""
        return self.get_moe_linguistic_params()

    # ── Checkpoint utilities ───────────────────────────────────────────────────

    def load_llm_pretrained(self, state_dict: Dict[str, torch.Tensor]):
        """
        Initialize LLM from a pre-trained checkpoint.
        MoE visual expert weights are initialized randomly (new parameters).
        Linguistic expert weights are copied from the pre-trained attention/FFN.
        """
        # Map standard LLM weight names to our MoE structure
        new_state = {}
        for k, v in state_dict.items():
            # Attention projections -> linguistic expert (expert_id=1)
            if "self_attn.q_proj" in k:
                new_k = k.replace("self_attn.q_proj", "attn.q_projs.1")
                new_state[new_k] = v
            elif "self_attn.k_proj" in k:
                new_k = k.replace("self_attn.k_proj", "attn.k_projs.1")
                new_state[new_k] = v
            elif "self_attn.v_proj" in k:
                new_k = k.replace("self_attn.v_proj", "attn.v_projs.1")
                new_state[new_k] = v
            elif "self_attn.o_proj" in k:
                new_k = k.replace("self_attn.o_proj", "attn.o_projs.1")
                new_state[new_k] = v
            # FFN projections -> linguistic expert (expert_id=1)
            elif "mlp.gate_proj" in k:
                new_k = k.replace("mlp.gate_proj", "ffn.gate_projs.1")
                new_state[new_k] = v
            elif "mlp.up_proj" in k:
                new_k = k.replace("mlp.up_proj", "ffn.up_projs.1")
                new_state[new_k] = v
            elif "mlp.down_proj" in k:
                new_k = k.replace("mlp.down_proj", "ffn.down_projs.1")
                new_state[new_k] = v
            # Norms and embeddings pass through
            elif "model.embed_tokens" in k:
                new_state[k.replace("model.", "")] = v
            elif "model.norm" in k:
                new_state[k.replace("model.", "")] = v
            elif "lm_head" in k:
                new_state[k] = v
            elif "input_layernorm" in k:
                new_k = k.replace("input_layernorm", "norm1")
                new_state[new_k] = v
            elif "post_attention_layernorm" in k:
                new_k = k.replace("post_attention_layernorm", "norm2")
                new_state[new_k] = v
            else:
                new_state[k] = v

        missing, unexpected = self.llm.load_state_dict(new_state, strict=False)
        return missing, unexpected


# ── Sampling helper ───────────────────────────────────────────────────────────

def _sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Top-p (nucleus) sampling."""
    if temperature != 1.0:
        logits = logits / temperature

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ── Factory functions ─────────────────────────────────────────────────────────

def build_navil_2b(special_token_ids: Optional[Dict[str, int]] = None) -> NaViL:
    cfg = get_navil_2b_config()
    if special_token_ids:
        cfg.begin_of_image_token_id = special_token_ids.get("begin_of_image", -1)
        cfg.end_of_image_token_id   = special_token_ids.get("end_of_image", -1)
        cfg.end_of_line_token_id    = special_token_ids.get("end_of_line", -1)
        cfg.end_of_scale_token_id   = special_token_ids.get("end_of_scale", -1)
        cfg.image_patch_token_id    = special_token_ids.get("image_patch", -1)
    return NaViL(cfg)


def build_navil_9b(special_token_ids: Optional[Dict[str, int]] = None) -> NaViL:
    cfg = get_navil_9b_config()
    if special_token_ids:
        cfg.begin_of_image_token_id = special_token_ids.get("begin_of_image", -1)
        cfg.end_of_image_token_id   = special_token_ids.get("end_of_image", -1)
        cfg.end_of_line_token_id    = special_token_ids.get("end_of_line", -1)
        cfg.end_of_scale_token_id   = special_token_ids.get("end_of_scale", -1)
        cfg.image_patch_token_id    = special_token_ids.get("image_patch", -1)
    return NaViL(cfg)


# ── Ablation model builders (Sec. 3.2) ────────────────────────────────────────

def build_ablation_visual_encoder_depth_width(
    depth: int,
    width: int,
    mlp_width: int,
    num_heads: int,
    llm_size: str = "0.6B",
) -> NaViL:
    """Build a NaViL variant for the visual encoder depth/width ablation study."""
    from config import VisualEncoderConfig, LLMConfig, ConnectorConfig

    visual = VisualEncoderConfig(
        depth=depth,
        width=width,
        mlp_width=mlp_width,
        num_heads=num_heads,
    )
    # Use a small LLM for ablation (600M)
    llm = LLMConfig(
        depth=24,
        width=1024,
        mlp_width=4096,
        num_heads=16,
        num_kv_heads=8,
        vocab_size=92544,
    )
    connector = ConnectorConfig(
        pixel_shuffle_factor=2,
        input_dim=width,
        output_dim=1024,
    )
    from config import NaViLConfig, MoEConfig
    cfg = NaViLConfig(
        visual_encoder=visual,
        llm=llm,
        moe=MoEConfig(),
        connector=connector,
    )
    return NaViL(cfg)
