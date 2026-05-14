import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from config import Config
from utils import NormalizationUtils, ScaledLearnableParameter


class NGPTEmbeddings(nn.Module):
    """
    Manages the input token embeddings (E_input) and the final projection layer
    to vocabulary logits (E_output), incorporating the learnable scaling factor s_z.
    """

    def __init__(self, config: Config):
        """
        Initializes the NGPTEmbeddings module.

        Args:
            config: An instance of the Config dataclass.
        """
        super().__init__()
        self.config = config

        # Input token embeddings (E_input)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        # Initialize token embeddings. Paper Section A.6: "initialized by sampling from a
        # zero-mean normal distribution with a standard deviation of 0.02 for GPT and
        # 1/√d_model for nGPT."
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=config.init_std_dev)

        # Output projection matrix (E_output)
        # We model E_output as a linear layer. Paper Section 2.1: "E_output in R^(V x d_model)"
        # implies a linear projection from d_model to V.
        # It is usually tied to token_embeddings.weight but the paper doesn't specify tying.
        # "unless they are tied to be equivalent". We assume they are NOT tied by default.
        self.unembed_linear = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.unembed_linear.weight, mean=0.0, std=config.init_std_dev)

        # Logits scaling factor s_z (Section 2.1, Eq. 3)
        self.s_z = ScaledLearnableParameter(
            size=(config.vocab_size,),
            s_init=config.ngpt_specific_config.s_z_init,
            s_scale=config.ngpt_specific_config.s_z_scale_factor,
            name="s_z"
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for input embeddings.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length).

        Returns:
            The embedded input tensor of shape (batch_size, sequence_length, d_model).
        """
        # (B, T) -> (B, T, D_model)
        h_in = self.token_embeddings(input_ids)
        return h_in

    def get_output_logits(self, h_final: torch.Tensor) -> torch.Tensor:
        """
        Computes the final logits using the output embedding matrix and scaling factor s_z.

        Args:
            h_final: The hidden state after the last Transformer block,
                     shape (batch_size, sequence_length, d_model).

        Returns:
            The scaled logits of shape (batch_size, sequence_length, vocab_size).
        """
        # z_raw = E_output * h_final (Section 2.1)
        z_raw = self.unembed_linear(h_final)

        # Apply element-wise scaling with s_z (Section 2.1, Eq. 3)
        # s_z is (vocab_size,), z_raw is (B, T, V). This will broadcast correctly.
        z = z_raw * self.s_z.get_effective_value()
        return z

    def post_optimizer_norm(self) -> None:
        """
        Normalizes the input and output embedding matrices after each optimizer step.
        (Section 2.6, Step 2: "normalize matrices E_input, E_output along their embedding dimension.")
        """
        NormalizationUtils.normalize_embedding_dim(self.token_embeddings.weight)
        NormalizationUtils.normalize_embedding_dim(self.unembed_linear.weight)


class NGPTAttention(nn.Module):
    """
    Implements the multi-head self-attention mechanism with NGPT's specific
    normalizations, QK scaling, and modified softmax scaling.
    """

    def __init__(self, config: Config):
        """
        Initializes the NGPTAttention module.

        Args:
            config: An instance of the Config dataclass.
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k  # d_model // n_heads

        # Linear projections for query, key, value (Section 2.3.1)
        # Biases are typically omitted in Transformers for these projections.
        # Paper 2.4.1 explicitly says "we omit bias terms" for MLP, implying general practice.
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_k, self.d_model, bias=False)

        # QK scaling factor s_qk (Section 2.3.2, Eq. 15, 16)
        # s_qk is a vector of scaling factors for the i-th head (d_k dimensions).
        # To apply it to all heads, its size should be (n_heads, d_k).
        self.s_qk = ScaledLearnableParameter(
            size=(self.n_heads, self.d_k),
            s_init=config.ngpt_specific_config.s_qk_init,
            s_scale=config.ngpt_specific_config.s_qk_scale_factor,
            name="s_qk"
        )

        # Softmax scaling factor (Section 2.3.2)
        # Changed from 1/√d_k to √d_k in nGPT.
        self.scale_factor_softmax = math.sqrt(self.d_k)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits the input tensor into multiple heads.
        Input shape: (batch_size, sequence_length, n_heads * d_k)
        Output shape: (batch_size, n_heads, sequence_length, d_k)
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines multiple heads back into a single tensor.
        Input shape: (batch_size, n_heads, sequence_length, d_k)
        Output shape: (batch_size, sequence_length, n_heads * d_k)
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_k)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, current_pos: int) -> torch.Tensor:
        """
        Performs the forward pass for the attention mechanism.

        Args:
            x: Input hidden state, shape (batch_size, sequence_length, d_model).
            mask: Causal attention mask, shape (1, 1, sequence_length, sequence_length).
            current_pos: Starting position for RoPE application.

        Returns:
            Output of the attention block, shape (batch_size, sequence_length, d_model).
        """
        batch_size, seq_len, d_model = x.size()

        # Project input to Q, K, V
        q_raw = self.q_proj(x)  # (B, T, n_heads * d_k)
        k_raw = self.k_proj(x)  # (B, T, n_heads * d_k)
        v_raw = self.v_proj(x)  # (B, T, n_heads * d_k)

        # Split into multiple heads
        q_raw = self._split_heads(q_raw)  # (B, n_heads, T, d_k)
        k_raw = self._split_heads(k_raw)  # (B, n_heads, T, d_k)
        v = self._split_heads(v_raw)    # (B, n_heads, T, d_k)

        # Apply Rotary Positional Embeddings (RoPE) (Section 2.3.1)
        q_rope = NormalizationUtils.apply_rope(q_raw, current_pos, self.config.rope_base)
        k_rope = NormalizationUtils.apply_rope(k_raw, current_pos, self.config.rope_base)

        # Apply QK normalization and scaling (Section 2.3.2, Eq. 15, 16)
        # s_qk.get_effective_value() is (n_heads, d_k).
        # Unsqueeze for broadcasting with (B, n_heads, T, d_k).
        s_qk_effective = self.s_qk.get_effective_value().unsqueeze(2) # (n_heads, 1, d_k)

        q = NormalizationUtils.norm(q_rope) * s_qk_effective
        k = NormalizationUtils.norm(k_rope) * s_qk_effective

        # Compute attention scores: Q K^T (Section 2.3.1)
        # (B, n_heads, T, d_k) @ (B, n_heads, d_k, T) -> (B, n_heads, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # Scale attention scores (Section 2.3.2)
        attn_scores = attn_scores * self.scale_factor_softmax

        # Apply causal mask (Section 2.3.1)
        # M is a matrix that prevents attending to future tokens by setting corresponding entries to -inf.
        attn_scores = attn_scores.masked_fill(mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, n_heads, T, T)

        # Compute weighted sum of V (Section 2.3.1)
        # (B, n_heads, T, T) @ (B, n_heads, T, d_k) -> (B, n_heads, T, d_k)
        attn_output = torch.matmul(attn_weights, v)

        # Combine heads
        attn_output = self._combine_heads(attn_output) # (B, T, n_heads * d_k)

        # Final linear projection (Section 2.3.1)
        output = self.o_proj(attn_output) # (B, T, d_model)

        return output

    def post_optimizer_norm(self) -> None:
        """
        Normalizes the weights of the attention projection matrices after each optimizer step.
        (Section 2.6, Step 2: "normalize matrices W_q, W_k, W_v, W_o along their embedding dimension.")
        """
        NormalizationUtils.normalize_embedding_dim(self.q_proj.weight)
        NormalizationUtils.normalize_embedding_dim(self.k_proj.weight)
        NormalizationUtils.normalize_embedding_dim(self.v_proj.weight)
        NormalizationUtils.normalize_embedding_dim(self.o_proj.weight)


class NGPTMLP(nn.Module):
    """
    Implements the Multi-Layer Perceptron (MLP) block with SwiGLU activation
    and intermediate scaling factors s_u and s_nu.
    """

    def __init__(self, config: Config):
        """
        Initializes the NGPTMLP module.

        Args:
            config: An instance of the Config dataclass.
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_mlp = config.d_mlp

        # Linear projections for the two branches of SwiGLU (Section 2.4.1)
        # "we omit bias terms" (Section 2.4.1)
        self.u_proj = nn.Linear(self.d_model, self.d_mlp, bias=False)
        self.nu_proj = nn.Linear(self.d_model, self.d_mlp, bias=False)
        self.o_proj = nn.Linear(self.d_mlp, self.d_model, bias=False)

        # Initialize linear layer weights (Section A.6)
        # This is handled by NGPTModel._init_weights applying to all submodules.

        # Scaling factors s_u and s_nu (Section 2.4.2, Eq. 20, 21)
        self.s_u = ScaledLearnableParameter(
            size=(self.d_mlp,),
            s_init=config.ngpt_specific_config.s_u_init,
            s_scale=config.ngpt_specific_config.s_u_scale_factor,
            name="s_u"
        )
        self.s_nu = ScaledLearnableParameter(
            size=(self.d_mlp,),
            s_init=config.ngpt_specific_config.s_nu_init,
            s_scale=config.ngpt_specific_config.s_nu_scale_factor,
            name="s_nu"
        )

        # Rescaling factor for nu branch (Section 2.4.2, Eq. 21, Appendix A.1)
        self.sqrt_d_model = math.sqrt(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the MLP block.

        Args:
            x: Input hidden state, shape (batch_size, sequence_length, d_model).

        Returns:
            Output of the MLP block, shape (batch_size, sequence_length, d_model).
        """
        # Project input to u_raw and nu_raw
        u_raw = self.u_proj(x)  # (B, T, d_mlp)
        nu_raw = self.nu_proj(x) # (B, T, d_mlp)

        # Apply scaling factors (Section 2.4.2, Eq. 20, 21)
        # s_u and s_nu effective values are (d_mlp,), will broadcast with (B, T, d_mlp).
        u = u_raw * self.s_u.get_effective_value()
        nu = nu_raw * self.s_nu.get_effective_value() * self.sqrt_d_model

        # SwiGLU activation (Section 2.4.1)
        swiglu_output = u * F.silu(nu) # F.silu is SiLU(nu) = nu * sigmoid(nu)

        # Final linear projection
        output = self.o_proj(swiglu_output) # (B, T, d_model)

        return output

    def post_optimizer_norm(self) -> None:
        """
        Normalizes the weights of the MLP projection matrices after each optimizer step.
        (Section 2.6, Step 2: "normalize matrices W_u, W_nu and W_oMLP along their embedding dimension.")
        """
        NormalizationUtils.normalize_embedding_dim(self.u_proj.weight)
        NormalizationUtils.normalize_embedding_dim(self.nu_proj.weight)
        NormalizationUtils.normalize_embedding_dim(self.o_proj.weight)


class NGPTBlock(nn.Module):
    """
    Encapsulates a single Transformer block, combining NGPTAttention and NGPTMLP
    with the NGPT-specific update equations involving eigen learning rates (alpha_A, alpha_M).
    """

    def __init__(self, config: Config):
        """
        Initializes an NGPTBlock.

        Args:
            config: An instance of the Config dataclass.
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        self.attention = NGPTAttention(config)
        self.mlp = NGPTMLP(config)

        # Eigen learning rates alpha_A and alpha_M (Section 2.2.2, Eq. 10, 11)
        # These are d_model-dimensional vectors.
        self.alpha_A = ScaledLearnableParameter(
            size=(self.d_model,),
            s_init=config.ngpt_specific_config.alpha_A_init,
            s_scale=config.ngpt_specific_config.alpha_A_scale_factor,
            name="alpha_A"
        )
        self.alpha_M = ScaledLearnableParameter(
            size=(self.d_model,),
            s_init=config.ngpt_specific_config.alpha_M_init,
            s_scale=config.ngpt_specific_config.alpha_M_scale_factor,
            name="alpha_M"
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, current_pos: int) -> torch.Tensor:
        """
        Performs the forward pass for an NGPT Transformer block.

        Args:
            x: Input hidden state, shape (batch_size, sequence_length, d_model).
            mask: Causal attention mask, shape (1, 1, sequence_length, sequence_length).
            current_pos: Starting position for RoPE application in attention.

        Returns:
            Output hidden state of the block, shape (batch_size, sequence_length, d_model).
        """
        # Attention block update (Section 2.2.2, Eq. 10)
        # hA_raw is the output of the attention mechanism.
        hA_raw = self.attention(x, mask, current_pos)
        # hA is the normalized output of the attention block.
        hA = NormalizationUtils.norm(hA_raw)
        # Update hidden state using eigen learning rates and re-normalize.
        # alpha_A.get_effective_value() is (d_model,), will broadcast with (B, T, d_model).
        x = NormalizationUtils.norm(x + self.alpha_A.get_effective_value() * (hA - x))

        # MLP block update (Section 2.2.2, Eq. 11)
        # hM_raw is the output of the MLP.
        hM_raw = self.mlp(x)
        # hM is the normalized output of the MLP block.
        hM = NormalizationUtils.norm(hM_raw)
        # Update hidden state using eigen learning rates and re-normalize.
        # alpha_M.get_effective_value() is (d_model,), will broadcast with (B, T, d_model).
        x = NormalizationUtils.norm(x + self.alpha_M.get_effective_value() * (hM - x))

        return x

    def post_optimizer_norm(self) -> None:
        """
        Calls normalization methods for all sub-components (attention and MLP)
        within this block.
        """
        self.attention.post_optimizer_norm()
        self.mlp.post_optimizer_norm()


class NGPTModel(nn.Module):
    """
    The main NGPT model, orchestrating the sequence of token embeddings,
    Transformer blocks, and final logits projection.
    """

    def __init__(self, config: Config):
        """
        Initializes the NGPTModel.

        Args:
            config: An instance of the Config dataclass.
        """
        super().__init__()
        self.config = config

        self.embeddings = NGPTEmbeddings(config)
        self.blocks = nn.ModuleList([NGPTBlock(config) for _ in range(config.n_layers)])

        # Apply custom weight initialization to all linear layers.
        # NGPTEmbeddings already handles its own initialization.
        self.apply(self._init_weights)
        print("NGPTModel initialized with custom weight initialization.")

    def _init_weights(self, module: nn.Module) -> None:
        """
        Custom weight initialization for linear layers.
        (Section A.6: "All matrix parameters are initialized by sampling from a
        zero-mean normal distribution with a standard deviation of 0.02 for GPT
        and 1/√d_model for nGPT.")

        Args:
            module: The module to initialize.
        """
        if isinstance(module, nn.Linear):
            if module.weight is not None and module.weight.dim() > 1:
                # Standard deviation for output matrices scaled by sqrt(2*n_layers) for GPT.
                # nGPT does not specify this for output matrices specifically,
                # but says "The initialization of matrix parameters is not important
                # for nGPT because they are normalized afterwards."
                # We'll stick to config.init_std_dev for all for consistency with normalization.
                # However, for GPT baseline model_type, if the config `init_std_dev`
                # already considers this, then it's fine.
                # The config `init_std_dev` is already set based on model_type.
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std_dev)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # nn.Embedding is initialized in NGPTEmbeddings explicitly.

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates a causal attention mask. This mask prevents attention to future tokens.

        Args:
            seq_len: The sequence length.
            device: The device to place the mask on.

        Returns:
            A causal mask tensor of shape (1, 1, seq_len, seq_len).
            Values are 0.0 for allowed attention, -inf for masked attention.
        """
        # Create an upper triangular matrix of ones, with the diagonal also ones
        # Example for seq_len=4:
        # [[1., 0., 0., 0.],
        #  [1., 1., 0., 0.],
        #  [1., 1., 1., 0.],
        #  [1., 1., 1., 1.]]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        
        # Invert the boolean mask: False for allowed, True for masked (future tokens)
        # Fill True with -inf, False with 0.0
        # The design document refers to `mask == 0` for `masked_fill`, meaning `0` means masked.
        # This aligns with a mask where valid positions are 1 and masked positions are 0.
        # Let's adjust for consistency with the common pattern of `1` for allowed, `0` for masked
        # when using `masked_fill(mask == 0, float('-inf'))`.
        # So the `mask` should be a boolean tensor where `True` is allowed and `False` is masked.
        # `torch.tril(torch.ones(seq_len, seq_len))` creates a mask where `True` values allow attention.
        # The attention formulation is `softmax(QK^T + M)V`, where M sets future tokens to -inf.
        # If mask contains 0s for future tokens, then `masked_fill(mask == 0, float('-inf'))` is correct.
        
        # Causal mask (upper triangle including diagonal, is 0. Lower triangle is 1)
        # If j > i, M_i,j = -inf (cannot attend to future)
        # if j <= i, M_i,j = 0 (can attend to past/self)
        # This corresponds to a lower triangular matrix of ones, upper triangular (excl. diag) of zeros.
        # For example, seq_len=4:
        # [[1, 0, 0, 0],
        #  [1, 1, 0, 0],
        #  [1, 1, 1, 0],
        #  [1, 1, 1, 1]]
        # `torch.tril(torch.ones(...))` creates exactly this.
        # Then, `masked_fill(mask == 0, float('-inf'))` will correctly mask future tokens.
        
        # Add batch and head dimensions for broadcasting with attention scores.
        return mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) \
            -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Performs the forward pass for the entire NGPT model.

        Args:
            input_ids: Input token IDs, shape (batch_size, sequence_length).
            targets: Optional; Target token IDs for language modeling loss,
                     shape (batch_size, sequence_length).

        Returns:
            A tuple containing:
            - loss: The computed cross-entropy loss if `targets` are provided, otherwise None.
            - logits: The predicted logits for each token, shape (batch_size, sequence_length, vocab_size).
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.size()

        # Get initial hidden state from embeddings
        h = self.embeddings(input_ids) # (B, T, D_model)

        # Generate causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)

        # Process through Transformer blocks
        # current_pos_for_rope is 0 assuming fixed sequence length training.
        current_pos_for_rope = 0
        for block in self.blocks:
            h = block(h, causal_mask, current_pos_for_rope)

        # Compute logits
        logits = self.embeddings.get_output_logits(h) # (B, T, V)

        loss = None
        if targets is not None:
            # Compute cross-entropy loss
            # Reshape logits to (B * T, V) and targets to (B * T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return loss, logits

    def post_optimizer_norm(self) -> None:
        """
        Applies normalization to all relevant parameters across the entire model
        after each optimizer step.
        (Section 2.6, Step 2: "After each training step (and, optionally, during the forward pass),
        normalize matrices E_input, E_output, W_q, W_k, W_v, W_o, W_u, W_nu and W_oMLP
        along their embedding dimension.")
        """
        self.embeddings.post_optimizer_norm()
        for block in self.blocks:
            block.post_optimizer_norm()

