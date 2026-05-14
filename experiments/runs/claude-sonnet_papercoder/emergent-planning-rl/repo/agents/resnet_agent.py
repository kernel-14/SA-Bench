## agents/resnet_agent.py
"""ResNet agent for the emergent planning interpretability pipeline.

This module implements the ResNet agent analyzed in Appendix G of the paper
"Interpreting Emergent Planning in Model-Free Reinforcement Learning". It
provides an alternative architecture to the DRC agent to test whether emergent
planning is architecture-agnostic.

Key finding from Appendix G: despite lacking recurrent connections, the ResNet
agent appears to perform iterative planning *across layers* — C_B representations
peak around layer 10, C_A around layer 16, suggesting a sequential box-then-agent
planning process within a single forward pass.

Architecture (paper Appendix G):
    - 24 simplified residual blocks, 32 channels throughout
    - No pooling or downsampling — spatial dims preserved at 8×8 throughout
    - Each block: conv → layer_norm → relu → conv → layer_norm → add_residual → relu
    - After final block: flatten → Linear(2048→256) → ReLU → policy/value heads
    - Probes applied after the **final ReLU** of each residual block

From config.yaml (agent_variants.resnet):
    obs_channels: 7, num_blocks: 24, channels: 32, grid_h: 8, grid_w: 8
    total_steps: 250_000_000, intervention_alpha: 4.0

Critical note on what gets probed (Appendix G):
    "probing the hidden state of this agent for C_A and C_B after the final
    ReLU at each layer" — hidden_states[i] is the output after the final ReLU
    of block i, shape (B, 32, 8, 8), matching DRC cell state spatial structure.

Example:
    >>> import torch
    >>> agent = ResNetAgent()
    >>> obs = torch.zeros(2, 8, 8, 7)  # HWC from SokobanEnv
    >>> logits, value, hidden_states = agent.forward(obs)
    >>> logits.shape
    torch.Size([2, 5])
    >>> len(hidden_states)
    24
    >>> hidden_states[0].shape
    torch.Size([2, 32, 8, 8])
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ResNetBlock(nn.Module):
    """Simplified residual block for the ResNet agent.

    Implements the block structure described in paper Appendix G:
        conv → layer_norm → relu → conv → layer_norm → add_residual → relu

    No downsampling or pooling — spatial dimensions are preserved at 8×8
    throughout, maintaining the spatial bijection with the Sokoban grid that
    enables 1×1 and 3×3 probes to operate on the hidden states.

    The **final ReLU output** is what linear probes operate on. This is the
    hidden state at each layer referenced in Appendix G: "probing the hidden
    state of this agent for C_A and C_B after the final ReLU at each layer."

    LayerNorm is applied over the full [channels, grid_h, grid_w] shape,
    normalizing over (C, H, W) jointly for each batch element. This is the
    standard interpretation of "layer norm" for spatial feature maps.

    Attributes:
        channels: Number of input and output channels (32 throughout).
        grid_h: Spatial height of feature maps (8).
        grid_w: Spatial width of feature maps (8).
        conv1: First 3×3 convolution with same-padding.
        norm1: LayerNorm over [channels, grid_h, grid_w].
        conv2: Second 3×3 convolution with same-padding.
        norm2: LayerNorm over [channels, grid_h, grid_w].

    Example:
        >>> block = ResNetBlock(channels=32, grid_h=8, grid_w=8)
        >>> x = torch.zeros(2, 32, 8, 8)
        >>> out = block(x)
        >>> out.shape
        torch.Size([2, 32, 8, 8])
    """

    def __init__(
        self,
        channels: int = 32,
        grid_h: int = 8,
        grid_w: int = 8,
    ) -> None:
        """Initialize the ResNetBlock.

        Args:
            channels: Number of input and output channels. Matches
                config.yaml agent_variants.resnet.channels = 32.
                All blocks have the same channel count — no downsampling.
            grid_h: Spatial height of feature maps. Matches
                config.yaml agent_variants.resnet.grid_h = 8.
                Required for LayerNorm normalized_shape specification.
            grid_w: Spatial width of feature maps. Matches
                config.yaml agent_variants.resnet.grid_w = 8.
                Required for LayerNorm normalized_shape specification.
        """
        super().__init__()

        self.channels: int = channels
        self.grid_h: int = grid_h
        self.grid_w: int = grid_w

        # First convolution: same-padding (padding=1 for kernel_size=3)
        # preserves 8×8 spatial dimensions.
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # LayerNorm over [C, H, W] = [32, 8, 8].
        # For input shape (B, C, H, W), this normalizes over the last 3 dims
        # (C, H, W) jointly for each batch element.
        self.norm1: nn.LayerNorm = nn.LayerNorm(
            normalized_shape=[channels, grid_h, grid_w]
        )

        # Second convolution: same architecture as conv1.
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # Second LayerNorm: applied after conv2, before residual addition.
        self.norm2: nn.LayerNorm = nn.LayerNorm(
            normalized_shape=[channels, grid_h, grid_w]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block computation.

        Implements the block structure from paper Appendix G:
            residual = x
            x = conv1(x) → norm1(x) → relu(x)
            x = conv2(x) → norm2(x)
            x = x + residual   (residual connection)
            x = relu(x)        (final ReLU — probes operate on this output)

        The final ReLU output is what ResNetAgent collects as hidden_states[i]
        and what linear probes operate on throughout the interpretability pipeline.

        Args:
            x: Input feature map of shape (B, channels, grid_h, grid_w)
               = (B, 32, 8, 8). Must match the channel and spatial dimensions
               specified at construction time.

        Returns:
            Output feature map of shape (B, channels, grid_h, grid_w)
            = (B, 32, 8, 8). This is the hidden state at this block,
            after the final ReLU, as referenced in paper Appendix G.
        """
        # Save residual for skip connection.
        residual: torch.Tensor = x

        # First sub-layer: conv → norm → relu.
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        # Second sub-layer: conv → norm.
        x = self.conv2(x)
        x = self.norm2(x)

        # Residual addition: add the input before the final activation.
        x = x + residual

        # Final ReLU: this is the output that probes operate on.
        # ResNetAgent.forward() collects x at this point as hidden_states[i].
        x = F.relu(x)

        return x


class ResNetAgent(nn.Module):
    """Feedforward ResNet actor-critic agent for Sokoban.

    Implements the ResNet agent described in paper Appendix G. Unlike the DRC
    agent, this agent is stateless (no recurrent connections) — it processes
    each observation independently without maintaining hidden state across steps.

    The agent performs iterative computation *across layers* rather than across
    ticks. The paper finds that C_B representations peak around block 10 and
    C_A around block 16, suggesting the agent first plans box movements in early
    layers, then agent movements in later layers.

    Architecture:
        input_conv: Conv2d(7→32, k=3, p=1) → ReLU
        blocks: 24 × ResNetBlock(32, 8, 8)
        output_mlp: Flatten → Linear(2048→256) → ReLU
        policy_head: Linear(256→5)
        value_head: Linear(256→1)

    The hidden state at block i is the output of blocks[i].forward(x), which
    is the tensor after the final ReLU of that block. These 24 hidden states
    (each shape (B, 32, 8, 8)) are returned by forward() and used by the
    probing and intervention pipeline.

    Attributes:
        obs_channels: Number of input observation channels (7).
        num_blocks: Number of residual blocks (24).
        channels: Channel dimension throughout the network (32).
        grid_h: Spatial height (8).
        grid_w: Spatial width (8).
        n_actions: Number of discrete actions (5).
        input_conv: Initial projection from obs_channels to channels.
        blocks: ModuleList of 24 ResNetBlock instances.
        output_mlp: Flatten + Linear(2048→256) + ReLU.
        policy_head: Linear(256→5) for action logits.
        value_head: Linear(256→1) for value estimate.

    Example:
        >>> agent = ResNetAgent(obs_channels=7, num_blocks=24, channels=32,
        ...                     grid_h=8, grid_w=8)
        >>> obs = torch.zeros(2, 8, 8, 7)  # HWC from SokobanEnv
        >>> logits, value, hidden_states = agent.forward(obs)
        >>> logits.shape
        torch.Size([2, 5])
        >>> value.shape
        torch.Size([2, 1])
        >>> len(hidden_states)
        24
        >>> hidden_states[9].shape  # block 10 (0-indexed: 9), C_B peaks here
        torch.Size([2, 32, 8, 8])
        >>> hidden_states[15].shape  # block 16 (0-indexed: 15), C_A peaks here
        torch.Size([2, 32, 8, 8])
    """

    def __init__(
        self,
        obs_channels: int = 7,
        num_blocks: int = 24,
        channels: int = 32,
        grid_h: int = 8,
        grid_w: int = 8,
        n_actions: int = 5,
    ) -> None:
        """Initialize the ResNet agent.

        Constructs the input projection, residual block stack, output MLP,
        and policy/value heads. All spatial dimensions are preserved at
        (grid_h, grid_w) = (8, 8) throughout the network.

        Args:
            obs_channels: Number of channels in the symbolic observation.
                Matches config.yaml agent_variants.resnet.obs_channels = 7.
            num_blocks: Number of simplified residual blocks.
                Matches config.yaml agent_variants.resnet.num_blocks = 24.
                Paper Appendix G: "24 simplified residual blocks."
            channels: Channel dimension throughout the network.
                Matches config.yaml agent_variants.resnet.channels = 32.
                Paper Appendix G: "32 channels for consistency with DRC agent."
            grid_h: Spatial height of feature maps.
                Matches config.yaml agent_variants.resnet.grid_h = 8.
            grid_w: Spatial width of feature maps.
                Matches config.yaml agent_variants.resnet.grid_w = 8.
            n_actions: Number of discrete actions.
                Matches config.yaml env.n_actions = 5.
        """
        super().__init__()

        self.obs_channels: int = obs_channels
        self.num_blocks: int = num_blocks
        self.channels: int = channels
        self.grid_h: int = grid_h
        self.grid_w: int = grid_w
        self.n_actions: int = n_actions

        # ------------------------------------------------------------------
        # Input projection: obs (B, 7, 8, 8) → feature map (B, 32, 8, 8).
        # Same-padding (padding=1 for kernel_size=3) preserves 8×8 spatial dims.
        # Followed by ReLU to introduce non-linearity before the first block.
        # ------------------------------------------------------------------
        self.input_conv: nn.Conv2d = nn.Conv2d(
            in_channels=obs_channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # ------------------------------------------------------------------
        # Residual block stack: 24 blocks, each preserving (B, 32, 8, 8).
        # Each block's final ReLU output is collected as a hidden state for
        # probing. The paper finds C_B peaks ~block 10, C_A peaks ~block 16.
        # ------------------------------------------------------------------
        self.blocks: nn.ModuleList = nn.ModuleList([
            ResNetBlock(
                channels=channels,
                grid_h=grid_h,
                grid_w=grid_w,
            )
            for _ in range(num_blocks)
        ])

        # ------------------------------------------------------------------
        # Output MLP: flatten spatial dims then project to 256-dim vector.
        # From paper Appendix G: "flattened, passed through an MLP of
        # dimensionality 256, and then passed to policy and value heads."
        # Spatial flat dim: channels * grid_h * grid_w = 32 * 8 * 8 = 2048.
        # ------------------------------------------------------------------
        spatial_flat_dim: int = channels * grid_h * grid_w  # 32 * 8 * 8 = 2048

        self.output_mlp: nn.Sequential = nn.Sequential(
            nn.Flatten(),                                    # (B, 2048)
            nn.Linear(spatial_flat_dim, 256, bias=True),    # (B, 256)
            nn.ReLU(),
        )

        # Policy head: (B, 256) → (B, n_actions=5)
        self.policy_head: nn.Linear = nn.Linear(
            in_features=256,
            out_features=n_actions,
            bias=True,
        )

        # Value head: (B, 256) → (B, 1)
        self.value_head: nn.Linear = nn.Linear(
            in_features=256,
            out_features=1,
            bias=True,
        )

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Run the full ResNet forward pass for one observation.

        Processes the observation through the input projection, all 24 residual
        blocks, and the output MLP to produce policy logits, a value estimate,
        and the hidden states at each block.

        The hidden states are collected after the final ReLU of each block,
        matching the paper's description: "probing the hidden state of this
        agent for C_A and C_B after the final ReLU at each layer."

        Args:
            obs: Symbolic observation tensor. Accepted shapes:
                - (B, 7, 8, 8): CHW format (already transposed)
                - (B, 8, 8, 7): HWC format from SokobanEnv.get_symbolic_obs()
                - (8, 8, 7): Single unbatched HWC (auto-batched to (1, 7, 8, 8))
                - (7, 8, 8): Single unbatched CHW (auto-batched to (1, 7, 8, 8))

        Returns:
            Tuple of three elements:
            1. policy_logits: Tensor of shape (B, n_actions=5). Raw logits
               (not softmaxed) for the action distribution. Used by the trainer
               for loss computation and by get_action for sampling/argmax.
            2. value: Tensor of shape (B, 1). Scalar value estimate V(s_t).
               Used by V-trace for advantage computation in IMPALATrainer.
            3. hidden_states: List of 24 tensors, each of shape (B, 32, 8, 8).
               hidden_states[i] is the output after the final ReLU of block i
               (0-indexed). These are what linear probes operate on:
               - hidden_states[9]  ≈ block 10: C_B peaks here (Appendix G)
               - hidden_states[15] ≈ block 16: C_A peaks here (Appendix G)
               The spatial structure (B, 32, 8, 8) matches DRC cell states,
               enabling the same 1×1 and 3×3 probe architecture.

        Example:
            >>> obs = torch.zeros(1, 8, 8, 7)  # HWC from SokobanEnv
            >>> logits, value, hidden_states = agent.forward(obs)
            >>> logits.shape
            torch.Size([1, 5])
            >>> value.shape
            torch.Size([1, 1])
            >>> len(hidden_states)
            24
            >>> hidden_states[0].shape
            torch.Size([1, 32, 8, 8])
        """
        # ------------------------------------------------------------------
        # Step 1: Normalize observation to (B, C, H, W) CHW format.
        # ------------------------------------------------------------------
        obs_chw: torch.Tensor = self._normalize_obs(obs)

        # ------------------------------------------------------------------
        # Step 2: Input projection + ReLU.
        # obs_chw: (B, 7, 8, 8) → x: (B, 32, 8, 8)
        # ------------------------------------------------------------------
        x: torch.Tensor = F.relu(self.input_conv(obs_chw))

        # ------------------------------------------------------------------
        # Step 3: Pass through all 24 residual blocks, collecting hidden states.
        # Each block's output (after its final ReLU) is stored as a hidden state.
        # ------------------------------------------------------------------
        hidden_states: List[torch.Tensor] = []

        for block in self.blocks:
            x = block(x)                  # (B, 32, 8, 8) — includes final ReLU
            hidden_states.append(x)       # collect after final ReLU of each block

        # ------------------------------------------------------------------
        # Step 4: Output MLP: flatten → Linear(2048→256) → ReLU.
        # x: (B, 32, 8, 8) → o: (B, 256)
        # ------------------------------------------------------------------
        o: torch.Tensor = self.output_mlp(x)

        # ------------------------------------------------------------------
        # Step 5: Policy and value heads.
        # ------------------------------------------------------------------
        policy_logits: torch.Tensor = self.policy_head(o)   # (B, 5)
        value: torch.Tensor = self.value_head(o)             # (B, 1)

        return policy_logits, value, hidden_states

    def get_action(
        self,
        obs: torch.Tensor,
        greedy: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Select an action given an observation.

        Wraps forward() to provide a convenient interface for environment
        interaction. Handles both stochastic sampling (training) and greedy
        selection (evaluation).

        During training (greedy=False): samples from the categorical distribution
        parameterized by policy_logits, as described in paper Section E.4.
        At test time (greedy=True): selects the action with the highest logit,
        as described in paper Section E.4 ("acts greedily by always performing
        the action with the greatest logit").

        Note: Unlike DRCAgent.get_action, this method does not take or return
        a recurrent state, since the ResNet is a feedforward (stateless) agent.

        Args:
            obs: Symbolic observation tensor. Accepted shapes:
                - (B, 7, 8, 8): CHW format
                - (B, 8, 8, 7): HWC format from SokobanEnv
                - (8, 8, 7): Single unbatched HWC (auto-batched)
                - (7, 8, 8): Single unbatched CHW (auto-batched)
            greedy: If True, select the action with the highest logit (argmax).
                If False, sample from the categorical distribution.
                Default False (training mode). Set True for evaluation.

        Returns:
            Tuple of four elements:
            1. action: int, the selected action in {0, 1, 2, 3, 4}.
               For batched inputs (B>1), returns the action for the first
               element (index 0).
            2. log_prob: Tensor of shape (B,), log probability of the selected
               action under the current policy. Used by V-trace for importance
               sampling ratios in IMPALATrainer.
            3. value: Tensor of shape (B,), scalar value estimate (squeezed
               from (B, 1) to (B,)). Used by V-trace for advantage computation.
            4. hidden_states: List of 24 tensors, each of shape (B, 32, 8, 8).
               Hidden states at each block after the final ReLU. Used by the
               probing and intervention pipeline.

        Example:
            >>> obs = torch.zeros(1, 8, 8, 7)
            >>> action, log_prob, value, hidden_states = agent.get_action(
            ...     obs, greedy=True
            ... )
            >>> isinstance(action, int)
            True
            >>> log_prob.shape
            torch.Size([1])
            >>> value.shape
            torch.Size([1])
            >>> len(hidden_states)
            24
        """
        # Run full forward pass.
        policy_logits: torch.Tensor
        value_raw: torch.Tensor
        hidden_states: List[torch.Tensor]

        policy_logits, value_raw, hidden_states = self.forward(obs)

        # Build categorical distribution from logits.
        dist: Categorical = Categorical(logits=policy_logits)

        # Select action: greedy (argmax) or stochastic (sample).
        if greedy:
            action_tensor: torch.Tensor = policy_logits.argmax(dim=-1)
        else:
            action_tensor = dist.sample()

        # Compute log probability of the selected action.
        log_prob: torch.Tensor = dist.log_prob(action_tensor)

        # Squeeze value from (B, 1) to (B,).
        value: torch.Tensor = value_raw.squeeze(-1)

        # Return scalar action (int) for single-environment interaction.
        action: int = int(action_tensor[0].item())

        return action, log_prob, value, hidden_states

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observation tensor to (B, C, H, W) CHW format.

        SokobanEnv.get_symbolic_obs() returns observations in HWC format
        (H, W, C) = (8, 8, 7). PyTorch convolutions expect BCHW format.
        This method handles the transposition and auto-batching.

        Accepted input shapes:
        - (8, 8, 7): Single unbatched HWC → (1, 7, 8, 8)
        - (B, 8, 8, 7): Batched HWC → (B, 7, 8, 8)
        - (B, 7, 8, 8): Already in CHW format → returned as-is
        - (7, 8, 8): Single unbatched CHW → (1, 7, 8, 8)

        Args:
            obs: Observation tensor in any of the above formats.

        Returns:
            Observation tensor in (B, C, H, W) = (B, 7, 8, 8) format,
            on the same device as the input tensor.

        Raises:
            ValueError: If the observation shape is not recognized as a valid
                Sokoban symbolic observation format.
        """
        if obs.dim() == 3:
            # Single unbatched observation.
            if obs.shape[-1] == self.obs_channels:
                # (H, W, C) = (8, 8, 7) → (1, 7, 8, 8)
                obs = obs.permute(2, 0, 1).unsqueeze(0)
            elif obs.shape[0] == self.obs_channels:
                # (C, H, W) = (7, 8, 8) → (1, 7, 8, 8)
                obs = obs.unsqueeze(0)
            else:
                raise ValueError(
                    f"Unrecognized 3D observation shape {obs.shape}. "
                    f"Expected (H, W, {self.obs_channels}) or "
                    f"({self.obs_channels}, H, W)."
                )
        elif obs.dim() == 4:
            # Batched observation.
            if obs.shape[-1] == self.obs_channels:
                # (B, H, W, C) = (B, 8, 8, 7) → (B, 7, 8, 8)
                obs = obs.permute(0, 3, 1, 2)
            elif obs.shape[1] == self.obs_channels:
                # (B, C, H, W) = (B, 7, 8, 8) → already correct
                pass
            else:
                raise ValueError(
                    f"Unrecognized 4D observation shape {obs.shape}. "
                    f"Expected (B, H, W, {self.obs_channels}) or "
                    f"(B, {self.obs_channels}, H, W)."
                )
        else:
            raise ValueError(
                f"Observation must be 3D or 4D tensor, got {obs.dim()}D "
                f"with shape {obs.shape}."
            )

        return obs.contiguous()
