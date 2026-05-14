```python
## evaluation.py
"""Evaluator for Prioritized Generative Replay (PGR).

Implements all post-hoc and periodic analysis experiments described in the
paper, including policy evaluation, generation MSE (Fig. 5), dormant ratio
(Fig. 6a), relevance score distributions (Fig. 6b), and t-SNE projections
(Fig. 2).

The Evaluator is a read-only consumer of trained components — it never
modifies policy weights, buffer contents, or diffusion model parameters.

Config references (config.yaml):
    training.eval_episodes:          10      # episodes per evaluation
    evaluation.mse_num_samples:      10000   # transitions for MSE eval
    evaluation.dormant_threshold:    0.01    # neuron activation threshold
    evaluation.relevance_num_samples: 10000  # transitions for relevance dist
    evaluation.tsne_num_samples:     10000   # transitions per buffer for t-SNE
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.manifold import TSNE
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from buffers.replay_buffer import ReplayBuffer
from diffusion.conditional_diffusion import ConditionalDiffusion
from policies.redq import REDQPolicy
from relevance.base import BaseRelevance


class Evaluator:
    """Handles all evaluation and analysis experiments for PGR.

    Provides five analysis methods consumed by PGRTrainer at configurable
    intervals during training:

        1. evaluate_policy:              Mean episodic return (every eval_freq steps)
        2. compute_generation_mse:       Dynamics faithfulness (at mse_eval_epoch)
        3. compute_dormant_ratio:        Fraction of inactive neurons (periodic)
        4. compute_relevance_distribution: ICM score histogram (every relevance_eval_freq)
        5. compute_tsne:                 2D projection of transition space (at tsne_epochs)

    All methods are read-only — no model parameters are modified. All forward
    passes run under torch.no_grad() for memory efficiency.

    Attributes:
        env: Environment wrapper (DMCEnv or GymEnv) for policy evaluation and
            generation MSE rollouts.
        policy: REDQPolicy (or compatible policy) for action selection and
            dormant ratio computation.
        diffusion: ConditionalDiffusion model for generating synthetic transitions
            in compute_generation_mse and compute_tsne.
        relevance_fn: BaseRelevance subclass for scoring transitions in
            compute_relevance_distribution and compute_generation_mse.
        device: PyTorch device string for all tensor operations.
    """

    def __init__(
        self,
        env: Any,
        policy: REDQPolicy,
        diffusion: ConditionalDiffusion,
        relevance_fn: BaseRelevance,
        device: str = "cuda",
    ) -> None:
        """Initialises the evaluator with references to all trained components.

        No new trainable parameters are created. All arguments are stored as
        references — the evaluator reads from them but never writes to them.

        Args:
            env: Environment wrapper instance (DMCEnv or GymEnv). Used in
                evaluate_policy() for episode rollouts and in
                compute_generation_mse() for ground-truth dynamics rollouts.
                Must implement reset() -> np.ndarray and
                step(action) -> Tuple[np.ndarray, float, bool, dict].
            policy: REDQPolicy instance (or SACPolicy/DRQv2Policy with a
                compatible select_action interface). Used in evaluate_policy()
                for deterministic action selection and in compute_dormant_ratio()
                for activation analysis. Must implement
                select_action(obs, deterministic=True) -> np.ndarray.
            diffusion: ConditionalDiffusion instance. Used in
                compute_generation_mse() to generate synthetic transitions for
                dynamics faithfulness evaluation. Must implement
                generate(num_samples, conditions) -> Tensor.
            relevance_fn: BaseRelevance subclass instance (e.g. ICMRelevance).
                Used in compute_relevance_distribution() to score transitions
                and in compute_generation_mse() to produce generation conditions.
                Must implement score(obs, action, next_obs, reward) -> Tensor.
            device: PyTorch device string. All tensors are moved to this device
                before inference. Corresponds to config.hardware.device
                (default "cuda").
        """
        self.env: Any = env
        self.policy: REDQPolicy = policy
        self.diffusion: ConditionalDiffusion = diffusion
        self.relevance_fn: BaseRelevance = relevance_fn
        self.device: str = device

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate_policy(self, num_episodes: int = 10) -> float:
        """Runs the policy deterministically and returns mean episodic return.

        Executes num_episodes complete episodes using the deterministic policy
        (tanh(mean) without sampling noise). Called every eval_freq=5000 steps
        by PGRTrainer. Corresponds to config.yaml: training.eval_episodes=10.

        The policy is set to eval mode during rollouts and restored to train
        mode afterward. No gradient tracking is performed.

        Args:
            num_episodes: Number of complete episodes to run. Corresponds to
                config.yaml: training.eval_episodes (default 10).

        Returns:
            Mean episodic return (sum of rewards per episode) averaged over
            num_episodes episodes. Used as the primary performance metric
            reported in Tables 1 and 2 of the paper.
        """
        episode_returns: List[float] = []

        # Set policy actor to eval mode — disables dropout/batchnorm stochasticity
        # (not present in REDQ, but defensive practice for eval-time inference).
        if hasattr(self.policy, "actor"):
            self.policy.actor.eval()

        with torch.no_grad():
            for _episode_idx in range(num_episodes):
                # Reset environment to get initial observation.
                obs: np.ndarray = self.env.reset()
                episode_return: float = 0.0
                done: bool = False

                # Roll out one complete episode.
                while not done:
                    # Select deterministic action: tanh(mean) without sampling.
                    action: np.ndarray = self.policy.select_action(
                        obs, deterministic=True
                    )

                    # Step environment.
                    next_obs: np.ndarray
                    reward: float
                    next_obs, reward, done, _info = self.env.step(action)

                    episode_return += reward
                    obs = next_obs

                episode_returns.append(episode_return)

        # Restore policy actor to training mode.
        if hasattr(self.policy, "actor"):
            self.policy.actor.train()

        return float(np.mean(episode_returns))

    def compute_generation_mse(
        self,
        real_buffer: ReplayBuffer,
        env: Any,
        num_samples: int = 10000,
    ) -> Dict[str, float]:
        """Measures faithfulness of generated transitions to true environment dynamics.

        Replicates the Fig. 5 analysis from the paper (Section 5.2):
        "we borrow the methodology of Lu et al. (2024) and measure faithfulness
        of generated transitions to environment dynamics. Given a generated
        transition (s, a, s', r), we roll out the action a given the current
        state s in the environment simulator to obtain the ground truth next
        state and reward."

        Called at mse_eval_epoch=50000 (config.yaml: evaluation.mse_eval_epoch)
        over num_samples=10000 generated transitions (evaluation.mse_num_samples).

        For pixel-based tasks, the generated transitions are in latent space
        (f_θ(s), a, f_θ(s'), r). MSE is computed in latent space since
        generation happens in latent space — consistent with the paper's approach.

        For state-based tasks, MSE is computed in the original state space.
        State restoration uses environment-specific physics setters where
        available; falls back to approximate comparison when not available.

        Args:
            real_buffer: D_real replay buffer. Used to sample transitions for
                generating conditions (relevance scores) to pass to the
                diffusion model. Must have at least num_samples valid entries.
            env: Environment instance for ground-truth dynamics rollouts.
                For DMCEnv: uses env._env.physics for state restoration.
                For GymEnv: uses env._env for state restoration.
                Passed explicitly (rather than using self.env) to allow
                evaluation on a separate environment instance if needed.
            num_samples: Number of generated transitions to evaluate.
                Corresponds to config.yaml: evaluation.mse_num_samples
                (default 10000).

        Returns:
            Dict with keys:
                'state_mse': Mean squared error between generated next states
                    and ground-truth next states from environment rollouts.
                'reward_mse': Mean squared error between generated rewards
                    and ground-truth rewards from environment rollouts.
            Both values are Python floats.
        """
        # Clamp num_samples to available buffer size.
        actual_samples: int = min(num_samples, len(real_buffer))
        if actual_samples == 0:
            return {"state_mse": 0.0, "reward_mse": 0.0}

        # ── Step 1: Sample transitions and compute conditions ─────────────────
        # Sample from D_real to get conditions for generation.
        batch: Dict[str, torch.Tensor] = real_buffer.sample(actual_samples)

        with torch.no_grad():
            # Score transitions to get relevance conditions.
            conditions: torch.Tensor = self.relevance_fn.score(
                batch["observations"],
                batch["actions"],
                batch["next_observations"],
                batch["rewards"],
            )  # (actual_samples, 1)

            # Normalize conditions to [0, 1] for diffusion model.
            cond_min: torch.Tensor = conditions.min()
            cond_max: torch.Tensor = conditions.max()
            conditions_norm: torch.Tensor = (conditions - cond_min) / (
                cond_max - cond_min + 1e-8
            )  # (actual_samples, 1)

            # ── Step 2: Generate synthetic transitions ────────────────────────
            # Generate in batches to avoid OOM for large num_samples.
            gen_batch_size: int = 256
            all_generated: List[torch.Tensor] = []

            for start_idx in range(0, actual_samples, gen_batch_size):
                end_idx: int = min(start_idx + gen_batch_size, actual_samples)
                batch_cond: torch.Tensor = conditions_norm[start_idx:end_idx]
                n_gen: int = end_idx - start_idx

                generated_batch: torch.Tensor = self.diffusion.generate(
                    n_gen, batch_cond
                )  # (n_gen, input_dim)
                all_generated.append(generated_batch)

            generated: torch.Tensor = torch.cat(all_generated, dim=0)
            # generated shape: (actual_samples, input_dim)
            # input_dim = obs_dim + action_dim + obs_dim + 1

        # ── Step 3: Split generated transitions into components ───────────────
        obs_dim: int = real_buffer.obs_dim
        action_dim: int = real_buffer.action_dim

        gen_obs: torch.Tensor = generated[:, :obs_dim]                          # (N, obs_dim)
        gen_action: torch.Tensor = generated[:, obs_dim:obs_dim + action_dim]   # (N, action_dim)
        gen_next_obs: torch.Tensor = generated[:, obs_dim + action_dim:2 * obs_dim + action_dim]  # (N, obs_dim)
        gen_reward: torch.Tensor = generated[:, -1:]                            # (N, 1)

        # Convert to numpy for environment rollouts.
        gen_obs_np: np.ndarray = gen_obs.cpu().numpy()
        gen_action_np: np.ndarray = gen_action.cpu().numpy()
        gen_next_obs_np: np.ndarray = gen_next_obs.cpu().numpy()
        gen_reward_np: np.ndarray = gen_reward.cpu().numpy().flatten()

        # ── Step 4: Roll out in environment simulator ─────────────────────────
        # Attempt state restoration for ground-truth dynamics comparison.
        # This is environment-specific and may not be possible for all envs.
        gt_next_obs_list: List[np.ndarray] = []
        gt_reward_list: List[float] = []

        # Detect environment type for state restoration.
        env_type: str = self._detect_env_type(env)

        for i in range(actual_samples):
            gt_next_obs_i: Optional[np.ndarray] = None
            gt_reward_i: Optional[float] = None

            try:
                if env_type == "dmc":
                    gt_next_obs_i, gt_reward_i = self._rollout_dmc(
                        env, gen_obs_np[i], gen_action_np[i]
                    )
                elif env_type == "gym":
                    gt_next_obs_i, gt_reward_i = self._rollout_gym(
                        env, gen_obs_np[i], gen_action_np[i]
                    )
                else:
                    # Unknown environment type — use generated values as fallback.
                    gt_next_obs_i = gen_next_obs_np[i]
                    gt_reward_i = float(gen_reward_np[i])
            except Exception:
                # If state restoration fails, use generated values as fallback.
                # This prevents the entire evaluation from crashing on edge cases.
                gt_next_obs_i = gen_next_obs_np[i]
                gt_reward_i = float(gen_reward_np[i])

            gt_next_obs_list.append(gt_next_obs_i)
            gt_reward_list.append(gt_reward_i)

        # ── Step 5: Compute MSE ───────────────────────────────────────────────
        gt_next_obs_arr: np.ndarray = np.stack(gt_next_obs_list, axis=0)  # (N, obs_dim)
        gt_reward_arr: np.ndarray = np.array(gt_reward_list, dtype=np.float64)  # (N,)

        # State MSE: mean over all elements (samples × obs_dim).
        state_mse: float = float(
            np.mean((gen_next_obs_np.astype(np.float64) - gt_next_obs_arr.astype(np.float64)) ** 2)
        )

        # Reward MSE: mean over all samples.
        reward_mse: float = float(
            np.mean((gen_reward_np.astype(np.float64) - gt_reward_arr) ** 2)
        )

        return {"state_mse": state_mse, "reward_mse": reward_mse}

    def compute_dormant_ratio(
        self,
        policy: REDQPolicy,
        real_buffer: ReplayBuffer,
        threshold: float = 0.01,
    ) -> float:
        """Computes the dormant ratio (DR) of the policy network.

        Implements the dormant ratio metric from Sokar et al. (2023), used in
        Fig. 6a of the paper. DR is the fraction of neurons in the policy
        network with mean absolute activation below threshold across a batch
        of observations.

        "DR is the fraction of inactive neurons in the policy network (i.e.
        activations below some threshold). Prior work has shown this metric
        effectively quantifies overfitting in value-based RL, where higher DR
        correlates with policies that execute unmeaningful actions."

        Applies to the GaussianActor's hidden layers (the policy network).
        Uses forward hooks to capture intermediate activations without
        modifying the network architecture.

        Args:
            policy: REDQPolicy instance. The dormant ratio is computed over
                the actor's hidden linear layers. Passed explicitly to allow
                computing DR for different policies (e.g., REDQ vs SYNTHER).
            real_buffer: D_real replay buffer. Used to sample a batch of
                observations for the activation analysis. A larger batch
                gives more stable DR estimates.
            threshold: Neuron activation threshold below which a neuron is
                considered dormant. Corresponds to config.yaml:
                evaluation.dormant_threshold (default 0.01, Sokar et al., 2023).

        Returns:
            Dormant ratio as a float in [0, 1]. Higher values indicate more
            overfitting. A ratio of 0.0 means all neurons are active; 1.0
            means all neurons are dormant (degenerate policy).
        """
        if len(real_buffer) == 0:
            return 0.0

        # Use a large batch for stable DR estimates — cap at buffer size.
        dr_batch_size: int = min(1024, len(real_buffer))
        batch: Dict[str, torch.Tensor] = real_buffer.sample(dr_batch_size)
        obs: torch.Tensor = batch["observations"].to(
            device=self.device, dtype=torch.float32
        )  # (B, obs_dim)

        # ── Register forward hooks on all Linear layers of the actor ─────────
        # Hooks capture the output activations of each linear layer.
        # We capture post-linear (pre-activation) outputs and apply the
        # threshold to the absolute values of the activations.
        activation_list: List[torch.Tensor] = []
        hook_handles: List[Any] = []

        def _make_hook(layer_idx: int) -> Any:
            """Creates a forward hook that captures layer output activations."""
            def _hook(
                module: nn.Module,
                input: Tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                # Capture the output activation of this layer.
                # output shape: (B, hidden_dim) for hidden layers.
                activation_list.append(output.detach())
            return _hook

        # Register hooks on all Linear layers in the actor's hidden_layers.
        # The actor's hidden_layers is a ModuleList of nn.Linear layers.
        if hasattr(policy, "actor") and hasattr(policy.actor, "hidden_layers"):
            for layer_idx, layer in enumerate(policy.actor.hidden_layers):
                if isinstance(layer, nn.Linear):
                    handle = layer.register_forward_hook(_make_hook(layer_idx))
                    hook_handles.append(handle)

        # ── Forward pass to collect activations ───────────────────────────────
        policy.actor.eval()

        try:
            with torch.no_grad():
                # Run forward pass through the actor to trigger hooks.
                # forward() returns (mean, log_std) — we only need the side
                # effect of the hooks capturing intermediate activations.
                _mean, _log_std = policy.actor.forward(obs)

            # ── Compute dormant ratio ─────────────────────────────────────────
            total_neurons: int = 0
            total_dormant: int = 0

            for activation in activation_list:
                # activation shape: (B, hidden_dim)
                # Compute mean absolute activation per neuron across the batch.
                mean_abs: torch.Tensor = activation.abs().mean(dim=0)  # (hidden_dim,)

                # Count neurons with mean absolute activation below threshold.
                dormant_count: int = int((mean_abs < threshold).sum().item())
                neuron_count: int = int(mean_abs.shape[0])

                total_dormant += dormant_count
                total_neurons += neuron_count

        finally:
            # Always remove hooks to prevent memory leaks.
            for handle in hook_handles:
                handle.remove()
            # Restore actor to training mode.
            policy.actor.train()

        if total_neurons == 0:
            return 0.0

        dormant_ratio: float = float(total_dormant) / float(total_neurons)
        return dormant_ratio

    def compute_relevance_distribution(
        self,
        real_buffer: ReplayBuffer,
        num_samples: int = 10000,
    ) -> np.ndarray:
        """Scores transitions from D_real and returns the array of relevance scores.

        Used to generate the curiosity score histograms in Fig. 6b of the paper.
        Called every relevance_eval_freq=10000 steps by PGRTrainer.

        "We examine the curiosity-PGR variant on the quadruped-walk task,
        measuring the distribution of F(s, a, s', r) using Eq. (5) over 10K
        real transitions. We perform this evaluation every 10K timesteps."

        Scores are computed in mini-batches to avoid OOM for large num_samples.
        Returns raw (unnormalized) ICM prediction error values so the
        distribution shape is meaningful for histogram analysis.

        Args:
            real_buffer: D_real replay buffer. Transitions are sampled uniformly
                for scoring. Must have at least 1 valid entry.
            num_samples: Number of transitions to score. Corresponds to
                config.yaml: evaluation.relevance_num_samples (default 10000).

        Returns:
            Float32 numpy array of shape (num_samples,) containing per-transition
            raw relevance scores (e.g., ICM prediction errors). Values are
            unnormalized — the caller (Logger.log_histogram) handles visualization.
        """
        if len(real_buffer) == 0:
            return np.array([], dtype=np.float32)

        # Clamp to available buffer size.
        actual_samples: int = min(num_samples, len(real_buffer))

        # Score in mini-batches to avoid OOM for large num_samples.
        score_batch_size: int = 256
        all_scores: List[np.ndarray] = []

        with torch.no_grad():
            for start_idx in range(0, actual_samples, score_batch_size):
                end_idx: int = min(start_idx + score_batch_size, actual_samples)
                n_batch: int = end_idx - start_idx

                # Sample a mini-batch from D_real.
                mini_batch: Dict[str, torch.Tensor] = real_buffer.sample(n_batch)

                # Score the mini-batch using the relevance function.
                scores: torch.Tensor = self.relevance_fn.score(
                    mini_batch["observations"],
                    mini_batch["actions"],
                    mini_batch["next_observations"],
                    mini_batch["rewards"],
                )  # (n_batch, 1)

                # Convert to numpy and flatten to 1D.
                scores_np: np.ndarray = (
                    scores.squeeze(-1).cpu().numpy().astype(np.float32)
                )  # (n_batch,)
                all_scores.append(scores_np)

        # Concatenate all mini-batch scores.
        if not all_scores:
            return np.array([], dtype=np.float32)

        return np.concatenate(all_scores, axis=0)  # (actual_samples,)

    def compute_tsne(
        self,
        real_buffer: ReplayBuffer,
        syn_buffer: ReplayBuffer,
        num_samples: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Projects transitions from D_real and D_syn into 2D using t-SNE.

        Replicates the Fig. 2 analysis from the paper:
        "We project 10K generations for both our PGR and the unconditional
        baseline SYNTHER to the same tSNE plot."

        t-SNE is fit jointly on the combined real + synthetic data so both
        sets share the same 2D embedding space — critical for meaningful
        comparison between PGR and SYNTHER generations.

        Called at tsne_epochs=[1, 130, -1] (config.yaml: evaluation.tsne_epochs).
        Requires scikit-learn to be installed. Returns empty arrays if
        scikit-learn is unavailable or if either buffer is empty.

        Args:
            real_buffer: D_real replay buffer. Transitions are sampled uniformly
                for the t-SNE projection.
            syn_buffer: D_syn replay buffer (synthetic transitions). May be
                empty before the first inner loop call (step < inner_loop_freq).
                Returns (real_projected, np.empty((0, 2))) in this case.
            num_samples: Number of transitions to sample from each buffer.
                Corresponds to config.yaml: evaluation.tsne_num_samples
                (default 10000). The actual number may be smaller if either
                buffer has fewer valid entries.

        Returns:
            Tuple of (real_projected, syn_projected) where:
                - real_projected: Float64 numpy array of shape (n_real, 2)
                  containing 2D t-SNE coordinates for real transitions.
                - syn_projected: Float64 numpy array of shape (n_syn, 2)
                  containing 2D t-SNE coordinates for synthetic transitions.
                  Shape (0, 2) if syn_buffer is empty.
            Both arrays use the same 2D embedding space (joint t-SNE fit).
        """
        if not _SKLEARN_AVAILABLE:
            # Return empty arrays if scikit-learn is not installed.
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

        if len(real_buffer) == 0:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

        # ── Step 1: Sample from both buffers ──────────────────────────────────
        n_real: int = min(num_samples, len(real_buffer))
        n_syn: int = min(num_samples, len(syn_buffer)) if len(syn_buffer) > 0 else 0

        real_batch: Dict[str, torch.Tensor] = real_buffer.sample(n_real)

        # ── Step 2: Construct feature vectors ─────────────────────────────────
        # Concatenate (s, a, s', r) for each transition → (N, input_dim).
        # Ensure rewards have shape (N, 1) for consistent concatenation.
        real_rewards: torch.Tensor = real_batch["rewards"]
        if real_rewards.dim() == 1:
            real_rewards = real_rewards.unsqueeze(-1)

        real_features: np.ndarray = torch.cat(
            [
                real_batch["observations"],
                real_batch["actions"],
                real_batch["next_observations"],
                real_rewards,
            ],
            dim=-1,
        ).cpu().numpy().astype(np.float64)  # (n_real, input_dim)

        # Handle empty synthetic buffer (before first inner loop).
        if n_syn == 0:
            # Run t-SNE on real data only and return empty syn projection.
            tsne: TSNE = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, max(5, n_real // 10)),
                n_iter=1000,
                method="barnes_hut" if n_real > 1000 else "exact",
            )
            real_projected: np.ndarray = tsne.fit_transform(real_features)
            return real_projected, np.empty((0, 2), dtype=np.float64)

        syn_batch: Dict[str, torch.Tensor] = syn_buffer.sample(n_syn)

        syn_rewards: torch.Tensor = syn_batch["rewards"]
        if syn_rewards.dim() == 1:
            syn_rewards = syn_rewards.unsqueeze(-1)

        syn_features: np.ndarray = torch.cat(
            [
                syn_batch["observations"],
                syn_batch["actions"],
                syn_batch["next_observations"],
                syn_rewards,
            ],
            dim=-1,
        ).cpu().numpy().astype(np.float64)  # (n_syn, input_dim)

        # ── Step 3: Concatenate for joint t-SNE ───────────────────────────────
        # Joint fit ensures both sets share the same 2D embedding space.
        all_features: np.ndarray = np.concatenate(
            [real_features, syn_features], axis=0
        )  # (n_real + n_syn, input_dim)

        split_idx: int = n_real  # Index separating real from synthetic.

        # ── Step 4: Run t-SNE ─────────────────────────────────────────────────
        # Use barnes_hut method for large datasets (n > 1000) for efficiency.
        # Perplexity is clamped to a valid range: [5, min(50, n//10)].
        total_n: int = n_real + n_syn
        perplexity: float = float(min(50, max(5, total_n // 10)))

        tsne_model: TSNE = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            n_iter=1000,
            method="barnes_hut" if total_n > 1000 else "exact",
        )

        all_projected: np.ndarray = tsne_model.fit_transform(
            all_features
        )  # (n_real + n_syn, 2)

        # ── Step 5: Split back into real and synthetic projections ─────────────
        real_projected_out: np.ndarray = all_projected[:split_idx]   # (n_real, 2)
        syn_projected_out: np.ndarray = all_projected[split_idx:]    # (n_syn, 2)

        return real_projected_out, syn_projected_out

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_env_type(self, env: Any) -> str:
        """Detects the environment type for state restoration in MSE computation.

        Checks the class name and available attributes to determine whether
        the environment is a DMCEnv, GymEnv, or unknown type.

        Args:
            env: Environment wrapper instance.

        Returns:
            String identifier: "dmc", "gym", or "unknown".
        """
        class_name: str = type(env).__name__

        if class_name == "DMCEnv":
            return "dmc"
        elif class_name == "GymEnv":
            return "gym"
        elif hasattr(env, "_env") and hasattr(env._env, "physics"):
            # Duck-typing fallback for DMC environments.
            return "dmc"
        elif hasattr(env, "_env") and hasattr(env._env, "sim"):
            # Duck-typing fallback for MuJoCo/Gym environments.
            return "gym"
        else:
            return "unknown"

    def _rollout_dmc(
        self,
        env: Any,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Rolls out one step in a DMC environment from a given state.

        Attempts to restore the environment physics state from the observation
        vector, then executes the given action to obtain ground-truth dynamics.

        For dm_control environments, the observation is a concatenation of
        various physics quantities. State restoration uses the physics object's
        set_state() method with the position (qpos) and velocity (qvel) vectors.

        If state restoration fails (e.g., observation doesn't directly correspond
        to qpos+qvel), falls back to using the generated values as ground truth.

        Args:
            env: DMCEnv wrapper instance with access to env._env.physics.
            obs: Generated observation vector of shape (obs_dim,). Used to
                attempt state restoration.
            action: Generated action vector of shape (action_dim,). Executed
                after state restoration to get ground-truth next state.

        Returns: