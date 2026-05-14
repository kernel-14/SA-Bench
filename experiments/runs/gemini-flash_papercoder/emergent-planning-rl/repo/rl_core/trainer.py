```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import gymnasium.vector
from typing import Any, Dict, List, Tuple, Callable, Optional

# Attempt to import Config and Logger from their respective modules
try:
    from config import Config
    from utils.logger import Logger
    from environments.base_environment import BaseEnvironment
    from agents.base_agent import BaseAgentModel
    from environments.sokoban import SokobanEnv # Needed for env factory
    # Potentially MiniPacManEnv for later, but not in this file scope yet
except ImportError:
    # Dummy classes for standalone testing or if dependencies are not yet available
    print("Warning: Could not import core dependencies. Using dummy classes.")
    class Config:
        def __init__(self, data: Dict = None): self._data = data if data is not None else {}
        def get(self, key: str, default: Any = None) -> Any:
            keys = key.split('.')
            current = self._data
            for k in keys:
                if isinstance(current, dict) and k in current: current = current[k]
                else: return default
            return current
        def set(self, key: str, value: Any) -> None: pass
        def save(self, output_path: str) -> None: pass

    class Logger:
        def __init__(self, config: Config): pass
        def log_info(self, message: str) -> None: print(f"INFO: {message}")
        def log_metric(self, name: str, value: float, step: int = 0, tag: str = 'train') -> None: print(f"METRIC: {tag}/{name} @ {step}: {value}")
        def log_figure(self, name: str, fig: Any, step: int = 0) -> None: print(f"FIGURE: {name} @ {step}")
        def save_model_weights(self, model: Any, path: str) -> None: print(f"Saving dummy model to {path}")
        def load_model_weights(self, model: Any, path: str) -> None: print(f"Loading dummy model from {path}")
        def close(self) -> None: print("Logger closed.")

    class BaseEnvironment:
        def __init__(self, config: Config) -> None: self.config = config
        def reset(self, level_config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]: return np.zeros((8,8,7)), {}
        def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]: return np.zeros((8,8,7)), 0.0, False, {'is_success': False}
        def render(self) -> np.ndarray: return np.zeros((10,10,3), dtype=np.uint8)
        def get_state(self) -> np.ndarray: return np.zeros((8,8,7))
        def set_state(self, state: np.ndarray) -> None: pass
        def get_action_space_size(self) -> int: return 5
        def get_observation_space_shape(self) -> Tuple[int, ...]: return (8,8,7)
        def simulate_future_state(self, current_state: np.ndarray, action: int) -> Dict[str, Any]: return {'next_state_2d': current_state}
        @property
        def episode_length_max(self) -> int: return 120 # Added property for evaluation

    class BaseAgentModel(nn.Module):
        def __init__(self, config: Config) -> None:
            super().__init__()
            self.device = torch.device("cpu")
            self.action_space_size = 5
            self.config = config
            self.grid_height = 8
            self.grid_width = 8
            self.convlstm_channels = 32 # Dummy for DRCAgent. _init_hidden_states requires this
        def _build_model_architecture(self) -> None: pass
        def forward(self, obs: torch.Tensor, hidden_state: Any = None) -> Tuple[torch.Tensor, Any]: return torch.randn(obs.shape[0], 256), None
        def get_action_logits(self, model_output_vector: torch.Tensor) -> torch.Tensor: return torch.randn(model_output_vector.shape[0], self.action_space_size)
        def get_value_estimate(self, model_output_vector: torch.Tensor) -> torch.Tensor: return torch.randn(model_output_vector.shape[0], 1)
        def get_cell_state(self, layer_idx: int, tick_idx: int = -1) -> torch.Tensor: return torch.randn(8,8,32)
        def set_cell_state(self, layer_idx: int, tick_idx: int, new_state: torch.Tensor) -> None: pass
        def act(self, obs: np.ndarray, hidden_state: Any = None, greedy: bool = True) -> Tuple[int, Any, torch.Tensor, torch.Tensor]:
            dummy_obs_tensor = torch.from_numpy(obs).float().to(self.device).permute(2, 0, 1).unsqueeze(0)
            model_output_vector, new_hidden_state = self.forward(dummy_obs_tensor, hidden_state)
            policy_logits = self.get_action_logits(model_output_vector).squeeze(0)
            value_estimate = self.get_value_estimate(model_output_vector).squeeze(0)
            action = torch.argmax(policy_logits, dim=-1).item() if greedy else dist.Categorical(logits=policy_logits).sample().item()
            return action, new_hidden_state, policy_logits, value_estimate
        def _init_hidden_states(self, device: torch.device, channels: int, H: int, W: int) -> List[torch.Tensor]:
            # Dummy implementation assuming D=3 ConvLSTM layers
            return [torch.zeros(1, channels, H, W, device=device) for _ in range(3)]


class IMPALATrainer:
    """
    Implements the IMPALA reinforcement learning algorithm. Manages agent-environment
    interaction, gathers trajectories, computes V-trace updates, and optimizes the agent.
    """

    def __init__(self, agent_model: BaseAgentModel, env: BaseEnvironment, config: Config, logger: Logger) -> None:
        """
        Initializes the IMPALA trainer.

        Args:
            agent_model (BaseAgentModel): The agent model to be trained.
            env (BaseEnvironment): A prototype environment instance (used to create vectorized environments).
            config (Config): The configuration object.
            logger (Logger): The logger instance for logging metrics and checkpoints.
        """
        self.agent_model: BaseAgentModel = agent_model
        self.env_prototype: BaseEnvironment = env # Storing prototype to create vector envs
        self.config: Config = config
        self.logger: Logger = logger
        self.device: torch.device = self.agent_model.device

        # Hyperparameters from config
        self.unroll_length: int = self.config.get('rl_training.unroll_length', 20)
        self.batch_size: int = self.config.get('rl_training.batch_size', 16) # Number of parallel environments
        self.discount_rate: float = self.config.get('rl_training.discount_rate', 0.97) # gamma
        self.v_trace_lambda: float = self.config.get('rl_training.v_trace_lambda', 0.97) # lambda
        self.l2_action_logits_penalty: float = self.config.get('rl_training.l2_action_logits_penalty', 0.001)
        self.l2_policy_value_heads_reg: float = self.config.get('rl_training.l2_policy_value_heads_reg', 1e-5)
        self.entropy_penalty: float = self.config.get('rl_training.entropy_penalty', 0.01)
        self.rho_max_policy: float = self.config.get('rl_training.rho_max_policy', 1.0) # Clipping for policy gradient
        self.rho_max_value: float = self.config.get('rl_training.rho_max_value', 1.0) # Clipping for value target

        # Optimizer setup
        self.optimizer: optim.Adam = optim.Adam(
            self.agent_model.parameters(),
            lr=self.config.get('rl_training.learning_rate', 0.0004)
        )
        
        # Learning rate scheduler (linear decay)
        self.initial_lr: float = self.config.get('rl_training.learning_rate', 0.0004)
        self.final_lr: float = self.config.get('rl_training.learning_rate_final', 0.0)
        # The scheduler step will be called manually in the training loop based on total steps
        
        self.logger.log_info(f"IMPALATrainer initialized for agent: {type(agent_model).__name__}")
        self.logger.log_info(f"Training on device: {self.device}")
        self.logger.log_info(f"Unroll Length: {self.unroll_length}, Batch Size: {self.batch_size}")


    def _create_env_callable(self) -> Callable[[], BaseEnvironment]:
        """
        Creates a callable that returns a new instance of the environment type.
        This is needed for gymnasium.vector.SyncVectorEnv.
        """
        env_class = self.env_prototype.__class__
        return lambda: env_class(self.config)


    def train(self, num_transitions: int, checkpoint_interval: int) -> None:
        """
        Orchestrates the main IMPALA training loop.

        Args:
            num_transitions (int): Total number of environment transitions to train for.
            checkpoint_interval (int): How often (in transitions) to save a model checkpoint.
        """
        self.logger.log_info(f"Starting IMPALA training for {num_transitions} transitions.")
        self.agent_model.train() # Set agent to training mode

        # Setup vectorized environment
        env_fns = [self._create_env_callable() for _ in range(self.batch_size)]
        vector_env = gym.vector.SyncVectorEnv(env_fns)

        # Initial reset of all parallel environments
        current_obs_batch_np, info_batch = vector_env.reset()
        # Convert initial observations to Torch tensor, permute to (B, C, H, W)
        current_obs_batch_t = torch.from_numpy(current_obs_batch_np).float().to(self.device)
        current_obs_batch_t = current_obs_batch_t.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)

        # Initialize hidden states for all parallel agents
        # Assuming BaseAgentModel subclasses provide a way to initialize (e.g., for DRCAgent, it's a list of D states)
        # For a batch, the internal _prev_h_states/_prev_c_states in agent_model.forward handles batching implicitly from initial_h_states
        hidden_state_batch: Any = None # DRCAgent will initialize to zeros if None on first forward

        current_total_steps: int = 0
        episode_rewards: List[float] = [0.0] * self.batch_size
        episode_counts: List[int] = [0] * self.batch_size
        
        while current_total_steps < num_transitions:
            # Store data for one unroll segment
            unroll_rewards: List[torch.Tensor] = []
            unroll_dones: List[torch.Tensor] = []
            unroll_obs_t: List[torch.Tensor] = []
            unroll_behavior_actions: List[torch.Tensor] = []
            unroll_behavior_policy_logits: List[torch.Tensor] = []
            unroll_learner_value_estimates: List[torch.Tensor] = [] # V(s_t) predicted by learner

            # Collect `unroll_length` steps for all `batch_size` parallel environments
            for t in range(self.unroll_length):
                unroll_obs_t.append(current_obs_batch_t.clone())

                # Agent acts (behavior policy) - sample action
                # `act` returns action (int), new_hidden_state (Any), policy_logits (Tensor), value_estimate (Tensor)
                # policy_logits and value_estimate are for the current state `current_obs_batch_t`
                # hidden_state is for the batch (list of D h/c states, or None for ResNet)
                
                # Need to run forward pass separately to store current learner policy/value
                # and then act to get chosen action and actual behavior logits
                with torch.no_grad():
                    # Get learner's current estimates for current_obs_batch_t
                    model_output_vector, next_hidden_state_for_behavior = self.agent_model.forward(current_obs_batch_t, hidden_state_batch)
                    learner_policy_logits_t = self.agent_model.get_action_logits(model_output_vector)
                    learner_value_estimates_t = self.agent_model.get_value_estimate(model_output_vector).squeeze(-1) # (B,)
                    
                    unroll_learner_value_estimates.append(learner_value_estimates_t.clone())

                    # Sample action from learner's policy for behavior policy
                    action_distribution = dist.Categorical(logits=learner_policy_logits_t)
                    actions_batch_t = action_distribution.sample() # (B,)
                    
                    # Store log_probs of chosen actions under behavior policy
                    unroll_behavior_policy_logits.append(learner_policy_logits_t.clone())
                    unroll_behavior_actions.append(actions_batch_t.clone())

                # Environment step
                next_obs_batch_np, rewards_batch_np, dones_batch_np, info_batch = vector_env.step(actions_batch_t.cpu().numpy())
                
                rewards_batch_t = torch.from_numpy(rewards_batch_np).float().to(self.device) # (B,)
                dones_batch_t = torch.from_numpy(dones_batch_np).bool().to(self.device) # (B,)

                unroll_rewards.append(rewards_batch_t)
                unroll_dones.append(dones_batch_t)

                current_total_steps += self.batch_size # Each step in vector_env advances total_steps by batch_size

                # Update current_obs and hidden_state for next iteration
                current_obs_batch_np = next_obs_batch_np
                current_obs_batch_t = torch.from_numpy(current_obs_batch_np).float().to(self.device)
                current_obs_batch_t = current_obs_batch_t.permute(0, 3, 1, 2)
                
                # Hidden state is updated implicitly by agent_model.forward,
                # then stored as `self._prev_h_states`, `self._prev_c_states`
                # So we just pass None for the next forward, and the agent_model handles it.
                hidden_state_batch = next_hidden_state_for_behavior # This carries the h/c states from previous step (t) to next (t+1)

                # Handle episode termination for environments that are done
                for env_idx in range(self.batch_size):
                    episode_rewards[env_idx] += rewards_batch_np[env_idx]
                    if dones_batch_np[env_idx]:
                        episode_counts[env_idx] += 1
                        self.logger.log_metric('episode_reward', episode_rewards[env_idx], step=current_total_steps, tag='train')
                        self.logger.log_metric('episodes_completed', episode_counts[env_idx], step=current_total_steps, tag='train')
                        episode_rewards[env_idx] = 0.0 # Reset reward for new episode
                        # Hidden states for done environments are reset to initial (zeros)
                        # The agent's `forward` method implicitly handles this when it encounters a new episode.

            # After collecting unroll_length transitions for all environments
            # Stack all collected tensors
            unroll_rewards_t = torch.stack(unroll_rewards, dim=0) # (U, B)
            unroll_dones_t = torch.stack(unroll_dones, dim=0) # (U, B)
            unroll_obs_t = torch.stack(unroll_obs_t, dim=0) # (U, B, C, H, W)
            unroll_behavior_actions_t = torch.stack(unroll_behavior_actions, dim=0) # (U, B)
            unroll_behavior_policy_logits_t = torch.stack(unroll_behavior_policy_logits, dim=0) # (U, B, ActionSpace)
            unroll_learner_value_estimates_t = torch.stack(unroll_learner_value_estimates, dim=0) # (U, B)

            # Get learner's value estimate for the state *after* the unroll (bootstrap value)
            with torch.no_grad():
                next_model_output_vector, _ = self.agent_model.forward(current_obs_batch_t, hidden_state_batch)
                next_learner_value_estimate = self.agent_model.get_value_estimate(next_model_output_vector).squeeze(-1) # (B,)
            
            # --- V-trace and Loss Calculation ---
            # Compute policy_logits and value_estimates for all unroll states using the *current* learner's parameters
            # Need to detach hidden_state_batch from previous step's graph to avoid issues
            h_states_for_learner = [h.detach() for h in hidden_state_batch[0]] if hidden_state_batch else None
            c_states_for_learner = [c.detach() for c in hidden_state_batch[1]] if hidden_state_batch else None
            
            # Reshape unroll_obs_t to (U*B, C, H, W) for a single forward pass
            unroll_obs_flat = unroll_obs_t.view(-1, *unroll_obs_t.shape[2:]) # (U*B, C, H, W)

            # Perform forward pass with `hidden_state_batch` to get updated states and correct value/logits for learner
            learner_policy_logits_flat, learner_value_estimates_flat, _ = self.agent_model.forward(
                unroll_obs_flat, (h_states_for_learner, c_states_for_learner) if h_states_for_learner else None
            )
            learner_policy_logits = learner_policy_logits_flat.view(self.unroll_length, self.batch_size, -1)
            learner_value_estimates = learner_value_estimates_flat.view(self.unroll_length, self.batch_size)
            
            # This is where the actual V-trace calculations happen
            v_trace_targets, policy_gradient_coefs, entropy_term = self._calculate_v_trace(
                unroll_rewards_t,
                unroll_dones_t,
                unroll_behavior_actions_t,
                unroll_behavior_policy_logits_t,
                learner_policy_logits,
                learner_value_estimates,
                next_learner_value_estimate
            )

            # --- Loss computation ---
            total_loss, policy_loss, value_loss, entropy_loss, l2_reg_loss = self._compute_losses(
                policy_gradient_coefs,
                entropy_term,
                v_trace_targets,
                learner_value_estimates,
                learner_policy_logits, # Use for L2 regularization
                self.agent_model
            )

            # --- Optimization Step ---
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Update learning rate (manual linear decay)
            progress_ratio = current_total_steps / num_transitions
            new_lr = self.initial_lr - progress_ratio * (self.initial_lr - self.final_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            # --- Logging ---
            if current_total_steps % (checkpoint_interval // 10) < self.batch_size: # Log more frequently than save
                self.logger.log_metric('total_loss', total_loss.item(), step=current_total_steps, tag='train')
                self.logger.log_metric('policy_loss', policy_loss.item(), step=current_total_steps, tag='train')
                self.logger.log_metric('value_loss', value_loss.item(), step=current_total_steps, tag='train')
                self.logger.log_metric('entropy_loss', entropy_loss.item(), step=current_total_steps, tag='train')
                self.logger.log_metric('l2_reg_loss', l2_reg_loss.item(), step=current_total_steps, tag='train')
                self.logger.log_metric('learning_rate', new_lr, step=current_total_steps, tag='train')

            # --- Checkpointing ---
            if current_total_steps % checkpoint_interval < self.batch_size:
                checkpoint_name = f"agent_checkpoint_{current_total_steps}.pth"
                self.logger.save_model_weights(self.agent_model, checkpoint_name)
                # For emergence studies, also save checkpoints for a specific agent type and total steps
                agent_type_name = self.config.get('agent.type')
                if agent_type_name == 'DRCAgent' and current_total_steps <= 50000000 and current_total_steps % 1000000 < self.batch_size:
                     self.logger.save_model_weights(self.agent_model, f"emergence_checkpoint_{current_total_steps}.pth")


        # Save final model
        self.logger.save_model_weights(self.agent_model, "agent_final.pth")
        self.logger.log_info("IMPALA training finished.")
        vector_env.close()

    def _calculate_v_trace(
        self,
        rewards: torch.Tensor, # (U, B)
        dones: torch.Tensor,   # (U, B)
        behavior_actions: torch.Tensor, # (U, B)
        behavior_policy_logits: torch.Tensor, # (U, B, A)
        learner_policy_logits: torch.Tensor, # (U, B, A)
        learner_value_estimates: torch.Tensor, # (U, B)
        next_learner_value_estimate: torch.Tensor, # (B,) for V(s_k)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates V-trace targets and policy gradient coefficients.
        Based on Espeholt et al. (2018), Appendix B.1.
        """
        unroll_length, batch_size = rewards.shape
        action_space_size = learner_policy_logits.shape[-1]

        # Convert logits to log_softmax for probability calculation
        learner_policy_log_probs = torch.log_softmax(learner_policy_logits, dim=-1)
        behavior_policy_log_probs = torch.log_softmax(behavior_policy_logits, dim=-1)

        # Get log_probs for the actually taken actions
        learner_action_log_probs = torch.gather(learner_policy_log_probs, -1, behavior_actions.unsqueeze(-1)).squeeze(-1) # (U, B)
        behavior_action_log_probs = torch.gather(behavior_policy_log_probs, -1, behavior_actions.unsqueeze(-1)).squeeze(-1) # (U, B)

        # Calculate importance sampling ratios (rhos)
        rhos = torch.exp(learner_action_log_probs - behavior_action_log_probs) # (U, B)

        # Clip rhos for policy and value updates
        clipped_rhos_policy = torch.min(rhos, torch.tensor(self.rho_max_policy, device=self.device)) # (U, B)
        clipped_rhos_value = torch.min(rhos, torch.tensor(self.rho_max_value, device=self.device))   # (U, B)

        v_trace_targets = torch.zeros_like(rewards, device=self.device)
        
        # Initialize `v_s_plus_1` for the backward pass
        # This is the V-trace value target for the state s_{k} (after the unroll).
        # If the episode ends at s_{k-1} (done[k-1] == True), then v_s_plus_1 for k-1 should be 0.
        # However, the formula from Espeholt et al. (2018) treats `next_learner_value_estimate` as V(s_k).
        # And `dones[t]` is typically for transition `s_t -> s_{t+1}`. If `s_{t+1}` is terminal, `dones[t]` is True.
        # So `1 - dones[t]` implies if `s_{t+1}` is non-terminal.
        
        # Bootstrap value from the state after the unroll
        v_s_plus_1 = next_learner_value_estimate # (B,)
        
        # Backward pass to compute V-trace targets
        for t in reversed(range(unroll_length)):
            is_terminal_next_state = dones[t].float() # 1.0 if s_{t+1} is terminal, 0.0 otherwise
            
            # learner_value_estimates for s_{t+1} needed for bootstrapping part `V(s_{s+1})`
            # If t+1 is beyond unroll, use `next_learner_value_estimate`
            bootstrap_V_s_plus_1 = learner_value_estimates[t+1] if t + 1 < unroll_length else next_learner_value_estimate
            
            # V-trace value target update (Eq. 3 from Espeholt et al. 2018, Appendix B.1)
            # v_s = r_s + gamma_s * ( (1-d_{s+1}) * v_{s+1} * lambda_c_s + (1-d_{s+1}) * V(s_{s+1}) * (1 - lambda_c_s) )
            # where lambda_c_s = lambda * c_s and c_s = min(rho_s, 1.0) for value targets.
            # We use `clipped_rhos_value[t]` as `c_s` and `self.v_trace_lambda` as `lambda`

            # Note: the paper uses `(1-d_{s+1})` which is `(1 - is_terminal_next_state)`.
            # This factor makes terms zero if `s_{s+1}` is terminal.
            v_trace_target_t = rewards[t] + self.discount_rate * (1 - is_terminal_next_state) * (
                self.v_trace_lambda * clipped_rhos_value[t] * v_s_plus_1 +
                (1 - self.v_trace_lambda * clipped_rhos_value[t]) * bootstrap_V_s_plus_1
            )
            v_trace_targets[t] = v_trace_target_t
            v_s_plus_1 = v_trace_target_t # For the next iteration (t-1), this becomes `v_{t+1}`

        # Calculate advantages for policy update
        # Advantage = V-trace target - learner's value estimate
        advantages = v_trace_targets - learner_value_estimates # (U, B)
        policy_gradient_coefs = clipped_rhos_policy * advantages # (U, B)

        # Calculate entropy term
        # Entropy of the learner's policy (average over actions and batch)
        policy_dist = dist.Categorical(logits=learner_policy_logits) # (U, B, A)
        entropy_term = policy_dist.entropy().mean() # Scalar
        
        return v_trace_targets, policy_gradient_coefs, entropy_term


    def _compute_losses(
        self,
        policy_gradient_coefs: torch.Tensor, # (U, B)
        entropy_term: torch.Tensor,          # Scalar
        v_trace_targets: torch.Tensor,       # (U, B)
        learner_value_estimates: torch.Tensor, # (U, B)
        learner_policy_logits: torch.Tensor,   # (U, B, A)
        model: nn.Module                       # Agent model for L2 regularization
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the total loss for IMPALA.
        """
        # Policy Loss (A_t * log(pi_learner(a_t|s_t))) + L2 on logits
        # We want to maximize expected advantage, so negate for gradient descent
        policy_loss = - (policy_gradient_coefs * learner_action_log_probs).mean() # (U, B) -> scalar

        # Value Loss (MSE between V-trace targets and learner's value estimates)
        value_loss = 0.5 * ((v_trace_targets - learner_value_estimates)**2).mean()

        # Entropy Loss (maximize entropy for exploration)
        entropy_loss = - self.entropy_penalty * entropy_term

        # L2 regularization on policy and value head parameters
        l2_reg_loss = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            if 'policy_value_heads' in name and param.requires_grad:
                l2_reg_loss += self.l2_policy_value_heads_reg * torch.sum(param**2)
        
        # L2 penalty on action logits
        # This penalizes large logits, which can make policy distributions too sharp.
        # `learner_policy_logits` is (U, B, A)
        l2_action_logits_loss = self.l2_action_logits_penalty * torch.sum(learner_policy_logits**2) / (learner_policy_logits.numel())
        # Use numel for averaging over all elements, or sum and let mean() handle it if taken over batch/unroll.
        # The paper implies sum of logits^2 over all, then multiplied by penalty. So `torch.sum(logits**2)`.

        total_loss = policy_loss + value_loss + entropy_loss + l2_reg_loss + l2_action_logits_loss

        return total_loss, policy_loss, value_loss, entropy_loss, l2_reg_loss + l2_action_logits_loss


    def evaluate_behavior(self, levels: List[np.ndarray], num_thinking_steps: int, policy_fn: Optional[Callable] = None) -> float:
        """
        Evaluates the agent's performance on a given set of levels, optionally with thinking steps.

        Args:
            levels (List[np.ndarray]): A list of initial symbolic observations for levels to evaluate.
            num_thinking_steps (int): Number of internal computational steps the agent performs
                                      before taking its first action in an episode.
            policy_fn (Optional[Callable]): A callable that defines how the agent acts. If None,
                                           `agent_model.act()` with `greedy=config.rl_training.inference_greedy`
                                           is used. (This parameter is kept for API consistency, but the design implies
                                           using agent_model.act directly).

        Returns:
            float: The success rate (percentage of levels solved).
        """
        self.logger.log_info(f"Evaluating agent behavior on {len(levels)} levels with {num_thinking_steps} thinking steps.")
        self.agent_model.eval() # Set agent to evaluation mode
        
        solved_count: int = 0
        
        inference_greedy: bool = self.config.get('rl_training.inference_greedy', True)
        
        for level_idx, initial_level_state in enumerate(levels):
            # Create a fresh environment for each evaluation episode to ensure isolation
            eval_env: BaseEnvironment = self.env_prototype.__class__(self.config)
            
            # Reset environment using the specific initial level state
            current_obs_np, info = eval_env.reset(level_config={'level_type': 'custom', 'initial_state': initial_level_state, 'seed': level_idx})
            current_obs_t = torch.from_numpy(current_obs_np).float().to(self.device).permute(2, 0, 1).unsqueeze(0)

            # Initialize hidden state for the agent
            # DRCAgent will initialize to zeros if None on first forward.
            # ResNetAgent (feedforward) will always ignore hidden_state_input.
            # For DRCAgent, ensure the initial state is proper (list of D zero tensors, not just None)
            hidden_state: Any = None
            if hasattr(self.agent_model, '_init_hidden_states'): # Check if it's a DRC agent
                hidden_state_h = self.agent_model._init_hidden_states(self.device, self.agent_model.convlstm_channels, self.agent_model.grid_height, self.agent_model.grid_width)
                hidden_state_c = self.agent_model._init_hidden_states(self.device, self.agent_model.convlstm_channels, self.agent_model.grid_height, self.agent_model.grid_width)
                hidden_state = (hidden_state_h, hidden_state_c)


            with torch.no_grad():
                # --- "Thinking Steps" Phase ---
                for _ in range(num_thinking_steps):
                    # Agent performs internal computation without environment interaction.
                    # This means we just run forward (or act), and the environment's state does not change.
                    _, hidden_state, _, _ = self.agent_model.act(current_obs_np, hidden_state, greedy=inference_greedy)

                # --- Active Episode Phase ---
                episode_done: bool = False
                episode_steps: int = 0
                max_episode_steps: int = eval_env.episode_length_max # Using env's max episode length
                
                while not episode_done and episode_steps < max_episode_steps:
                    action_int, hidden_state, _, _ = self.agent_model.act(current_obs_np, hidden_state, greedy=inference_greedy)
                    
                    next_obs_np, reward_float, episode_done, info_dict = eval_env.step(action_int)
                    
                    current_obs_np = next_obs_np
                    episode_steps += 1

                    # Check for success condition from info_dict (