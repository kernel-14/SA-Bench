"""
main.py
Entry-point script for orchestrating the implementation of the MR.Q algorithm.
It loads configurations, initializes environments, setups the model, manages 
training, and periodically evaluates policy performance across benchmarks.
"""

import os
import torch
import torch.optim as optim
from utils import Utils
from dataset_loader import DatasetLoader
from model import Model
from replay_buffer import ReplayBuffer
from trainer import Trainer
from evaluation import Evaluation


class Main:
    """
    Core class to manage the MR.Q pipeline.
    - Loads configuration file.
    - Initializes DatasetLoader for loading environments.
    - Instantiates model, replay buffer, trainer, and evaluation components.
    - Executes the training and evaluation workflow.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Constructor for Main. Handles configuration loading and pipeline setup.

        Args:
            config_path (str): Path to configuration YAML file.
        """
        # Load and validate configuration
        self.config = Utils.load_config(config_path)
        self.training_config = self.config.get("training", {})
        self.model_config = self.config.get("model", {})
        self.eval_config = self.config.get("evaluation", {})
        self.env_config = self.config.get("environments", {})

        # Initialize dataset loader for RL environments
        self.dataset_loader = DatasetLoader(self.config)

        # Retrieve specific configurations
        self.epochs = self.training_config.get("epochs", 1_000_000)
        self.checkpoint_dir = self.config.get("logging", {}).get("checkpoint_dir", "./checkpoints")
        self.log_dir = self.config.get("logging", {}).get("log_dir", "./logs")

    def run_experiment(self):
        """
        Run the end-to-end training and evaluation of the MR.Q algorithm.
        """
        # Step 1: Environment Initialization
        print("Initializing environments...")
        train_envs = self._initialize_train_environments()
        eval_envs = self._initialize_eval_environments()

        # Step 2: Create Model
        print("Initializing model...")
        input_shape, action_dim, discrete = self._get_env_specs(train_envs[0])
        model = Model({
            **self.model_config,
            "observation_shape": input_shape,
            "action_dim": action_dim,
            "discrete_action_space": discrete,
            "is_image_input": len(input_shape) > 1
        })

        # Step 3: Replay Buffer
        print("Initializing replay buffer...")
        replay_buffer_capacity = self.training_config.get("replay_buffer_size", 1_000_000)
        replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

        # Step 4: Optimizer Setup
        optimizer = optim.AdamW(model.parameters(),
                                lr=self.training_config.get("learning_rate", 1e-4),
                                weight_decay=self.config["optimizer"]["weight_decay"])

        # Step 5: Trainer Initialization
        print("Initializing trainer...")
        trainer = Trainer(model, optimizer, replay_buffer, self.config)

        # Step 6: Evaluation Component
        print("Initializing evaluation module...")
        evaluation = Evaluation(model, eval_envs, metrics=["mean", "median", "IQM"], config=self.config)

        # Step 7: Training Loop
        print("Starting training loop...")
        self._run_training_loop(trainer, train_envs, evaluation)

    def _initialize_train_environments(self):
        """
        Initialize training environments based on the list of tasks in the config.

        Returns:
            List[Env]: List of Gym/DM/Atari training environments.
        """
        env_list = []
        for env_name in (
            self.env_config.get("gym_tasks", [])
            + self.env_config.get("dm_control_proprioceptive_tasks", [])
            + self.env_config.get("dm_control_visual_tasks", [])
        ):
            env = self.dataset_loader.load_env(env_name)
            env_list.append(env)
        return env_list

    def _initialize_eval_environments(self):
        """
        Initialize evaluation environments for periodic agent testing.

        Returns:
            List[Env]: List of Gym/DM/Atari evaluation environments.
        """
        eval_env_list = []
        for env_name in (
            self.env_config.get("atari_tasks", [])
            + self.env_config.get("dm_control_visual_tasks", [])
        ):
            env = self.dataset_loader.load_env(env_name)
            eval_env_list.append(env)
        return eval_env_list

    def _get_env_specs(self, environment):
        """
        Extract environment-specific state and action dimensions.

        Args:
            environment: Gym/DM/Atari environment object.

        Returns:
            Tuple: (input_shape, action_dim, discrete)
                - input_shape: Shape of environment observations.
                - action_dim: Size of action space.
                - discrete: Boolean, whether action space is discrete.
        """
        obs_space = environment.observation_space
        action_space = environment.action_space

        # Determine observation shape
        if isinstance(obs_space, gym.spaces.Box):
            input_shape = obs_space.shape
        else:
            raise ValueError(f"Unsupported observation space: {obs_space}")

        # Determine action space properties
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n
            discrete = True
        elif isinstance(action_space, gym.spaces.Box):
            action_dim = action_space.shape[0]
            discrete = False
        else:
            raise ValueError(f"Unsupported action space: {action_space}")

        return input_shape, action_dim, discrete

    def _run_training_loop(self, trainer: Trainer, train_envs, evaluator: Evaluation):
        """
        Training and evaluation routine for the MR.Q algorithm.

        Args:
            trainer (Trainer): The Trainer instance for managing model training.
            train_envs (List[Env]): List of training environments.
            evaluator (Evaluation): The Evaluation instance for agent testing.
        """
        step_counter = 0
        checkpoint_freq = self.training_config.get("target_network_update_frequency", 250)
        eval_freq = self.eval_config.get("num_episodes", 10)

        # Interact with environment and train
        print("Beginning training...")
        for epoch in range(self.epochs):
            for train_env in train_envs:
                obs = train_env.reset()
                done = False
                while not done:
                    state_tensor = torch.tensor(obs, device=trainer.device, dtype=torch.float32).unsqueeze(0)

                    # Sample action from policy
                    with torch.no_grad():
                        action = trainer.model.policy_head(
                            trainer.model.forward_state(state_tensor)).cpu().numpy()

                    action = action.argmax(-1) if trainer.model.discrete_action_space else action

                    # Interaction with environment
                    next_obs, reward, done, info = train_env.step(action)
                    transition = {
                        "state": obs,
                        "action": action,
                        "reward": reward,
                        "next_state": next_obs,
                        "terminal": done,
                    }
                    # Store transition in replay buffer
                    trainer.replay_buffer.add_sample(transition)
                    obs = next_obs

                    # Increment global step
                    step_counter += 1

                # Perform one optimization step if enough samples
                if trainer.replay_buffer.size >= trainer.batch_size:
                    trainer.train_one_epoch()

                # Periodic Evaluation
                if step_counter % eval_freq == 0:
                    eval_results = evaluator.evaluate_policy()
                    print(f"Evaluation at step {step_counter}: {eval_results}")
                    evaluator.visualize_learning_curves(eval_results, self.log_dir)

                # Save Checkpoints
                if step_counter % checkpoint_freq == 0:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                    Utils.save_model(trainer.model, trainer.optimizer, checkpoint_path, epoch)
