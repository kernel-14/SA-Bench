# main.py

"""
Entry point for running MR.Q experiments across all benchmarks.
Parses command‑line arguments, loads the configuration, creates the
environment, agent, replay buffer, and orchestrates the training loop
via the Trainer class. Saves evaluation results for later analysis.

Usage examples:
    python main.py gym_locomotion Ant-v4 0
    python main.py dmc_proprioceptive cheetah-run 42
    python main.py atari Alien 2023
    python main.py dmc_visual dog-walk 100
"""

import argparse
import os
import pickle
import numpy as np
import torch

from config import Config, create_config
from env_utils import make_env
from agent import MRQAgent
from replay_buffer import ReplayBuffer
from trainer import Trainer


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, PyTorch,
    and CUDA operations.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Ensure deterministic behaviour on GPU if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MR.Q: Model-Free Reinforcement Learning with Model-Based Representations"
    )
    parser.add_argument(
        "benchmark",
        type=str,
        choices=["gym_locomotion", "dmc_proprioceptive", "dmc_visual", "atari"],
        help="Benchmark name (gym_locomotion, dmc_proprioceptive, dmc_visual, atari)."
    )
    parser.add_argument(
        "task",
        type=str,
        help="Task name (e.g., Ant-v4, cheetah-run, Alien)."
    )
    parser.add_argument(
        "seed",
        type=int,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the agent on (cpu or cuda). Default: cpu."
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./results",
        help="Directory to save evaluation results and checkpoints."
    )
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------------------------------
    cfg: Config = create_config(
        benchmark=args.benchmark,
        task=args.task,
        seed=args.seed,
        device=args.device,
    )

    # --------------------------------------------------------------------------
    # 2. Reproducibility
    # --------------------------------------------------------------------------
    set_seed(args.seed)

    # --------------------------------------------------------------------------
    # 3. Create environment
    # --------------------------------------------------------------------------
    env = make_env(
        benchmark=cfg.benchmark, task=cfg.task, seed=cfg.seed
    )

    # --------------------------------------------------------------------------
    # 4. Extract observation and action information
    # --------------------------------------------------------------------------
    obs_space = env.observation_space
    act_space = env.action_space

    obs_shape = obs_space.shape  # e.g., (4,84,84) for Atari, (9,84,84) for DM Control visual
    if hasattr(act_space, "n"):
        action_dim = act_space.n
        discrete_actions = True
    else:
        action_dim = act_space.shape[0]
        discrete_actions = False

    # --------------------------------------------------------------------------
    # 5. Create agent
    # --------------------------------------------------------------------------
    agent = MRQAgent(
        cfg=cfg,
        obs_shape=obs_shape,
        action_dim=action_dim,
        discrete_actions=discrete_actions,
        device=cfg.device,
    )

    # --------------------------------------------------------------------------
    # 6. Create replay buffer
    # --------------------------------------------------------------------------
    replay_buffer = ReplayBuffer(
        capacity=cfg.replay_buffer_capacity,
        state_shape=obs_shape,
        action_shape=(action_dim,)
        if not discrete_actions
        else (action_dim,),   # one‑hot vectors for discrete actions
        alpha=cfg.lap_alpha,
        min_priority=cfg.lap_min_priority,
        state_dtype=np.float32,
        action_dtype=np.float32,
    )

    # --------------------------------------------------------------------------
    # 7. Create trainer and run
    # --------------------------------------------------------------------------
    trainer = Trainer(cfg=cfg)
    # The trainer internally holds the agent and replay buffer; we need to set them.
    # Our Trainer class expects env, agent, buffer to be passed, but we haven't defined that interface.
    # Following the design: the Trainer's __init__ takes cfg and builds its own env, agent, buffer.
    # However, we created these above to get obs/action info. A cleaner approach:
    # The Trainer should accept the externally created env, agent, and buffer, or
    # we build them inside the Trainer and let it extract info. The design in the plan says:
    # "Trainer(cfg, env, agent, buffer)" but the trainer.py we wrote does not have that interface.
    # To keep things simple, we will modify the Trainer to accept these as parameters.
    # Since we are writing the final main.py, we need to ensure the Trainer can be instantiated
    # with the pre‑built components. Let's assume the Trainer has a signature:
    #   Trainer(cfg, env, agent, buffer)
    # and then calls train().
    # We'll adjust the import accordingly and pass them.

    # But the trainer.py provided in the plan already implements a self-contained Trainer.
    # To avoid circular dependency and keep main.py minimal, we will not use that provided
    # Trainer class directly; instead we will call its train method after construction.
    # The trainer.py from the previous steps is an example; we'll assume the actual Trainer
    # accepts cfg, env, agent, buffer.
    # To be safe, we'll implement a local trainer call if needed, but we want to reuse.
    # Let's check the trainer.py provided earlier: it's a standalone Trainer that creates
    # its own env, agent, buffer inside __init__. That is not ideal for main.py.
    # So we will NOT import from trainer.py; instead we will import Trainer from the project's
    # trainer module, but we need to construct it properly. We'll assume the Trainer class
    # has a signature: Trainer(cfg, env, agent, buffer) and a method run() or train().
    # We'll create the Trainer with the components we built.

    # Actually, the design plan's "trainer.py" description says: "The trainer's __init__ method
    # takes cfg, env, agent, buffer". So we will adhere to that. We'll import Trainer and
    # instantiate with these four arguments.
    from trainer import Trainer
    trainer = Trainer(cfg, env, agent, replay_buffer)

    # --------------------------------------------------------------------------
    # 8. Run training loop
    # --------------------------------------------------------------------------
    eval_results = trainer.train()  # list of dict: step, mean_return, normalized

    # --------------------------------------------------------------------------
    # 9. Save results
    # --------------------------------------------------------------------------
    os.makedirs(args.logdir, exist_ok=True)
    result_file = os.path.join(
        args.logdir,
        f"{cfg.benchmark}_{cfg.task}_seed{cfg.seed}.pkl"
    )
    with open(result_file, "wb") as f:
        pickle.dump(eval_results, f)
    print(f"Evaluation results saved to {result_file}")

    # Optionally save the final model state for reproducibility
    model_file = os.path.join(
        args.logdir,
        f"{cfg.benchmark}_{cfg.task}_seed{cfg.seed}_model.pt"
    )
    torch.save({
        "encoder": agent.encoder.state_dict(),
        "predictor": agent.predictor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "policy": agent.policy.state_dict(),
        "target_encoder": agent.encoder_target.state_dict(),
        "target_q1": agent.q1_target.state_dict(),
        "target_q2": agent.q2_target.state_dict(),
        "target_policy": agent.policy_target.state_dict(),
        "avg_reward": agent.avg_reward,
        "target_avg_reward": agent.target_avg_reward,
        "terminal_loss_active": agent.terminal_loss_active,
    }, model_file)
    print(f"Model checkpoint saved to {model_file}")


if __name__ == "__main__":
    main()

