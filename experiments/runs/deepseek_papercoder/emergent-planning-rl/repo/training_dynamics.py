# training_dynamics.py
"""
Training-dynamics analysis for "Interpreting Emergent Planning in Model-Free RL".

Implements the experiments described in Section 6.2 and Appendices C.1-C.4.
For a sequence of agent checkpoints (every 1M transitions up to 50M), the
analyser:

1. Trains linear probes (1×1) on the final ConvLSTM layer to measure how well
   the agent represents C_A and C_B.
2. Evaluates the extra percentage of medium-difficulty Sokoban levels solved
   when the agent is given 5 forced "thinking steps" (extra test-time compute).
3. Optionally, using the trained probes, quantifies the iterative plan
   refinement during thinking steps (macro F1 improvement from first to last
   internal tick).
4. Produces correlation plots (macro F1 vs extra solved, plan refinement vs
   extra solved) to show that planning-relevant representations and planning-
   like behaviour emerge concurrently.

All parameters are read from the `config.yaml` file via a `Config` object.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project‑internal imports
# ---------------------------------------------------------------------------
from utils import Config, set_seed, load_boxoban_levels
from environment import SokobanEnv
from model import DRCNetwork
from dataset import ProbeDataset, ConceptLabeler
from probes import LinearProbe, ProbeTrainer, compute_metrics


# ===========================================================================
#  TrainingDynamicsAnalyzer
# ===========================================================================
class TrainingDynamicsAnalyzer:
    """
    Orchestrates the training‑dynamics interpretability experiments.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing the saved DRC agent checkpoints
        (e.g., ``checkpoint_step_1000000.pt``).
    config : Config
        Configuration object loaded from ``config.yaml``.
    device : str, optional
        Torch device (default: auto‑detect).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        config: Config,
        device: Optional[str] = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Output base for dynamics results
        self.output_dir = os.path.join(config.output_dir, "dynamics")
        os.makedirs(self.output_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Load Boxoban level strings from disk (used for probe datasets
        # and for extra‑planning gain)
        # ------------------------------------------------------------------
        self.train_levels = load_boxoban_levels(
            config.dataset["training_levels_path"]
        )
        self.valid_levels = load_boxoban_levels(
            config.dataset["validation_levels_path"]
        )

        # Medium / hard levels (the paper uses medium for the extra‑solved metric)
        medium_path = config.dataset.get("medium_levels_path", "")
        hard_path = config.dataset.get("hard_levels_path", "")
        self.medium_levels = (
            load_boxoban_levels(medium_path) if medium_path else []
        )
        self.hard_levels = (
            load_boxoban_levels(hard_path) if hard_path else []
        )

        # ------------------------------------------------------------------
        # Extract dynamics hyperparameters from config
        # ------------------------------------------------------------------
        dyn = config.dynamics
        self.checkpoint_interval = config.training["checkpoint_interval"]
        self.max_checkpoints = dyn["num_checkpoints"]
        self.thinking_steps = dyn["thinking_steps"]
        self.medium_eval_count = dyn.get("medium_eval_levels", 1000)
        self.hard_eval_count = dyn.get("hard_eval_levels", 1000)
        self.probe_train_episodes = dyn["probe_train_episodes"]
        self.probe_test_episodes = dyn["probe_test_episodes"]

        # Probing hyperparameters (will be passed to ProbeTrainer / ProbeDataset)
        probing_cfg = config.probing
        self.probe_epochs = probing_cfg["epochs"]
        self.probe_lr = probing_cfg["learning_rate"]
        self.probe_wd = probing_cfg["weight_decay"]
        self.probe_batch_size = probing_cfg["batch_size"]
        self.probe_seeds = probing_cfg["seeds"]

        # Environment creation parameters
        self.env_kwargs = {
            "level_strings": [],  # will be set per environment
            "max_steps_range": (
                config.env.get("max_steps", 115),
                config.env.get("max_steps", 120),
            ),
            "step_penalty": config.env.get("step_penalty", -0.01),
            "box_on_target_reward": config.env.get("box_on_target_reward", 1.0),
            "box_off_target_reward": config.env.get("box_off_target_reward", -1.0),
            "level_solve_reward": config.env.get("level_solve_reward", 10.0),
            "num_boxes": config.env.get("num_boxes", 4),
            "num_targets": config.env.get("num_targets", 4),
            "seed": config.seed,
        }

        # Agent architecture parameters (needed to instantiate model from checkpoint)
        self.agent_cfg = config.agent

        # Results storage for plotting
        self.results = {
            "steps": [],
            "f1_CA": [],
            "f1_CB": [],
            "extra_pct": [],
            "delta_f1_CA": [],
            "delta_f1_CB": [],
        }

    # ------------------------------------------------------------------
    #  Utilities: checkpoint discovery, model loading, env creation
    # ------------------------------------------------------------------
    def _list_checkpoints(self) -> List[Tuple[int, str]]:
        """
        Scan `checkpoint_dir` for files matching ``checkpoint_*.pt`` and return
        a sorted list of (step, path) tuples.
        """
        if not os.path.isdir(self.checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        ckpts = []
        for fname in os.listdir(self.checkpoint_dir):
            if fname.startswith("checkpoint_") and fname.endswith(".pt"):
                # Extract the step number (the part before .pt after 'checkpoint_')
                # Example: checkpoint_step_1000000.pt -> step=1000000
                try:
                    step_str = fname[len("checkpoint_"):-len(".pt")]
                    # file may have prefix like "step_"
                    if step_str.startswith("step_"):
                        step_str = step_str[5:]
                    step = int(step_str)
                    ckpts.append((step, os.path.join(self.checkpoint_dir, fname)))
                except ValueError:
                    continue

        ckpts.sort(key=lambda x: x[0])
        # Limit to the requested number of checkpoints
        max_step = self.max_checkpoints * self.checkpoint_interval
        ckpts = [(s, p) for s, p in ckpts if s <= max_step]
        if not ckpts:
            raise RuntimeError("No valid checkpoints found in {}".format(self.checkpoint_dir))
        return ckpts

    def _load_model_from_checkpoint(self, ckpt_path: str) -> DRCNetwork:
        """
        Instantiate a fresh DRCNetwork and load its weights from the checkpoint.
        """
        model = DRCNetwork(self.agent_cfg)
        model.to(self.device)
        # The checkpoint may contain a dict with key 'model_state_dict'
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            # Assume the checkpoint is the state dict itself
            state = checkpoint
        model.load_state_dict(state, strict=False)  # strict=False to ignore potential missing heads (value/policy)
        model.eval()
        return model

    def _make_env(self, level_strings: List[str]) -> SokobanEnv:
        """Convenience: create a SokobanEnv with pre‑loaded levels."""
        env = SokobanEnv(level_strings=level_strings, **self.env_kwargs)
        return env

    # ------------------------------------------------------------------
    #  Caching helpers
    # ------------------------------------------------------------------
    def _cache_path_for_checkpoint(self, step: int, sub: str) -> str:
        """Return a cache directory for a specific checkpoint and sub‑type."""
        return os.path.join(self.output_dir, f"checkpoint_{step}", sub)

    # ------------------------------------------------------------------
    #  1.  Probe per checkpoint
    # ------------------------------------------------------------------
    def probe_for_checkpoint(
        self,
        ckpt_path: str,
        num_train_episodes: Optional[int] = None,
        num_test_episodes: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        For a single checkpoint: load the model, generate/load probe datasets
        for the training and test splits of Boxoban unfiltered levels (if not
        cached), train 1×1 linear probes on the *final* ConvLSTM layer for C_A
        and C_B, and return the macro F1 metrics.

        Parameters
        ----------
        ckpt_path : str
            Path to the checkpoint file.
        num_train_episodes : int, optional
            Number of episodes to use for probe training dataset.
            Default: ``dynamics.probe_train_episodes``.
        num_test_episodes : int, optional
            Number of episodes to use for probe testing dataset.
            Default: ``dynamics.probe_test_episodes``.
        force : bool
            If ``True``, regenerate probe datasets even if cached.

        Returns
        -------
        dict
            Dictionary containing:
                - ``step`` : int – training step of the checkpoint.
                - ``macro_f1_CA`` : float – mean macro F1 for C_A over seeds.
                - ``std_CA`` : float – std of macro F1 for C_A.
                - ``macro_f1_CB`` : float – mean macro F1 for C_B.
                - ``std_CB`` : float – std of macro F1 for C_B.
                - ``probe_paths_CA`` : list of str – paths to saved CA probe models.
                - ``probe_paths_CB`` : list of str – paths to saved CB probe models.
        """
        # Extract step number from ckpt_path (needed for caching)
        basename = os.path.splitext(os.path.basename(ckpt_path))[0]
        # typical filename: checkpoint_step_1000000 or similar
        parts = basename.split("_")
        step = int(parts[-1])  # simple extract

        probe_train_eps = num_train_episodes if num_train_episodes is not None else self.probe_train_episodes
        probe_test_eps = num_test_episodes if num_test_episodes is not None else self.probe_test_episodes

        # Determine the final layer index
        final_layer = self.agent_cfg["layers"] - 1   # layers count from 0

        # Build probe config dictionary (used to pass probing hyperparameters)
        probe_cfg = {
            "epochs": self.probe_epochs,
            "learning_rate": self.probe_lr,
            "weight_decay": self.probe_wd,
            "batch_size": self.probe_batch_size,
            "probe_kernel_sizes": [1],  # only 1x1 for dynamics
            "seeds": self.probe_seeds,
            "num_classes": 5,
            "in_channels": self.agent_cfg["channels"],
            # dataset cache dir will be set per checkpoint below
        }

        # Cache paths for this checkpoint
        base_cache = self._cache_path_for_checkpoint(step, "probe_dataset")
        os.makedirs(base_cache, exist_ok=True)

        # ------------------------------------------------------------------
        # 1. Generate / load probe dataset (train & test splits)
        # ------------------------------------------------------------------
        # We'll use ProbeDataset with a custom cache_dir set in probe config.
        # The ProbeDataset constructor uses `config.probing.get('dataset_cache_dir')`.
        # So we'll temporarily override it.
        # Build a shallow config dict for dataset.
        dataset_cfg = dict(self.config.probing)
        dataset_cfg["dataset_cache_dir"] = base_cache  # single directory, splits handled inside

        # Instantiate a ProbeDataset for the final layer?
        # ProbeDataset is used to store all layers, we can reuse it.
        # We'll create two instances: one for train, one for test.
        # But the ProbeDataset.generate stores all data; we can generate both splits sequentially.

        # Load model once for dataset generation.
        model = self._load_model_from_checkpoint(ckpt_path)
        labeler = ConceptLabeler()

        # Helper to get/generate a split dataset
        def _get_split(split_name, levels, num_eps):
            split_cache = os.path.join(base_cache, f"probe_{split_name}.pt")
            if not force and os.path.exists(split_cache):
                # Load existing tensors
                data = torch.load(split_cache, map_location='cpu')
                # Return tensors for the final layer and corresponding labels
                return {
                    "cell_layer": data[f"cell_l{final_layer}"],
                    "label_A": data["label_A"],
                    "label_B": data["label_B"],
                }
            # Generate
            env = self._make_env(levels)
            ds = ProbeDataset(
                model=model,
                env=env,
                levels=levels,
                labeler=labeler,
                config=Config(dataset_cfg) if isinstance(Config, type) else Config(dataset_cfg),  # hack: but we can pass dict; Config class accepts dict.
            )
            # ProbeDataset.generate expects a `split` string; it will create subdirectories.
            # To avoid multiple subdirectories, we'll just call generate and then load.
            ds.generate(num_episodes=num_eps, split=split_name, use_greedy=True)
            # The generate method saves to cache_dir/split/probe_data.pt.
            # We'll copy or directly load from there.
            data = torch.load(os.path.join(base_cache, split_name, "probe_data.pt"), map_location='cpu')
            # Also save a combined file for quicker future loading
            torch.save(data, split_cache)
            return {
                "cell_layer": data[f"cell_l{final_layer}"],
                "label_A": data["label_A"],
                "label_B": data["label_B"],
            }

        train_data = _get_split("train", self.train_levels, probe_train_eps)
        test_data  = _get_split("test", self.valid_levels, probe_test_eps)

        # ------------------------------------------------------------------
        # 2. Train probes for C_A and C_B
        # ------------------------------------------------------------------
        # We'll create a custom ProbeTrainer that works with the loaded tensors
        # directly, because ProbeTrainer expects a ProbeDataset instance.
        # Another approach: create a ProbeDataset instance, set its internal tensors
        # manually (bypassing generate/load). We can subclass or simply directly
        # use the tensors with our own training loop. Simpler: use ProbeTrainer with
        # a custom dataset that yields (cell, label). Since we already have tensors,
        # we can create a torch.utils.data.TensorDataset and directly train.

        # We'll define a local function that trains a probe on a given (x, y) dataset,
        # returns probe and macro F1. This avoids relying on ProbeDataset's interface.

        def train_probe_on_tensors(
            concept: str,
            cell_tensor: torch.Tensor,   # (N, 32, 8, 8)
            label_tensor: torch.Tensor,  # (N, 8, 8)
            seed: int,
        ) -> Tuple[LinearProbe, float]:
            set_seed(seed)
            probe_net = LinearProbe(
                in_channels=32,
                num_classes=5,
                kernel_size=1,
                bias=True,
            ).to(self.device)

            # Split train/test already pre‑computed, so we use provided test_data.
            # We'll train on train_data and eval on test_data.
            # But we need separate train and test cell/labels.
            # For the split, we will pass them as parameters.
            pass  # we'll handle differently: run full training using ProbeTrainer with loaded dataset.

        # Actually, using ProbeTrainer is more reliable if we can create a ProbeDataset
        # with the correct internal tensors. We can manually set the attributes of a
        # ProbeDataset instance (not ideal). Instead, we can implement a minimal
        # wrapper that mimics the `get_dataloader` method.

        # Let's design a clean approach: We'll implement a thin wrapper that holds
        # the cell/label tensors and can produce DataLoaders. This is simpler.

        class _SimpleProbeDataset:
            def __init__(self, cell_layer, label_A, label_B):
                self.cell_layer = cell_layer
                self.label_A = label_A
                self.label_B = label_B

            def get_dataloader(self, layer_idx: int, concept: str, batch_size: int, shuffle: bool, num_workers: int = 0):
                # layer_idx is ignored because we already selected layer
                if concept == 'C_A':
                    y = self.label_A
                else:
                    y = self.label_B
                dataset = torch.utils.data.TensorDataset(self.cell_layer, y)
                return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        train_split = _SimpleProbeDataset(train_data["cell_layer"], train_data["label_A"], train_data["label_B"])
        test_split = _SimpleProbeDataset(test_data["cell_layer"], test_data["label_A"], test_data["label_B"])

        # Instantiate ProbeTrainer with a custom config
        class _ProbeConfig:
            def __init__(self, d):
                self.epochs = d["epochs"]
                self.learning_rate = d["learning_rate"]
                self.weight_decay = d["weight_decay"]
                self.batch_size = d["batch_size"]
                self.probe_kernel_sizes = d["probe_kernel_sizes"]
                self.seeds = d["seeds"]
                self.num_classes = d["num_classes"]

        probe_cfg_obj = _ProbeConfig(probe_cfg)

        # For each concept, train probes.
        all_results = {}
        for concept in ("C_A", "C_B"):
            # Since ProbeTrainer needs a single dataset that provides both train and test,
            # we need to adapt. The ProbeTrainer in probes.py expects to call `dataset.load('train')`
            # and `dataset.load('test')`. Our _SimpleProbeDataset doesn't have that.
            # The easiest way: we modify ProbeTrainer or create a new one.
            # To avoid rewriting, we will subclass ProbeTrainer to accept split datasets.

            # Instead, we can use the ProbeTrainer's `train_single_probe` method if we already
            # have a DataLoader. But `train_single_probe` calls `_get_dataloader('train')`.
            # So we need to adapt. Maybe it's simplest to just implement our own training loop
            # here.

        # Given the complexity, I'll implement a self-contained training routine.
        # This keeps the module independent.

        def train_and_eval_probe(
            concept: str,
            train_cell: torch.Tensor,
            train_label: torch.Tensor,
            test_cell: torch.Tensor,
            test_label: torch.Tensor,
            seed: int,
        ) -> Tuple[LinearProbe, Dict]:
            set_seed(seed)
            probe = LinearProbe(
                in_channels=32,
                num_classes=5,
                kernel_size=1,
                bias=True,
            ).to(self.device)
            optimizer = torch.optim.AdamW(probe.parameters(), lr=probe_cfg["learning_rate"], weight_decay=probe_cfg["weight_decay"])
            loss_fn = torch.nn.CrossEntropyLoss()

            # Train
            probe.train()
            for epoch in range(probe_cfg["epochs"]):
                indices = torch.randperm(train_cell.size(0))
                for i in range(0, train_cell.size(0), probe_cfg["batch_size"]):
                    batch_idx = indices[i:i+probe_cfg["batch_size"]]
                    x = train_cell[batch_idx].to(self.device)
                    y = train_label[batch_idx].to(self.device)
                    logits = probe(x)  # (B,5,8,8)
                    B, C, H, W = logits.shape
                    logits_flat = logits.permute(0,2,3,1).reshape(-1, C)
                    y_flat = y.reshape(-1)
                    loss = loss_fn(logits_flat, y_flat)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Evaluate
            probe.eval()
            preds_all = []
            labels_all = []
            with torch.no_grad():
                for i in range(0, test_cell.size(0), probe_cfg["batch_size"]):
                    x = test_cell[i:i+probe_cfg["batch_size"]].to(self.device)
                    y = test_label[i:i+probe_cfg["batch_size"]].to(self.device)
                    logits = probe(x)
                    preds_all.append(logits.argmax(dim=1).cpu().numpy())
                    labels_all.append(y.cpu().numpy())
            y_pred = np.concatenate(preds_all).flatten()
            y_true = np.concatenate(labels_all).flatten()
            metrics = compute_metrics(y_true, y_pred, num_classes=5)
            return probe, metrics

        # Train for each seed and concept
        f1_list_CA = []
        f1_list_CB = []
        probe_paths_CA = []
        probe_paths_CB = []
        probe_save_dir = os.path.join(self.output_dir, f"checkpoint_{step}", "probes")
        os.makedirs(probe_save_dir, exist_ok=True)

        for seed in range(self.probe_seeds):
            # C_A
            probe_CA, metrics_CA = train_and_eval_probe(
                "C_A",
                train_data["cell_layer"],
                train_data["label_A"],
                test_data["cell_layer"],
                test_data["label_A"],
                seed,
            )
            f1_list_CA.append(metrics_CA["macro_f1"])
            # Save probe
            ca_path = os.path.join(probe_save_dir, f"probe_CA_seed{seed}.pt")
            torch.save(probe_CA.state_dict(), ca_path)
            probe_paths_CA.append(ca_path)

            # C_B
            probe_CB, metrics_CB = train_and_eval_probe(
                "C_B",
                train_data["cell_layer"],
                train_data["label_B"],
                test_data["cell_layer"],
                test_data["label_B"],
                seed,
            )
            f1_list_CB.append(metrics_CB["macro_f1"])
            cb_path = os.path.join(probe_save_dir, f"probe_CB_seed{seed}.pt")
            torch.save(probe_CB.state_dict(), cb_path)
            probe_paths_CB.append(cb_path)

        # Aggregate
        res = {
            "step": step,
            "macro_f1_CA": np.mean(f1_list_CA),
            "std_CA": np.std(f1_list_CA),
            "macro_f1_CB": np.mean(f1_list_CB),
            "std_CB": np.std(f1_list_CB),
            "probe_paths_CA": probe_paths_CA,
            "probe_paths_CB": probe_paths_CB,
        }
        return res

    # ------------------------------------------------------------------
    #  2.  Extra planning gain
    # ------------------------------------------------------------------
    def extra_planning_gain(
        self,
        ckpt_path: str,
        thinking_steps: int = 5,
        num_levels: Optional[int] = None,
        level_type: str = "medium",
        force: bool = False,
    ) -> float:
        """
        Measure the percentage of levels (from a predefined set) that are solved
        **only** when the agent is given ``thinking_steps`` forced stationary
        steps at the beginning of the episode.

        Parameters
        ----------
        ckpt_path : str
            Path to checkpoint.
        thinking_steps : int
            Number of extra stationary steps (default 5).
        num_levels : int, optional
            Number of levels to evaluate (default from config).
        level_type : str
            ``"medium"`` or ``"hard"`` (default medium).
        force : bool
            If ``True``, recompute even if cached result exists.

        Returns
        -------
        float
            Extra solved percentage (0‑100).
        """
        # Use caching to avoid recomputation
        basename = os.path.splitext(os.path.basename(ckpt_path))[0]
        step = int(basename.split("_")[-1])
        cache_dir = self._cache_path_for_checkpoint(step, "extra_gain")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"extra_solve_{level_type}.json")
        if not force and os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
                return data["extra_pct"]

        # Model loading
        model = self._load_model_from_checkpoint(ckpt_path)

        # Choose levels
        if level_type == "medium":
            levels = self.medium_levels
            max_eval = self.medium_eval_count
        else:
            levels = self.hard_levels
            max_eval = self.hard_eval_count
        num_lvl = num_levels if num_levels is not None else max_eval
        num_lvl = min(num_lvl, len(levels))
        eval_levels = levels[:num_lvl]

        # Helper: run one episode with optional thinking
        def _run_episode(env, model, use_thinking: bool):
            obs = env.reset()
            state = model.initial_state(batch_size=1)
            if use_thinking:
                # forced stationary steps
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                for _ in range(thinking_steps):
                    # forward pass; we ignore action, just update state
                    logits, value, new_state = model(obs_tensor, state, num_ticks=model.internal_ticks)
                    state = new_state
            # Now act greedily until done
            while not env.done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits, value, new_state = model(obs_tensor, state, num_ticks=model.internal_ticks)
                action = logits.argmax(dim=-1).item()
                next_obs, reward, done, _ = env.step(action)
                obs = next_obs
                state = new_state
            # Determine if solved
            return env.grid.sum() > 0 and np.sum(env.grid == 4) == env.num_boxes  # 4 = BOX_ON_TARGET

        # Evaluate baseline (no thinking)
        env_base = self._make_env(eval_levels)
        solved_base = 0
        env_base.reset()  # reset to first level
        for lvl_idx, lvl_str in enumerate(eval_levels):
            env_base.set_level(lvl_str)
            if _run_episode(env_base, model, use_thinking=False):
                solved_base += 1
        solved_base_ct = solved_base

        # Evaluate with thinking
        env_think = self._make_env(eval_levels)
        solved_with = 0
        for lvl_idx, lvl_str in enumerate(eval_levels):
            env_think.set_level(lvl_str)
            if _run_episode(env_think, model, use_thinking=True):
                solved_with += 1

        extra_solved = solved_with - solved_base
        pct = (extra_solved / num_lvl) * 100.0

        # Cache result
        with open(cache_file, "w") as f:
            json.dump({"step": step, "solved_base": solved_base, "solved_with": solved_with, "extra_pct": pct}, f)

        return pct

    # ------------------------------------------------------------------
    #  3.  Plan refinement metrics (iterative plan improvement)
    # ------------------------------------------------------------------
    def plan_refinement_metrics(
        self,
        ckpt_path: str,
        probes_CA: LinearProbe,
        probes_CB: LinearProbe,
        thinking_steps: int = 5,
        num_levels: Optional[int] = None,
        level_type: str = "medium",
        force: bool = False,
    ) -> Dict[str, float]:
        """
        Quantify the agent's ability to iteratively refine its internal plan
        during extra thinking steps.  The plan is decoded using the supplied
        linear probes, and macro F1 is computed for the plan at the first vs.
        the last internal tick of the stationary period.

        Parameters
        ----------
        ckpt_path : str
        probes_CA : LinearProbe
            Trained 1×1 probe for C_A (on the final layer).
        probes_CB : LinearProbe
            Trained 1×1 probe for C_B (on the final layer).
        thinking_steps : int
            Number of forced stationary steps (default 5, i.e. 15 ticks).
        num_levels : int, optional
            Number of levels to evaluate (default from config).
        level_type : str
            ``"medium"`` (default) or ``"hard"``.
        force : bool
            If True, recompute even if cached.

        Returns
        -------
        dict
            ``{'delta_f1_CA': float, 'delta_f1_CB': float,
               'f1_tick1_CA': float, 'f1_tick1_CB': float,
               'f1_tick15_CA': float, 'f1_tick15_CB': float}``
        """
        basename = os.path.splitext(os.path.basename(ckpt_path))[0]
        step = int(basename.split("_")[-1])
        cache_dir = self._cache_path_for_checkpoint(step, "plan_refinement")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"plan_ref_{level_type}.json")
        if not force and os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)

        model = self._load_model_from_checkpoint(ckpt_path)
        num_ticks = model.internal_ticks

        # Choose levels
        if level_type == "medium":
            levels = self.medium_levels
            max_eval = self.medium_eval_count
        else:
            levels = self.hard_levels
            max_eval = self.hard_eval_count
        num_lvl = num_levels if num_levels is not None else max_eval
        num_lvl = min(num_lvl, len(levels))
        eval_levels = levels[:num_lvl]

        # We will collect episode predictions (per level) for both ticks.
        # We need to run each level once with thinking steps, then label the episode.
        # We'll accumulate per-level F1s for tick 1 and tick 15.
        f1_ca_tick1 = []
        f1_ca_tick15 = []
        f1_cb_tick1 = []
        f1_cb_tick15 = []

        labeler = ConceptLabeler()

        for lvl_str in eval_levels:
            env = self._make_env([lvl_str])
            obs = env.reset()
            state = model.initial_state(batch_size=1)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Record cell states for each tick during thinking steps
            cell_states_ticks = []  # list of (layer_idx) cell state tensor (32,8,8)
            for think_step in range(thinking_steps):
                # full forward with all cell states per tick
                logits, value, new_state, cell_states_per_tick = model.forward_with_all_cell_states(
                    obs_tensor, state, num_ticks=num_ticks
                )
                # cell_states_per_tick is list of length num_ticks, each a list of cell states per layer
                for tick_idx in range(num_ticks):
                    # we care about the final layer (index = model.layers - 1)
                    layer_idx = model.layers - 1
                    c_tensor = cell_states_per_tick[tick_idx][layer_idx].squeeze(0)  # (32,8,8)
                    cell_states_ticks.append(c_tensor)
                state = new_state

            # After thinking steps, continue acting while recording trajectory for labelling
            trajectory = []  # list of dicts: action, agent_pos, push_event
            while not env.done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits, value, new_state = model(obs_tensor, state, num_ticks=num_ticks)
                action = logits.argmax(dim=-1).item()
                prev_agent_pos = env.agent_pos
                # capture before step to compute push event
                # We'll rely on the environment's internal `last_push_event` attribute, added earlier.
                next_obs, reward, done, info = env.step(action)
                # Record
                push_event = env.last_push_event if hasattr(env, 'last_push_event') else None
                trajectory.append({
                    'action': action,
                    'agent_pos': env.agent_pos,
                    'push_event': push_event,
                })
                obs = next_obs
                state = new_state

            # Label the initial state using the whole trajectory
            labels_A, labels_B = labeler.label_episode(trajectory)  # shape (T,8,8); we need step 0
            label_A_initial = labels_A[0]  # (8,8) class indices
            label_B_initial = labels_B[0]

            # Evaluate plan at first tick and last tick
            first_tick_cell = cell_states_ticks[0].unsqueeze(0)  # (1,32,8,8)
            last_tick_cell  = cell_states_ticks[-1].unsqueeze(0)

            # Move probes to correct device
            probes_CA.to(self.device)
            probes_CB.to(self.device)

            def eval_one(cell, probe, label):
                with torch.no_grad():
                    logits = probe(cell)   # (1,5,8,8)
                    preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (8,8)
                # compute macro F1 using compute_metrics
                met = compute_metrics(label.flatten(), preds.flatten(), num_classes=5)
                return met['macro_f1']

            # Need to ensure compute_metrics is imported correctly.
            # We'll import it at top.

            # Evaluate
            f1t1_ca = eval_one(first_tick_cell, probes_CA, label_A_initial)
            f1t15_ca = eval_one(last_tick_cell, probes_CA, label_A_initial)
            f1t1_cb = eval_one(first_tick_cell, probes_CB, label_B_initial)
            f1t15_cb = eval_one(last_tick_cell, probes_CB, label_B_initial)

            f1_ca_tick1.append(f1t1_ca)
            f1_ca_tick15.append(f1t15_ca)
            f1_cb_tick1.append(f1t1_cb)
            f1_cb_tick15.append(f1t15_cb)

        # Average across levels
        avg_f1_ca_t1 = float(np.mean(f1_ca_tick1))
        avg_f1_ca_t15 = float(np.mean(f1_ca_tick15))
        avg_f1_cb_t1 = float(np.mean(f1_cb_tick1))
        avg_f1_cb_t15 = float(np.mean(f1_cb_tick15))
        delta_ca = avg_f1_ca_t15 - avg_f1_ca_t1
        delta_cb = avg_f1_cb_t15 - avg_f1_cb_t1

        result = {
            "delta_f1_CA": delta_ca,
            "delta_f1_CB": delta_cb,
            "f1_tick1_CA": avg_f1_ca_t1,
            "f1_tick1_CB": avg_f1_cb_t1,
            "f1_tick15_CA": avg_f1_ca_t15,
            "f1_tick15_CB": avg_f1_cb_t15,
        }

        with open(cache_file, "w") as f:
            json.dump(result, f)
        return result

    # ------------------------------------------------------------------
    #  4.  Full analysis across checkpoints
    # ------------------------------------------------------------------
    def run_full_analysis(self, force_recompute: bool = False) -> Dict:
        """
        Iterates over all available checkpoints (1M to max), collects
        the metrics, and generates the correlation plots described in
        the paper (Figure 9, 35, 39).  Results are saved to the output
        directory and also returned as a dictionary.

        Parameters
        ----------
        force_recompute : bool
            If True, ignore cached metrics and recompute everything.

        Returns
        -------
        dict
            Collected results for all checkpoints.
        """
        ckpts = self._list_checkpoints()
        print(f"Found {len(ckpts)} checkpoints to analyse.")

        # Re-init results if re-running
        self.results = {k: [] for k in self.results}

        # Progress bar
        for step, ckpt_path in tqdm(ckpts, desc="Checkpoints"):
            # 1. Probe metrics
            probe_res = self.probe_for_checkpoint(
                ckpt_path, force=force_recompute
            )
            self.results["steps"].append(step)
            self.results["f1_CA"].append(probe_res["macro_f1_CA"])
            self.results["f1_CB"].append(probe_res["macro_f1_CB"])

            # 2. Extra planning gain
            extra_pct = self.extra_planning_gain(
                ckpt_path, thinking_steps=self.thinking_steps,
                level_type="medium", force=force_recompute
            )
            self.results["extra_pct"].append(extra_pct)

            # 3. Plan refinement (requires trained probes; load from probe_res paths)
            # We'll load one probe per concept (seed 0) for simplicity.
            ca_path = probe_res["probe_paths_CA"][0]
            cb_path = probe_res["probe_paths_CB"][0]
            probe_CA = LinearProbe(32, 5, 1, bias=True)
            probe_CA.load_state_dict(torch.load(ca_path, map_location=self.device))
            probe_CA.to(self.device)
            probe_CB = LinearProbe(32, 5, 1, bias=True)
            probe_CB.load_state_dict(torch.load(cb_path, map_location=self.device))
            probe_CB.to(self.device)

            plan_ref = self.plan_refinement_metrics(
                ckpt_path, probe_CA, probe_CB,
                thinking_steps=self.thinking_steps,
                level_type="medium", force=force_recompute
            )
            self.results["delta_f1_CA"].append(plan_ref["delta_f1_CA"])
            self.results["delta_f1_CB"].append(plan_ref["delta_f1_CB"])

        # Generate and save plots
        self._save_plots()

        # Save aggregated CSV for convenience
        import csv
        csv_path = os.path.join(self.output_dir, "dynamics_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "f1_CA", "f1_CB", "extra_pct", "delta_f1_CA", "delta_f1_CB"])
            for i in range(len(self.results["steps"])):
                writer.writerow([
                    self.results["steps"][i],
                    self.results["f1_CA"][i],
                    self.results["f1_CB"][i],
                    self.results["extra_pct"][i],
                    self.results["delta_f1_CA"][i],
                    self.results["delta_f1_CB"][i],
                ])

        return self.results

    def _save_plots(self) -> None:
        """Internal helper to produce and save all required plots."""
        steps = np.array(self.results["steps"])
        f1_CA = np.array(self.results["f1_CA"])
        f1_CB = np.array(self.results["f1_CB"])
        extra_pct = np.array(self.results["extra_pct"])
        delta_CA = np.array(self.results["delta_f1_CA"])
        delta_CB = np.array(self.results["delta_f1_CB"])

        # Figure 9: extra_pct vs macro F1 for C_A and C_B
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(f1_CA, extra_pct, c='blue', label='C_A')
        ax.scatter(f1_CB, extra_pct, c='orange', label='C_B')
        ax.set_xlabel('Macro F1 (final layer)')
        ax.set_ylabel('Extra levels solved (%)')
        ax.set_title('Figure 9: Emergence of planning ability')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "figure9.png"))
        plt.close(fig)

        # Figure 35: concept representation emergence over training steps
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps / 1e6, f1_CA, marker='o', label='C_A')
        ax.plot(steps / 1e6, f1_CB, marker='s', label='C_B')
        ax.set_xlabel('Training steps (millions)')
        ax.set_ylabel('Macro F1 (final layer)')
        ax.set_title('Figure 35: Concept representation during training')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "figure35.png"))
        plt.close(fig)

        # Figure 39 (optional plan refinement vs extra solved)
        if len(delta_CA) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(extra_pct, delta_CA, c='blue', label='C_A')
            ax.scatter(extra_pct, delta_CB, c='orange', label='C_B')
            ax.set_xlabel('Extra levels solved (%)')
            ax.set_ylabel('Δ Macro F1 (tick15 - tick1)')
            ax.set_title('Figure 39: Plan refinement co‑emergence')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(self.output_dir, "figure39.png"))
            plt.close(fig)

        print(f"Plots saved to {self.output_dir}")


# ---------------------------------------------------------------------------
#  Minimal self‑test (run if executed as main)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("TrainingDynamicsAnalyzer self-test not implemented.")
    print("Usage: configure config.yaml and run main.py with --mode dynamics")
