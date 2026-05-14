```python
## main.py
"""Main orchestrator for reproducing experiments from the adversarial leaderboard paper.

This module provides the CLI entry point and the Main class that wires together
all experimental components:
  1. De-anonymization experiments (Section 2): identity-probing and training-based
     detectors, Tables 2, 3, 7, Figures 2, 3.
  2. Leaderboard manipulation simulation (Section 3): Bradley-Terry model,
     adversarial voting, Tables 4, 5, 8, 9.
  3. Mitigation experiments (Section 4): cost model, malicious user detection,
     perturbed leaderboard, Figures 4, 5, 6.

Usage:
    python main.py --config config.yaml all
    python main.py --config config.yaml collect
    python main.py --config config.yaml deanonymize
    python main.py --config config.yaml simulate
    python main.py --config config.yaml mitigate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Internal project imports — ordered by dependency graph to avoid circular
# imports. config.py has no internal deps and must be imported first.
# ---------------------------------------------------------------------------
from config import Config, ModelConfig
from utils.logger import get_logger
from utils.cache import Cache
from data_structures import ResponseDataset, SimulationResult
from api_client import APIClient
from dataset_loader import DatasetLoader
from data_collector import DataCollector
from deanonymization.identity_probing import IdentityProbingDetector, IDENTITY_PROMPTS
from deanonymization.training_based import TrainingBasedDetector
from simulation.voting_data_loader import VotingDataLoader
from simulation.bradley_terry import BradleyTerryModel
from simulation.attack_simulator import AttackSimulator
from mitigations.cost_model import CostModel
from mitigations.malicious_user_detector import MaliciousUserDetector
from mitigations.perturbed_leaderboard import PerturbedLeaderboard
from evaluation.metrics import Metrics
from evaluation.visualizer import Visualizer

# ---------------------------------------------------------------------------
# Module-level logger — initialized before Main so that early errors are logged.
# ---------------------------------------------------------------------------
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# File paths for persisted intermediate results.
# These are relative to config.output_dir and match the Shared Knowledge spec.
# ---------------------------------------------------------------------------
_IDENTITY_RESPONSES_FILE: str = "responses/identity_responses.json"
_TRAINING_RESPONSES_FILE: str = "responses/training_responses.json"
_SIMULATION_RESULTS_FILE: str = "simulation/simulation_results.json"

# ---------------------------------------------------------------------------
# Table 2 model subset (7 models shown in the paper's Table 2).
# ---------------------------------------------------------------------------
_TABLE2_MODELS: List[str] = [
    "claude-3-5-sonnet-20240620",
    "gemini-1.5-pro",
    "gpt-4o-mini-2024-07-18",
    "gemma-2-27b-it",
    "llama-3.1-70b-instruct",
    "mixtral-8x7b-instruct-v0.1",
    "qwen2-72b-instruct",
]

# ---------------------------------------------------------------------------
# Table 3 model subset (same 7 models as Table 2).
# ---------------------------------------------------------------------------
_TABLE3_MODELS: List[str] = _TABLE2_MODELS


class Main:
    """Top-level orchestrator for all paper reproduction experiments.

    Initializes all dependencies (API clients, cache, loaders, visualizer)
    and provides four run methods corresponding to the paper's experimental
    sections. Maintains shared state (bt_model, votes_df, simulation_results)
    as instance attributes so that run_mitigations() can access results from
    run_simulation() without re-computation.

    Attributes:
        config: The global Config object loaded from config.yaml.
        clients: Dict mapping provider name to initialized APIClient instance.
        cache: Disk-based Cache for persisting API responses.
        dataset_loader: DatasetLoader for prompt sampling from source datasets.
        data_collector: DataCollector for orchestrating API calls.
        visualizer: Visualizer for rendering figures and saving CSV tables.
        bt_model: Fitted BradleyTerryModel. Set by run_simulation(), used by
            run_mitigations(). None until run_simulation() is called.
        votes_df: Voting records DataFrame from VotingDataLoader. Set by
            run_simulation(), used by run_mitigations(). None until
            run_simulation() is called.
        simulation_results: List of SimulationResult objects from the main
            simulation experiments. Set by run_simulation(), used by
            run_mitigations(). None until run_simulation() is called.

    Example:
        >>> main = Main(config_path="config.yaml")
        >>> main.run_all()
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize the Main orchestrator.

        Loads configuration, creates output directories, initializes API clients
        from environment variables, and constructs all stateless dependencies.
        No API calls or heavy computation are performed at init time.

        Args:
            config_path: Path to the YAML configuration file. Defaults to
                "config.yaml" in the current working directory.

        Raises:
            FileNotFoundError: If config_path does not exist.
            ValueError: If the configuration file contains invalid values.
            SystemExit: If a fatal initialization error occurs (logged before exit).
        """
        logger.info("=" * 70)
        logger.info("Initializing Main orchestrator from config: '%s'", config_path)
        logger.info("=" * 70)

        # --- Load configuration ---
        try:
            self.config: Config = Config.from_yaml(config_path)
        except FileNotFoundError as exc:
            logger.error("Configuration file not found: %s", exc)
            sys.exit(1)
        except (ValueError, KeyError) as exc:
            logger.error("Configuration error: %s", exc)
            sys.exit(1)

        logger.info(
            "Config loaded: %d models, %d prompt categories, "
            "output_dir='%s', cache_dir='%s'.",
            len(self.config.models),
            len(self.config.prompt_categories),
            self.config.output_dir,
            self.config.cache_dir,
        )

        # --- Ensure output directory structure exists ---
        # Config.__post_init__ already creates these, but we re-create here
        # to be explicit and defensive.
        for subdir in ("tables", "figures", "responses", "simulation"):
            Path(self.config.output_dir, subdir).mkdir(parents=True, exist_ok=True)

        # --- Initialize API clients from environment variables ---
        # Read key_env_vars from config.yaml api section.
        api_cfg: Dict[str, Any] = self.config.raw.get("api", {})
        key_env_vars: Dict[str, str] = api_cfg.get(
            "key_env_vars",
            {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
                "together": "TOGETHER_API_KEY",
            },
        )

        self.clients: Dict[str, APIClient] = {}
        for provider, env_var in key_env_vars.items():
            api_key: str = os.environ.get(env_var, "")
            if not api_key:
                logger.warning(
                    "API key for provider '%s' not found in environment "
                    "variable '%s'. Models using this provider will not be "
                    "queryable.",
                    provider,
                    env_var,
                )
                continue
            try:
                self.clients[provider] = APIClient(provider=provider, api_key=api_key)
                logger.info(
                    "Initialized APIClient for provider '%s'.", provider
                )
            except ValueError as exc:
                logger.warning(
                    "Failed to initialize APIClient for provider '%s': %s. "
                    "Skipping.",
                    provider,
                    exc,
                )

        logger.info(
            "Initialized %d/%d API clients: %s.",
            len(self.clients),
            len(key_env_vars),
            sorted(self.clients.keys()),
        )

        # --- Initialize stateless dependencies ---
        self.cache: Cache = Cache(cache_dir=self.config.cache_dir)
        self.dataset_loader: DatasetLoader = DatasetLoader(config=self.config)
        self.data_collector: DataCollector = DataCollector(
            config=self.config,
            clients=self.clients,
            cache=self.cache,
        )
        self.visualizer: Visualizer = Visualizer(output_dir=self.config.output_dir)

        # --- Shared state set by run_simulation(), used by run_mitigations() ---
        self.bt_model: Optional[BradleyTerryModel] = None
        self.votes_df = None  # pd.DataFrame, typed loosely to avoid pandas import at top
        self.simulation_results: Optional[List[SimulationResult]] = None

        logger.info("Main orchestrator initialized successfully.")

    # -----------------------------------------------------------------------
    # Private helper methods
    # -----------------------------------------------------------------------

    def _output_path(self, *parts: str) -> str:
        """Construct a full output path under config.output_dir.

        Args:
            *parts: Path components to join under config.output_dir.

        Returns:
            Full path string, e.g. "outputs/tables/table2.csv".

        Example:
            >>> self._output_path("tables", "table2.csv")
            'outputs/tables/table2.csv'
        """
        return os.path.join(self.config.output_dir, *parts)

    def _load_or_collect_identity_responses(
        self,
    ) -> Dict[str, Dict[str, List[str]]]:
        """Load identity-probing responses from disk or collect via API.

        Checks for a cached JSON file at outputs/responses/identity_responses.json.
        If found, loads and returns it. Otherwise, calls
        DataCollector.collect_all_identity_responses() and saves the result.

        Returns:
            Nested dict: model_name -> prompt_string -> List[str] of responses.
        """
        save_path: str = self._output_path(_IDENTITY_RESPONSES_FILE)

        if os.path.exists(save_path):
            logger.info(
                "Loading cached identity-probing responses from '%s'.", save_path
            )
            try:
                with open(save_path, "r", encoding="utf-8") as fh:
                    identity_responses: Dict[str, Dict[str, List[str]]] = json.load(fh)
                n_models: int = len(identity_responses)
                logger.info(
                    "Loaded identity responses for %d models from cache.", n_models
                )
                return identity_responses
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed to load cached identity responses from '%s': %s. "
                    "Re-collecting via API.",
                    save_path,
                    exc,
                )

        # Collect via API.
        logger.info(
            "Collecting identity-probing responses for %d models × %d prompts × "
            "%d queries each. This may take a while.",
            len(self.config.models),
            len(IDENTITY_PROMPTS),
            self.config.n_identity_queries,
        )
        t_start: float = time.time()
        identity_responses = self.data_collector.collect_all_identity_responses()
        elapsed: float = time.time() - t_start
        logger.info(
            "Identity-probing collection complete in %.1f seconds.", elapsed
        )

        # Save to disk for future runs.
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            with open(save_path, "w", encoding="utf-8") as fh:
                json.dump(identity_responses, fh, indent=2, ensure_ascii=False)
            logger.info(
                "Saved identity-probing responses to '%s'.", save_path
            )
        except OSError as exc:
            logger.warning(
                "Failed to save identity responses to '%s': %s. "
                "Continuing without saving.",
                save_path,
                exc,
            )

        return identity_responses

    def _load_or_collect_training_responses(self) -> ResponseDataset:
        """Load training-based detector responses from disk or collect via API.

        Checks for a cached JSON file at outputs/responses/training_responses.json.
        If found, loads and returns it. Otherwise, loads prompts from source
        datasets and collects responses via API.

        Returns:
            ResponseDataset containing all collected model responses.
        """
        save_path: str = self._output_path(_TRAINING_RESPONSES_FILE)

        if os.path.exists(save_path):
            logger.info(
                "Loading cached training responses from '%s'.", save_path
            )
            try:
                dataset: ResponseDataset = ResponseDataset.load(save_path)
                if len(dataset) > 0:
                    logger.info(
                        "Loaded ResponseDataset with %d (category, prompt, model) "
                        "entries from cache.",
                        len(dataset),
                    )
                    return dataset
                else:
                    logger.warning(
                        "Cached ResponseDataset at '%s' is empty. Re-collecting.",
                        save_path,
                    )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed to load cached training responses from '%s': %s. "
                    "Re-collecting via API.",
                    save_path,
                    exc,
                )

        # Load prompts from source datasets.
        logger.info("Loading prompts from source datasets...")
        t_start = time.time()
        prompts_by_category: Dict[str, List[str]] = (
            self.dataset_loader.load_all_categories()
        )
        elapsed = time.time() - t_start
        logger.info(
            "Loaded prompts in %.1f seconds: %s.",
            elapsed,
            {cat: len(prompts) for cat, prompts in prompts_by_category.items()},
        )

        # Estimate API cost before collecting.
        cost_cfg: Dict[str, Any] = (
            self.config.raw.get("mitigations", {})
            .get("cost_model", {})
        )
        c_per_prompt: float = float(cost_cfg.get("c_per_prompt", 2.2))
        total_prompts: int = sum(len(p) for p in prompts_by_category.values())
        estimated_cost: float = total_prompts * c_per_prompt
        logger.info(
            "Estimated API cost for training response collection: "
            "$%.2f (%d total prompts × $%.2f/prompt).",
            estimated_cost,
            total_prompts,
            c_per_prompt,
        )

        # Collect responses via API.
        logger.info(
            "Collecting training-based detector responses for %d models × "
            "%d categories × %d prompts × %d responses each.",
            len(self.config.models),
            len(prompts_by_category),
            self.config.n_prompts_per_category,
            self.config.n_responses_per_model,
        )
        t_start = time.time()
        dataset = self.data_collector.collect_all_responses(prompts_by_category)
        elapsed = time.time() - t_start
        logger.info(
            "Training response collection complete in %.1f seconds. "
            "ResponseDataset has %d entries.",
            elapsed,
            len(dataset),
        )

        # Save to disk.
        try:
            dataset.save(save_path)
            logger.info("Saved ResponseDataset to '%s'.", save_path)
        except OSError as exc:
            logger.warning(
                "Failed to save ResponseDataset to '%s': %s. "
                "Continuing without saving.",
                save_path,
                exc,
            )

        return dataset

    def _initialize_bt_model(self) -> tuple:
        """Initialize the BradleyTerryModel from voting data.

        Loads voting data from the configured path (or HuggingFace fallback),
        fits the BT model, and returns both the model and the votes DataFrame.

        Returns:
            Tuple of (BradleyTerryModel, pd.DataFrame) where the DataFrame
            has columns ['model_a', 'model_b', 'winner'].

        Raises:
            SystemExit: If voting data cannot be loaded from any source.
        """
        sim_cfg: Dict[str, Any] = self.config.raw.get("simulation", {})
        voting_data_path: str = str(
            sim_cfg.get("voting_data_path", "data/chatbot_arena_votes.json")
        )
        hf_fallback: str = str(
            sim_cfg.get(
                "voting_data_hf_fallback", "lmsys/chatbot_arena_conversations"
            )
        )

        # Try local file first, then HuggingFace fallback.
        if not os.path.exists(voting_data_path):
            logger.warning(
                "Local voting data file not found at '%s'. "
                "Falling back to HuggingFace dataset '%s'. "
                "Note: this may not match the paper's 1.67M vote dataset.",
                voting_data_path,
                hf_fallback,
            )
            effective_path: str = hf_fallback
        else:
            effective_path = voting_data_path

        logger.info("Loading voting data from '%s'.", effective_path)
        t_start: float = time.time()
        try:
            voting_loader: VotingDataLoader = VotingDataLoader(
                data_path=effective_path
            )
        except (FileNotFoundError, ValueError, ImportError) as exc:
            logger.error(
                "Failed to load voting data from '%s': %s. "
                "Cannot run simulation without voting data.",
                effective_path,
                exc,
            )
            sys.exit(1)

        votes_df = voting_loader.load_votes()
        elapsed: float = time.time() - t_start
        logger.info(
            "Loaded %d voting records in %.1f seconds. "
            "%d unique models, %d unique pairs.",
            len(votes_df),
            elapsed,
            len(voting_loader.get_model_list()),
            len(voting_loader.get_pair_counts()),
        )

        # Fit Bradley-Terry model.
        logger.info("Fitting Bradley-Terry model...")
        t_start = time.time()
        win_matrix, model_names = voting_loader.get_win_matrix()
        bt_scale_factor: float = float(
            sim_cfg.get("bt_scale_factor", 1.0)
        )
        bt_model: BradleyTerryModel = BradleyTerryModel(
            scale_factor=bt_scale_factor
        )
        try:
            bt_model.fit(win_matrix, model_names)
        except (ValueError, RuntimeError) as exc:
            logger.error("Failed to fit Bradley-Terry model: %s", exc)
            sys.exit(1)

        elapsed = time.time() - t_start
        logger.info(
            "Bradley-Terry model fitted in %.1f seconds. "
            "%d models ranked.",
            elapsed,
            len(bt_model.ratings),
        )

        # Log top-10 ranked models for verification.
        top10: List[tuple] = bt_model.get_rankings()[:10]
        logger.info(
            "Top-10 ranked models: %s",
            [(name, f"{rating:.4f}") for name, rating in top10],
        )

        return bt_model, votes_df, voting_loader

    def _save_simulation_results(
        self, results: List[SimulationResult], filename: str
    ) -> None:
        """Serialize a list of SimulationResult objects to a JSON file.

        Args:
            results: List of SimulationResult objects to save.
            filename: Filename (not full path) within the simulation/ subdirectory.
        """
        save_path: str = self._output_path("simulation", filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            with open(save_path, "w", encoding="utf-8") as fh:
                json.dump(
                    [r.to_dict() for r in results],
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )
            logger.info(
                "Saved %d simulation results to '%s'.", len(results), save_path
            )
        except OSError as exc:
            logger.warning(
                "Failed to save simulation results to '%s': %s.",
                save_path,
                exc,
            )

    def _load_simulation_results(
        self, filename: str
    ) -> Optional[List[SimulationResult]]:
        """Load SimulationResult objects from a JSON file.

        Args:
            filename: Filename within the simulation/ subdirectory.

        Returns:
            List of SimulationResult objects, or None if the file does not exist
            or cannot be parsed.
        """
        load_path: str = self._output_path("simulation", filename)
        if not os.path.exists(load_path):
            return None
        try:
            with open(load_path, "r", encoding="utf-8") as fh:
                raw_list: List[Dict[str, Any]] = json.load(fh)
            results: List[SimulationResult] = [
                SimulationResult.from_dict(d) for d in raw_list
            ]
            logger.info(
                "Loaded %d simulation results from '%s'.",
                len(results),
                load_path,
            )
            return results
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as exc:
            logger.warning(
                "Failed to load simulation results from '%s': %s.",
                load_path,
                exc,
            )
            return None

    # -----------------------------------------------------------------------
    # Public run methods
    # -----------------------------------------------------------------------

    def run_deanonymization(self) -> None:
        """Run all de-anonymization experiments from Section 2 of the paper.

        Reproduces:
          - Table 2: Identity-probing accuracy for 7 selected models.
          - Table 7: Identity-probing accuracy for all 22 models.
          - Table 3: Feature comparison (Length, BoW, TF-IDF) on English prompts.
          - Figure 2: PCA scatter plots of BoW features for 3 specific prompts.
          - Figure 3: Heatmap of BoW detection accuracy across categories × models.

        Data collection is performed on demand if cached responses are not found.
        All results are saved to outputs/tables/ and outputs/figures/.
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: De-anonymization Experiments (Section 2)")
        logger.info("=" * 60)
        t_phase_start: float = time.time()

        # ----------------------------------------------------------------
        # Step 1: Identity-Probing Detector (Section 2.4.1)
        # ----------------------------------------------------------------
        logger.info("--- Step 1: Identity-Probing Detector ---")
        t_step: float = time.time()

        try:
            identity_responses: Dict[str, Dict[str, List[str]]] = (
                self._load_or_collect_identity_responses()
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Identity-probing data collection failed: %s. "
                "Skipping identity-probing evaluation.",
                exc,
                exc_info=True,
            )
            identity_responses = {}

        if identity_responses:
            try:
                ip_detector: IdentityProbingDetector = IdentityProbingDetector(
                    config=self.config
                )
                identity_df = ip_detector.evaluate_all(
                    identity_responses=identity_responses,
                    model_configs=self.config.models,
                )

                # Save Table 7 (all 22 models).
                table7_path: str = self._output_path(
                    "tables", "table7_identity_probing_all.csv"
                )
                self.visualizer.save_table_as_csv(identity_df, table7_path)
                logger.info("Table 7 saved to '%s'.", table7_path)

                # Save Table 2 (7 selected models subset).
                table2_df = ip_detector.filter_table_for_paper(
                    identity_df, selected_models=_TABLE2_MODELS
                )
                table2_path: str = self._output_path(
                    "tables", "table2_identity_probing_subset.csv"
                )
                self.visualizer.save_table_as_csv(table2_df, table2_path)
                logger.info("Table 2 saved to '%s'.", table2_path)

                # Log summary statistics.
                summary_df = Metrics.summarize_deanonymization(identity_df)
                logger.info(
                    "Identity-probing summary (mean accuracy per prompt):\n%s",
                    summary_df.loc[["Mean"]].to_string() if "Mean" in summary_df.index else "N/A",
                )

            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Identity-probing evaluation failed: %s",
                    exc,
                    exc_info=True,
                )

        logger.info(
            "Step 1 complete in %.1f seconds.", time.time() - t_step
        )

        # ----------------------------------------------------------------
        # Step 2: Training-Based Detector — Data Collection
        # ----------------------------------------------------------------
        logger.info("--- Step 2: Training-Based Detector Data Collection ---")
        t_step = time.time()

        try:
            training_dataset: ResponseDataset = (
                self._load_or_collect_training_responses()
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Training response collection failed: %s. "
                "Skipping training-based detector evaluation.",
                exc,
                exc_info=True,
            )
            logger.info(
                "Step 2 failed in %.1f seconds.", time.time() - t_step
            )
            logger.info(
                "Phase 1 (partial) complete in %.1f seconds.",
                time.time() - t_phase_start,
            )
            return

        logger.info(
            "Step 2 complete in %.1f seconds. "
            "ResponseDataset has %d entries.",
            time.time() - t_step,
            len(training_dataset),
        )

        # ----------------------------------------------------------------
        # Step 3: Feature Comparison — Table 3
        # ----------------------------------------------------------------
        logger.info("--- Step 3: Feature Comparison (Table 3) ---")
        t_step = time.time()

        try:
            tbd: TrainingBasedDetector = TrainingBasedDetector(config=self.config)

            table3_df = tbd.evaluate_feature_comparison(
                dataset=training_dataset,
                category="english",
            )

            # Filter to the 7 models shown in Table 3 of the paper.
            available_table3_models: List[str] = [
                m for m in _TABLE3_MODELS if m in table3_df.index
            ]
            if available_table3_models:
                table3_subset = table3_df.loc[available_table3_models]
            else:
                table3_subset = table3_df

            table3_path: str = self._output_path(
                "tables", "table3_feature_comparison.csv"
            )
            self.visualizer.save_table_as_csv(table3_subset, table3_path)
            logger.info("Table 3 saved to '%s'.", table3_path)

            # Also save the full table (all 22 models).
            table3_full_path: str = self._output_path(
                "tables", "table3_feature_comparison_all_models.csv"
            )
            self.visualizer.save_table_as_csv(table3_df, table3_full_path)
            logger.info(
                "Table 3 (all models) saved to '%s'.", table3_full_path
            )

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Feature comparison evaluation failed: %s",
                exc,
                exc_info=True,
            )

        logger.info(
            "Step 3 complete in %.1f seconds.", time.time() - t_step
        )

        # ----------------------------------------------------------------
        # Step 4: Detection Accuracy Heatmap — Figure 3
        # ----------------------------------------------------------------
        logger.info("--- Step 4: Detection Accuracy Heatmap (Figure 3) ---")
        t_step = time.time()

        try:
            # Read primary feature type from config.
            tbd_cfg: Dict[str, Any] = self.config.raw.get(
                "training_based_detector", {}
            )
            primary_feature_type: str = str(
                tbd_cfg.get("primary_feature_type", "bow")
            )

            accuracy_df = tbd.evaluate_all_models_categories(
                dataset=training_dataset,
                feature_type=primary_feature_type,
            )

            # Save heatmap figure (Figure 3).
            figure3_path: str = self._output_path(
                "figures", "figure3_detection_accuracy_heatmap.png"
            )
            self.visualizer.plot_detection_accuracy_heatmap(
                accuracy_df, figure3_path
            )
            logger.info("Figure 3 saved to '%s'.", figure3_path)

            # Save underlying data as CSV.
            figure3_data_path: str = self._output_path(
                "tables", "figure3_accuracy_data.csv"
            )
            self.visualizer.save_table_as_csv(accuracy_df, figure3_data_path)
            logger.info(
                "Figure 3 data saved to '%s'.", figure3_data_path
            )

            # Log summary.
            logger.info(
                "Detection accuracy heatmap: shape=%s, "
                "overall mean=%.1f%%.",
                accuracy_df.shape,
                float(accuracy_df.values.mean()) if not accuracy_df.empty else 0.0,
            )

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Detection accuracy heatmap generation failed: %s",
                exc,
                exc_info=True,
            )

        logger.info(
            "Step 4 complete in %.1f seconds.", time.time() - t_step
        )

        # ----------------------------------------------------------------
        # Step 5: PCA Visualization — Figure 2
        # ----------------------------------------------------------------
        logger.info("--- Step 5: PCA BoW Visualization (Figure 2) ---")
        t_step = time.time()

        try:
            # Read PCA prompts from config.
            pca_prompts: List[str] = list(
                tbd_cfg.get("pca_visualization_prompts", [])
            )

            if not pca_prompts:
                logger.warning(
                    "No PCA visualization prompts configured in "
                    "training_based_detector.pca_visualization_prompts. "
                    "Skipping Figure 2."
                )
            else:
                model_names_for_pca: List[str] = [
                    m.name for m in self.config.models
                ]
                pca_data: Dict[str, Any] = tbd.build_pca_visualization_data(
                    dataset=training