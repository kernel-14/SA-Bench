```python
## mitigations/cost_model.py
"""Attack cost model for the adversarial leaderboard manipulation paper.

This module implements the cost model from Section 4.1 of the paper
"Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards."

The total attack cost formula from Section 4.1:
    Total cost = ceil(N / m) × c_account + N × c_action + c_detector

Where:
  - N = total number of actions (interactions or votes) required
  - m = maximum actions permitted per user account
  - c_account = cost to obtain one user account
  - c_action = cost per individual action
  - c_detector = one-time fixed cost to build the training-based detector

Paper alignment:
  - Section 4.1: "The total cost of the attack consists of three components:
    Training detector cost c_detector, Account maintenance cost, Action cost."
  - Appendix A.3: "the cost is at most $440" for 200 prompts.
  - Section 4.2.4: "approximately $20 per prompt (or per action)" for prompt
    uniqueness mitigation.
  - config.yaml: mitigations.cost_model.c_detector: 440.0
  - config.yaml: mitigations.cost_model.c_prompt_uniqueness_per_action: 20.0
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from data_structures import SimulationResult
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default cost constants from config.yaml mitigations.cost_model section.
# These are module-level constants for documentation purposes; the actual
# values are passed in at construction time by main.py.
# ---------------------------------------------------------------------------

# config.yaml: mitigations.cost_model.c_detector: 440.0
_DEFAULT_C_DETECTOR: float = 440.0

# config.yaml: mitigations.cost_model.c_prompt_uniqueness_per_action: 20.0
_DEFAULT_C_PROMPT_UNIQUENESS: float = 20.0

# ---------------------------------------------------------------------------
# Valid mitigation type identifiers for the summarize() method.
# ---------------------------------------------------------------------------
_VALID_MITIGATION_TYPES: frozenset = frozenset(
    {
        "baseline",
        "authentication",
        "rate_limiting",
        "captcha",
        "prompt_uniqueness",
    }
)

# ---------------------------------------------------------------------------
# Column names for the summary DataFrame.
# ---------------------------------------------------------------------------
_COL_TARGET_MODEL: str = "target_model"
_COL_TARGET_RANK: str = "target_rank"
_COL_N_VOTES: str = "n_votes"
_COL_N_INTERACTIONS: str = "n_interactions"
_COL_ACHIEVED: str = "achieved"
_COL_BASELINE_COST: str = "baseline_cost_usd"


class CostModel:
    """Computes the total cost of an adversarial leaderboard manipulation attack.

    Implements the cost formula from Section 4.1 of the paper:
        Total cost = ceil(N / m) × c_account + N × c_action + c_detector

    Provides methods for computing costs under different mitigation scenarios
    (authentication, rate limiting, CAPTCHA, prompt uniqueness) and a
    summarize() method that builds a comparison table across all scenarios
    for a list of simulation results.

    Attributes:
        c_detector: One-time fixed cost of building the training-based
            de-anonymization detector. Default $440.0 per Appendix A.3:
            "the cost is at most $440" for 200 prompts × ~$2.20/prompt.

    Example:
        >>> cost_model = CostModel(c_detector=440.0)
        >>> cost_model.compute_baseline_cost()
        440.0
        >>> cost_model.compute_cost(n_actions=1000, m=100, c_account=5.0, c_action=0.0)
        490.0  # ceil(1000/100)*5.0 + 1000*0.0 + 440.0 = 50.0 + 0.0 + 440.0 = 490.0
        >>> cost_model.compute_with_captcha(n_actions=1000, c_captcha=0.001)
        441.0  # 1000*0.001 + 440.0 = 1.0 + 440.0 = 441.0
    """

    def __init__(self, c_detector: float = _DEFAULT_C_DETECTOR) -> None:
        """Initialize the CostModel with the fixed detector training cost.

        Args:
            c_detector: One-time cost of building the training-based detector
                offline. Default $440.0 per config.yaml
                mitigations.cost_model.c_detector and Appendix A.3 of the paper.
                This cost is incurred regardless of the number of adversarial
                actions and is independent of any mitigation strategy.

        Raises:
            ValueError: If c_detector is negative (cost cannot be negative).

        Example:
            >>> model = CostModel(c_detector=440.0)
            >>> model.c_detector
            440.0
            >>> model = CostModel()  # Uses default $440.0
            >>> model.c_detector
            440.0
        """
        if c_detector < 0.0:
            raise ValueError(
                f"c_detector must be non-negative, got {c_detector}. "
                f"The detector training cost cannot be negative."
            )

        self.c_detector: float = c_detector

        logger.info(
            "CostModel initialized with c_detector=$%.2f.",
            self.c_detector,
        )

    def compute_cost(
        self,
        n_actions: int,
        m: int,
        c_account: float,
        c_action: float,
    ) -> float:
        """Compute the total attack cost using the paper's cost formula.

        Implements the formula from Section 4.1:
            Total cost = ceil(N / m) × c_account + N × c_action + c_detector

        The three cost components are:
          1. Account maintenance cost: ceil(N / m) × c_account
             - ceil(N / m) is the minimum number of accounts needed to
               distribute N actions with at most m actions per account.
             - A partial batch still requires a full account (hence ceil).
          2. Action cost: N × c_action
             - Aggregate cost of all N individual actions.
          3. Detector training cost: c_detector (fixed, one-time).

        Args:
            n_actions: Total number of actions (interactions or votes) required
                to achieve the attack objective. Must be non-negative.
                Use n_interactions (not n_votes) from SimulationResult since
                all interactions incur costs (API calls, CAPTCHA fees, etc.).
            m: Maximum number of actions permitted per user account. Must be
                positive. When m is very large (e.g., m = n_actions), only
                one account is needed: ceil(n_actions / n_actions) = 1.
            c_account: Cost to obtain one user account (USD). Non-negative.
                Examples: $0 (no authentication), $5 (phone verification),
                $10 (credit card verification).
            c_action: Cost per individual action (USD). Non-negative.
                Examples: $0 (no per-action cost), $0.001 (CAPTCHA solving),
                $20 (prompt uniqueness — new detector per action).

        Returns:
            Total attack cost in USD as a float. Always >= c_detector.

        Raises:
            ValueError: If m <= 0 (division by zero in ceil(N / m)).
            ValueError: If n_actions < 0, c_account < 0, or c_action < 0.

        Example:
            >>> model = CostModel(c_detector=440.0)
            >>> # 1000 actions, max 100 per account, $5/account, $0/action
            >>> model.compute_cost(1000, 100, 5.0, 0.0)
            490.0  # ceil(1000/100)*5 + 1000*0 + 440 = 50 + 0 + 440
            >>> # 1000 actions, unlimited per account, $0/account, $0.001/action
            >>> model.compute_cost(1000, 1000, 0.0, 0.001)
            441.0  # ceil(1000/1000)*0 + 1000*0.001 + 440 = 0 + 1 + 440
        """
        # --- Input validation ---
        if m <= 0:
            raise ValueError(
                f"compute_cost: m must be positive (got m={m}). "
                f"m represents the maximum actions per account and cannot be "
                f"zero or negative."
            )
        if n_actions < 0:
            raise ValueError(
                f"compute_cost: n_actions must be non-negative (got {n_actions})."
            )
        if c_account < 0.0:
            raise ValueError(
                f"compute_cost: c_account must be non-negative (got {c_account})."
            )
        if c_action < 0.0:
            raise ValueError(
                f"compute_cost: c_action must be non-negative (got {c_action})."
            )

        # --- Handle zero-action edge case ---
        # If no actions are needed (e.g., target rank already achieved),
        # only the fixed detector cost applies.
        if n_actions == 0:
            logger.debug(
                "compute_cost: n_actions=0, returning c_detector=%.2f.",
                self.c_detector,
            )
            return self.c_detector

        # --- Compute account maintenance cost ---
        # ceil(N / m) = number of accounts needed to distribute N actions
        # with at most m actions per account.
        n_accounts: int = math.ceil(n_actions / m)
        account_maintenance_cost: float = float(n_accounts) * c_account

        # --- Compute action cost ---
        action_cost: float = float(n_actions) * c_action

        # --- Sum all three components ---
        total_cost: float = account_maintenance_cost + action_cost + self.c_detector

        logger.debug(
            "compute_cost: n_actions=%d, m=%d, c_account=%.4f, c_action=%.4f "
            "-> n_accounts=%d, account_cost=%.4f, action_cost=%.4f, "
            "c_detector=%.4f, total=%.4f.",
            n_actions,
            m,
            c_account,
            c_action,
            n_accounts,
            account_maintenance_cost,
            action_cost,
            self.c_detector,
            total_cost,
        )

        return total_cost

    def compute_baseline_cost(self) -> float:
        """Compute the attack cost with no mitigations in place.

        Models the baseline scenario from Section 4.1:
        "Without mitigations, a single user can place as many actions per
        account as desired and thus only a single account is necessary.
        Further, the cost per action is minimal. Therefore, the total cost
        is dominated by the training detector cost c_detector which we
        estimated in Appendix B.1 to be $440."

        In the no-mitigation scenario:
          - A single account handles all actions (m effectively unlimited).
          - Account creation is free (c_account = 0).
          - Each action is free (c_action = 0).
          - Only the one-time detector training cost applies.

        Returns:
            The fixed detector training cost c_detector (default $440.0).
            This is the minimum possible attack cost — all mitigations
            increase the total cost above this baseline.

        Example:
            >>> model = CostModel(c_detector=440.0)
            >>> model.compute_baseline_cost()
            440.0
            >>> model = CostModel(c_detector=500.0)
            >>> model.compute_baseline_cost()
            500.0
        """
        logger.debug(
            "compute_baseline_cost: returning c_detector=%.2f (no mitigations).",
            self.c_detector,
        )
        return self.c_detector

    def compute_with_authentication(
        self,
        n_actions: int,
        m: int,
        c_account: float,
    ) -> float:
        """Compute attack cost under the authentication mitigation (Section 4.2.1).

        Models the authentication defense from Section 4.2.1:
        "The most effective method to increase the cost per account c_account
        is to enforce authentication on Chatbot Arena through integration with
        existing digital identity providers... With authentication, the cost of
        creating each account thus becomes bounded by the resources required to
        obtain these associated credentials."

        Authentication increases c_account (cost per account) but does not
        add per-action cost. The attacker must create ceil(N / m) accounts,
        each costing c_account to obtain.

        Args:
            n_actions: Total number of actions required. Use n_interactions
                from SimulationResult.
            m: Maximum actions per account before the account is flagged or
                rate-limited. Represents the behavioral limit enforced by the
                authentication system.
            c_account: Cost to obtain one authenticated account (USD).
                Examples:
                  - $0: No authentication (baseline).
                  - $1–$5: Email/social media verification.
                  - $5–$20: Phone number verification.
                  - $20+: Credit card or government ID verification.

        Returns:
            Total attack cost in USD: ceil(N/m) × c_account + c_detector.

        Raises:
            ValueError: If m <= 0, n_actions < 0, or c_account < 0.

        Example:
            >>> model = CostModel(c_detector=440.0)
            >>> # 1000 actions, max 100/account, $5/account
            >>> model.compute_with_authentication(1000, 100, 5.0)
            490.0  # ceil(1000/100)*5 + 0 + 440 = 50 + 0 + 440
            >>> # 1000 actions, max 10/account, $10/account
            >>> model.compute_with_authentication(1000, 10, 10.0)
            1440.0  # ceil(1000/10)*10 + 0 + 440 = 1000 + 0 + 440
        """
        logger.debug(
            "compute_with_authentication: n_actions=%d, m=%d, c_account=%.4f.",
            n_actions,
            m,
            c_account,
        )
        return self.compute_cost(
            n_actions=n_actions,
            m=m,
            c_account=c_account,
            c_action=0.0,
        )

    def compute_with_rate_limiting(
        self,
        n_actions: int,
        m: int,
        c_account: float,
    ) -> float:
        """Compute attack cost under the rate limiting mitigation (Section 4.2.2).

        Models the rate limiting defense from Section 4.2.2:
        "Reducing m through temporal rate limits on actions for each account
        is also an effective strategy. Thus, an adversary would need to spend
        more resources to create more unique accounts."

        Rate limiting specifically targets m (maximum actions per account),
        forcing the attacker to distribute actions across more accounts.
        The cost formula is structurally identical to authentication — both
        increase ceil(N/m) × c_account — but the conceptual focus differs:
          - Authentication: increases c_account (harder to create each account).
          - Rate limiting: decreases m (fewer actions allowed per account).

        The paper suggests setting m to a quantile of the benign user query
        distribution (e.g., the median) to minimize impact on legitimate users
        while maximizing cost for adversaries.

        Args:
            n_actions: Total number of actions required. Use n_interactions
                from SimulationResult.
            m: Maximum actions per account enforced by the rate limit. Should
                be set to a low quantile of the benign user query distribution
                (e.g., median number of queries per session).
            c_account: Cost to obtain one user account (USD). May be $0 if
                account creation is free but rate-limited.

        Returns:
            Total attack cost in USD: ceil(N/m) × c_account + c_detector.

        Raises:
            ValueError: If m <= 0, n_actions < 0, or c_account < 0.

        Example:
            >>> model = CostModel(c_detector=440.0)
            >>> # 1000 actions, rate limit of 50/account, $0/account (free accounts)
            >>> model.compute_with_rate_limiting(1000, 50, 0.0)
            440.0  # ceil(1000/50)*0 + 0 + 440 = 0 + 0 + 440 (free accounts don't help)
            >>> # 1000 actions, rate limit of 50/account, $2/account
            >>> model.compute_with_rate_limiting(1000, 50, 2.0)
            480.0  # ceil(1000/50)*2 + 0 + 440 = 40 + 0 + 440
        """
        logger.debug(
            "compute_with_rate_limiting: n_actions=%d, m=%d, c_account=%.4f.",
            n_actions,
            m,
            c_account,
        )
        return self.compute_cost(
            n_actions=n_actions,
            m=m,
            c_account=c_account,
            c_action=0.0,
        )

    def compute_with_captcha(
        self,
        n_actions: int,
        c_captcha: float,
    ) -> float:
        """Compute attack cost under the CAPTCHA mitigation (Section 4.2.4).

        Models the CAPTCHA defense from Section 4.2.4:
        "Requiring a CAPTCHA per impression/vote: this makes the cost
        c_action = N × c_CAPTCHA, since automated CAPTCHA-solving services
        typically charge on a per-CAPTCHA basis."

        CAPTCHA increases c_action (per-action cost) rather than c_account.
        A single account is sufficient since the bottleneck is per-action cost,
        not account creation. With m = n_actions, ceil(n_actions / n_actions) = 1,
        so the account maintenance term is 1 × 0 = 0.

        Args:
            n_actions: Total number of actions (each requiring a CAPTCHA solve).
                Use n_interactions from SimulationResult.
            c_captcha: Cost per CAPTCHA solve (USD). Automated CAPTCHA-solving
                services typically charge $0.001–$0.003 per solve.

        Returns:
            Total attack cost in USD: n_actions × c_captcha + c_detector.

        Raises:
            ValueError: If n_actions < 0 or c_captcha < 0.

        Example:
            >>> model = CostModel(c_detector=440.0)
            >>> # 1000 actions, $0.001/CAPTCHA
            >>> model.compute_with_captcha(1000, 0.001)
            441.0  # 1000*0.001 + 440 = 1.0 + 440
            >>> # 100000 actions, $0.003/CAPTCHA
            >>> model.compute_with_captcha(100000, 0.003)
            740.0  # 100000*0.003 + 440 = 300 + 440
        """
        logger.debug(
            "compute_with_captcha: n_actions=%d, c_captcha=%.6f.",
            n_actions,
            c_captcha,
        )
        # Single account handles all actions (m = n_actions or 1 if n_actions=0).
        # Use max(n_actions, 1) to avoid m=0 when n_actions=0.
        effective_m: int = max(n_actions, 1)
        return self.compute_cost(
            n_actions=n_actions,
            m=effective_m,
            c_account=0.0,
            c_action=c_captcha,
        )

    def compute_with_prompt_uniqueness(
        self,
        n_actions: int,
        c_per_prompt: float = _DEFAULT_C_PROMPT_UNIQUENESS,
    ) -> float:
        """Compute attack cost under the prompt uniqueness mitigation (Section 4.2.4).

        Models the prompt uniqueness defense from Section 4.2.4:
        "Enforcing prompt uniqueness: A potentially more effective mitigation
        is to reject or downweight previously used prompts when updating the
        Bradley-Terry coefficient leaderboard. This forces attackers to generate
        new prompts and train corresponding detectors for each action. As
        detailed in Appendix A.3, this approach would introduce a cost of
        approximately $20 per prompt (or per action)."

        Each action now requires a fresh prompt plus new detector training,
        making c_action = c_per_prompt = $20 per action. This is the most
        expensive per-action mitigation — for 1,000 votes, total ≈ $20,440.

        Args:
            n_actions: Total number of actions (each requiring a new prompt
                and detector). Use n_interactions from SimulationResult.
            c_per_prompt: Cost per unique prompt (USD). Default $20.0 per
                config.yaml mitigations.cost_model.c_prompt_uniqueness_per_action
                and Section 4.2.4 of the paper.

        Returns:
            Total attack cost in USD: n_actions × c_per_prompt + c_detector.

        Raises:
            ValueError: If n_actions < 0 or c_per_prompt < 0.

        Example:
            >>> model = CostModel(c_detector=440.0)
            >>> # 1000 actions, $20/prompt
            >>> model.compute_with_prompt_uniqueness(1000, 20.0)
            20440.0  # 1000*20 + 440 = 20000 + 440
            >>> # 100 actions, $20/prompt (default)
            >>> model.compute_with_prompt_uniqueness(100)
            2440.0  # 100*20 + 440 = 2000 + 440
        """
        logger.debug(
            "compute_with_prompt_uniqueness: n_actions=%d, c_per_prompt=%.4f.",
            n_actions,
            c_per_prompt,
        )
        # Single account handles all actions (m = n_actions or 1 if n_actions=0).
        effective_m: int = max(n_actions, 1)
        return self.compute_cost(
            n_actions=n_actions,
            m=effective_m,
            c_account=0.0,
            c_action=c_per_prompt,
        )

    def summarize(
        self,
        simulation_results: List[SimulationResult],
        mitigation_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """Build a cost comparison table across mitigation scenarios.

        For each SimulationResult, computes the attack cost under the baseline
        (no mitigation) and each specified mitigation scenario. Returns a
        DataFrame suitable for direct CSV export via Visualizer.save_table_as_csv.

        The cost is computed using n_interactions (total actions including
        abstentions) rather than n_votes, because all interactions incur costs
        (API calls, CAPTCHA fees, etc.) regardless of whether a vote was cast.

        Args:
            simulation_results: List of SimulationResult objects from
                AttackSimulator. Each provides target_model, target_rank,
                achieved, n_votes, and n_interactions fields.
            mitigation_configs: List of mitigation scenario dicts. Each dict
                must have a 'name' (str) and 'type' (str) key. Additional keys
                depend on the mitigation type:
                  - 'authentication': requires 'm' (int), 'c_account' (float)
                  - 'rate_limiting': requires 'm' (int), 'c_account' (float)
                  - 'captcha': requires 'c_captcha' (float)
                  - 'prompt_uniqueness': optionally 'c_per_prompt' (float,
                    default $20.0)
                If None, uses a default set of representative scenarios.

        Returns:
            pd.DataFrame with columns:
              - 'target_model': model name string
              - 'target_rank': desired rank (int)
              - 'n_votes': adversarial votes cast (int)
              - 'n_interactions': total interactions (int)
              - 'achieved': whether target rank was reached (bool)
              - 'baseline_cost_usd': cost with no mitigations ($440.0)
              - One column per mitigation config named by config['name']
            Rows where achieved=False have costs computed from n_interactions
            at max_interactions (may be very high) and are flagged in the
            'achieved' column.

        Example:
            >>> results = [SimulationResult("llama-13b", 128, True, 126, 10000)]
            >>> configs = [
            ...     {'name': 'Auth (phone)', 'type': 'authentication',
            ...      'm': 100, 'c_account': 5.0},
            ...     {'name': 'CAPTCHA', 'type': 'captcha', 'c_captcha': 0.001},
            ... ]
            >>> df = cost_model.summarize(results, configs)
            >>> 'baseline_cost_usd' in df.columns
            True
            >>> 'Auth (phone)' in df.columns
            True
        """
        # --- Use default mitigation configs if none provided ---
        if mitigation_configs is None:
            mitigation_configs = self._default_mitigation_configs()
            logger.info(
                "summarize: using %d default mitigation configs.",
                len(mitigation_configs),
            )

        logger.info(
            "summarize: computing costs for %d simulation results × "
            "%d mitigation scenarios (+1 baseline).",
            len(simulation_results),
            len(mitigation_configs),
        )

        # --- Build rows for the output DataFrame ---
        rows: List[Dict[str, Any]] = []

        for result in simulation_results:
            # Use n_interactions as N (total actions) for cost computation.
            # All interactions incur costs, not just votes.
            n_actions: int = result.n_interactions

            # Build the row dict starting with metadata columns.
            row: Dict[str, Any] = {
                _COL_TARGET_MODEL: result.target_model,
                _COL_TARGET_RANK: result.target_rank,
                _COL_N_VOTES: result.n_votes,
                _COL_N_INTERACTIONS: result.n_interactions,
                _COL_ACHIEVED: result.achieved,
            }

            # --- Baseline cost (no mitigations) ---
            baseline_cost: float = self.compute_baseline_cost()
            row[_COL_BASELINE_COST] = round(baseline_cost, 4)

            # --- Per-mitigation costs ---
            for mit_cfg in mitigation_configs:
                mit_name: str = str(mit_cfg.get("name", "unknown"))
                mit_type: str = str(mit_cfg.get("type", "baseline"))

                try:
                    mit_cost: float = self._compute_mitigation_cost(
                        n_actions=n_actions,
                        mit_cfg=mit_cfg,
                    )
                    row[mit_name] = round(mit_cost, 4)
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "summarize: failed to compute cost for mitigation "
                        "'%s' (type='%s') on result target_model='%s': %s. "
                        "Storing NaN.",
                        mit_name,
                        mit_type,
                        result.target_model,
                        exc,
                    )
                    row[mit_name] = float("nan")

            rows.append(row)

        # --- Construct DataFrame ---
        if not rows:
            logger.warning(
                "summarize: no simulation results provided. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        df: pd.DataFrame = pd.DataFrame(rows)

        # Ensure consistent column ordering: metadata first, then costs.
        metadata_cols: List[str] = [
            _COL_TARGET_MODEL,
            _COL_TARGET_RANK,
            _COL_N_VOTES,
            _COL_N_INTERACTIONS,
            _COL_ACHIEVED,
            _COL_BASELINE_COST,
        ]
        mitigation_cols: List[str] = [
            str(cfg.get("name", "unknown")) for cfg in mitigation_configs
        ]
        # Only include columns that actually exist in the DataFrame.
        ordered_cols: List[str] = [
            c for c in metadata_cols if c in df.columns
        ] + [
            c for c in mitigation_cols if c in df.columns
        ]
        df = df[ordered_cols]

        logger.info(
            "summarize: built cost summary DataFrame with shape %s. "
            "Columns: %s.",
            df.shape,
            list(df.columns),
        )

        return df

    # -----------------------------------------------------------------------
    # Private helper methods
    # -----------------------------------------------------------------------

    def _compute_mitigation_cost(
        self,
        n_actions: int,
        mit_cfg: Dict[str, Any],
    ) -> float:
        """Dispatch to the appropriate compute_with_* method based on mitigation type.

        Args:
            n_actions: Total number of actions (n_interactions from SimulationResult).
            mit_cfg: Mitigation configuration dict with 'type' key and
                type-specific parameter keys.

        Returns:
            Total attack cost in USD for this mitigation scenario.

        Raises:
            ValueError: If the mitigation type is unknown or required parameters
                are missing from mit_cfg.
        """
        mit_type: str = str(mit_cfg.get("type", "baseline")).lower().strip()

        if mit_type == "baseline":
            return self.compute_baseline_cost()

        elif mit_type == "authentication":
            m: int = int(mit_cfg.get("m", 100))
            c_account: float = float(mit_cfg.get("c_account", 5.0))
            return self.compute_with_authentication(
                n_actions=n_actions,
                m=m,
                c_account=c_account,
            )

        elif mit_type == "rate_limiting":
            m = int(mit_cfg.get("m", 50))
            c_account = float(mit_cfg.get("c_account", 0.0))
            return self.compute_with_rate_limiting(
                n_actions=n_actions,
                m=m,
                c_account=c_account,
            )

        elif mit_type == "captcha":
            c_captcha: float = float(mit_cfg.get("c_captcha", 0.001))
            return self.compute_with_captcha(
                n_actions=n_actions,
                c_captcha=c_captcha,
            )

        elif mit_type == "prompt_uniqueness":
            c_per_prompt: float = float(
                mit_cfg.get("c_per_prompt", _DEFAULT_C_PROMPT_UNIQUENESS)
            )
            return self.compute_with_prompt_uniqueness(
                n_actions=n_actions,
                c_per_prompt=c_per_prompt,
            )

        else:
            raise ValueError(
                f"_compute_mitigation_cost: unknown mitigation type '{mit_type}'. "
                f"Must be one of {sorted(_VALID_MITIGATION_TYPES)}."
            )

    def _default_mitigation_configs(self) -> List[Dict[str, Any]]:
        """Return a default set of representative mitigation scenarios.

        Provides a sensible default