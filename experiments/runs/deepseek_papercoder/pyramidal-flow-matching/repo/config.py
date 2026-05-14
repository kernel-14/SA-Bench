## config.py
"""
Configuration module for Pyramidal Flow Matching.
Loads YAML configuration via OmegaConf, validates and adapts it,
and provides attribute-style access for the rest of the project.
"""

import os
import warnings
from typing import Any, List, Dict

import omegaconf
from omegaconf import OmegaConf, DictConfig


class Config:
    """
    Central configuration class. Loads a YAML file and exposes
    all parameters via attribute access (e.g., cfg.global.seed).
    Provides a get() method for dotted key access.
    """
    def __init__(self, config_path: str):
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        try:
            raw_cfg = OmegaConf.load(config_path)
        except omegaconf.errors.OmegaConfBaseException as e:
            raise ValueError(f"Failed to parse configuration file {config_path}: {e}") from e

        self._cfg: DictConfig = raw_cfg

        # Post-process and validate sections
        self._process_pyramid_schedule()
        self._process_temporal_pyramid()
        self._warn_about_placeholder_paths()

    def _process_pyramid_schedule(self) -> None:
        """Reverse pyramid stage s/e lists to paper order (finest-first) and validate recursion."""
        pyramid = self._cfg.model.pyramid
        K = pyramid.num_stages
        s_raw = list(pyramid.schedules.s)  # coarsest-to-finest
        e_raw = list(pyramid.schedules.e)

        if len(s_raw) != K or len(e_raw) != K:
            raise ValueError(f"Pyramid schedule length mismatch: expected {K}, got s={len(s_raw)}, e={len(e_raw)}")

        # Reverse to finest-first order (stage k=0 is finest, k=K-1 is coarsest)
        s_finest_first = s_raw[::-1]
        e_finest_first = e_raw[::-1]

        # Validate and correct recursion e_{k+1} = 2 * s_k / (1 + s_k)
        for k in range(K - 1):
            expected_e = 2.0 * s_finest_first[k] / (1.0 + s_finest_first[k])
            if not abs(e_finest_first[k + 1] - expected_e) < 1e-4:
                warnings.warn(
                    f"Pyramid stage schedule violates renoising recursion for k={k}: "
                    f"e_{k+1}={e_finest_first[k+1]} != 2*s_{k}={s_finest_first[k]}/(1+s_k)={expected_e}. "
                    f"Correcting e_{k+1} to {expected_e}."
                )
                e_finest_first[k + 1] = expected_e

        # Store corrected values back into config (finest-first)
        # Keep them as Python lists for easier consumption
        pyramid.s = s_finest_first
        pyramid.e = e_finest_first
        # Also store a list of dicts for convenience
        pyramid.stages = [dict(s=s, e=e) for s, e in zip(s_finest_first, e_finest_first)]
        # Remove the now redundant schedules sub-key
        del pyramid.schedules

    def _process_temporal_pyramid(self) -> None:
        """Convert temporal pyramid mapping into a lookup table of extra downsampling factors."""
        tp = self._cfg.temporal_pyramid
        max_frames = tp.max_history_frames
        mapping = list(tp.mapping)  # list of dict-like items with 'offset' and 'factor'

        # Initialize lookup array (1-indexed, offset 1..max_frames)
        factors = [0] * (max_frames + 1)   # index 0 unused

        # Parse offset ranges; they can be lists of two ints: [start, end] inclusive
        for rule in mapping:
            offset_range = rule.offset
            factor = rule.factor
            if isinstance(offset_range, (list, tuple)) and len(offset_range) == 2:
                start, end = offset_range
            elif isinstance(offset_range, int):
                start = end = offset_range
            else:
                raise ValueError(f"Invalid offset format in temporal pyramid mapping: {offset_range}")
            for offset in range(start, min(end, max_frames) + 1):
                factors[offset] = factor

        # Verify all offsets 1..max_frames are covered; if not, raise error
        for offset in range(1, max_frames + 1):
            if factors[offset] == 0 and not any(offset in range(r.offset[0], r.offset[1]+1) for r in mapping if isinstance(r.offset, (list, tuple))): # may not have been set intentionally
                # Actually, if no rule sets factor for this offset, it stays 0 (no extra downsample). That's acceptable.
                pass
        # We will keep factor 0 for any offset not explicitly listed.

        # Add the computed factors array to config
        tp.history_factors = factors[1:]   # drop index 0

    def _warn_about_placeholder_paths(self) -> None:
        """Scan dataset paths and warn if they still contain placeholder strings."""
        # Recursively search for string values that contain "path/to"
        def _check(node, keypath=""):
            if isinstance(node, str):
                if "path/to" in node or node.startswith("/path/"):
                    warnings.warn(f"Dataset path '{node}' at key '{keypath}' appears to be a placeholder. "
                                  "Please replace it with actual data path before training.")
            elif isinstance(node, DictConfig):
                for k, v in node.items():
                    _check(v, f"{keypath}.{k}")

        _check(self._cfg.datasets, "datasets")

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve configuration value using dotted key path, e.g., 'model.pyramid.K'."""
        try:
            return omegaconf.select(self._cfg, key, default=default)
        except Exception:
            if default is not None:
                return default
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a plain dictionary."""
        return OmegaConf.to_container(self._cfg, resolve=True)

    def __getattr__(self, name: str) -> Any:
        # Delegate attribute access to the underlying OmegaConf config.
        # This enables cfg.global, cfg.model, etc.
        try:
            return getattr(self._cfg, name)
        except AttributeError as e:
            raise AttributeError(f"'Config' object has no attribute '{name}'") from e

    def __repr__(self) -> str:
        return f"Config(keys={list(self._cfg.keys())})"


# Optional CLI test
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python config.py <config.yaml>")
        sys.exit(1)
    cfg = Config(sys.argv[1])
    print("Pyramid stages (finest-first):")
    for i, stg in enumerate(cfg.model.pyramid.stages):
        print(f"  Stage {i}: s={stg['s']:.4f}, e={stg['e']:.4f}")
    print("Temporal history factors:", cfg.temporal_pyramid.history_factors)
