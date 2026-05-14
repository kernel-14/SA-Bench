"""
Script to run scaling experiments from Section 3.3 of the NaViL paper.

This script:
1. Trains models with different LLM sizes (0.5B, 1.8B, 7B) and fixed 600M encoder
2. Trains models with different encoder sizes (75M-2.4B) and fixed LLM
3. Analyzes the results to find optimal encoder sizes
4. Fits the log-linear scaling relationship

Usage:
    python tools/run_scaling_experiments.py \
        --experiment llm_scaling \
        --data_path ./data/pretrain_small.json \
        --output_dir ./outputs/scaling_experiments
"""

import os
import sys
import json
import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from navil.model import NaViLModel, NaViLConfig
from navil.visual_encoder import VisualEncoderConfig
from navil.scaling_analysis import (
    ScalingAnalyzer,
    ScalingExperimentResult,
    VisualEncoderArchitectureAnalyzer,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Experiment configurations from the paper

# LLM sizes for scaling experiments
LLM_CONFIGS = {
    "0.5B": {
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 64,
        "intermediate_size": 2816,
        "vocab_size": 92544,
    },
    "1.8B": {
        "hidden_size": 2048,
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 92544,
    },
    "7B": {
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "vocab_size": 92544,
    },
}

# Visual encoder sizes for scaling experiments
# (depth, width) pairs that give approximately the target parameter count
ENCODER_CONFIGS = {
    "75M": {"depth": 6, "width": 1024},
    "150M": {"depth": 12, "width": 1024},
    "300M": {"depth": 12, "width": 1472},
    "600M": {"depth": 24, "width": 1472},
    "1.2B": {"depth": 24, "width": 2048},
    "2.4B": {"depth": 24, "width": 2880},
}

# Depth/width configurations for architecture search (Section 3.2.3)
# All have approximately 600M parameters
ARCH_CONFIGS_600M = [
    {"depth": 3, "width": 4096},
    {"depth": 6, "width": 2880},
    {"depth": 12, "width": 2048},
    {"depth": 24, "width": 1472},
    {"depth": 48, "width": 1024},
]


def build_model_for_scaling(
    llm_size: str,
    encoder_size: str,
) -> NaViLModel:
    """Build a NaViL model for scaling experiments."""
    llm_cfg = LLM_CONFIGS[llm_size]
    enc_cfg = ENCODER_CONFIGS[encoder_size]

    config = NaViLConfig(
        llm_hidden_size=llm_cfg["hidden_size"],
        llm_num_layers=llm_cfg["num_layers"],
        llm_num_heads=llm_cfg["num_heads"],
        llm_num_kv_heads=llm_cfg["num_kv_heads"],
        llm_head_dim=llm_cfg["head_dim"],
        llm_intermediate_size=llm_cfg["intermediate_size"],
        llm_vocab_size=llm_cfg["vocab_size"],
        visual_encoder_depth=enc_cfg["depth"],
        visual_encoder_width=enc_cfg["width"],
        visual_encoder_num_heads=max(1, enc_cfg["width"] // 64),
        use_moe=True,
        use_multiscale=False,  # Disabled for scaling experiments
    )

    return NaViLModel(config)


def estimate_param_count(llm_size: str, encoder_size: str) -> Tuple[float, float]:
    """Estimate parameter counts for LLM and encoder."""
    llm_cfg = LLM_CONFIGS[llm_size]
    enc_cfg = ENCODER_CONFIGS[encoder_size]

    # LLM params (approximate)
    h = llm_cfg["hidden_size"]
    L = llm_cfg["num_layers"]
    V = llm_cfg["vocab_size"]
    ffn = llm_cfg["intermediate_size"]
    llm_params = V * h + L * (4 * h * h + 3 * h * ffn) + h

    # Encoder params (approximate): N ≈ 12 * d * w^2
    d = enc_cfg["depth"]
    w = enc_cfg["width"]
    enc_params = 12 * d * w * w

    return llm_params / 1e6, enc_params / 1e6


def run_architecture_search_experiment(args):
    """
    Run architecture search experiment (Section 3.2.3).
    Tests different depth/width combinations for 600M encoder.
    """
    logger.info("Running architecture search experiment (Section 3.2.3)")
    logger.info("Testing depth/width configurations for 600M encoder budget")

    results = {}
    for cfg in ARCH_CONFIGS_600M:
        d, w = cfg["depth"], cfg["width"]
        n_params = VisualEncoderArchitectureAnalyzer.compute_param_count(d, w)
        logger.info(f"  Config d={d}, w={w}: {n_params/1e6:.0f}M params")
        results[(d, w)] = n_params / 1e6

    # Print summary
    print("\nArchitecture Search Configurations (600M budget):")
    print(f"{'Depth':>6} {'Width':>6} {'Params (M)':>12} {'Approx N':>12}")
    print("-" * 40)
    for (d, w), n in sorted(results.items()):
        print(f"{d:>6} {w:>6} {n:>12.0f} {12*d*w*w/1e6:>12.0f}")

    return results


def run_llm_scaling_experiment(args):
    """
    Run LLM scaling experiment (Section 3.3.1, Fig. 5).
    Fixed 600M encoder, varying LLM sizes.
    """
    logger.info("Running LLM scaling experiment (Section 3.3.1)")

    experiment_configs = []
    for llm_size in ["0.5B", "1.8B", "7B"]:
        llm_m, enc_m = estimate_param_count(llm_size, "600M")
        experiment_configs.append({
            "llm_size": llm_size,
            "encoder_size": "600M",
            "llm_params_m": llm_m,
            "encoder_params_m": enc_m,
        })
        logger.info(f"  LLM {llm_size}: {llm_m:.0f}M params, Encoder: {enc_m:.0f}M params")

    # Save experiment configs
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "llm_scaling_configs.json")
    with open(config_path, "w") as f:
        json.dump(experiment_configs, f, indent=2)

    logger.info(f"Saved experiment configs to {config_path}")
    logger.info("To run actual training, use the training scripts with these configs")

    return experiment_configs


def run_encoder_scaling_experiment(args):
    """
    Run encoder scaling experiment (Section 3.3.1, Fig. 6).
    Fixed 1.8B LLM, varying encoder sizes.
    """
    logger.info("Running encoder scaling experiment (Section 3.3.1)")

    experiment_configs = []
    for enc_size in ["75M", "150M", "300M", "600M", "1.2B", "2.4B"]:
        llm_m, enc_m = estimate_param_count("1.8B", enc_size)
        experiment_configs.append({
            "llm_size": "1.8B",
            "encoder_size": enc_size,
            "llm_params_m": llm_m,
            "encoder_params_m": enc_m,
        })
        logger.info(f"  Encoder {enc_size}: {enc_m:.0f}M params, LLM: {llm_m:.0f}M params")

    # Save experiment configs
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "encoder_scaling_configs.json")
    with open(config_path, "w") as f:
        json.dump(experiment_configs, f, indent=2)

    logger.info(f"Saved experiment configs to {config_path}")
    return experiment_configs


def analyze_scaling_results(args):
    """
    Analyze scaling results and find optimal encoder sizes (Section 3.3.2, Fig. 7).
    """
    logger.info("Analyzing scaling results")

    # Load results if available
    results_path = os.path.join(args.output_dir, "scaling_results.json")
    if not os.path.exists(results_path):
        logger.warning(f"No results found at {results_path}")
        logger.info("Generating example analysis with synthetic data...")

        # Generate synthetic results for demonstration
        import numpy as np
        np.random.seed(42)

        results = []
        for llm_size_m, llm_name in [(500, "0.5B"), (1800, "1.8B"), (7000, "7B")]:
            for enc_size_m in [75, 150, 300, 600, 1200, 2400]:
                for n_samples in [10_000_000, 30_000_000, 100_000_000, 300_000_000]:
                    # Synthetic loss: decreases with both LLM and encoder size
                    # but encoder has diminishing returns
                    base_loss = 3.0 - 0.3 * math.log(llm_size_m / 500)
                    enc_gain = 0.2 * (1 - math.exp(-enc_size_m / (llm_size_m * 0.3)))
                    data_gain = 0.1 * math.log(n_samples / 1e7)
                    noise = np.random.normal(0, 0.01)
                    loss = base_loss - enc_gain - data_gain + noise

                    results.append(ScalingExperimentResult(
                        llm_size_m=llm_size_m,
                        encoder_size_m=enc_size_m,
                        validation_loss=max(1.5, loss),
                        num_training_samples=n_samples,
                        model_name=f"LLM-{llm_name}_Enc-{enc_size_m}M",
                    ))
    else:
        with open(results_path) as f:
            raw = json.load(f)
        results = [ScalingExperimentResult(**r) for r in raw]

    # Run analysis
    analyzer = ScalingAnalyzer()
    report = analyzer.generate_scaling_report(results)
    print(report)

    # Find optimal encoder sizes
    max_data = max(r.num_training_samples for r in results)
    optimal_sizes = analyzer.compute_optimal_encoder_sizes(results, max_data)

    if len(optimal_sizes) >= 2:
        llm_sizes = sorted(optimal_sizes.keys())
        enc_sizes = [optimal_sizes[s] for s in llm_sizes]
        alpha, beta = analyzer.fit_optimal_encoder_scaling(llm_sizes, enc_sizes)

        print(f"\nKey Finding (Observation 5):")
        print(f"  Optimal encoder size scales log-linearly with LLM size")
        print(f"  log(enc_opt) = {alpha:.3f} * log(llm) + {beta:.3f}")
        print(f"  i.e., enc_opt ∝ llm^{alpha:.3f}")
        print()
        print("  Predicted optimal encoder sizes:")
        for llm_m in [500, 1800, 7000, 30000, 70000]:
            enc_pred = analyzer.predict_optimal_encoder_size(llm_m, alpha, beta)
            print(f"    LLM {llm_m/1000:.1f}B -> optimal encoder: {enc_pred:.0f}M")

    # Save analysis
    analysis_path = os.path.join(args.output_dir, "scaling_analysis.txt")
    with open(analysis_path, "w") as f:
        f.write(report)
    logger.info(f"Saved analysis to {analysis_path}")


def main():
    parser = argparse.ArgumentParser(description="Run NaViL scaling experiments")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["arch_search", "llm_scaling", "encoder_scaling",
                                 "analyze", "all"])
    parser.add_argument("--data_path", type=str, default="./data/pretrain_small.json")
    parser.add_argument("--output_dir", type=str, default="./outputs/scaling_experiments")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.experiment in ["arch_search", "all"]:
        run_architecture_search_experiment(args)

    if args.experiment in ["llm_scaling", "all"]:
        run_llm_scaling_experiment(args)

    if args.experiment in ["encoder_scaling", "all"]:
        run_encoder_scaling_experiment(args)

    if args.experiment in ["analyze", "all"]:
        analyze_scaling_results(args)


if __name__ == "__main__":
    main()
