"""
Experiment: Theorem 2 Iteration-Scaling Validation.

Validates Theorem 2's prediction that DemDiv decreases as O(1/sqrt(T)):
    DemDiv(S*) <= epsilon + C/sqrt(T)

This experiment sweeps over iteration counts T at multiple problem scales,
then fits the 1/sqrt(T) curve to confirm the theoretical rate.

Key design choices:
- Uses n=50 and n=100 clients where the search space is large enough
  that more iterations provide meaningful improvement
- Disables early convergence so all T iterations run
- Tracks both final DemDiv and epsilon-satisfaction rate vs T

Author: Tenicka Norwood

Usage:
    python run_fairness_scaling.py                    # Standard (30 trials)
    python run_fairness_scaling.py --n_trials 10      # Quick test
    python run_fairness_scaling.py --n_trials 50      # Full publication
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

from fairswarm import FairSwarm, FairSwarmConfig
from fairswarm.core.client import create_synthetic_clients
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.fitness.fairness import DemographicFitness

from statistics_utils import get_git_hash, mean_ci

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScalingExperimentConfig:
    """Configuration for iteration-scaling experiment."""

    # Problem scales to test
    problem_scales: List[Tuple[int, int]] = None  # type: ignore  # (n_clients, coalition_size)
    n_demographic_groups: int = 4

    # Iteration counts to test (the key variable)
    iteration_counts: List[int] = None  # type: ignore

    # Fixed fairness parameters
    lambda_value: float = 0.7
    epsilon_target: float = 0.05

    # Experiment setup
    n_trials: int = 30
    seed: int = 42
    output_dir: str = "results/fairness_scaling"

    def __post_init__(self):
        if self.iteration_counts is None:
            self.iteration_counts = [10, 25, 50, 100, 200, 500]
        if self.problem_scales is None:
            # (n_clients, coalition_size) — larger instances need more iterations
            self.problem_scales = [(50, 15), (100, 25)]


def inv_sqrt_model(t: np.ndarray, c: float, eps_floor: float) -> np.ndarray:
    """Model: DemDiv = eps_floor + c / sqrt(T)."""
    return eps_floor + c / np.sqrt(t)


def run_scaling_experiment(config: ScalingExperimentConfig) -> Dict[str, Any]:
    """Run the iteration-scaling experiment across problem scales."""
    logger.info("Starting Theorem 2 iteration-scaling experiment")
    logger.info(f"Problem scales: {config.problem_scales}")
    logger.info(f"Iteration counts: {config.iteration_counts}")
    logger.info(f"Trials per (scale, T): {config.n_trials}")

    all_scale_results = {}

    for n_clients, coalition_size in config.problem_scales:
        scale_key = f"n{n_clients}_m{coalition_size}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Scale: {n_clients} clients, coalition size {coalition_size}")
        logger.info(f"{'='*60}")

        rng = np.random.default_rng(config.seed)

        clients = create_synthetic_clients(
            n_clients=n_clients,
            n_demographic_groups=config.n_demographic_groups,
            seed=config.seed,
        )

        target = DemographicDistribution(
            values=np.array([0.20, 0.35, 0.35, 0.10]),
            labels=("age_18_44", "age_45_64", "age_65_84", "age_85_plus"),
        )

        fitness_fn = DemographicFitness(target_distribution=target)

        results_by_T: Dict[int, List[float]] = {}

        total_runs = len(config.iteration_counts) * config.n_trials
        run_count = 0

        for T in config.iteration_counts:
            logger.info(f"  T={T} iterations ({config.n_trials} trials)")
            divergences = []

            for trial in range(config.n_trials):
                run_count += 1
                trial_seed = int(rng.integers(0, 2**31))

                fs_config = FairSwarmConfig(
                    swarm_size=30,
                    max_iterations=T,
                    coalition_size=coalition_size,
                    fairness_coefficient=0.5,
                    fairness_weight=config.lambda_value,
                    epsilon_fair=config.epsilon_target,
                )

                optimizer = FairSwarm(
                    clients=clients,
                    coalition_size=coalition_size,
                    config=fs_config,
                    target_distribution=target,
                    seed=trial_seed,
                )

                # Disable early convergence by setting threshold=0
                result = optimizer.optimize(
                    fitness_fn=fitness_fn,
                    n_iterations=T,
                    convergence_threshold=0.0,
                    verbose=False,
                )

                div = (
                    result.fairness.demographic_divergence
                    if result.fairness
                    else 0.0
                )
                divergences.append(div)

                if run_count % 20 == 0 or run_count == total_runs:
                    logger.info(
                        f"    Progress: {run_count}/{total_runs} "
                        f"({100 * run_count / total_runs:.0f}%) "
                        f"last DemDiv={div:.6f}"
                    )

            results_by_T[T] = divergences

        analysis = analyze_scaling(results_by_T, config)
        all_scale_results[scale_key] = {
            "n_clients": n_clients,
            "coalition_size": coalition_size,
            "results_by_T": {
                str(T): {
                    "divergences": divs,
                    "mean": float(np.mean(divs)),
                    "std": float(np.std(divs, ddof=1)) if len(divs) > 1 else 0.0,
                    "ci": mean_ci(divs).to_dict(),
                }
                for T, divs in results_by_T.items()
            },
            "analysis": analysis,
        }

    # Overall summary
    overall_summary = build_overall_summary(all_scale_results)

    return {
        "config": asdict(config),
        "scale_results": all_scale_results,
        "overall_summary": overall_summary,
        "timestamp": datetime.now().isoformat(),
        "code_version": get_git_hash(),
    }


def analyze_scaling(
    results_by_T: Dict[int, List[float]],
    config: ScalingExperimentConfig,
) -> Dict[str, Any]:
    """Analyze whether DemDiv follows 1/sqrt(T) scaling."""

    T_values = sorted(results_by_T.keys())
    mean_divs = [float(np.mean(results_by_T[T])) for T in T_values]
    std_divs = [
        float(np.std(results_by_T[T], ddof=1)) if len(results_by_T[T]) > 1 else 1e-10
        for T in T_values
    ]
    ci_divs = [mean_ci(results_by_T[T]) for T in T_values]

    T_arr = np.array(T_values, dtype=float)
    mean_arr = np.array(mean_divs)

    # Fit 1/sqrt(T) model: DemDiv = eps_floor + c / sqrt(T)
    try:
        se = np.array(std_divs) / np.sqrt(max(config.n_trials, 1))
        se = np.maximum(se, 1e-10)
        popt, pcov = curve_fit(
            inv_sqrt_model,
            T_arr,
            mean_arr,
            p0=[0.1, 0.01],
            bounds=([0, 0], [10, 1]),
            sigma=se,
            absolute_sigma=True,
        )
        c_fit, eps_floor_fit = popt
        c_std, eps_floor_std = np.sqrt(np.diag(pcov))
        fit_success = True
    except (RuntimeError, ValueError):
        c_fit, eps_floor_fit = 0.0, 0.0
        c_std, eps_floor_std = 0.0, 0.0
        fit_success = False

    # Compute R^2 of the 1/sqrt(T) fit
    if fit_success:
        predicted = inv_sqrt_model(T_arr, c_fit, eps_floor_fit)
        ss_res = np.sum((mean_arr - predicted) ** 2)
        ss_tot = np.sum((mean_arr - np.mean(mean_arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        r_squared = 0.0
        predicted = np.zeros_like(mean_arr)

    # Spearman correlation: DemDiv vs 1/sqrt(T) should be positive
    inv_sqrt_T = 1.0 / np.sqrt(T_arr)
    if len(T_arr) >= 3:
        spearman_r, spearman_p = scipy_stats.spearmanr(inv_sqrt_T, mean_arr)
    else:
        spearman_r, spearman_p = 0.0, 1.0

    # Check monotonicity: DemDiv should decrease with T
    is_monotone = all(
        mean_divs[i] >= mean_divs[i + 1] for i in range(len(mean_divs) - 1)
    )

    # Compute improvement ratios
    if mean_divs[0] > 0:
        total_reduction = 1 - mean_divs[-1] / mean_divs[0]
    else:
        total_reduction = 0.0

    # Epsilon satisfaction rate at each T
    satisfaction_rates = {}
    for T in T_values:
        n_satisfied = sum(
            1 for d in results_by_T[T] if d <= config.epsilon_target
        )
        satisfaction_rates[str(T)] = {
            "rate": n_satisfied / len(results_by_T[T]),
            "n_satisfied": n_satisfied,
            "n_total": len(results_by_T[T]),
        }

    # Pairwise comparison: T_min vs T_max
    if len(T_values) >= 2:
        from statistics_utils import compare_means
        pairwise = compare_means(
            results_by_T[T_values[0]],
            results_by_T[T_values[-1]],
        )
    else:
        pairwise = None

    # Determine validation status
    validates_scaling = (
        fit_success
        and r_squared >= 0.7
        and spearman_r > 0
        and total_reduction > 0.1
    )

    # Build summary
    if validates_scaling:
        summary = (
            f"VALIDATED: DemDiv follows 1/sqrt(T) with R^2={r_squared:.4f}. "
            f"Model: DemDiv = {eps_floor_fit:.4f} + {c_fit:.4f}/sqrt(T). "
            f"Reduction: {total_reduction * 100:.1f}% from T={T_values[0]} to T={T_values[-1]}. "
            f"Spearman rho={spearman_r:.3f} (p={spearman_p:.4f})."
        )
    else:
        summary = (
            f"R^2={r_squared:.4f}, "
            f"Spearman rho={spearman_r:.3f} (p={spearman_p:.4f}). "
            f"Reduction: {total_reduction * 100:.1f}% from T={T_values[0]} to T={T_values[-1]}."
        )

    return {
        "summary": summary,
        "validates_scaling": validates_scaling,
        "scaling_fit": {
            "model": "DemDiv = eps_floor + c / sqrt(T)",
            "c": float(c_fit),
            "c_std": float(c_std),
            "eps_floor": float(eps_floor_fit),
            "eps_floor_std": float(eps_floor_std),
            "r_squared": float(r_squared),
            "fit_success": fit_success,
        },
        "correlation": {
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        },
        "monotonicity": {
            "is_monotone_decreasing": is_monotone,
        },
        "total_reduction_pct": float(total_reduction * 100),
        "pairwise_Tmin_vs_Tmax": pairwise,
        "per_T_summary": {
            str(T): {
                "mean_div": float(mean_divs[i]),
                "std_div": float(std_divs[i]),
                "ci_lower": float(ci_divs[i].lower),
                "ci_upper": float(ci_divs[i].upper),
                "predicted_div": float(predicted[i]) if fit_success else None,
                "satisfaction_rate": satisfaction_rates[str(T)]["rate"],
            }
            for i, T in enumerate(T_values)
        },
        "satisfaction_rates": satisfaction_rates,
    }


def build_overall_summary(all_scale_results: Dict[str, Any]) -> str:
    """Build overall summary across all problem scales."""
    lines = ["Theorem 2 Iteration-Scaling Summary"]
    lines.append("=" * 50)

    all_validated = True
    for scale_key, data in all_scale_results.items():
        n = data["n_clients"]
        m = data["coalition_size"]
        analysis = data["analysis"]
        validated = analysis["validates_scaling"]
        r2 = analysis["scaling_fit"]["r_squared"]
        reduction = analysis["total_reduction_pct"]

        status = "VALIDATED" if validated else "NOT VALIDATED"
        lines.append(
            f"  n={n}, m={m}: {status} "
            f"(R^2={r2:.3f}, {reduction:.1f}% reduction)"
        )
        if not validated:
            all_validated = False

    if all_validated:
        lines.append(
            "\nConclusion: Theorem 2 scaling confirmed across all problem sizes."
        )
    else:
        lines.append(
            "\nConclusion: Scaling confirmed for some but not all problem sizes."
        )

    return "\n".join(lines)


def save_results(results: Dict[str, Any], output_dir: str) -> Path:
    """Save results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"fairness_scaling_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Theorem 2 Iteration-Scaling Validation"
    )
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="results/fairness_scaling"
    )
    parser.add_argument(
        "--lambda_value", type=float, default=0.7,
        help="Fixed fairness weight",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.05,
        help="Target epsilon-fairness",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=None,
        help="Iteration counts to test (default: 10 25 50 100 200 500)",
    )

    args = parser.parse_args()

    config = ScalingExperimentConfig(
        n_trials=args.n_trials,
        seed=args.seed,
        output_dir=args.output_dir,
        lambda_value=args.lambda_value,
        epsilon_target=args.epsilon,
    )
    if args.iterations:
        config.iteration_counts = args.iterations

    results = run_scaling_experiment(config)

    # Print summary
    print("\n" + "=" * 70)
    print("THEOREM 2 ITERATION-SCALING RESULTS")
    print("=" * 70)
    print(results["overall_summary"])

    for scale_key, data in results["scale_results"].items():
        n = data["n_clients"]
        m = data["coalition_size"]
        print(f"\n--- n={n}, coalition_size={m} ---")
        print(f"{'T':>6} | {'Mean DemDiv':>12} | {'95% CI':>24} | {'eps-sat':>8}")
        print("-" * 60)
        for T_str, tdata in data["analysis"]["per_T_summary"].items():
            print(
                f"{T_str:>6} | {tdata['mean_div']:>12.6f} | "
                f"[{tdata['ci_lower']:.6f}, {tdata['ci_upper']:.6f}] | "
                f"{tdata['satisfaction_rate']:>7.1%}"
            )

        fit = data["analysis"]["scaling_fit"]
        if fit["fit_success"]:
            print(
                f"Fit: DemDiv = {fit['eps_floor']:.4f} + "
                f"{fit['c']:.4f}/sqrt(T), R^2={fit['r_squared']:.4f}"
            )

    print("=" * 70)

    save_results(results, config.output_dir)


if __name__ == "__main__":
    main()
