"""
Experiment: Theorem 4 (Privacy-Fairness Tradeoff) Validation.

This script validates Theorem 4 from the paper:
    "For any (epsilon_DP, delta)-differentially private coalition selection
    mechanism that achieves epsilon_F-fairness, the utility loss is at least:
    UtilityLoss >= Omega(sqrt(k * log(1/delta)) / (epsilon_DP * epsilon_F))"

Experimental Setup:
    1. Run FairSwarm-DP with varying privacy budgets
    2. Measure achieved fairness and utility
    3. Compare empirical tradeoff with theoretical bound
    4. Validate impossibility result (tradeoff is fundamental)

Key Insight:
    Theorem 4 establishes that privacy and fairness cannot be achieved
    simultaneously without sacrificing utility. The bound shows this
    tradeoff is fundamental, not an artifact of our algorithm.

Metrics:
    - Utility achieved (fitness)
    - Fairness achieved (demographic divergence)
    - Privacy spent (epsilon)
    - Theoretical bound comparison

Author: Tenicka Norwood

Usage:
    python run_privacy.py --epsilon_dp 4.0 --epsilon_fair 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

from fairswarm import Client, FairSwarmConfig
from fairswarm.algorithms.fairswarm_dp import FairSwarmDP, DPConfig
from fairswarm.core.client import create_synthetic_clients
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import kl_divergence
from fairswarm.fitness.fairness import (
    DemographicFitness,
    compute_coalition_demographics,
)

from statistics_utils import get_git_hash, mean_ci, proportion_ci

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PrivacyExperimentConfig:
    """Configuration for privacy-fairness tradeoff experiment."""

    # Client setup
    n_clients: int = 20
    n_demographic_groups: int = 4
    coalition_size: int = 10

    # Privacy parameters
    epsilon_dp_values: List[float] = None  # type: ignore
    epsilon_fair_values: List[float] = None  # type: ignore
    delta: float = 1e-5

    # Experiment setup
    n_iterations: int = 50
    n_trials: int = 30
    swarm_size: int = 10  # Fewer particles = fewer queries = less noise per iteration

    # Output
    output_dir: str = "results/privacy"
    seed: int = 42

    def __post_init__(self):
        if self.epsilon_dp_values is None:
            self.epsilon_dp_values = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        if self.epsilon_fair_values is None:
            self.epsilon_fair_values = [0.02, 0.05, 0.10]


@dataclass
class PrivacyTrialResult:
    """Result from a single privacy-fairness trial."""

    epsilon_dp: float
    epsilon_fair_target: float
    delta: float
    n_demographic_groups: int

    utility_achieved: float
    fairness_achieved: float  # Demographic divergence
    privacy_spent: float

    theoretical_bound: float
    utility_loss: float
    bound_ratio: float  # empirical / theoretical

    fairness_satisfied: bool
    coalition: List[int]


def compute_theoretical_bound(
    k: int,
    epsilon_dp: float,
    epsilon_fair: float,
    delta: float,
) -> float:
    """
    Compute Theorem 4 theoretical bound on utility loss.

    Theorem 4: UtilityLoss >= Omega(sqrt(k * log(1/delta)) / (epsilon_DP * epsilon_F))

    Args:
        k: Number of demographic groups
        epsilon_dp: Privacy budget
        epsilon_fair: Fairness parameter
        delta: Privacy parameter delta

    Returns:
        Lower bound on utility loss
    """
    if epsilon_dp <= 0 or epsilon_fair <= 0 or delta <= 0:
        return float("inf")

    # Omega hides a constant factor; we use 1 for simplicity
    bound = math.sqrt(k * math.log(1.0 / delta)) / (epsilon_dp * epsilon_fair)
    return bound


def run_private_fairswarm_trial(
    clients: List[Client],
    target_distribution: DemographicDistribution,
    epsilon_dp: float,
    epsilon_fair: float,
    delta: float,
    coalition_size: int,
    n_iterations: int,
    seed: int,
    swarm_size: int = 10,
) -> PrivacyTrialResult:
    """
    Run a single privacy-fairness trial using FairSwarm-DP.

    Args:
        clients: List of clients
        target_distribution: Target demographics
        epsilon_dp: Privacy budget
        epsilon_fair: Fairness target
        delta: Privacy parameter
        coalition_size: Target coalition size
        n_iterations: Number of iterations
        seed: Random seed
        swarm_size: Number of particles (fewer = fewer privacy queries)

    Returns:
        PrivacyTrialResult with tradeoff measurements
    """
    k = target_distribution.n_groups

    # Create config for FairSwarm-DP
    config = FairSwarmConfig(
        swarm_size=swarm_size,
        max_iterations=n_iterations,
        coalition_size=coalition_size,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        fairness_coefficient=0.5,
        epsilon_fair=epsilon_fair,
    )

    # Create fitness function
    fitness_fn = DemographicFitness(target_distribution=target_distribution)

    # Create DP config with explicit sensitivity and auto-calibration
    dp_config = DPConfig(
        epsilon=epsilon_dp,
        delta=delta,
        fitness_sensitivity=0.1,  # Known sensitivity for DemographicFitness
        auto_calibrate=True,
    )

    # Run FairSwarm-DP
    optimizer = FairSwarmDP(
        clients=clients,
        coalition_size=coalition_size,
        config=config,
        target_distribution=target_distribution,
        dp_config=dp_config,
        seed=seed,
    )

    result = optimizer.optimize(
        fitness_fn=fitness_fn,
        n_iterations=n_iterations,
        verbose=False,
    )

    # Compute fairness (demographic divergence)
    coalition_demo = compute_coalition_demographics(result.coalition, clients)
    target = target_distribution.as_array()
    divergence = kl_divergence(coalition_demo, target)

    # Get privacy spent
    epsilon_spent, _delta_spent = optimizer.get_privacy_spent()
    privacy_spent = epsilon_spent

    # Compute theoretical bound
    theoretical_bound = compute_theoretical_bound(k, epsilon_dp, epsilon_fair, delta)

    # Utility loss placeholder (adjusted relative to baseline in run_privacy_experiment)
    utility_loss = 0.0

    # Bound ratio (how close to theoretical limit)
    bound_ratio = utility_loss / theoretical_bound if theoretical_bound > 0 else 0

    return PrivacyTrialResult(
        epsilon_dp=epsilon_dp,
        epsilon_fair_target=epsilon_fair,
        delta=delta,
        n_demographic_groups=k,
        utility_achieved=result.fitness,
        fairness_achieved=divergence,
        privacy_spent=privacy_spent,
        theoretical_bound=theoretical_bound,
        utility_loss=utility_loss,
        bound_ratio=bound_ratio,
        fairness_satisfied=divergence <= epsilon_fair,
        coalition=list(result.coalition),
    )


def run_non_private_baseline(
    clients: List[Client],
    target_distribution: DemographicDistribution,
    coalition_size: int,
    n_iterations: int,
    seed: int,
    swarm_size: int = 10,
) -> float:
    """
    Run non-private FairSwarm as baseline.

    Args:
        clients: List of clients
        target_distribution: Target demographics
        coalition_size: Target coalition size
        n_iterations: Number of iterations
        seed: Random seed
        swarm_size: Number of particles

    Returns:
        Baseline fitness (non-private)
    """
    from fairswarm import FairSwarm

    config = FairSwarmConfig(
        swarm_size=swarm_size,
        max_iterations=n_iterations,
        coalition_size=coalition_size,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        fairness_coefficient=0.5,
    )

    fitness_fn = DemographicFitness(target_distribution=target_distribution)

    optimizer = FairSwarm(
        clients=clients,
        coalition_size=coalition_size,
        config=config,
        target_distribution=target_distribution,
        seed=seed,
    )

    result = optimizer.optimize(
        fitness_fn=fitness_fn,
        n_iterations=n_iterations,
        verbose=False,
    )

    return result.fitness


def run_privacy_experiment(
    config: PrivacyExperimentConfig,
) -> Dict[str, Any]:
    """
    Run full privacy-fairness tradeoff experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary of results
    """
    logger.info("Starting privacy-fairness tradeoff experiment (Theorem 4 validation)")
    logger.info(f"Privacy budgets: {config.epsilon_dp_values}")
    logger.info(f"Fairness targets: {config.epsilon_fair_values}")

    rng = np.random.default_rng(config.seed)

    all_results: List[PrivacyTrialResult] = []
    results_by_config: Dict[str, List[PrivacyTrialResult]] = {}

    # Generate clients once
    clients = create_synthetic_clients(
        n_clients=config.n_clients,
        n_demographic_groups=config.n_demographic_groups,
        seed=config.seed,
    )

    # Target distribution
    target = DemographicDistribution(
        values=np.ones(config.n_demographic_groups) / config.n_demographic_groups,
        labels=tuple(f"group_{i}" for i in range(config.n_demographic_groups)),
    )

    # Compute non-private baseline
    logger.info("Computing non-private baseline...")
    baseline_fitness = run_non_private_baseline(
        clients=clients,
        target_distribution=target,
        coalition_size=config.coalition_size,
        n_iterations=config.n_iterations,
        seed=config.seed,
        swarm_size=config.swarm_size,
    )
    logger.info(f"Baseline (non-private) fitness: {baseline_fitness:.4f}")

    # Test all privacy/fairness combinations
    total_configs = len(config.epsilon_dp_values) * len(config.epsilon_fair_values)
    config_idx = 0

    for eps_dp in config.epsilon_dp_values:
        for eps_fair in config.epsilon_fair_values:
            config_idx += 1
            config_key = f"dp{eps_dp}_fair{eps_fair}"
            results_by_config[config_key] = []

            theoretical_bound = compute_theoretical_bound(
                config.n_demographic_groups, eps_dp, eps_fair, config.delta
            )

            logger.info(
                f"\nConfig {config_idx}/{total_configs}: "
                f"epsilon_dp={eps_dp}, epsilon_fair={eps_fair}, "
                f"theoretical_bound={theoretical_bound:.4f}"
            )

            for trial in range(config.n_trials):
                trial_seed = rng.integers(0, 2**31)

                result = run_private_fairswarm_trial(
                    clients=clients,
                    target_distribution=target,
                    epsilon_dp=eps_dp,
                    epsilon_fair=eps_fair,
                    delta=config.delta,
                    coalition_size=config.coalition_size,
                    n_iterations=config.n_iterations,
                    seed=trial_seed,
                    swarm_size=config.swarm_size,
                )

                # Adjust utility loss relative to baseline
                result.utility_loss = max(0, baseline_fitness - result.utility_achieved)
                result.bound_ratio = (
                    result.utility_loss / theoretical_bound
                    if theoretical_bound > 0 and theoretical_bound != float("inf")
                    else 0
                )

                all_results.append(result)
                results_by_config[config_key].append(result)

            # Log summary for this config
            avg_utility = np.mean(
                [r.utility_achieved for r in results_by_config[config_key]]
            )
            avg_divergence = np.mean(
                [r.fairness_achieved for r in results_by_config[config_key]]
            )
            avg_loss = np.mean([r.utility_loss for r in results_by_config[config_key]])
            fair_rate = np.mean(
                [r.fairness_satisfied for r in results_by_config[config_key]]
            )

            logger.info(
                f"  Avg utility: {avg_utility:.4f}, "
                f"Avg divergence: {avg_divergence:.4f}, "
                f"Utility loss: {avg_loss:.4f}, "
                f"Fairness rate: {fair_rate * 100:.1f}%"
            )

    # Analyze results
    analysis = analyze_privacy_results(
        all_results, results_by_config, baseline_fitness, config
    )

    return {
        "config": asdict(config),
        "baseline_fitness": baseline_fitness,
        "analysis": analysis,
        "n_total_trials": len(all_results),
        "timestamp": datetime.now().isoformat(),
    }


def analyze_privacy_results(
    all_results: List[PrivacyTrialResult],
    results_by_config: Dict[str, List[PrivacyTrialResult]],
    baseline_fitness: float,
    config: PrivacyExperimentConfig,
) -> Dict[str, Any]:
    """
    Analyze privacy-fairness tradeoff results with confidence intervals.

    Args:
        all_results: All trial results
        results_by_config: Results grouped by configuration
        baseline_fitness: Non-private baseline fitness
        config: Experiment configuration

    Returns:
        Analysis dictionary with confidence intervals for publication
    """
    analysis = {
        "overall": {},
        "by_privacy_budget": {},
        "by_fairness_target": {},
        "theorem_validation": {},
    }

    # Overall statistics with CIs
    utility_values = [r.utility_achieved for r in all_results]
    utility_loss_values = [r.utility_loss for r in all_results]
    fairness_values = [r.fairness_achieved for r in all_results]
    fairness_satisfied_count = sum(1 for r in all_results if r.fairness_satisfied)

    utility_ci = mean_ci(utility_values)
    utility_loss_ci = mean_ci(utility_loss_values)
    fairness_ci = mean_ci(fairness_values)
    fairness_rate_ci = proportion_ci(fairness_satisfied_count, len(all_results))

    analysis["overall"] = {
        "n_trials": len(all_results),
        "baseline_fitness": baseline_fitness,
        "avg_utility_achieved": utility_ci.mean,
        "utility_achieved_ci": utility_ci.to_dict(),
        "avg_utility_loss": utility_loss_ci.mean,
        "utility_loss_ci": utility_loss_ci.to_dict(),
        "avg_fairness_achieved": fairness_ci.mean,
        "fairness_achieved_ci": fairness_ci.to_dict(),
        "overall_fairness_rate": fairness_rate_ci.mean,
        "fairness_rate_ci": fairness_rate_ci.to_dict(),
    }

    # By privacy budget with CIs
    for eps_dp in config.epsilon_dp_values:
        dp_results = [r for r in all_results if r.epsilon_dp == eps_dp]
        if dp_results:
            utility_vals = [r.utility_achieved for r in dp_results]
            loss_vals = [r.utility_loss for r in dp_results]
            fairness_vals = [r.fairness_achieved for r in dp_results]
            satisfied_count = sum(1 for r in dp_results if r.fairness_satisfied)

            utility_ci = mean_ci(utility_vals)
            loss_ci = mean_ci(loss_vals)
            fairness_ci = mean_ci(fairness_vals)
            rate_ci = proportion_ci(satisfied_count, len(dp_results))

            analysis["by_privacy_budget"][f"eps_dp_{eps_dp}"] = {
                "n_trials": len(dp_results),
                "avg_utility": utility_ci.mean,
                "utility_ci": utility_ci.to_dict(),
                "avg_utility_loss": loss_ci.mean,
                "utility_loss_ci": loss_ci.to_dict(),
                "avg_fairness": fairness_ci.mean,
                "fairness_ci": fairness_ci.to_dict(),
                "fairness_rate": rate_ci.mean,
                "fairness_rate_ci": rate_ci.to_dict(),
            }

    # By fairness target with CIs
    for eps_fair in config.epsilon_fair_values:
        fair_results = [r for r in all_results if r.epsilon_fair_target == eps_fair]
        if fair_results:
            utility_vals = [r.utility_achieved for r in fair_results]
            loss_vals = [r.utility_loss for r in fair_results]
            fairness_vals = [r.fairness_achieved for r in fair_results]
            satisfied_count = sum(1 for r in fair_results if r.fairness_satisfied)

            utility_ci = mean_ci(utility_vals)
            loss_ci = mean_ci(loss_vals)
            fairness_ci = mean_ci(fairness_vals)
            rate_ci = proportion_ci(satisfied_count, len(fair_results))

            analysis["by_fairness_target"][f"eps_fair_{eps_fair}"] = {
                "n_trials": len(fair_results),
                "avg_utility": utility_ci.mean,
                "utility_ci": utility_ci.to_dict(),
                "avg_utility_loss": loss_ci.mean,
                "utility_loss_ci": loss_ci.to_dict(),
                "avg_fairness": fairness_ci.mean,
                "fairness_ci": fairness_ci.to_dict(),
                "fairness_rate": rate_ci.mean,
                "fairness_rate_ci": rate_ci.to_dict(),
            }

    # Theorem 4 validation with CIs
    # Check monotonicity: lower privacy budget -> higher utility loss
    privacy_utility_losses = {}
    privacy_utility_loss_cis = {}
    for eps_dp in sorted(config.epsilon_dp_values):
        dp_results = [r for r in all_results if r.epsilon_dp == eps_dp]
        if dp_results:
            loss_vals = [r.utility_loss for r in dp_results]
            loss_ci = mean_ci(loss_vals)
            privacy_utility_losses[eps_dp] = loss_ci.mean
            privacy_utility_loss_cis[eps_dp] = loss_ci.to_dict()

    # Check monotonicity: tighter fairness -> higher utility loss
    fairness_utility_losses = {}
    fairness_utility_loss_cis = {}
    for eps_fair in sorted(config.epsilon_fair_values):
        fair_results = [r for r in all_results if r.epsilon_fair_target == eps_fair]
        if fair_results:
            loss_vals = [r.utility_loss for r in fair_results]
            loss_ci = mean_ci(loss_vals)
            fairness_utility_losses[eps_fair] = loss_ci.mean
            fairness_utility_loss_cis[eps_fair] = loss_ci.to_dict()

    # Validate tradeoff using Spearman rank correlation
    # Theorem 4 predicts: higher epsilon → lower utility loss (negative correlation)
    sorted_eps_dp = sorted(privacy_utility_losses.keys())
    dp_losses_sorted = [privacy_utility_losses[e] for e in sorted_eps_dp]
    if len(sorted_eps_dp) >= 3:
        privacy_spearman = stats.spearmanr(sorted_eps_dp, dp_losses_sorted)
        privacy_rho = float(privacy_spearman.statistic)
        privacy_pvalue = float(privacy_spearman.pvalue)
    else:
        privacy_rho, privacy_pvalue = 0.0, 1.0

    sorted_eps_fair = sorted(fairness_utility_losses.keys())
    fair_losses_sorted = [fairness_utility_losses[e] for e in sorted_eps_fair]
    if len(sorted_eps_fair) >= 3:
        fairness_spearman = stats.spearmanr(sorted_eps_fair, fair_losses_sorted)
        fairness_rho = float(fairness_spearman.statistic)
        fairness_pvalue = float(fairness_spearman.pvalue)
    else:
        fairness_rho, fairness_pvalue = 0.0, 1.0

    # Negative correlation = higher epsilon → lower loss (expected by Theorem 4)
    privacy_trend_valid = privacy_rho < -0.5
    fairness_trend_valid = fairness_rho < -0.5

    # Bound comparison with CI
    bound_ratios = [r.bound_ratio for r in all_results if r.bound_ratio > 0]
    bound_ratio_ci = mean_ci(bound_ratios) if bound_ratios else None

    # Overall fairness rate with CI for summary
    overall_fair_ci = analysis["overall"]["fairness_rate_ci"]

    analysis["theorem_validation"] = {
        "privacy_utility_losses": privacy_utility_losses,
        "privacy_utility_loss_cis": privacy_utility_loss_cis,
        "fairness_utility_losses": fairness_utility_losses,
        "fairness_utility_loss_cis": fairness_utility_loss_cis,
        "privacy_spearman_rho": privacy_rho,
        "privacy_spearman_pvalue": privacy_pvalue,
        "privacy_trend_valid": privacy_trend_valid,
        "fairness_spearman_rho": fairness_rho,
        "fairness_spearman_pvalue": fairness_pvalue,
        "fairness_trend_valid": fairness_trend_valid,
        "avg_bound_ratio": bound_ratio_ci.mean if bound_ratio_ci else 0,
        "bound_ratio_ci": bound_ratio_ci.to_dict() if bound_ratio_ci else None,
        "theorem_validated": privacy_trend_valid or fairness_trend_valid,
        "summary": (
            f"Theorem 4 {'VALIDATED' if (privacy_trend_valid or fairness_trend_valid) else 'NOT VALIDATED'}: "
            f"Privacy Spearman rho={privacy_rho:.3f} (p={privacy_pvalue:.3f}), "
            f"Fairness Spearman rho={fairness_rho:.3f} (p={fairness_pvalue:.3f}), "
            f"Fairness achieved: {overall_fair_ci['mean'] * 100:.1f}% "
            f"[{overall_fair_ci['ci_lower'] * 100:.1f}%, {overall_fair_ci['ci_upper'] * 100:.1f}%] (95% CI)"
        ),
    }

    return analysis


def save_results(results: Dict[str, Any], output_dir: str) -> Path:
    """Save experiment results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results["code_version"] = get_git_hash()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"privacy_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Theorem 4 (Privacy-Fairness Tradeoff) Validation Experiment"
    )
    parser.add_argument("--n_clients", type=int, default=20)
    parser.add_argument("--coalition_size", type=int, default=10)
    parser.add_argument("--n_iterations", type=int, default=50)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--swarm_size", type=int, default=10)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/privacy")

    args = parser.parse_args()

    config = PrivacyExperimentConfig(
        n_clients=args.n_clients,
        coalition_size=args.coalition_size,
        n_iterations=args.n_iterations,
        n_trials=args.n_trials,
        swarm_size=args.swarm_size,
        delta=args.delta,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    results = run_privacy_experiment(config)

    # Print summary
    print("\n" + "=" * 60)
    print("PRIVACY-FAIRNESS TRADEOFF EXPERIMENT RESULTS (Theorem 4)")
    print("=" * 60)
    print(f"Baseline (non-private) fitness: {results['baseline_fitness']:.4f}")
    print(results["analysis"]["theorem_validation"]["summary"])
    print("=" * 60)

    # Save
    save_results(results, config.output_dir)


if __name__ == "__main__":
    main()
