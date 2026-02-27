"""
Experiment: Ablation Studies for FairSwarm.

This script performs ablation studies to understand the contribution
of each component of the FairSwarm algorithm.

Ablations:
    1. Fairness Gradient: Compare with/without fairness-aware velocity
    2. Adaptive Lambda: Compare fixed vs adaptive fairness weight
    3. Swarm Size: Effect of particle population
    4. Coalition Size: Effect of selection size
    5. Component Weights: Sensitivity to fitness component weights

Author: Tenicka Norwood

Usage:
    python run_ablation.py --ablation fairness_gradient
    python run_ablation.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from fairswarm import Client, FairSwarm, FairSwarmConfig
from fairswarm.core.client import create_synthetic_clients
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.fitness.fairness import AccuracyFairnessFitness

from statistics_utils import compare_means, get_git_hash, mean_ci

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation experiment."""

    n_clients: int = 20
    n_demographic_groups: int = 4
    coalition_size: int = 10
    n_iterations: int = 100
    n_trials: int = 30

    output_dir: str = "results/ablation"
    seed: int = 42


@dataclass
class AblationResult:
    """Result from an ablation trial."""

    variant: str
    trial: int
    fitness: float
    fairness_divergence: float
    convergence_iteration: Optional[int]
    config_params: Dict[str, Any]


def run_ablation_fairness_gradient(
    clients: List[Client],
    target: DemographicDistribution,
    config: AblationConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Ablation: Effect of fairness gradient (novel contribution).

    Compares:
    - Full FairSwarm (with fairness gradient)
    - FairSwarm without fairness gradient (c3 = 0)
    """
    logger.info("Running ablation: Fairness Gradient")

    # Use multi-objective fitness (accuracy + fairness) to properly evaluate
    # the benefit of the fairness gradient in a realistic setting
    fitness_fn = AccuracyFairnessFitness(
        target_distribution=target, fairness_weight=0.3
    )
    results: List[AblationResult] = []

    variants = {
        "full_fairswarm": {"fairness_coefficient": 0.5},
        "no_fairness_gradient": {"fairness_coefficient": 0.0},
    }

    for variant_name, params in variants.items():
        logger.info(f"  Variant: {variant_name}")

        for trial in range(config.n_trials):
            trial_seed = rng.integers(0, 2**31)

            fs_config = FairSwarmConfig(
                swarm_size=30,
                max_iterations=config.n_iterations,
                coalition_size=config.coalition_size,
                fairness_coefficient=params["fairness_coefficient"],
                fairness_weight=0.3,
            )

            optimizer = FairSwarm(
                clients=clients,
                coalition_size=config.coalition_size,
                config=fs_config,
                target_distribution=target,
                seed=trial_seed,
            )

            result = optimizer.optimize(fitness_fn, n_iterations=config.n_iterations)

            results.append(
                AblationResult(
                    variant=variant_name,
                    trial=trial,
                    fitness=result.fitness,
                    fairness_divergence=result.fairness.demographic_divergence
                    if result.fairness
                    else 0.0,
                    convergence_iteration=result.convergence.convergence_iteration
                    if result.convergence
                    else None,
                    config_params=params,
                )
            )

    # Analyze with confidence intervals
    analysis = {}
    for variant_name in variants:
        variant_results = [r for r in results if r.variant == variant_name]
        fitness_values = [r.fitness for r in variant_results]
        fairness_values = [r.fairness_divergence for r in variant_results]

        fitness_ci = mean_ci(fitness_values)
        fairness_ci = mean_ci(fairness_values)

        analysis[variant_name] = {
            "avg_fitness": fitness_ci.mean,
            "fitness_ci": fitness_ci.to_dict(),
            "avg_fairness": fairness_ci.mean,
            "fairness_ci": fairness_ci.to_dict(),
            "n_trials": len(variant_results),
        }

    # Statistical comparison of variants
    full_fitness = [r.fitness for r in results if r.variant == "full_fairswarm"]
    no_grad_fitness = [
        r.fitness for r in results if r.variant == "no_fairness_gradient"
    ]
    full_fairness = [
        r.fairness_divergence for r in results if r.variant == "full_fairswarm"
    ]
    no_grad_fairness = [
        r.fairness_divergence for r in results if r.variant == "no_fairness_gradient"
    ]

    fitness_comparison = compare_means(full_fitness, no_grad_fitness)
    fairness_comparison = compare_means(
        no_grad_fairness, full_fairness
    )  # Note: reversed for improvement

    # Improvement from fairness gradient
    full = analysis["full_fairswarm"]
    no_grad = analysis["no_fairness_gradient"]

    improvement_pct = (
        (no_grad["avg_fairness"] - full["avg_fairness"]) / no_grad["avg_fairness"] * 100
        if no_grad["avg_fairness"] > 0
        else 0
    )

    analysis["improvement"] = {
        "fitness_change": full["avg_fitness"] - no_grad["avg_fitness"],
        "fairness_improvement": no_grad["avg_fairness"] - full["avg_fairness"],
        "fairness_improvement_pct": improvement_pct,
        "fitness_comparison": fitness_comparison,
        "fairness_comparison": fairness_comparison,
    }

    # Build summary with CI
    full_ci = full["fairness_ci"]
    no_grad_ci = no_grad["fairness_ci"]

    return {
        "ablation": "fairness_gradient",
        "analysis": analysis,
        "conclusion": (
            f"Fairness gradient improves demographic balance by {improvement_pct:.1f}% "
            f"(full: {full['avg_fairness']:.4f} [{full_ci['ci_lower']:.4f}, {full_ci['ci_upper']:.4f}] vs "
            f"no gradient: {no_grad['avg_fairness']:.4f} [{no_grad_ci['ci_lower']:.4f}, {no_grad_ci['ci_upper']:.4f}], "
            f"p={fairness_comparison['p_value']:.4f}, 95% CI)"
        ),
    }


def run_ablation_swarm_size(
    clients: List[Client],
    target: DemographicDistribution,
    config: AblationConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Ablation: Effect of swarm size (number of particles).
    """
    logger.info("Running ablation: Swarm Size")

    fitness_fn = AccuracyFairnessFitness(
        target_distribution=target, fairness_weight=0.3
    )
    results: List[AblationResult] = []

    swarm_sizes = [10, 20, 30, 50, 100]

    for swarm_size in swarm_sizes:
        logger.info(f"  Swarm size: {swarm_size}")

        for trial in range(config.n_trials):
            trial_seed = rng.integers(0, 2**31)

            fs_config = FairSwarmConfig(
                swarm_size=swarm_size,
                max_iterations=config.n_iterations,
                coalition_size=config.coalition_size,
                fairness_coefficient=0.5,
                fairness_weight=0.3,
            )

            optimizer = FairSwarm(
                clients=clients,
                coalition_size=config.coalition_size,
                config=fs_config,
                target_distribution=target,
                seed=trial_seed,
            )

            result = optimizer.optimize(fitness_fn, n_iterations=config.n_iterations)

            results.append(
                AblationResult(
                    variant=f"swarm_{swarm_size}",
                    trial=trial,
                    fitness=result.fitness,
                    fairness_divergence=result.fairness.demographic_divergence
                    if result.fairness
                    else 0.0,
                    convergence_iteration=result.convergence.convergence_iteration
                    if result.convergence
                    else None,
                    config_params={"swarm_size": swarm_size},
                )
            )

    # Analyze with confidence intervals
    analysis = {}
    for swarm_size in swarm_sizes:
        variant_results = [r for r in results if r.variant == f"swarm_{swarm_size}"]
        fitness_values = [r.fitness for r in variant_results]
        fairness_values = [r.fairness_divergence for r in variant_results]

        fitness_ci = mean_ci(fitness_values)
        fairness_ci = mean_ci(fairness_values)

        analysis[f"swarm_{swarm_size}"] = {
            "swarm_size": swarm_size,
            "avg_fitness": fitness_ci.mean,
            "fitness_ci": fitness_ci.to_dict(),
            "avg_fairness": fairness_ci.mean,
            "fairness_ci": fairness_ci.to_dict(),
            "n_trials": len(variant_results),
        }

    # Find optimal
    best_size = max(analysis.keys(), key=lambda k: analysis[k]["avg_fitness"])
    best_ci = analysis[best_size]["fitness_ci"]

    return {
        "ablation": "swarm_size",
        "analysis": analysis,
        "best_swarm_size": analysis[best_size]["swarm_size"],
        "conclusion": (
            f"Optimal swarm size: {analysis[best_size]['swarm_size']} particles "
            f"(fitness: {analysis[best_size]['avg_fitness']:.4f} "
            f"[{best_ci['ci_lower']:.4f}, {best_ci['ci_upper']:.4f}], 95% CI)"
        ),
    }


def run_ablation_coalition_size(
    clients: List[Client],
    target: DemographicDistribution,
    config: AblationConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Ablation: Effect of coalition size.
    """
    logger.info("Running ablation: Coalition Size")

    fitness_fn = AccuracyFairnessFitness(
        target_distribution=target, fairness_weight=0.3
    )
    results: List[AblationResult] = []

    coalition_sizes = [3, 5, 8, 10, 15]

    for coalition_size in coalition_sizes:
        if coalition_size > len(clients):
            continue

        logger.info(f"  Coalition size: {coalition_size}")

        for trial in range(config.n_trials):
            trial_seed = rng.integers(0, 2**31)

            fs_config = FairSwarmConfig(
                swarm_size=30,
                max_iterations=config.n_iterations,
                coalition_size=coalition_size,
                fairness_coefficient=0.5,
                fairness_weight=0.3,
            )

            optimizer = FairSwarm(
                clients=clients,
                coalition_size=coalition_size,
                config=fs_config,
                target_distribution=target,
                seed=trial_seed,
            )

            result = optimizer.optimize(fitness_fn, n_iterations=config.n_iterations)

            results.append(
                AblationResult(
                    variant=f"coalition_{coalition_size}",
                    trial=trial,
                    fitness=result.fitness,
                    fairness_divergence=result.fairness.demographic_divergence
                    if result.fairness
                    else 0.0,
                    convergence_iteration=result.convergence.convergence_iteration
                    if result.convergence
                    else None,
                    config_params={"coalition_size": coalition_size},
                )
            )

    # Analyze with confidence intervals
    analysis = {}
    for coalition_size in coalition_sizes:
        if coalition_size > len(clients):
            continue
        variant_results = [
            r for r in results if r.variant == f"coalition_{coalition_size}"
        ]
        if variant_results:
            fitness_values = [r.fitness for r in variant_results]
            fairness_values = [r.fairness_divergence for r in variant_results]

            fitness_ci = mean_ci(fitness_values)
            fairness_ci = mean_ci(fairness_values)

            analysis[f"coalition_{coalition_size}"] = {
                "coalition_size": coalition_size,
                "avg_fitness": fitness_ci.mean,
                "fitness_ci": fitness_ci.to_dict(),
                "avg_fairness": fairness_ci.mean,
                "fairness_ci": fairness_ci.to_dict(),
                "n_trials": len(variant_results),
            }

    # Find best fairness
    if analysis:
        best_fairness_key = min(
            analysis.keys(), key=lambda k: analysis[k]["avg_fairness"]
        )
        best_ci = analysis[best_fairness_key]["fairness_ci"]
        conclusion = (
            f"Best fairness at coalition size {analysis[best_fairness_key]['coalition_size']}: "
            f"divergence {analysis[best_fairness_key]['avg_fairness']:.4f} "
            f"[{best_ci['ci_lower']:.4f}, {best_ci['ci_upper']:.4f}] (95% CI)"
        )
    else:
        conclusion = "No valid coalition sizes tested"

    return {
        "ablation": "coalition_size",
        "analysis": analysis,
        "conclusion": conclusion,
    }


def run_ablation_component_weights(
    clients: List[Client],
    target: DemographicDistribution,
    config: AblationConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Ablation: Sensitivity to fitness component weights.
    """
    logger.info("Running ablation: Component Weights")

    results: List[AblationResult] = []

    # Different weight configurations
    weight_configs = [
        {"name": "accuracy_focused", "w_acc": 0.7, "w_fair": 0.2, "w_cost": 0.1},
        {"name": "fairness_focused", "w_acc": 0.3, "w_fair": 0.6, "w_cost": 0.1},
        {"name": "balanced", "w_acc": 0.4, "w_fair": 0.4, "w_cost": 0.2},
        {"name": "cost_aware", "w_acc": 0.4, "w_fair": 0.3, "w_cost": 0.3},
    ]

    for weight_config in weight_configs:
        logger.info(f"  Weight config: {weight_config['name']}")

        # Use AccuracyFairnessFitness with varying fairness weight
        fitness_fn = AccuracyFairnessFitness(
            target_distribution=target,
            fairness_weight=weight_config["w_fair"],
        )

        for trial in range(config.n_trials):
            trial_seed = rng.integers(0, 2**31)

            fs_config = FairSwarmConfig(
                swarm_size=30,
                max_iterations=config.n_iterations,
                coalition_size=config.coalition_size,
                fairness_coefficient=0.5,
                fairness_weight=weight_config["w_fair"],
                weight_accuracy=weight_config["w_acc"],
                weight_fairness=weight_config["w_fair"],
                weight_cost=weight_config["w_cost"],
            )

            optimizer = FairSwarm(
                clients=clients,
                coalition_size=config.coalition_size,
                config=fs_config,
                target_distribution=target,
                seed=trial_seed,
            )

            result = optimizer.optimize(fitness_fn, n_iterations=config.n_iterations)

            results.append(
                AblationResult(
                    variant=weight_config["name"],
                    trial=trial,
                    fitness=result.fitness,
                    fairness_divergence=result.fairness.demographic_divergence
                    if result.fairness
                    else 0.0,
                    convergence_iteration=result.convergence.convergence_iteration
                    if result.convergence
                    else None,
                    config_params=weight_config,
                )
            )

    # Analyze with confidence intervals
    analysis = {}
    for weight_config in weight_configs:
        variant_results = [r for r in results if r.variant == weight_config["name"]]
        fitness_values = [r.fitness for r in variant_results]
        fairness_values = [r.fairness_divergence for r in variant_results]

        fitness_ci = mean_ci(fitness_values)
        fairness_ci = mean_ci(fairness_values)

        analysis[weight_config["name"]] = {
            "weights": {k: v for k, v in weight_config.items() if k != "name"},
            "avg_fitness": fitness_ci.mean,
            "fitness_ci": fitness_ci.to_dict(),
            "avg_fairness": fairness_ci.mean,
            "fairness_ci": fairness_ci.to_dict(),
            "n_trials": len(variant_results),
        }

    # Find best fairness config
    best_fairness_config = min(
        analysis.keys(), key=lambda k: analysis[k]["avg_fairness"]
    )
    best_ci = analysis[best_fairness_config]["fairness_ci"]

    return {
        "ablation": "component_weights",
        "analysis": analysis,
        "best_config": best_fairness_config,
        "conclusion": (
            f"Best fairness with '{best_fairness_config}' weights: "
            f"divergence {analysis[best_fairness_config]['avg_fairness']:.4f} "
            f"[{best_ci['ci_lower']:.4f}, {best_ci['ci_upper']:.4f}] (95% CI)"
        ),
    }


def run_all_ablations(config: AblationConfig) -> Dict[str, Any]:
    """Run all ablation studies."""
    logger.info("Running all ablation studies")

    rng = np.random.default_rng(config.seed)

    # Generate clients
    clients = create_synthetic_clients(
        n_clients=config.n_clients,
        n_demographic_groups=config.n_demographic_groups,
        seed=config.seed,
    )

    # Target distribution
    target = DemographicDistribution(
        values=np.array([0.20, 0.35, 0.35, 0.10]),
        labels=("age_18_44", "age_45_64", "age_65_84", "age_85_plus"),
    )

    results = {
        "config": asdict(config),
        "ablations": {},
        "timestamp": datetime.now().isoformat(),
    }

    # Run each ablation
    results["ablations"]["fairness_gradient"] = run_ablation_fairness_gradient(
        clients, target, config, rng
    )

    results["ablations"]["swarm_size"] = run_ablation_swarm_size(
        clients, target, config, rng
    )

    results["ablations"]["coalition_size"] = run_ablation_coalition_size(
        clients, target, config, rng
    )

    results["ablations"]["component_weights"] = run_ablation_component_weights(
        clients, target, config, rng
    )

    # Build analysis summary highlighting key findings
    # The fairness gradient ablation is the most important - it validates the novel contribution
    fg_ablation = results["ablations"].get("fairness_gradient", {})
    fg_analysis = fg_ablation.get("analysis", {})
    fg_improvement = fg_analysis.get("improvement", {})

    improvement_pct = fg_improvement.get("fairness_improvement_pct", 0)
    fairness_comparison = fg_improvement.get("fairness_comparison", {})
    p_value = fairness_comparison.get("p_value", 1.0) if fairness_comparison else 1.0
    significant = (
        fairness_comparison.get("significant", False) if fairness_comparison else False
    )

    # Create analysis summary for the experiment suite
    summary_parts = [
        f"Fairness gradient (novel contribution) improves demographic balance by {improvement_pct:.1f}%"
    ]
    if significant:
        summary_parts.append(f" (p={p_value:.4f}, statistically significant)")
    else:
        summary_parts.append(f" (p={p_value:.4f})")

    results["analysis"] = {
        "summary": "".join(summary_parts),
        "key_finding": fg_ablation.get("conclusion", ""),
        "fairness_gradient_validated": improvement_pct > 10
        and significant,  # >10% improvement, significant
    }

    return results


def save_results(results: Dict[str, Any], output_dir: str) -> Path:
    """Save experiment results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results["code_version"] = get_git_hash()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"ablation_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="FairSwarm Ablation Studies")
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=[
            "all",
            "fairness_gradient",
            "swarm_size",
            "coalition_size",
            "component_weights",
        ],
    )
    parser.add_argument("--n_clients", type=int, default=20)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/ablation")

    args = parser.parse_args()

    config = AblationConfig(
        n_clients=args.n_clients,
        n_trials=args.n_trials,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    results = run_all_ablations(config)

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)

    for ablation_name, ablation_results in results["ablations"].items():
        print(f"\n{ablation_name.upper()}:")
        print(f"  {ablation_results['conclusion']}")

    print("=" * 60)

    # Save
    save_results(results, config.output_dir)


if __name__ == "__main__":
    main()
