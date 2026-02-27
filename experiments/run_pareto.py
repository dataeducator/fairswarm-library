"""
Experiment: Pareto Front Analysis for FairSwarm.

Sweeps across weight combinations to map the accuracy-fairness
Pareto frontier, demonstrating that FairSwarm dominates baselines
across the full tradeoff curve.

CIA-Integrity: Reproducible seeds, validated inputs, deterministic training.
CIA-Availability: Parallel execution, graceful error handling.

Output: JSON with Pareto-optimal points for each algorithm.

Author: Tenicka Norwood

Usage:
    python experiments/run_pareto.py --parallel
    python experiments/run_pareto.py --n_trials 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from run_real_fl import (
    FederatedFitness,
    build_fairswarm_clients,
    generate_federated_dataset,
    RealFLExperimentConfig,
)
from fairswarm import FairSwarm, FairSwarmConfig
from fairswarm.demographics.distribution import DemographicDistribution
from statistics_utils import get_git_hash, mean_ci

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ParetoConfig:
    """Configuration for Pareto front analysis."""

    n_clients: int = 50
    k: int = 4
    coalition_fraction: float = 0.3
    n_fl_rounds: int = 3
    local_epochs: int = 5
    learning_rate: float = 0.01
    n_trials: int = 10
    n_iterations: int = 50
    swarm_size: int = 20
    non_iid_alpha: float = 0.5
    n_samples_total: int = 20000
    n_features: int = 20
    output_dir: str = "results/pareto"
    seed: int = 42

    # Weight grid: (w_accuracy, w_fairness, w_cost)
    # w_cost is fixed at 0.1; w_accuracy + w_fairness = 0.9
    weight_grid: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.80, 0.10, 0.10),
            (0.70, 0.20, 0.10),
            (0.60, 0.30, 0.10),
            (0.50, 0.40, 0.10),
            (0.40, 0.50, 0.10),
            (0.30, 0.60, 0.10),
            (0.20, 0.70, 0.10),
        ]
    )


def _run_pareto_point(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run a single Pareto point: one weight combination, one trial.

    Returns results for FairSwarm and baselines at this weight setting.
    """
    w_acc = args["w_accuracy"]
    w_fair = args["w_fairness"]
    w_cost = args["w_cost"]
    trial_seed = args["trial_seed"]
    trial_idx = args["trial_idx"]
    n_clients = args["n_clients"]
    k = args["k"]

    config = RealFLExperimentConfig(
        n_clients_values=[n_clients],
        k_values=[k],
        coalition_fraction=args["coalition_fraction"],
        n_fl_rounds=args["n_fl_rounds"],
        local_epochs=args["local_epochs"],
        learning_rate=args["learning_rate"],
        n_trials=1,
        n_iterations=args["n_iterations"],
        swarm_size=args["swarm_size"],
        non_iid_alpha=args["non_iid_alpha"],
        n_samples_total=args["n_samples_total"],
        n_features=args["n_features"],
        seed=trial_seed,
    )

    coalition_size = max(3, int(n_clients * config.coalition_fraction))

    # Generate data
    fed_dataset = generate_federated_dataset(
        n_clients=n_clients,
        n_demographic_groups=k,
        n_features=config.n_features,
        n_samples_total=config.n_samples_total,
        non_iid_alpha=config.non_iid_alpha,
        seed=trial_seed,
    )

    clients = build_fairswarm_clients(fed_dataset)
    target_dist = DemographicDistribution(values=fed_dataset.target_distribution)

    results: List[Dict[str, Any]] = []

    # FairSwarm with specified weights
    try:
        fitness_fn = FederatedFitness(
            fed_dataset=fed_dataset,
            target_distribution=target_dist,
            n_fl_rounds=config.n_fl_rounds,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            weight_accuracy=w_acc,
            weight_fairness=w_fair,
            weight_cost=w_cost,
            seed=trial_seed,
        )

        fs_config = FairSwarmConfig(
            swarm_size=config.swarm_size,
            max_iterations=config.n_iterations,
            coalition_size=coalition_size,
            inertia=0.7,
            cognitive=1.5,
            social=1.5,
            fairness_coefficient=0.5,
            fairness_weight=w_fair,
            adaptive_fairness=True,
            velocity_max=4.0,
            convergence_threshold=1e-6,
            patience=10,
        )

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=fs_config,
            target_distribution=target_dist,
            seed=trial_seed,
        )

        result = optimizer.optimize(
            fitness_fn=fitness_fn, n_iterations=config.n_iterations, verbose=False
        )
        final_eval = fitness_fn.evaluate(result.coalition, clients)

        results.append(
            {
                "algorithm": "FairSwarm",
                "w_accuracy": w_acc,
                "w_fairness": w_fair,
                "w_cost": w_cost,
                "trial_idx": trial_idx,
                "auc_roc": final_eval.components.get("accuracy", 0.5),
                "demographic_divergence": final_eval.components.get(
                    "divergence", float("inf")
                ),
                "equalized_odds_gap": final_eval.components.get(
                    "equalized_odds_gap", 0.0
                ),
            }
        )
    except Exception as e:
        logger.error(f"FairSwarm failed at weights ({w_acc}, {w_fair}): {e}")

    # Standard PSO (same weights, no fairness gradient)
    try:
        fitness_fn2 = FederatedFitness(
            fed_dataset=fed_dataset,
            target_distribution=target_dist,
            n_fl_rounds=config.n_fl_rounds,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            weight_accuracy=w_acc,
            weight_fairness=w_fair,
            weight_cost=w_cost,
            seed=trial_seed + 1,
        )

        pso_config = FairSwarmConfig(
            swarm_size=config.swarm_size,
            max_iterations=config.n_iterations,
            coalition_size=coalition_size,
            inertia=0.7,
            cognitive=1.5,
            social=1.5,
            fairness_coefficient=0.0,
            fairness_weight=w_fair,
            adaptive_fairness=False,
            velocity_max=4.0,
            convergence_threshold=1e-6,
            patience=10,
        )

        optimizer2 = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target_dist,
            seed=trial_seed + 1,
        )

        result2 = optimizer2.optimize(
            fitness_fn=fitness_fn2, n_iterations=config.n_iterations, verbose=False
        )
        final_eval2 = fitness_fn2.evaluate(result2.coalition, clients)

        results.append(
            {
                "algorithm": "Standard PSO",
                "w_accuracy": w_acc,
                "w_fairness": w_fair,
                "w_cost": w_cost,
                "trial_idx": trial_idx,
                "auc_roc": final_eval2.components.get("accuracy", 0.5),
                "demographic_divergence": final_eval2.components.get(
                    "divergence", float("inf")
                ),
                "equalized_odds_gap": final_eval2.components.get(
                    "equalized_odds_gap", 0.0
                ),
            }
        )
    except Exception as e:
        logger.error(f"Standard PSO failed at weights ({w_acc}, {w_fair}): {e}")

    return results


def run_pareto_analysis(
    pareto_config: ParetoConfig, parallel: bool = True
) -> Dict[str, Any]:
    """Run full Pareto front analysis."""
    rng = np.random.default_rng(pareto_config.seed)
    total_start = time.time()

    # Build task list
    tasks: List[Dict[str, Any]] = []
    for w_acc, w_fair, w_cost in pareto_config.weight_grid:
        for trial_idx in range(pareto_config.n_trials):
            trial_seed = int(rng.integers(0, 2**31))
            tasks.append(
                {
                    "w_accuracy": w_acc,
                    "w_fairness": w_fair,
                    "w_cost": w_cost,
                    "trial_idx": trial_idx,
                    "trial_seed": trial_seed,
                    "n_clients": pareto_config.n_clients,
                    "k": pareto_config.k,
                    "coalition_fraction": pareto_config.coalition_fraction,
                    "n_fl_rounds": pareto_config.n_fl_rounds,
                    "local_epochs": pareto_config.local_epochs,
                    "learning_rate": pareto_config.learning_rate,
                    "n_iterations": pareto_config.n_iterations,
                    "swarm_size": pareto_config.swarm_size,
                    "non_iid_alpha": pareto_config.non_iid_alpha,
                    "n_samples_total": pareto_config.n_samples_total,
                    "n_features": pareto_config.n_features,
                }
            )

    all_results: List[Dict[str, Any]] = []
    n_tasks = len(tasks)

    print(
        f"\nPareto Front Analysis: {n_tasks} tasks "
        f"({len(pareto_config.weight_grid)} weight combos x {pareto_config.n_trials} trials x 2 algorithms)"
    )

    if parallel:
        n_workers = max(2, (os.cpu_count() or 4) - 2)
        print(f"Running on {n_workers} workers...")

        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_pareto_point, t): i for i, t in enumerate(tasks)
            }
            for future in as_completed(futures):
                try:
                    all_results.extend(future.result())
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                completed += 1
                if completed % 10 == 0:
                    print(
                        f"\r  Progress: {completed}/{n_tasks} ({100 * completed / n_tasks:.0f}%)",
                        end="",
                        flush=True,
                    )
        print()
    else:
        for i, task in enumerate(tasks):
            all_results.extend(_run_pareto_point(task))
            if (i + 1) % 5 == 0:
                print(f"\r  Progress: {i + 1}/{n_tasks}", end="", flush=True)
        print()

    total_time = time.time() - total_start

    # Analyze: group by (algorithm, weight_combo)
    analysis: Dict[str, Any] = {"pareto_points": {}, "dominance": {}}

    for alg in ["FairSwarm", "Standard PSO"]:
        alg_results = [r for r in all_results if r["algorithm"] == alg]
        alg_points: List[Dict[str, Any]] = []

        for w_acc, w_fair, w_cost in pareto_config.weight_grid:
            matching = [
                r
                for r in alg_results
                if abs(r["w_accuracy"] - w_acc) < 0.01
                and abs(r["w_fairness"] - w_fair) < 0.01
            ]
            if not matching:
                continue

            auc_values = [r["auc_roc"] for r in matching]
            div_values = [r["demographic_divergence"] for r in matching]
            eqodds_values = [r["equalized_odds_gap"] for r in matching]

            alg_points.append(
                {
                    "w_accuracy": w_acc,
                    "w_fairness": w_fair,
                    "w_cost": w_cost,
                    "n_trials": len(matching),
                    "auc_roc": mean_ci(auc_values).to_dict(),
                    "demographic_divergence": mean_ci(div_values).to_dict(),
                    "equalized_odds_gap": mean_ci(eqodds_values).to_dict(),
                }
            )

        analysis["pareto_points"][alg] = alg_points

    # Dominance analysis: at each weight, does FairSwarm dominate Standard PSO?
    fs_points = analysis["pareto_points"].get("FairSwarm", [])
    pso_points = analysis["pareto_points"].get("Standard PSO", [])
    dominance_count = 0
    total_comparisons = 0

    for fs_p in fs_points:
        for pso_p in pso_points:
            if abs(fs_p["w_accuracy"] - pso_p["w_accuracy"]) < 0.01:
                total_comparisons += 1
                fs_auc = fs_p["auc_roc"]["mean"]
                pso_auc = pso_p["auc_roc"]["mean"]
                fs_div = fs_p["demographic_divergence"]["mean"]
                pso_div = pso_p["demographic_divergence"]["mean"]

                # FairSwarm dominates if better on fairness and no worse on AUC
                if fs_div <= pso_div and fs_auc >= pso_auc - 0.005:
                    dominance_count += 1

    analysis["dominance"] = {
        "fairswarm_dominates": dominance_count,
        "total_comparisons": total_comparisons,
        "dominance_rate": dominance_count / total_comparisons
        if total_comparisons > 0
        else 0.0,
    }

    output = {
        "config": asdict(pareto_config),
        "analysis": analysis,
        "raw_results": all_results,
        "execution_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
    }

    # Save
    output["code_version"] = get_git_hash()
    output_path = Path(pareto_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"pareto_results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nPareto analysis complete in {total_time:.1f}s")
    print(
        f"FairSwarm dominates at {dominance_count}/{total_comparisons} weight settings"
    )
    print(f"Results saved to {filename}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto Front Analysis for FairSwarm")
    parser.add_argument(
        "--n_trials", type=int, default=10, help="Trials per weight combo"
    )
    parser.add_argument("--n_clients", type=int, default=50, help="Number of clients")
    parser.add_argument("--k", type=int, default=4, help="Demographic groups")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="results/pareto", help="Output directory"
    )
    args = parser.parse_args()

    config = ParetoConfig(
        n_clients=args.n_clients,
        k=args.k,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    run_pareto_analysis(config, parallel=args.parallel)


if __name__ == "__main__":
    main()
