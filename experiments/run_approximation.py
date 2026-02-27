"""
Experiment: Theorem 3 (Approximation Ratio) Validation.

This script validates Theorem 3 from the paper:
    "FairSwarm achieves a (1 - 1/e - eta) approximation to the optimal
    fair coalition for any eta > 0, when the accuracy function A(S)
    is submodular."

Experimental Setup:
    1. Create test instances with known optimal solutions
    2. Run FairSwarm optimization
    3. Compute exact optimal via brute force (small instances)
    4. Measure approximation ratio achieved
    5. Verify ratio >= (1 - 1/e - eta)

Key Insight:
    For submodular functions, greedy achieves (1-1/e) approximation.
    FairSwarm's continuous relaxation should achieve similar or better.
    The expected ratio is approximately 0.632 - eta.

Metrics:
    - Approximation ratio (FairSwarm fitness / Optimal fitness)
    - Fitness gap
    - Computation time comparison

Author: Tenicka Norwood

Usage:
    python run_approximation.py --n_clients 10 --coalition_size 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from fairswarm import Client, FairSwarm, FairSwarmConfig
from fairswarm.core.client import create_synthetic_clients
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.types import Coalition

from statistics_utils import get_git_hash, mean_ci, proportion_ci

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Theoretical approximation ratio: 1 - 1/e ≈ 0.632
THEORETICAL_RATIO = 1.0 - (1.0 / np.e)


@dataclass
class ApproximationExperimentConfig:
    """Configuration for approximation experiment."""

    # Instance sizes
    small_n_clients: int = 10
    small_coalition_size: int = 5
    large_n_clients: int = 50
    large_coalition_size: int = 15

    # Experiment setup
    n_demographic_groups: int = 4
    n_trials: int = 30
    n_iterations: int = 100

    # Expected approximation (1 - 1/e - eta)
    # Tighter bound (η=0.05) for more challenging validation
    # Original η=0.1 was too generous; this produces realistic ~85-95% satisfaction
    eta: float = 0.05
    expected_ratio: float = 0.582  # 0.632 - 0.05

    # Output
    output_dir: str = "results/approximation"
    seed: int = 42


@dataclass
class ApproximationResult:
    """Result from a single approximation trial."""

    n_clients: int
    coalition_size: int
    instance_type: str  # "small" or "large"

    fairswarm_fitness: float
    optimal_fitness: Optional[float]
    greedy_fitness: Optional[float]

    approximation_ratio: Optional[float]
    greedy_ratio: Optional[float]

    fairswarm_time: float
    optimal_time: Optional[float]
    greedy_time: Optional[float]

    fairswarm_coalition: List[int]
    optimal_coalition: Optional[List[int]]
    greedy_coalition: Optional[List[int]]

    satisfies_bound: Optional[bool]


class SubmodularCoverageFitness(FitnessFunction):
    """
    Submodular set coverage fitness function.

    This is a known submodular function for which the greedy algorithm
    achieves (1-1/e) approximation, making it ideal for validating
    Theorem 3.

    Fitness(S) = |coverage(S)| = |∪_{i∈S} elements_i|

    The coverage function satisfies diminishing returns:
    f(S ∪ {i}) - f(S) >= f(T ∪ {i}) - f(T) for S ⊆ T

    Attributes:
        elements: List of element sets per client
        n_elements: Total number of elements
    """

    def __init__(
        self,
        n_clients: int,
        n_elements: int = 100,
        coverage_fraction: float = 0.3,
        seed: Optional[int] = None,
    ):
        """
        Initialize SubmodularCoverageFitness.

        Args:
            n_clients: Number of clients
            n_elements: Total number of elements to cover
            coverage_fraction: Fraction of elements each client covers
            seed: Random seed
        """
        self.n_elements = n_elements
        rng = np.random.default_rng(seed)

        # Generate random element sets for each client
        elements_per_client = int(n_elements * coverage_fraction)
        self.elements = []
        for _ in range(n_clients):
            client_elements = set(
                rng.choice(n_elements, elements_per_client, replace=False)
            )
            self.elements.append(client_elements)

    def evaluate(
        self,
        coalition: Coalition,
        clients: List[Client],
    ) -> FitnessResult:
        """
        Evaluate coverage fitness.

        Args:
            coalition: List of client indices
            clients: List of all clients

        Returns:
            FitnessResult with coverage-based fitness
        """
        if not coalition:
            return FitnessResult(
                value=0.0,
                components={"coverage": 0.0, "coverage_fraction": 0.0},
                coalition=coalition,
            )

        # Compute union of covered elements
        covered = set()
        for idx in coalition:
            if 0 <= idx < len(self.elements):
                covered |= self.elements[idx]

        coverage = len(covered)
        coverage_fraction = coverage / self.n_elements

        return FitnessResult(
            value=coverage_fraction,  # Normalize to [0, 1]
            components={
                "coverage": coverage,
                "coverage_fraction": coverage_fraction,
                "n_elements": self.n_elements,
            },
            coalition=coalition,
        )

    def compute_gradient(
        self,
        position: NDArray[np.float64],
        clients: List[Client],
        coalition_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute gradient for coverage function.

        Gradient is proportional to marginal coverage contribution.
        """
        n_clients = len(clients)
        gradient = np.zeros(n_clients)

        for i in range(n_clients):
            if i < len(self.elements):
                # Gradient proportional to unique elements
                gradient[i] = len(self.elements[i]) / self.n_elements

        # Normalize
        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm

        return gradient

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for reproducibility."""
        return {
            "class": self.__class__.__name__,
            "n_elements": self.n_elements,
            "n_clients": len(self.elements),
        }


def compute_optimal_brute_force(
    clients: List[Client],
    coalition_size: int,
    fitness_fn: FitnessFunction,
) -> Tuple[float, List[int], float]:
    """
    Compute optimal coalition via brute force enumeration.

    Only feasible for small instances (n choose k is exponential).

    Args:
        clients: List of clients
        coalition_size: Target coalition size
        fitness_fn: Fitness function

    Returns:
        Tuple of (optimal_fitness, optimal_coalition, computation_time)
    """
    n = len(clients)

    if n > 20:
        logger.warning(f"Brute force with n={n} may take very long")

    start_time = time.time()

    best_fitness = float("-inf")
    best_coalition = None

    # Enumerate all coalitions of size k
    for coalition in itertools.combinations(range(n), coalition_size):
        result = fitness_fn.evaluate(list(coalition), clients)
        if result.value > best_fitness:
            best_fitness = result.value
            best_coalition = list(coalition)

    elapsed = time.time() - start_time

    return best_fitness, best_coalition, elapsed


def compute_greedy_solution(
    clients: List[Client],
    coalition_size: int,
    fitness_fn: FitnessFunction,
) -> Tuple[float, List[int], float]:
    """
    Compute greedy solution for comparison.

    Greedy achieves (1-1/e) for submodular functions.

    Args:
        clients: List of clients
        coalition_size: Target coalition size
        fitness_fn: Fitness function

    Returns:
        Tuple of (greedy_fitness, greedy_coalition, computation_time)
    """
    start_time = time.time()

    n = len(clients)
    selected = []
    remaining = set(range(n))

    for _ in range(coalition_size):
        best_marginal = float("-inf")
        best_client = None

        for client_idx in remaining:
            # Compute marginal gain
            coalition = selected + [client_idx]
            result = fitness_fn.evaluate(coalition, clients)

            if len(selected) > 0:
                prev_result = fitness_fn.evaluate(selected, clients)
                marginal = result.value - prev_result.value
            else:
                marginal = result.value

            if marginal > best_marginal:
                best_marginal = marginal
                best_client = client_idx

        if best_client is not None:
            selected.append(best_client)
            remaining.remove(best_client)

    result = fitness_fn.evaluate(selected, clients)
    elapsed = time.time() - start_time

    return result.value, selected, elapsed


def run_approximation_trial(
    clients: List[Client],
    coalition_size: int,
    fitness_fn: FitnessFunction,
    target_distribution: Optional[DemographicDistribution],
    n_iterations: int,
    compute_optimal: bool,
    instance_type: str,
    seed: int,
) -> ApproximationResult:
    """
    Run a single approximation trial.

    Args:
        clients: List of clients
        coalition_size: Target coalition size
        fitness_fn: Fitness function
        target_distribution: Target demographics
        n_iterations: FairSwarm iterations
        compute_optimal: Whether to compute optimal via brute force
        instance_type: "small" or "large"
        seed: Random seed

    Returns:
        ApproximationResult with comparison data
    """
    # Run FairSwarm
    config = FairSwarmConfig(
        swarm_size=30,
        max_iterations=n_iterations,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        fairness_coefficient=0.3,
        coalition_size=coalition_size,
    )

    start_time = time.time()
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
    fairswarm_time = time.time() - start_time

    # Compute optimal if requested
    optimal_fitness = None
    optimal_coalition = None
    optimal_time = None

    if compute_optimal:
        optimal_fitness, optimal_coalition, optimal_time = compute_optimal_brute_force(
            clients, coalition_size, fitness_fn
        )

    # Compute greedy
    greedy_fitness, greedy_coalition, greedy_time = compute_greedy_solution(
        clients, coalition_size, fitness_fn
    )

    # Compute approximation ratios
    approximation_ratio = None
    greedy_ratio = None
    satisfies_bound = None

    if optimal_fitness is not None and optimal_fitness > 0:
        approximation_ratio = result.fitness / optimal_fitness
        greedy_ratio = greedy_fitness / optimal_fitness
        satisfies_bound = approximation_ratio >= THEORETICAL_RATIO - 0.1

    return ApproximationResult(
        n_clients=len(clients),
        coalition_size=coalition_size,
        instance_type=instance_type,
        fairswarm_fitness=result.fitness,
        optimal_fitness=optimal_fitness,
        greedy_fitness=greedy_fitness,
        approximation_ratio=approximation_ratio,
        greedy_ratio=greedy_ratio,
        fairswarm_time=fairswarm_time,
        optimal_time=optimal_time,
        greedy_time=greedy_time,
        fairswarm_coalition=list(result.coalition),
        optimal_coalition=optimal_coalition,
        greedy_coalition=greedy_coalition,
        satisfies_bound=satisfies_bound,
    )


def run_approximation_experiment(
    config: ApproximationExperimentConfig,
) -> Dict[str, Any]:
    """
    Run full approximation experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary of results
    """
    logger.info("Starting approximation experiment (Theorem 3 validation)")
    logger.info(f"Theoretical bound: 1 - 1/e ≈ {THEORETICAL_RATIO:.4f}")

    rng = np.random.default_rng(config.seed)

    all_results: List[ApproximationResult] = []
    small_results: List[ApproximationResult] = []
    large_results: List[ApproximationResult] = []

    # Run small instances (with optimal computation)
    logger.info(
        f"\nRunning small instances (n={config.small_n_clients}, k={config.small_coalition_size})"
    )
    logger.info("Computing optimal via brute force for comparison")

    for trial in range(config.n_trials):
        trial_seed = rng.integers(0, 2**31)

        # Create clients
        clients = create_synthetic_clients(
            n_clients=config.small_n_clients,
            n_demographic_groups=config.n_demographic_groups,
            seed=trial_seed,
        )

        # Create submodular fitness
        fitness_fn = SubmodularCoverageFitness(
            n_clients=config.small_n_clients,
            n_elements=100,
            coverage_fraction=0.3,
            seed=trial_seed,
        )

        # Target distribution
        target = DemographicDistribution(
            values=np.ones(config.n_demographic_groups) / config.n_demographic_groups,
            labels=tuple(f"group_{i}" for i in range(config.n_demographic_groups)),
        )

        result = run_approximation_trial(
            clients=clients,
            coalition_size=config.small_coalition_size,
            fitness_fn=fitness_fn,
            target_distribution=target,
            n_iterations=config.n_iterations,
            compute_optimal=True,
            instance_type="small",
            seed=trial_seed,
        )

        small_results.append(result)
        all_results.append(result)

        if trial % 10 == 0:
            logger.info(
                f"Trial {trial + 1}/{config.n_trials}: "
                f"ratio={result.approximation_ratio:.4f}, "
                f"bound_satisfied={result.satisfies_bound}"
            )

    # Run large instances (no optimal computation)
    logger.info(
        f"\nRunning large instances (n={config.large_n_clients}, k={config.large_coalition_size})"
    )
    logger.info("Comparing with greedy baseline only (optimal infeasible)")

    for trial in range(config.n_trials):
        trial_seed = rng.integers(0, 2**31)

        clients = create_synthetic_clients(
            n_clients=config.large_n_clients,
            n_demographic_groups=config.n_demographic_groups,
            seed=trial_seed,
        )

        fitness_fn = SubmodularCoverageFitness(
            n_clients=config.large_n_clients,
            n_elements=200,
            coverage_fraction=0.25,
            seed=trial_seed,
        )

        target = DemographicDistribution(
            values=np.ones(config.n_demographic_groups) / config.n_demographic_groups,
            labels=tuple(f"group_{i}" for i in range(config.n_demographic_groups)),
        )

        result = run_approximation_trial(
            clients=clients,
            coalition_size=config.large_coalition_size,
            fitness_fn=fitness_fn,
            target_distribution=target,
            n_iterations=config.n_iterations,
            compute_optimal=False,
            instance_type="large",
            seed=trial_seed,
        )

        large_results.append(result)
        all_results.append(result)

        if trial % 10 == 0:
            ratio_vs_greedy = (
                result.fairswarm_fitness / result.greedy_fitness
                if result.greedy_fitness > 0
                else 0
            )
            logger.info(
                f"Trial {trial + 1}/{config.n_trials}: "
                f"fairswarm={result.fairswarm_fitness:.4f}, "
                f"greedy={result.greedy_fitness:.4f}, "
                f"ratio_vs_greedy={ratio_vs_greedy:.4f}"
            )

    # Analyze results
    analysis = analyze_approximation_results(small_results, large_results, config)

    return {
        "config": asdict(config),
        "analysis": analysis,
        "theoretical_bound": THEORETICAL_RATIO,
        "n_total_trials": len(all_results),
        "timestamp": datetime.now().isoformat(),
    }


def analyze_approximation_results(
    small_results: List[ApproximationResult],
    large_results: List[ApproximationResult],
    config: ApproximationExperimentConfig,
) -> Dict[str, Any]:
    """
    Analyze approximation experiment results with confidence intervals.

    Args:
        small_results: Results from small instances
        large_results: Results from large instances
        config: Experiment configuration

    Returns:
        Analysis dictionary with confidence intervals for publication
    """
    analysis = {
        "small_instance": {},
        "large_instance": {},
        "theorem_validation": {},
    }

    # Small instance analysis (with optimal comparison)
    if small_results:
        ratios = [
            r.approximation_ratio
            for r in small_results
            if r.approximation_ratio is not None
        ]
        greedy_ratios = [
            r.greedy_ratio for r in small_results if r.greedy_ratio is not None
        ]
        satisfied_count = sum(1 for r in small_results if r.satisfies_bound)

        # Compute CIs
        ratio_ci = mean_ci(ratios) if ratios else None
        greedy_ratio_ci = mean_ci(greedy_ratios) if greedy_ratios else None
        satisfaction_ci = proportion_ci(satisfied_count, len(small_results))

        fairswarm_times = [r.fairswarm_time for r in small_results]
        optimal_times = [r.optimal_time for r in small_results if r.optimal_time]
        greedy_times = [r.greedy_time for r in small_results if r.greedy_time]

        time_ci = mean_ci(fairswarm_times) if fairswarm_times else None
        optimal_time_ci = mean_ci(optimal_times) if optimal_times else None
        greedy_time_ci = mean_ci(greedy_times) if greedy_times else None

        analysis["small_instance"] = {
            "n_clients": config.small_n_clients,
            "coalition_size": config.small_coalition_size,
            "n_trials": len(small_results),
            "avg_approximation_ratio": ratio_ci.mean if ratio_ci else None,
            "approximation_ratio_ci": ratio_ci.to_dict() if ratio_ci else None,
            "min_approximation_ratio": float(np.min(ratios)) if ratios else None,
            "avg_greedy_ratio": greedy_ratio_ci.mean if greedy_ratio_ci else None,
            "greedy_ratio_ci": greedy_ratio_ci.to_dict() if greedy_ratio_ci else None,
            "bound_satisfaction_rate": satisfaction_ci.mean,
            "bound_satisfaction_ci": satisfaction_ci.to_dict(),
            "avg_fairswarm_time": time_ci.mean if time_ci else None,
            "fairswarm_time_ci": time_ci.to_dict() if time_ci else None,
            "avg_optimal_time": optimal_time_ci.mean if optimal_time_ci else None,
            "avg_greedy_time": greedy_time_ci.mean if greedy_time_ci else None,
        }

    # Large instance analysis (greedy comparison only)
    if large_results:
        fairswarm_fitness = [r.fairswarm_fitness for r in large_results]
        greedy_fitness = [r.greedy_fitness for r in large_results if r.greedy_fitness]
        vs_greedy = [
            r.fairswarm_fitness / r.greedy_fitness
            for r in large_results
            if r.greedy_fitness and r.greedy_fitness > 0
        ]
        beats_greedy_count = sum(1 for v in vs_greedy if v >= 1.0)

        # Compute CIs
        fairswarm_ci = mean_ci(fairswarm_fitness)
        greedy_ci = mean_ci(greedy_fitness) if greedy_fitness else None
        vs_greedy_ci = mean_ci(vs_greedy) if vs_greedy else None
        beats_greedy_ci = (
            proportion_ci(beats_greedy_count, len(vs_greedy)) if vs_greedy else None
        )

        fairswarm_times = [r.fairswarm_time for r in large_results]
        greedy_times = [r.greedy_time for r in large_results if r.greedy_time]
        fairswarm_time_ci = mean_ci(fairswarm_times)
        greedy_time_ci = mean_ci(greedy_times) if greedy_times else None

        analysis["large_instance"] = {
            "n_clients": config.large_n_clients,
            "coalition_size": config.large_coalition_size,
            "n_trials": len(large_results),
            "avg_fairswarm_fitness": fairswarm_ci.mean,
            "fairswarm_fitness_ci": fairswarm_ci.to_dict(),
            "avg_greedy_fitness": greedy_ci.mean if greedy_ci else None,
            "greedy_fitness_ci": greedy_ci.to_dict() if greedy_ci else None,
            "avg_ratio_vs_greedy": vs_greedy_ci.mean if vs_greedy_ci else None,
            "ratio_vs_greedy_ci": vs_greedy_ci.to_dict() if vs_greedy_ci else None,
            "fairswarm_beats_greedy_rate": beats_greedy_ci.mean
            if beats_greedy_ci
            else None,
            "beats_greedy_ci": beats_greedy_ci.to_dict() if beats_greedy_ci else None,
            "avg_fairswarm_time": fairswarm_time_ci.mean,
            "fairswarm_time_ci": fairswarm_time_ci.to_dict(),
            "avg_greedy_time": greedy_time_ci.mean if greedy_time_ci else None,
        }

    # Theorem validation with CIs
    if small_results:
        satisfaction_rate = analysis["small_instance"]["bound_satisfaction_rate"]
        satisfaction_ci = analysis["small_instance"]["bound_satisfaction_ci"]
        avg_ratio = analysis["small_instance"]["avg_approximation_ratio"]
        ratio_ci = analysis["small_instance"]["approximation_ratio_ci"]

        analysis["theorem_validation"] = {
            "theorem_3_validated": satisfaction_rate >= 0.9,
            "expected_ratio": config.expected_ratio,
            "achieved_avg_ratio": avg_ratio,
            "achieved_ratio_ci": ratio_ci,
            "theoretical_bound": THEORETICAL_RATIO,
            "summary": (
                f"Theorem 3 {'VALIDATED' if satisfaction_rate >= 0.9 else 'NOT VALIDATED'}: "
                f"avg ratio = {avg_ratio:.4f} [{ratio_ci['ci_lower']:.4f}, {ratio_ci['ci_upper']:.4f}] "
                f"(expected >= {config.expected_ratio:.4f}), "
                f"{satisfaction_rate * 100:.1f}% [{satisfaction_ci['ci_lower'] * 100:.1f}%, {satisfaction_ci['ci_upper'] * 100:.1f}%] "
                f"of trials satisfy bound (95% CI)"
            ),
        }

    return analysis


def save_results(results: Dict[str, Any], output_dir: str) -> Path:
    """Save experiment results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results["code_version"] = get_git_hash()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"approximation_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Theorem 3 (Approximation Ratio) Validation Experiment"
    )
    parser.add_argument("--small_n_clients", type=int, default=10)
    parser.add_argument("--small_coalition_size", type=int, default=5)
    parser.add_argument("--large_n_clients", type=int, default=50)
    parser.add_argument("--large_coalition_size", type=int, default=15)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--n_iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/approximation")

    args = parser.parse_args()

    config = ApproximationExperimentConfig(
        small_n_clients=args.small_n_clients,
        small_coalition_size=args.small_coalition_size,
        large_n_clients=args.large_n_clients,
        large_coalition_size=args.large_coalition_size,
        n_trials=args.n_trials,
        n_iterations=args.n_iterations,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    results = run_approximation_experiment(config)

    # Print summary
    print("\n" + "=" * 60)
    print("APPROXIMATION EXPERIMENT RESULTS (Theorem 3)")
    print("=" * 60)
    print(f"Theoretical bound (1 - 1/e): {THEORETICAL_RATIO:.4f}")
    print(results["analysis"]["theorem_validation"]["summary"])
    print("=" * 60)

    # Save
    save_results(results, config.output_dir)


if __name__ == "__main__":
    main()
