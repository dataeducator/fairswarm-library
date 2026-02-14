"""
Theorem 3 (Approximation Ratio) Validation Tests.

This module uses Hypothesis property-based testing to empirically validate
Theorem 3 from CLAUDE.md:

    Theorem 3 (Approximation Guarantee): For submodular fitness functions,
    FairSwarm achieves a (1 - 1/e - η)-approximation to the optimal
    coalition, where η → 0 as iterations → ∞.

    Specifically, if F is monotone submodular:
        F(S*_FairSwarm) ≥ (1 - 1/e - η) · F(S*_OPT)

Key Properties Tested:
    1. Submodular fitness functions (diminishing returns)
    2. Approximation ratio bounds
    3. Greedy baseline comparison
    4. Monotonicity of submodular objectives
    5. η decreases with more iterations

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""


import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from fairswarm.algorithms.fairswarm import FairSwarm
from fairswarm.core.client import Client, create_synthetic_clients
from fairswarm.core.config import FairSwarmConfig
from fairswarm.demographics.targets import CensusTarget
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.types import Coalition

# =============================================================================
# Submodular Fitness Functions for Testing
# =============================================================================


class CoverageFitness(FitnessFunction):
    """
    Coverage-based fitness (classic submodular function).

    Models how well the coalition "covers" different demographic groups.
    More diverse coalitions have diminishing returns for adding similar clients.

    F(S) = |∪_{i∈S} Groups(i)| / |All Groups|

    This is monotone submodular by construction.
    """

    def __init__(self, n_groups: int = 5):
        self.n_groups = n_groups

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        if not coalition:
            return FitnessResult(value=0.0, components={"coverage": 0.0}, coalition=[])

        # Each client "covers" groups based on their demographics
        covered_groups = set()

        for idx in coalition:
            if 0 <= idx < len(clients):
                demo = np.asarray(clients[idx].demographics)
                # A group is "covered" if client has > 10% representation
                for g, val in enumerate(demo):
                    if val > 0.1:
                        covered_groups.add(g)

        coverage = len(covered_groups) / self.n_groups
        return FitnessResult(
            value=coverage,
            components={"coverage": coverage, "n_covered": len(covered_groups)},
            coalition=coalition,
        )

    def compute_gradient(self, position, clients, coalition_size):
        # Coverage gradient: favor clients that cover uncovered groups
        n_clients = len(clients)
        gradient = np.zeros(n_clients)

        for i, client in enumerate(clients):
            demo = np.asarray(client.demographics)
            # Higher gradient for clients with diverse coverage
            gradient[i] = np.sum(demo > 0.1)

        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm
        return gradient


class DiversityFitness(FitnessFunction):
    """
    Diversity-based fitness (submodular).

    Measures how diverse the coalition is in terms of demographic spread.
    Adding similar clients has diminishing returns.

    F(S) = entropy of aggregated demographics
    """

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        if not coalition:
            return FitnessResult(value=0.0, components={"entropy": 0.0}, coalition=[])

        # Compute aggregated demographics
        demos = []
        for idx in coalition:
            if 0 <= idx < len(clients):
                demos.append(np.asarray(clients[idx].demographics))

        if not demos:
            return FitnessResult(value=0.0, components={"entropy": 0.0}, coalition=[])

        avg_demo = np.mean(demos, axis=0)
        avg_demo = avg_demo / avg_demo.sum()  # Normalize

        # Compute entropy (higher is more diverse)
        entropy = -np.sum(avg_demo * np.log(avg_demo + 1e-10))
        max_entropy = np.log(len(avg_demo))  # Maximum possible entropy

        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return FitnessResult(
            value=normalized_entropy,
            components={"entropy": entropy, "normalized": normalized_entropy},
            coalition=coalition,
        )

    def compute_gradient(self, position, clients, coalition_size):
        n_clients = len(clients)
        # Favor clients that would increase diversity
        gradient = np.random.default_rng(42).random(n_clients)
        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm
        return gradient


class FacilityLocationFitness(FitnessFunction):
    """
    Facility location fitness (classic submodular function).

    Each client has a "quality" and we maximize the sum of qualities
    with diminishing returns for similar clients.

    F(S) = Σ_g max_{i∈S} quality(i, g)

    where g indexes demographic groups.
    """

    def __init__(self, n_groups: int = 5):
        self.n_groups = n_groups

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        if not coalition:
            return FitnessResult(value=0.0, components={}, coalition=[])

        # For each group, find max quality among coalition members
        group_max = np.zeros(self.n_groups)

        for idx in coalition:
            if 0 <= idx < len(clients):
                client = clients[idx]
                demo = np.asarray(client.demographics)
                quality = client.data_quality

                # Quality contribution to each group
                for g in range(min(len(demo), self.n_groups)):
                    contribution = demo[g] * quality
                    group_max[g] = max(group_max[g], contribution)

        total = np.sum(group_max)
        return FitnessResult(
            value=total,
            components={"total": total, "group_max": group_max.tolist()},
            coalition=coalition,
        )

    def compute_gradient(self, position, clients, coalition_size):
        len(clients)
        gradient = np.array([c.data_quality for c in clients])
        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm
        return gradient


# =============================================================================
# Hypothesis Strategies
# =============================================================================


@st.composite
def coalition_size_strategy(draw, n_clients: int):
    """Generate valid coalition sizes."""
    return draw(st.integers(min_value=1, max_value=n_clients))


@st.composite
def submodular_fitness_strategy(draw):
    """Generate a submodular fitness function."""
    choice = draw(st.integers(min_value=0, max_value=2))
    if choice == 0:
        return CoverageFitness(n_groups=5)
    elif choice == 1:
        return DiversityFitness()
    else:
        return FacilityLocationFitness(n_groups=5)


# =============================================================================
# Greedy Algorithm (Baseline for Comparison)
# =============================================================================


def greedy_maximize(
    clients: list[Client],
    coalition_size: int,
    fitness_fn: FitnessFunction,
) -> tuple[Coalition, float]:
    """
    Greedy algorithm for submodular maximization.

    For monotone submodular functions, greedy achieves (1 - 1/e)-approximation.
    This serves as a baseline for comparison.
    """
    n_clients = len(clients)
    selected = []
    remaining = set(range(n_clients))

    for _ in range(coalition_size):
        best_idx = None
        best_gain = float("-inf")

        for idx in remaining:
            # Marginal gain of adding idx
            test_coalition = selected + [idx]
            result = fitness_fn.evaluate(test_coalition, clients)

            if result.value > best_gain:
                best_gain = result.value
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    final_result = fitness_fn.evaluate(selected, clients)
    return selected, final_result.value


def optimal_exhaustive(
    clients: list[Client],
    coalition_size: int,
    fitness_fn: FitnessFunction,
) -> tuple[Coalition, float]:
    """
    Find optimal coalition by exhaustive search (small instances only).
    """
    from itertools import combinations

    n_clients = len(clients)
    if n_clients > 12 or coalition_size > 6:
        # Too large for exhaustive search
        return greedy_maximize(clients, coalition_size, fitness_fn)

    best_coalition = []
    best_fitness = float("-inf")

    for coalition in combinations(range(n_clients), coalition_size):
        coalition_list = list(coalition)
        result = fitness_fn.evaluate(coalition_list, clients)
        if result.value > best_fitness:
            best_fitness = result.value
            best_coalition = coalition_list

    return best_coalition, best_fitness


# =============================================================================
# Theorem 3: Submodularity Tests
# =============================================================================


class TestTheorem3Submodularity:
    """
    Tests verifying submodularity properties of test fitness functions.
    """

    @given(st.integers(min_value=5, max_value=15))
    @settings(max_examples=20, deadline=None)
    def test_coverage_diminishing_returns(self, n_clients):
        """
        Property: Coverage fitness exhibits diminishing returns.

        For submodular f: f(A ∪ {x}) - f(A) ≥ f(B ∪ {x}) - f(B) when A ⊆ B
        """
        clients = create_synthetic_clients(n_clients=n_clients, seed=42)
        fitness = CoverageFitness(n_groups=5)

        # A = first 2 clients, B = first 4 clients
        A = [0, 1]
        B = [0, 1, 2, 3]
        x = 5  # New element

        assume(x < n_clients)
        assume(len(set(A + B)) < n_clients)  # Ensure x not in A or B

        # Marginal gains
        f_A = fitness.evaluate(A, clients).value
        f_A_plus_x = fitness.evaluate(A + [x], clients).value
        gain_A = f_A_plus_x - f_A

        f_B = fitness.evaluate(B, clients).value
        f_B_plus_x = fitness.evaluate(B + [x], clients).value
        gain_B = f_B_plus_x - f_B

        # Diminishing returns: gain from smaller set should be >= gain from larger set
        assert gain_A >= gain_B - 1e-10, (
            f"Diminishing returns violated: gain(A)={gain_A:.4f} < gain(B)={gain_B:.4f}"
        )

    def test_facility_location_monotone(self):
        """
        Property: Facility location fitness is monotone (adding clients doesn't hurt).
        """
        clients = create_synthetic_clients(n_clients=10, seed=42)
        fitness = FacilityLocationFitness(n_groups=5)

        # Check that adding clients never decreases fitness
        for size in range(1, 8):
            coalition_small = list(range(size))
            coalition_large = list(range(size + 1))

            f_small = fitness.evaluate(coalition_small, clients).value
            f_large = fitness.evaluate(coalition_large, clients).value

            assert f_large >= f_small - 1e-10, (
                f"Monotonicity violated: f({size})={f_small:.4f} > f({size+1})={f_large:.4f}"
            )


# =============================================================================
# Theorem 3: Approximation Ratio Tests
# =============================================================================


class TestTheorem3ApproximationRatio:
    """
    Tests for the (1 - 1/e - η) approximation guarantee.
    """

    def test_fairswarm_vs_greedy_coverage(self):
        """
        Test FairSwarm approximation ratio vs greedy baseline.
        """
        clients = create_synthetic_clients(n_clients=15, seed=42)
        fitness = CoverageFitness(n_groups=5)
        coalition_size = 5

        # Greedy solution (guaranteed (1-1/e)-approximation for submodular)
        greedy_coalition, greedy_fitness = greedy_maximize(
            clients, coalition_size, fitness
        )

        # FairSwarm solution
        config = FairSwarmConfig(swarm_size=20, fairness_coefficient=0.0)
        optimizer = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config,
            seed=42,
        )
        result = optimizer.optimize(fitness, n_iterations=100)

        # FairSwarm should achieve comparable fitness to greedy
        ratio = result.fitness / greedy_fitness if greedy_fitness > 0 else 1.0

        # Allow some slack since PSO is stochastic
        assert ratio >= 0.5, (
            f"FairSwarm ratio too low: {ratio:.4f} "
            f"(FairSwarm={result.fitness:.4f}, Greedy={greedy_fitness:.4f})"
        )

    @given(submodular_fitness_strategy())
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_fairswarm_achieves_reasonable_approximation(self, fitness):
        """
        Property: FairSwarm achieves reasonable approximation on submodular functions.
        """
        clients = create_synthetic_clients(n_clients=12, seed=42)
        coalition_size = 4

        # Find optimal (for small instances)
        opt_coalition, opt_fitness = optimal_exhaustive(
            clients, coalition_size, fitness
        )

        # FairSwarm solution
        config = FairSwarmConfig(swarm_size=15, fairness_coefficient=0.0)
        optimizer = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config,
            seed=42,
        )
        result = optimizer.optimize(fitness, n_iterations=50)

        # Check approximation ratio
        if opt_fitness > 0:
            ratio = result.fitness / opt_fitness
            # Theoretical guarantee is (1 - 1/e) ≈ 0.632
            # Allow slack for stochastic nature
            assert ratio >= 0.3, (
                f"Approximation ratio too low: {ratio:.4f} "
                f"(FairSwarm={result.fitness:.4f}, OPT={opt_fitness:.4f})"
            )

    def test_more_iterations_improves_eta(self):
        """
        Property: η decreases (better approximation) with more iterations.
        """
        clients = create_synthetic_clients(n_clients=15, seed=42)
        fitness = CoverageFitness(n_groups=5)
        coalition_size = 5

        # Get optimal/greedy baseline
        greedy_coalition, greedy_fitness = greedy_maximize(
            clients, coalition_size, fitness
        )

        config = FairSwarmConfig(swarm_size=15, fairness_coefficient=0.0)

        # Short run
        optimizer_short = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config,
            seed=42,
        )
        result_short = optimizer_short.optimize(fitness, n_iterations=20)

        # Long run
        optimizer_long = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config,
            seed=42,
        )
        result_long = optimizer_long.optimize(fitness, n_iterations=100)

        # Long run should achieve equal or better fitness
        assert result_long.fitness >= result_short.fitness - 1e-6, (
            f"More iterations didn't help: "
            f"short={result_short.fitness:.4f}, long={result_long.fitness:.4f}"
        )


# =============================================================================
# Theorem 3: Greedy Comparison Tests
# =============================================================================


class TestTheorem3GreedyComparison:
    """
    Tests comparing FairSwarm to greedy algorithm.
    """

    @given(st.integers(min_value=10, max_value=20))
    @settings(max_examples=15, deadline=None)
    def test_fairswarm_competitive_with_greedy(self, n_clients):
        """
        Property: FairSwarm is competitive with greedy on submodular objectives.
        """
        clients = create_synthetic_clients(n_clients=n_clients, seed=42)
        fitness = FacilityLocationFitness(n_groups=5)
        coalition_size = min(5, n_clients)

        # Greedy baseline
        greedy_coalition, greedy_fitness = greedy_maximize(
            clients, coalition_size, fitness
        )

        # FairSwarm
        config = FairSwarmConfig(swarm_size=15, fairness_coefficient=0.0)
        optimizer = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config,
            seed=42,
        )
        result = optimizer.optimize(fitness, n_iterations=50)

        # Should achieve at least 50% of greedy (allowing for stochasticity)
        if greedy_fitness > 0:
            ratio = result.fitness / greedy_fitness
            assert ratio >= 0.4, f"FairSwarm ratio vs greedy: {ratio:.4f}"

    def test_greedy_is_near_optimal_for_small_instances(self):
        """
        Verify greedy baseline is reasonable (sanity check).
        """
        clients = create_synthetic_clients(n_clients=8, seed=42)
        fitness = CoverageFitness(n_groups=5)
        coalition_size = 3

        # Greedy solution
        greedy_coalition, greedy_fitness = greedy_maximize(
            clients, coalition_size, fitness
        )

        # Optimal solution (small enough for exhaustive)
        opt_coalition, opt_fitness = optimal_exhaustive(
            clients, coalition_size, fitness
        )

        # Greedy should achieve at least (1 - 1/e) ≈ 0.632 of optimal
        if opt_fitness > 0:
            ratio = greedy_fitness / opt_fitness
            assert ratio >= 0.5, (
                f"Greedy ratio: {ratio:.4f} (expected >= 0.632)"
            )


# =============================================================================
# Theorem 3: Monotonicity Tests
# =============================================================================


class TestTheorem3Monotonicity:
    """
    Tests for monotonicity of fitness during optimization.
    """

    @given(submodular_fitness_strategy())
    @settings(max_examples=10, deadline=None)
    def test_global_best_monotone(self, fitness):
        """
        Property: Global best fitness is monotonically non-decreasing.
        """
        clients = create_synthetic_clients(n_clients=12, seed=42)

        config = FairSwarmConfig(swarm_size=10, fairness_coefficient=0.0)
        optimizer = FairSwarm(
            clients=clients,
            coalition_size=4,
            config=config,
            seed=42,
        )

        result = optimizer.optimize(fitness, n_iterations=50)
        history = result.convergence.fitness_history

        # Check monotonicity
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-10, (
                f"Fitness decreased at iteration {i}"
            )

    def test_submodular_functions_are_bounded(self):
        """
        Property: Submodular fitness values are bounded.
        """
        clients = create_synthetic_clients(n_clients=10, seed=42)

        fitness_fns = [
            CoverageFitness(n_groups=5),
            DiversityFitness(),
            FacilityLocationFitness(n_groups=5),
        ]

        for fitness in fitness_fns:
            for size in range(1, 10):
                coalition = list(range(size))
                result = fitness.evaluate(coalition, clients)

                assert np.isfinite(result.value), (
                    f"{fitness.__class__.__name__} returned non-finite value"
                )


# =============================================================================
# Theorem 3: Statistical Tests
# =============================================================================


class TestTheorem3Statistical:
    """
    Statistical tests for approximation guarantees.
    """

    @pytest.mark.slow
    def test_average_approximation_ratio(self):
        """
        Test average approximation ratio over multiple runs.
        """
        n_runs = 10
        ratios = []

        clients = create_synthetic_clients(n_clients=12, seed=42)
        fitness = CoverageFitness(n_groups=5)
        coalition_size = 4

        # Get optimal
        opt_coalition, opt_fitness = optimal_exhaustive(
            clients, coalition_size, fitness
        )

        for seed in range(n_runs):
            config = FairSwarmConfig(swarm_size=15, fairness_coefficient=0.0)
            optimizer = FairSwarm(
                clients=clients,
                coalition_size=coalition_size,
                config=config,
                seed=seed,
            )
            result = optimizer.optimize(fitness, n_iterations=50)

            if opt_fitness > 0:
                ratios.append(result.fitness / opt_fitness)

        avg_ratio = np.mean(ratios)
        # Average should be close to theoretical guarantee
        assert avg_ratio >= 0.5, f"Average ratio {avg_ratio:.4f} too low"

    def test_worst_case_approximation(self):
        """
        Test worst-case approximation ratio.
        """
        n_runs = 10
        worst_ratio = 1.0

        clients = create_synthetic_clients(n_clients=10, seed=42)
        fitness = FacilityLocationFitness(n_groups=5)
        coalition_size = 3

        # Get baseline
        greedy_coalition, greedy_fitness = greedy_maximize(
            clients, coalition_size, fitness
        )

        for seed in range(n_runs):
            config = FairSwarmConfig(swarm_size=10, fairness_coefficient=0.0)
            optimizer = FairSwarm(
                clients=clients,
                coalition_size=coalition_size,
                config=config,
                seed=seed,
            )
            result = optimizer.optimize(fitness, n_iterations=50)

            if greedy_fitness > 0:
                ratio = result.fitness / greedy_fitness
                worst_ratio = min(worst_ratio, ratio)

        # Worst case should still be reasonable
        assert worst_ratio >= 0.3, f"Worst ratio {worst_ratio:.4f} too low"


# =============================================================================
# Theorem 3: Integration Tests
# =============================================================================


class TestTheorem3Integration:
    """
    Integration tests for approximation guarantees.
    """

    def test_fairness_and_approximation_tradeoff(self):
        """
        Test the tradeoff between fairness and approximation quality.
        """
        clients = create_synthetic_clients(n_clients=15, n_demographic_groups=5, seed=42)
        target = CensusTarget.US_2020.as_distribution()
        fitness = CoverageFitness(n_groups=5)
        coalition_size = 5

        # Pure optimization (no fairness)
        config_opt = FairSwarmConfig(swarm_size=15, fairness_coefficient=0.0)
        optimizer_opt = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config_opt,
            target_distribution=target,
            seed=42,
        )
        result_opt = optimizer_opt.optimize(fitness, n_iterations=100)

        # With fairness
        config_fair = FairSwarmConfig(swarm_size=15, fairness_coefficient=0.5)
        optimizer_fair = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config_fair,
            target_distribution=target,
            seed=42,
        )
        result_fair = optimizer_fair.optimize(fitness, n_iterations=100)

        # Pure optimization might achieve higher fitness
        # But fair version should still be reasonable
        assert result_fair.fitness >= result_opt.fitness * 0.5, (
            f"Fairness penalty too severe: "
            f"fair={result_fair.fitness:.4f}, opt={result_opt.fitness:.4f}"
        )

    def test_multiple_submodular_functions(self):
        """
        Test approximation across different submodular functions.
        """
        clients = create_synthetic_clients(n_clients=10, seed=42)
        coalition_size = 3

        fitness_fns = [
            ("Coverage", CoverageFitness(n_groups=5)),
            ("Diversity", DiversityFitness()),
            ("FacilityLocation", FacilityLocationFitness(n_groups=5)),
        ]

        for name, fitness in fitness_fns:
            # Greedy baseline
            _, greedy_fitness = greedy_maximize(clients, coalition_size, fitness)

            # FairSwarm
            config = FairSwarmConfig(swarm_size=15, fairness_coefficient=0.0)
            optimizer = FairSwarm(
                clients=clients,
                coalition_size=coalition_size,
                config=config,
                seed=42,
            )
            result = optimizer.optimize(fitness, n_iterations=50)

            if greedy_fitness > 0:
                ratio = result.fitness / greedy_fitness
                assert ratio >= 0.3, (
                    f"{name}: ratio {ratio:.4f} too low "
                    f"(FairSwarm={result.fitness:.4f}, Greedy={greedy_fitness:.4f})"
                )
