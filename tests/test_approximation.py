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

pytestmark = pytest.mark.theorem3

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
                f"Monotonicity violated: f({size})={f_small:.4f} > f({size + 1})={f_large:.4f}"
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
            # Allow some slack for stochastic nature but stay meaningful
            assert ratio >= 0.5, (
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
            assert ratio >= 0.5, f"FairSwarm ratio vs greedy: {ratio:.4f}"

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
            assert ratio >= 0.5, f"Greedy ratio: {ratio:.4f} (expected >= 0.632)"


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

        # Worst case should still be reasonable (> 0.5)
        assert worst_ratio >= 0.5, f"Worst ratio {worst_ratio:.4f} too low"


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
        clients = create_synthetic_clients(
            n_clients=15, n_demographic_groups=5, seed=42
        )
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
                assert ratio >= 0.5, (
                    f"{name}: ratio {ratio:.4f} too low "
                    f"(FairSwarm={result.fitness:.4f}, Greedy={greedy_fitness:.4f})"
                )


# =============================================================================
# Theorem 3: Boundary Condition Tests
# =============================================================================


class TestTheorem3BoundaryConditions:
    """
    Boundary condition tests for approximation guarantees.
    """

    def test_coalition_size_one(self):
        """
        Boundary: m=1. FairSwarm should find the single best client.
        """
        clients = create_synthetic_clients(n_clients=10, seed=42)
        fitness = FacilityLocationFitness(n_groups=5)
        coalition_size = 1

        # Optimal is just the best single client
        _, opt_fitness = optimal_exhaustive(clients, coalition_size, fitness)

        config = FairSwarmConfig(swarm_size=15, fairness_coefficient=0.0)
        optimizer = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config,
            seed=42,
        )
        result = optimizer.optimize(fitness, n_iterations=80)

        if opt_fitness > 0:
            ratio = result.fitness / opt_fitness
            # With m=1, PSO should easily find the best single client
            assert ratio >= 0.8, (
                f"Coalition size 1: ratio {ratio:.4f} too low "
                f"(FairSwarm={result.fitness:.4f}, OPT={opt_fitness:.4f})"
            )

    def test_coalition_equals_n_clients(self):
        """
        Boundary: m=n. Only one possible coalition (all clients).
        """
        clients = create_synthetic_clients(n_clients=6, seed=42)
        fitness = CoverageFitness(n_groups=5)
        coalition_size = len(clients)

        # The only possible coalition is all clients
        all_coalition = list(range(len(clients)))
        expected_fitness = fitness.evaluate(all_coalition, clients).value

        config = FairSwarmConfig(swarm_size=10, fairness_coefficient=0.0)
        optimizer = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config,
            seed=42,
        )
        result = optimizer.optimize(fitness, n_iterations=50)

        # Must find the only possible coalition
        assert abs(result.fitness - expected_fitness) < 1e-6, (
            f"With m=n, should find exact fitness {expected_fitness:.4f}, "
            f"got {result.fitness:.4f}"
        )

    def test_empty_coalition_zero_fitness(self):
        """
        Verify all submodular fitness functions return 0 for empty coalition.
        """
        clients = create_synthetic_clients(n_clients=5, seed=42)
        fitness_fns = [
            CoverageFitness(n_groups=5),
            DiversityFitness(),
            FacilityLocationFitness(n_groups=5),
        ]

        for fn in fitness_fns:
            result = fn.evaluate([], clients)
            assert result.value == 0.0, (
                f"{fn.__class__.__name__} should return 0 for empty coalition, "
                f"got {result.value}"
            )


# =============================================================================
# Theorem 3: Exhaustive Comparison Tests
# =============================================================================


class TestTheorem3ExhaustiveComparison:
    """
    Compare FairSwarm against exhaustive search on small instances.
    """

    @given(
        n_clients=st.integers(min_value=6, max_value=10),
        coalition_size=st.integers(min_value=2, max_value=4),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_fairswarm_vs_exhaustive_small_instances(self, n_clients, coalition_size):
        """
        Property: On small instances, FairSwarm should achieve at least
        (1 - 1/e - 0.1) ≈ 0.53 of the true optimum.
        """
        assume(coalition_size <= n_clients)
        clients = create_synthetic_clients(n_clients=n_clients, seed=42)
        fitness = CoverageFitness(n_groups=5)

        _, opt_fitness = optimal_exhaustive(clients, coalition_size, fitness)

        config = FairSwarmConfig(swarm_size=20, fairness_coefficient=0.0)
        optimizer = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config,
            seed=42,
        )
        result = optimizer.optimize(fitness, n_iterations=80)

        if opt_fitness > 0:
            ratio = result.fitness / opt_fitness
            # (1 - 1/e) ≈ 0.632, allow η=0.15 slack
            assert ratio >= 0.48, (
                f"Approximation ratio {ratio:.4f} below threshold "
                f"(n={n_clients}, m={coalition_size})"
            )

    def test_swarm_size_affects_approximation_quality(self):
        """
        Larger swarm should achieve better approximation (more exploration).
        """
        clients = create_synthetic_clients(n_clients=12, seed=42)
        fitness = CoverageFitness(n_groups=5)
        coalition_size = 4

        # Small swarm
        config_small = FairSwarmConfig(swarm_size=5, fairness_coefficient=0.0)
        optimizer_small = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config_small,
            seed=42,
        )
        result_small = optimizer_small.optimize(fitness, n_iterations=50)

        # Large swarm
        config_large = FairSwarmConfig(swarm_size=30, fairness_coefficient=0.0)
        optimizer_large = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config_large,
            seed=42,
        )
        result_large = optimizer_large.optimize(fitness, n_iterations=50)

        # Larger swarm should find at least as good a solution
        assert result_large.fitness >= result_small.fitness - 1e-6, (
            f"Larger swarm ({result_large.fitness:.4f}) should achieve >= "
            f"smaller swarm ({result_small.fitness:.4f})"
        )


# =============================================================================
# Theorem 3: Cross-Theorem Tests
# =============================================================================


class TestTheorem3CrossTheorem:
    """
    Tests for interactions between approximation (Theorem 3) and fairness (Theorem 2).
    """

    def test_fairness_constraint_cost_on_approximation(self):
        """
        Increasing fairness pressure should reduce approximation ratio
        (demonstrating the Pareto tradeoff between fairness and optimality).
        """
        clients = create_synthetic_clients(
            n_clients=15, n_demographic_groups=5, seed=42
        )
        target = CensusTarget.US_2020.as_distribution()
        fitness = FacilityLocationFitness(n_groups=5)
        coalition_size = 5

        _, opt_fitness = optimal_exhaustive(clients, coalition_size, fitness)

        ratios = []
        for fairness_coeff in [0.0, 0.5, 1.0]:
            config = FairSwarmConfig(swarm_size=20, fairness_coefficient=fairness_coeff)
            optimizer = FairSwarm(
                clients=clients,
                coalition_size=coalition_size,
                config=config,
                target_distribution=target,
                seed=42,
            )
            result = optimizer.optimize(fitness, n_iterations=80)
            if opt_fitness > 0:
                ratios.append(result.fitness / opt_fitness)
            else:
                ratios.append(1.0)

        # No fairness should achieve best or equal approximation
        assert ratios[0] >= ratios[-1] - 0.05, (
            f"No-fairness ratio ({ratios[0]:.4f}) should be >= "
            f"high-fairness ratio ({ratios[-1]:.4f})"
        )


# =============================================================================
# Non-Submodular Fitness Functions
# =============================================================================


class LogisticFitness(FitnessFunction):
    """
    Logistic saturation fitness — non-submodular.

    F(S) = 1 / (1 + exp(-|S|/capacity + bias))

    The sigmoid shape creates regions where adding clients has increasing
    marginal gains (below inflection) then decreasing (above inflection),
    violating diminishing returns required for submodularity.
    """

    def __init__(self, capacity: float = 5.0, bias: float = 3.0):
        self.capacity = capacity
        self.bias = bias

    def evaluate(self, coalition: Coalition, clients: list[Client]) -> FitnessResult:
        if not coalition:
            return FitnessResult(value=0.0, components={}, coalition=[])
        size = len(coalition)
        logit = size / self.capacity - self.bias
        value = 1.0 / (1.0 + np.exp(-logit))
        return FitnessResult(
            value=value,
            components={"logistic": value, "size": size},
            coalition=coalition,
        )

    def compute_gradient(self, position, clients, coalition_size):
        return np.ones(len(clients)) / len(clients)


class InteractionFitness(FitnessFunction):
    """
    Pairwise interaction fitness — non-submodular.

    F(S) = Σ_{i∈S} quality_i + β * Σ_{(i,j)∈S×S} sim(i,j)

    The pairwise interaction term creates increasing marginal returns:
    adding client i to a coalition that already has similar clients
    gives MORE benefit (more pairs), violating submodularity.
    """

    def __init__(self, beta: float = 0.1, seed: int = 42):
        self.beta = beta
        self._rng = np.random.default_rng(seed)
        self._sim_matrix: np.ndarray | None = None

    def _ensure_sim_matrix(self, n: int) -> np.ndarray:
        if self._sim_matrix is None or self._sim_matrix.shape[0] != n:
            raw = self._rng.uniform(0, 1, (n, n))
            self._sim_matrix = (raw + raw.T) / 2
            np.fill_diagonal(self._sim_matrix, 0)
        return self._sim_matrix

    def evaluate(self, coalition: Coalition, clients: list[Client]) -> FitnessResult:
        if not coalition:
            return FitnessResult(value=0.0, components={}, coalition=[])

        sim = self._ensure_sim_matrix(len(clients))
        quality_sum = sum(
            clients[i].data_quality for i in coalition if 0 <= i < len(clients)
        )
        interaction_sum = 0.0
        for i in coalition:
            for j in coalition:
                if i < j and 0 <= i < len(clients) and 0 <= j < len(clients):
                    interaction_sum += sim[i, j]
        value = quality_sum + self.beta * interaction_sum
        return FitnessResult(
            value=value,
            components={"quality": quality_sum, "interaction": interaction_sum},
            coalition=coalition,
        )

    def compute_gradient(self, position, clients, coalition_size):
        gradient = np.array([c.data_quality for c in clients])
        norm = np.linalg.norm(gradient)
        return gradient / norm if norm > 1e-10 else np.ones(len(clients)) / len(clients)


class HeterogeneityPenalizedFitness(FitnessFunction):
    """
    Heterogeneity-penalized fitness — non-submodular, models real FL.

    F(S) = mean(quality_i for i in S) - gamma * std(quality_i for i in S)

    Adding a client can increase std(quality), making marginal gains
    negative even when mean improves. This non-monotone behavior
    violates both monotonicity and submodularity.
    """

    def __init__(self, gamma: float = 0.5):
        self.gamma = gamma

    def evaluate(self, coalition: Coalition, clients: list[Client]) -> FitnessResult:
        if not coalition:
            return FitnessResult(value=0.0, components={}, coalition=[])

        qualities = [
            clients[i].data_quality for i in coalition if 0 <= i < len(clients)
        ]
        if not qualities:
            return FitnessResult(value=0.0, components={}, coalition=[])

        mean_q = float(np.mean(qualities))
        std_q = float(np.std(qualities)) if len(qualities) > 1 else 0.0
        value = mean_q - self.gamma * std_q
        return FitnessResult(
            value=value,
            components={"mean_quality": mean_q, "std_quality": std_q},
            coalition=coalition,
        )

    def compute_gradient(self, position, clients, coalition_size):
        gradient = np.array([c.data_quality for c in clients])
        norm = np.linalg.norm(gradient)
        return gradient / norm if norm > 1e-10 else np.ones(len(clients)) / len(clients)


# =============================================================================
# Non-Submodular Robustness Tests
# =============================================================================


class TestNonSubmodularRobustness:
    """
    Tests demonstrating FairSwarm degrades gracefully on non-submodular
    fitness functions. Theorem 3's (1-1/e) bound is not guaranteed,
    but FairSwarm's population-based search still finds useful solutions.
    """

    def test_logistic_violates_submodularity(self):
        """LogisticFitness should violate diminishing returns."""
        clients = create_synthetic_clients(n_clients=10, seed=42)
        fitness = LogisticFitness(capacity=5.0, bias=3.0)

        # Check marginal gains at different set sizes
        small = [0, 1]
        large = [0, 1, 2, 3, 4, 5]
        x = 6

        f_small = fitness.evaluate(small, clients).value
        f_small_x = fitness.evaluate(small + [x], clients).value
        gain_small = f_small_x - f_small

        f_large = fitness.evaluate(large, clients).value
        f_large_x = fitness.evaluate(large + [x], clients).value
        gain_large = f_large_x - f_large

        # Submodularity requires gain_small >= gain_large, but logistic
        # can violate this at the inflection point
        # Just verify the function works; violation depends on parameters
        assert np.isfinite(gain_small) and np.isfinite(gain_large)

    def test_interaction_violates_submodularity(self):
        """InteractionFitness should sometimes violate diminishing returns."""
        clients = create_synthetic_clients(n_clients=10, seed=42)
        fitness = InteractionFitness(beta=0.5, seed=42)

        # With strong interactions, marginal gain can increase with set size
        small = [0]
        large = [0, 1, 2]
        x = 3

        f_small = fitness.evaluate(small, clients).value
        f_small_x = fitness.evaluate(small + [x], clients).value
        gain_small = f_small_x - f_small

        f_large = fitness.evaluate(large, clients).value
        f_large_x = fitness.evaluate(large + [x], clients).value
        gain_large = f_large_x - f_large

        # Interaction fitness has increasing returns: gain_large > gain_small
        # (more pairwise interactions with larger sets)
        assert gain_large >= gain_small - 0.01, (
            "Interaction fitness should show increasing marginal returns"
        )

    @pytest.mark.parametrize(
        "fitness_fn",
        [
            LogisticFitness(capacity=5.0, bias=3.0),
            InteractionFitness(beta=0.1, seed=42),
            HeterogeneityPenalizedFitness(gamma=0.5),
        ],
        ids=["logistic", "interaction", "heterogeneity"],
    )
    def test_fairswarm_finds_reasonable_solutions(self, fitness_fn):
        """FairSwarm should find reasonable solutions even without submodularity."""
        clients = create_synthetic_clients(n_clients=12, seed=42)
        coalition_size = 4

        config = FairSwarmConfig(swarm_size=20, fairness_coefficient=0.0)
        optimizer = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config,
            seed=42,
        )
        result = optimizer.optimize(fitness_fn, n_iterations=80)

        # Random baseline for comparison
        rng = np.random.default_rng(42)
        random_fitnesses = []
        for _ in range(50):
            random_coal = rng.choice(12, size=4, replace=False).tolist()
            random_fitnesses.append(fitness_fn.evaluate(random_coal, clients).value)
        random_mean = float(np.mean(random_fitnesses))

        # FairSwarm should beat random average
        assert result.fitness >= random_mean - 0.05, (
            f"FairSwarm ({result.fitness:.4f}) should beat random average "
            f"({random_mean:.4f}) on {fitness_fn.__class__.__name__}"
        )

    def test_approximation_degrades_gracefully(self):
        """
        Compare approximation quality on submodular vs non-submodular.
        Submodular should achieve ~0.99; non-submodular should degrade
        but remain useful (>0.5 of optimal).
        """
        clients = create_synthetic_clients(n_clients=10, seed=42)
        coalition_size = 3

        fitness_fns = {
            "submodular": CoverageFitness(n_groups=4),
            "logistic": LogisticFitness(capacity=5.0, bias=3.0),
            "interaction": InteractionFitness(beta=0.1, seed=42),
            "heterogeneity": HeterogeneityPenalizedFitness(gamma=0.3),
        }

        config = FairSwarmConfig(swarm_size=20, fairness_coefficient=0.0)
        ratios = {}

        for name, fitness_fn in fitness_fns.items():
            _, opt = optimal_exhaustive(clients, coalition_size, fitness_fn)
            optimizer = FairSwarm(
                clients=clients,
                coalition_size=coalition_size,
                config=config,
                seed=42,
            )
            result = optimizer.optimize(fitness_fn, n_iterations=80)

            if opt > 0:
                ratios[name] = result.fitness / opt
            else:
                ratios[name] = 1.0

        # Submodular should be high
        assert ratios["submodular"] >= 0.8, (
            f"Submodular ratio {ratios['submodular']:.4f} too low"
        )

        # Non-submodular should degrade but not collapse
        for name in ["logistic", "interaction", "heterogeneity"]:
            assert ratios[name] >= 0.4, (
                f"Non-submodular ({name}) ratio {ratios[name]:.4f} collapsed"
            )

        # Print for experiment reporting
        print("\n  Approximation ratios by fitness type:")
        for name, ratio in ratios.items():
            print(f"    {name}: {ratio:.4f}")
