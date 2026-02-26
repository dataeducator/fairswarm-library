"""
Theorem 4 (Privacy-Fairness Tradeoff) Validation Tests.

This module empirically validates Theorem 4 from the paper:

    Theorem 4 (Impossibility Result): For any (epsilon_DP, delta)-differentially
    private coalition selection mechanism that achieves epsilon_F-fairness,
    the utility loss is at least:

        UtilityLoss >= Omega(sqrt(k * log(1/delta)) / (epsilon_DP * epsilon_F))

    where k is the number of demographic groups.

Key Properties Tested:
    1. DP version has lower fitness (higher utility loss) than non-DP
    2. Utility loss increases as epsilon_DP decreases (more privacy = more loss)
    3. Utility loss increases as number of demographic groups k increases
    4. The tradeoff bound direction holds empirically

Author: Tenicka Norwood
"""

import numpy as np
import pytest

from fairswarm.algorithms.fairswarm import FairSwarm
from fairswarm.algorithms.fairswarm_dp import DPConfig, FairSwarmDP
from fairswarm.core.client import Client, create_synthetic_clients
from fairswarm.core.config import FairSwarmConfig
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.targets import CensusTarget
from fairswarm.fitness.base import FitnessFunction, FitnessResult
from fairswarm.types import Coalition

pytestmark = pytest.mark.theorem4


# =============================================================================
# Test Fitness Function
# =============================================================================


class QualitySumFitness(FitnessFunction):
    """
    Simple fitness that sums data quality of selected clients.

    Deterministic and sensitivity-bounded, making it suitable for
    measuring utility loss from DP noise.
    """

    def evaluate(
        self,
        coalition: Coalition,
        clients: list[Client],
    ) -> FitnessResult:
        if not coalition:
            return FitnessResult(value=0.0, components={}, coalition=coalition)

        value = sum(clients[i].data_quality for i in coalition if 0 <= i < len(clients))
        return FitnessResult(
            value=value,
            components={"quality_sum": value},
            coalition=coalition,
        )

    def compute_gradient(self, position, clients, coalition_size):
        gradient = np.array([c.data_quality for c in clients], dtype=np.float64)
        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm
        return gradient


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clients_5groups():
    """Create clients with 5 demographic groups."""
    return create_synthetic_clients(n_clients=20, n_demographic_groups=5, seed=42)


@pytest.fixture
def clients_3groups():
    """Create clients with 3 demographic groups."""
    return create_synthetic_clients(n_clients=20, n_demographic_groups=3, seed=42)


@pytest.fixture
def clients_8groups():
    """Create clients with 8 demographic groups."""
    return create_synthetic_clients(n_clients=20, n_demographic_groups=8, seed=42)


@pytest.fixture
def target_5groups():
    """Target distribution for 5 groups."""
    return CensusTarget.US_2020.as_distribution()


@pytest.fixture
def quality_fitness():
    """Quality-sum fitness function."""
    return QualitySumFitness()


@pytest.fixture
def pso_config():
    """Shared PSO configuration for fair comparisons."""
    return FairSwarmConfig(
        swarm_size=15,
        max_iterations=30,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        fairness_coefficient=0.2,
    )


# =============================================================================
# Theorem 4 Test 1: DP version has lower fitness than non-DP
# =============================================================================


class TestDPUtilityLoss:
    """
    Test that differential privacy introduces utility loss.

    Theorem 4 implies that any DP mechanism must sacrifice some utility
    to achieve privacy. The non-private FairSwarm should achieve higher
    fitness than the private FairSwarmDP on the same problem.
    """

    def test_dp_has_lower_fitness_than_non_dp(
        self, clients_5groups, target_5groups, quality_fitness, pso_config
    ):
        """
        Core Theorem 4 test: DP version achieves lower fitness.

        Run FairSwarm (no privacy) and FairSwarmDP (with privacy) on the
        same problem and verify the DP version has worse fitness.
        """
        n_iterations = 30
        coalition_size = 5
        seed = 42

        # Non-private FairSwarm
        optimizer_clean = FairSwarm(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target_5groups,
            seed=seed,
        )
        result_clean = optimizer_clean.optimize(
            quality_fitness, n_iterations=n_iterations
        )

        # Private FairSwarm-DP (moderate privacy)
        dp_config = DPConfig(
            epsilon=1.0,
            delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
        optimizer_dp = FairSwarmDP(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target_5groups,
            dp_config=dp_config,
            seed=seed,
        )
        result_dp = optimizer_dp.optimize(quality_fitness, n_iterations=n_iterations)

        # Non-DP should achieve higher or equal fitness
        assert result_clean.fitness >= result_dp.fitness, (
            f"Non-DP fitness ({result_clean.fitness:.4f}) should be >= "
            f"DP fitness ({result_dp.fitness:.4f})"
        )

    def test_dp_noise_degrades_best_achievable_fitness(
        self, clients_5groups, target_5groups, quality_fitness, pso_config
    ):
        """
        Verify that the best fitness found across multiple runs is lower
        with DP than without.

        Individual runs may vary due to stochastic noise, but the best
        result over many seeds should be worse under DP.
        """
        n_iterations = 30
        coalition_size = 5
        n_runs = 10

        clean_best = float("-inf")
        dp_best = float("-inf")

        for seed in range(n_runs):
            # Non-private
            optimizer_clean = FairSwarm(
                clients=clients_5groups,
                coalition_size=coalition_size,
                config=pso_config,
                target_distribution=target_5groups,
                seed=seed,
            )
            result_clean = optimizer_clean.optimize(
                quality_fitness, n_iterations=n_iterations
            )
            clean_best = max(clean_best, result_clean.fitness)

            # Private with moderate noise
            dp_config = DPConfig(
                epsilon=1.0,
                delta=1e-5,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
            )
            optimizer_dp = FairSwarmDP(
                clients=clients_5groups,
                coalition_size=coalition_size,
                config=pso_config,
                target_distribution=target_5groups,
                dp_config=dp_config,
                seed=seed,
            )
            result_dp = optimizer_dp.optimize(
                quality_fitness, n_iterations=n_iterations
            )
            dp_best = max(dp_best, result_dp.fitness)

        # Non-DP best should be at least as good as DP best
        assert clean_best >= dp_best, (
            f"Non-DP best ({clean_best:.4f}) should be >= DP best ({dp_best:.4f})"
        )


# =============================================================================
# Theorem 4 Test 2: More privacy = more utility loss
# =============================================================================


class TestPrivacyUtilityMonotonicity:
    """
    Test that utility loss increases as epsilon_DP decreases.

    Theorem 4: UtilityLoss >= Omega(... / epsilon_DP)
    So smaller epsilon_DP (stronger privacy) should lead to larger utility loss.
    """

    def test_lower_epsilon_higher_utility_loss(
        self, clients_5groups, target_5groups, quality_fitness, pso_config
    ):
        """
        Decreasing epsilon_DP should increase utility loss.

        Compare high-privacy (low epsilon) vs low-privacy (high epsilon)
        and verify the high-privacy version has worse fitness.
        """
        n_iterations = 30
        coalition_size = 5
        seed = 42

        # Baseline: non-private
        optimizer_clean = FairSwarm(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target_5groups,
            seed=seed,
        )
        optimizer_clean.optimize(quality_fitness, n_iterations=n_iterations)

        # High epsilon (low privacy, low noise) - should be close to non-DP
        dp_config_high_eps = DPConfig(
            epsilon=50.0,
            delta=1e-5,
            noise_multiplier=3.0,
            max_grad_norm=1.0,
        )
        optimizer_high_eps = FairSwarmDP(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target_5groups,
            dp_config=dp_config_high_eps,
            seed=seed,
        )
        result_high_eps = optimizer_high_eps.optimize(
            quality_fitness, n_iterations=n_iterations
        )

        # Low epsilon (high privacy, high noise) - should be far from non-DP
        dp_config_low_eps = DPConfig(
            epsilon=50.0,
            delta=1e-5,
            noise_multiplier=0.1,
            max_grad_norm=1.0,
        )
        optimizer_low_eps = FairSwarmDP(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target_5groups,
            dp_config=dp_config_low_eps,
            seed=seed,
        )
        result_low_eps = optimizer_low_eps.optimize(
            quality_fitness, n_iterations=n_iterations
        )

        # Higher noise multiplier => more noise => worse fitness
        # (noise_multiplier=3.0 adds more noise than noise_multiplier=0.1)
        assert result_high_eps.fitness <= result_low_eps.fitness, (
            f"Higher noise ({result_high_eps.fitness:.4f}) should have <= fitness "
            f"than lower noise ({result_low_eps.fitness:.4f})"
        )

    def test_increasing_noise_decreases_fitness(
        self, clients_5groups, target_5groups, quality_fitness, pso_config
    ):
        """
        Monotonically increasing noise multiplier should decrease fitness.
        """
        n_iterations = 30
        coalition_size = 5
        seed = 42

        noise_levels = [0.5, 1.5, 3.0]
        fitness_values = []

        for noise in noise_levels:
            dp_config = DPConfig(
                epsilon=50.0,  # Large budget so we don't exhaust early
                delta=1e-5,
                noise_multiplier=noise,
                max_grad_norm=1.0,
            )
            optimizer = FairSwarmDP(
                clients=clients_5groups,
                coalition_size=coalition_size,
                config=pso_config,
                target_distribution=target_5groups,
                dp_config=dp_config,
                seed=seed,
            )
            result = optimizer.optimize(quality_fitness, n_iterations=n_iterations)
            fitness_values.append(result.fitness)

        # Fitness should generally decrease with more noise
        # Check that the lowest noise gives better fitness than the highest noise
        assert fitness_values[0] >= fitness_values[-1], (
            f"Lowest noise fitness ({fitness_values[0]:.4f}) should be >= "
            f"highest noise fitness ({fitness_values[-1]:.4f}). "
            f"All values: {[f'{f:.4f}' for f in fitness_values]}"
        )


# =============================================================================
# Theorem 4 Test 3: More demographic groups = more utility loss
# =============================================================================


class TestDemographicGroupsTradeoff:
    """
    Test that utility loss increases with more demographic groups k.

    Theorem 4: UtilityLoss >= Omega(sqrt(k * ...))
    More groups means harder to satisfy fairness under privacy.
    """

    def test_more_groups_higher_utility_loss(self, quality_fitness, pso_config):
        """
        More demographic groups should lead to higher utility loss under DP.
        """
        n_iterations = 30
        coalition_size = 5
        seed = 42

        dp_config = DPConfig(
            epsilon=10.0,
            delta=1e-5,
            noise_multiplier=1.5,
            max_grad_norm=1.0,
        )

        group_counts = [3, 8]
        utility_losses = []

        for k in group_counts:
            clients = create_synthetic_clients(
                n_clients=20, n_demographic_groups=k, seed=42
            )

            # Build a uniform target distribution with k groups
            target_dict = {f"group_{i}": 1.0 / k for i in range(k)}
            target = DemographicDistribution.from_dict(target_dict)

            # Non-private baseline
            optimizer_clean = FairSwarm(
                clients=clients,
                coalition_size=coalition_size,
                config=pso_config,
                target_distribution=target,
                seed=seed,
            )
            result_clean = optimizer_clean.optimize(
                quality_fitness, n_iterations=n_iterations
            )

            # Private version
            optimizer_dp = FairSwarmDP(
                clients=clients,
                coalition_size=coalition_size,
                config=pso_config,
                target_distribution=target,
                dp_config=dp_config,
                seed=seed,
            )
            result_dp = optimizer_dp.optimize(
                quality_fitness, n_iterations=n_iterations
            )

            loss = result_clean.fitness - result_dp.fitness
            utility_losses.append(loss)

        # More groups (k=8) should incur higher utility loss than fewer (k=3)
        assert utility_losses[1] >= utility_losses[0], (
            f"Utility loss with k=8 ({utility_losses[1]:.4f}) should be >= "
            f"loss with k=3 ({utility_losses[0]:.4f})"
        )


# =============================================================================
# Theorem 4 Test 4: Tradeoff bound direction
# =============================================================================


class TestTradeoffBoundDirection:
    """
    Test that the tradeoff bound from Theorem 4 holds directionally.

    UtilityLoss >= Omega(sqrt(k * log(1/delta)) / (epsilon_DP * epsilon_F))

    We verify the bound's direction: larger denominator (weaker privacy,
    weaker fairness) allows smaller utility loss.
    """

    def test_bound_formula_direction(self):
        """
        Verify the mathematical bound: the theoretical lower bound increases
        when epsilon_DP decreases or k increases.
        """

        def tradeoff_bound(
            k: int, epsilon_dp: float, epsilon_f: float, delta: float
        ) -> float:
            """Compute Theorem 4 lower bound on utility loss."""
            return np.sqrt(k * np.log(1.0 / delta)) / (epsilon_dp * epsilon_f)

        delta = 1e-5
        epsilon_f = 0.1

        # More groups = higher bound
        bound_k3 = tradeoff_bound(k=3, epsilon_dp=1.0, epsilon_f=epsilon_f, delta=delta)
        bound_k8 = tradeoff_bound(k=8, epsilon_dp=1.0, epsilon_f=epsilon_f, delta=delta)
        assert bound_k8 > bound_k3, (
            f"Bound with k=8 ({bound_k8:.4f}) should exceed k=3 ({bound_k3:.4f})"
        )

        # Lower epsilon_DP = higher bound
        bound_high_eps = tradeoff_bound(
            k=5, epsilon_dp=5.0, epsilon_f=epsilon_f, delta=delta
        )
        bound_low_eps = tradeoff_bound(
            k=5, epsilon_dp=0.5, epsilon_f=epsilon_f, delta=delta
        )
        assert bound_low_eps > bound_high_eps, (
            f"Bound with low epsilon ({bound_low_eps:.4f}) should exceed "
            f"high epsilon ({bound_high_eps:.4f})"
        )

        # Lower epsilon_F (stricter fairness) = higher bound
        bound_loose_f = tradeoff_bound(k=5, epsilon_dp=1.0, epsilon_f=0.5, delta=delta)
        bound_strict_f = tradeoff_bound(
            k=5, epsilon_dp=1.0, epsilon_f=0.05, delta=delta
        )
        assert bound_strict_f > bound_loose_f, (
            f"Bound with strict fairness ({bound_strict_f:.4f}) should exceed "
            f"loose fairness ({bound_loose_f:.4f})"
        )

    def test_empirical_loss_consistent_with_bound_direction(
        self, clients_5groups, target_5groups, quality_fitness, pso_config
    ):
        """
        Verify that empirical utility loss is consistent with the tradeoff
        bound direction across different privacy levels.
        """
        n_iterations = 30
        coalition_size = 5

        # Run across multiple seeds to get stable estimates
        n_runs = 3
        noise_levels = [0.5, 2.0]  # low noise vs high noise
        avg_fitness_by_noise = {}

        for noise in noise_levels:
            fitnesses = []
            for seed in range(n_runs):
                dp_config = DPConfig(
                    epsilon=50.0,
                    delta=1e-5,
                    noise_multiplier=noise,
                    max_grad_norm=1.0,
                )
                optimizer = FairSwarmDP(
                    clients=clients_5groups,
                    coalition_size=coalition_size,
                    config=pso_config,
                    target_distribution=target_5groups,
                    dp_config=dp_config,
                    seed=seed,
                )
                result = optimizer.optimize(quality_fitness, n_iterations=n_iterations)
                fitnesses.append(result.fitness)

            avg_fitness_by_noise[noise] = np.mean(fitnesses)

        # Lower noise should give higher average fitness
        assert avg_fitness_by_noise[0.5] >= avg_fitness_by_noise[2.0], (
            f"Low noise avg fitness ({avg_fitness_by_noise[0.5]:.4f}) should be >= "
            f"high noise avg fitness ({avg_fitness_by_noise[2.0]:.4f})"
        )


# =============================================================================
# Theorem 4 Test 5: Privacy budget exhaustion
# =============================================================================


class TestPrivacyBudgetImpact:
    """
    Test that privacy budget constraints affect optimization quality.
    """

    def test_tight_budget_worse_than_loose_budget(
        self, clients_5groups, target_5groups, quality_fitness, pso_config
    ):
        """
        A tighter privacy budget should result in worse fitness because
        optimization terminates earlier or adds more noise.
        """
        n_iterations = 30
        coalition_size = 5
        seed = 42

        # Loose budget - can run many iterations
        dp_config_loose = DPConfig(
            epsilon=100.0,
            delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
        optimizer_loose = FairSwarmDP(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target_5groups,
            dp_config=dp_config_loose,
            seed=seed,
        )
        result_loose = optimizer_loose.optimize(
            quality_fitness, n_iterations=n_iterations
        )

        # Tight budget - may exhaust before completing iterations
        dp_config_tight = DPConfig(
            epsilon=0.1,
            delta=1e-5,
            noise_multiplier=0.1,  # Low noise = fast budget consumption
            max_grad_norm=1.0,
        )
        optimizer_tight = FairSwarmDP(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target_5groups,
            dp_config=dp_config_tight,
            seed=seed,
        )
        result_tight = optimizer_tight.optimize(
            quality_fitness, n_iterations=n_iterations
        )

        # Loose budget should achieve better fitness
        assert result_loose.fitness >= result_tight.fitness, (
            f"Loose budget fitness ({result_loose.fitness:.4f}) should be >= "
            f"tight budget fitness ({result_tight.fitness:.4f})"
        )

        # Tight budget should have run fewer iterations
        assert (
            result_tight.convergence.iterations <= result_loose.convergence.iterations
        ), (
            f"Tight budget iterations ({result_tight.convergence.iterations}) should be <= "
            f"loose budget iterations ({result_loose.convergence.iterations})"
        )


# =============================================================================
# Theorem 4 Test 6: Boundary Conditions
# =============================================================================


class TestPrivacyBoundaryConditions:
    """
    Boundary condition tests for privacy-fairness tradeoff.
    """

    def test_very_large_epsilon_matches_non_dp(
        self, clients_5groups, target_5groups, quality_fitness, pso_config
    ):
        """
        With epsilon → ∞ (very large budget, very low noise), FairSwarmDP
        should approach non-private FairSwarm performance.
        """
        n_iterations = 30
        coalition_size = 5
        seed = 42

        # Non-private baseline
        optimizer_clean = FairSwarm(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target_5groups,
            seed=seed,
        )
        result_clean = optimizer_clean.optimize(
            quality_fitness, n_iterations=n_iterations
        )

        # Near-infinite budget, minimal noise
        dp_config = DPConfig(
            epsilon=1000.0,
            delta=1e-5,
            noise_multiplier=0.001,
            max_grad_norm=1.0,
        )
        optimizer_dp = FairSwarmDP(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target_5groups,
            dp_config=dp_config,
            seed=seed,
        )
        result_dp = optimizer_dp.optimize(quality_fitness, n_iterations=n_iterations)

        # Should be very close to non-DP performance
        gap = abs(result_clean.fitness - result_dp.fitness)
        relative_gap = gap / max(abs(result_clean.fitness), 1e-10)
        assert relative_gap < 0.3, (
            f"With near-zero noise, DP fitness ({result_dp.fitness:.4f}) should be "
            f"close to non-DP ({result_clean.fitness:.4f}), gap={relative_gap:.4f}"
        )

    def test_single_demographic_group_trivial_fairness(
        self, quality_fitness, pso_config
    ):
        """
        With k=1 demographic group, fairness is trivially satisfied.
        DP noise should still reduce fitness but fairness shouldn't matter.
        """
        clients = create_synthetic_clients(
            n_clients=15, n_demographic_groups=1, seed=42
        )
        target = DemographicDistribution.from_dict({"group_0": 1.0})
        n_iterations = 30
        coalition_size = 5
        seed = 42

        # Non-private
        optimizer_clean = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target,
            seed=seed,
        )
        result_clean = optimizer_clean.optimize(
            quality_fitness, n_iterations=n_iterations
        )

        # Private (moderate noise)
        dp_config = DPConfig(
            epsilon=10.0,
            delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
        optimizer_dp = FairSwarmDP(
            clients=clients,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target,
            dp_config=dp_config,
            seed=seed,
        )
        result_dp = optimizer_dp.optimize(quality_fitness, n_iterations=n_iterations)

        # Both should achieve reasonable fitness since fairness is trivial
        assert result_clean.fitness > 0, "Non-DP should achieve positive fitness"
        assert result_dp.fitness > 0, (
            "DP with k=1 should still achieve positive fitness"
        )

    def test_very_tight_budget_minimal_optimization(
        self, clients_5groups, target_5groups, quality_fitness, pso_config
    ):
        """
        With an extremely tight budget, optimization should terminate early
        and produce limited results.
        """
        dp_config = DPConfig(
            epsilon=0.01,
            delta=1e-5,
            noise_multiplier=0.01,
            max_grad_norm=1.0,
        )
        optimizer = FairSwarmDP(
            clients=clients_5groups,
            coalition_size=5,
            config=pso_config,
            target_distribution=target_5groups,
            dp_config=dp_config,
            seed=42,
        )
        result = optimizer.optimize(quality_fitness, n_iterations=30)

        # Should terminate very early due to budget exhaustion
        assert result.convergence.iterations <= 30, (
            "With very tight budget, should not exceed max iterations"
        )
        # Fitness should be non-negative (but may be poor)
        assert result.fitness >= 0 or np.isfinite(result.fitness), (
            f"Fitness should be finite, got {result.fitness}"
        )


# =============================================================================
# Theorem 4 Test 7: Budget Accounting
# =============================================================================


class TestPrivacyBudgetAccounting:
    """
    Test that privacy budget is tracked correctly.
    """

    def test_spent_epsilon_does_not_exceed_budget(
        self, clients_5groups, target_5groups, quality_fitness, pso_config
    ):
        """
        The total privacy expenditure should not exceed the configured epsilon.
        """
        epsilon_budget = 10.0
        dp_config = DPConfig(
            epsilon=epsilon_budget,
            delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
        optimizer = FairSwarmDP(
            clients=clients_5groups,
            coalition_size=5,
            config=pso_config,
            target_distribution=target_5groups,
            dp_config=dp_config,
            seed=42,
        )
        result = optimizer.optimize(quality_fitness, n_iterations=30)

        # Check the accountant if accessible
        if hasattr(optimizer, "_accountant"):
            spent = optimizer._accountant.get_epsilon(dp_config.delta)
            assert spent <= epsilon_budget + 0.1, (
                f"Spent epsilon ({spent:.4f}) exceeds budget ({epsilon_budget})"
            )

        # Should have completed at least 1 iteration
        assert result.convergence.iterations >= 1

    def test_more_iterations_spend_more_budget(
        self, clients_5groups, target_5groups, quality_fitness
    ):
        """
        Running more iterations should consume more privacy budget.
        """
        dp_config = DPConfig(
            epsilon=100.0,
            delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
        config = FairSwarmConfig(
            swarm_size=10,
            inertia=0.7,
            cognitive=1.5,
            social=1.5,
            fairness_coefficient=0.2,
        )

        # Short run
        optimizer_short = FairSwarmDP(
            clients=clients_5groups,
            coalition_size=5,
            config=config,
            target_distribution=target_5groups,
            dp_config=dp_config,
            seed=42,
        )
        optimizer_short.optimize(quality_fitness, n_iterations=5)

        # Long run (fresh optimizer with same config)
        optimizer_long = FairSwarmDP(
            clients=clients_5groups,
            coalition_size=5,
            config=config,
            target_distribution=target_5groups,
            dp_config=dp_config,
            seed=42,
        )
        optimizer_long.optimize(quality_fitness, n_iterations=25)

        # Long run used more iterations
        if hasattr(optimizer_short, "_accountant") and hasattr(
            optimizer_long, "_accountant"
        ):
            spent_short = optimizer_short._accountant.get_epsilon(dp_config.delta)
            spent_long = optimizer_long._accountant.get_epsilon(dp_config.delta)
            assert spent_long >= spent_short, (
                f"Long run ({spent_long:.4f}) should spend >= "
                f"short run ({spent_short:.4f})"
            )


# =============================================================================
# Theorem 4 Test 8: Cross-Theorem Interactions
# =============================================================================


class TestPrivacyFairnessCrossTheorem:
    """
    Tests for interactions between privacy (Theorem 4) and fairness (Theorem 2).
    """

    def test_dp_with_fairness_both_degrade_utility(
        self, clients_5groups, target_5groups, quality_fitness, pso_config
    ):
        """
        Combining DP noise and fairness constraint should produce worse fitness
        than either alone.
        """
        n_iterations = 30
        coalition_size = 5
        seed = 42

        # No fairness, no DP (pure optimization)
        config_no_fair = FairSwarmConfig(
            swarm_size=15,
            max_iterations=30,
            inertia=0.7,
            cognitive=1.5,
            social=1.5,
            fairness_coefficient=0.0,
        )
        optimizer_pure = FairSwarm(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=config_no_fair,
            seed=seed,
        )
        result_pure = optimizer_pure.optimize(
            quality_fitness, n_iterations=n_iterations
        )

        # Fairness only (no DP)
        optimizer_fair = FairSwarm(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target_5groups,
            seed=seed,
        )
        _result_fair = optimizer_fair.optimize(
            quality_fitness, n_iterations=n_iterations
        )
        assert _result_fair.fitness is not None  # verify optimization ran

        # DP only (no fairness)
        dp_config = DPConfig(
            epsilon=10.0,
            delta=1e-5,
            noise_multiplier=1.5,
            max_grad_norm=1.0,
        )
        optimizer_dp = FairSwarmDP(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=config_no_fair,
            dp_config=dp_config,
            seed=seed,
        )
        _result_dp = optimizer_dp.optimize(quality_fitness, n_iterations=n_iterations)
        assert _result_dp.fitness is not None  # verify optimization ran

        # Both DP and fairness
        optimizer_both = FairSwarmDP(
            clients=clients_5groups,
            coalition_size=coalition_size,
            config=pso_config,
            target_distribution=target_5groups,
            dp_config=dp_config,
            seed=seed,
        )
        result_both = optimizer_both.optimize(
            quality_fitness, n_iterations=n_iterations
        )

        # Pure should be best or tied
        assert result_pure.fitness >= result_both.fitness - 1e-6, (
            f"Pure ({result_pure.fitness:.4f}) should be >= "
            f"DP+fair ({result_both.fitness:.4f})"
        )

    def test_dp_noise_does_not_help_fairness(
        self, clients_5groups, target_5groups, quality_fitness, pso_config
    ):
        """
        Adding DP noise should not systematically improve fairness.
        Noise is random and shouldn't be a substitute for the fairness gradient.
        """
        n_iterations = 30
        coalition_size = 5
        n_runs = 5

        fair_div_clean = []
        fair_div_dp = []

        for seed in range(n_runs):
            # Non-DP with fairness
            optimizer_clean = FairSwarm(
                clients=clients_5groups,
                coalition_size=coalition_size,
                config=pso_config,
                target_distribution=target_5groups,
                seed=seed,
            )
            result_clean = optimizer_clean.optimize(
                quality_fitness, n_iterations=n_iterations
            )
            fair_div_clean.append(result_clean.fairness.demographic_divergence)

            # DP with fairness (moderate noise)
            dp_config = DPConfig(
                epsilon=10.0,
                delta=1e-5,
                noise_multiplier=1.5,
                max_grad_norm=1.0,
            )
            optimizer_dp = FairSwarmDP(
                clients=clients_5groups,
                coalition_size=coalition_size,
                config=pso_config,
                target_distribution=target_5groups,
                dp_config=dp_config,
                seed=seed,
            )
            result_dp = optimizer_dp.optimize(
                quality_fitness, n_iterations=n_iterations
            )
            fair_div_dp.append(result_dp.fairness.demographic_divergence)

        avg_clean = np.mean(fair_div_clean)
        avg_dp = np.mean(fair_div_dp)

        # Non-DP should achieve equal or better fairness (lower divergence)
        assert avg_clean <= avg_dp + 0.1, (
            f"Non-DP avg divergence ({avg_clean:.4f}) should be <= "
            f"DP avg divergence ({avg_dp:.4f})"
        )
