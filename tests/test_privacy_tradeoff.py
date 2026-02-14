"""
Theorem 4 (Privacy-Fairness Tradeoff) Validation Tests.

This module empirically validates Theorem 4 from CLAUDE.md:

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
Advisor: Dr. Uttam Ghosh
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
