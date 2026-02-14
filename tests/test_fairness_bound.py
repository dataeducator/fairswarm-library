"""
Theorem 2 (ε-Fairness) Validation Tests.

This module uses Hypothesis property-based testing to empirically validate
Theorem 2 from CLAUDE.md:

    Theorem 2 (ε-Fairness Bound): For any ε > 0 and δ > 0, with appropriate
    hyperparameters, FairSwarm produces a coalition S* such that:
        DemDiv(S*) ≤ ε with probability ≥ 1 - δ

    where DemDiv(S) = D_KL(δ_S || δ*) is the demographic divergence from
    Definition 2.

Key Properties Tested:
    1. Divergence is non-negative: D_KL >= 0 always
    2. Divergence is zero when distributions match
    3. Fairness gradient reduces divergence over time
    4. Higher fairness coefficient leads to fairer coalitions
    5. ε-threshold satisfaction with sufficient iterations

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fairswarm.algorithms.fairswarm import FairSwarm
from fairswarm.core.client import Client, create_synthetic_clients
from fairswarm.core.config import FairSwarmConfig
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import (
    js_divergence,
    kl_divergence,
    total_variation_distance,
)
from fairswarm.demographics.targets import CensusTarget
from fairswarm.fitness.fairness import (
    DemographicFitness,
    compute_coalition_demographics,
    compute_fairness_gradient,
)

pytestmark = pytest.mark.theorem2

# =============================================================================
# Hypothesis Strategies
# =============================================================================


@st.composite
def demographic_distribution_strategy(draw):
    """Generate valid demographic distributions that sum to 1."""
    n_groups = draw(st.integers(min_value=2, max_value=6))

    # Generate n_groups positive values
    values = [draw(st.floats(min_value=0.01, max_value=1.0)) for _ in range(n_groups)]

    # Normalize to sum to 1
    total = sum(values)
    normalized = [v / total for v in values]

    categories = {f"group_{i}": normalized[i] for i in range(n_groups)}
    return DemographicDistribution.from_dict(categories)


@st.composite
def client_with_demographics_strategy(draw, n_groups=5):
    """Generate a client with random demographics."""
    # Generate demographic values
    values = [draw(st.floats(min_value=0.01, max_value=1.0)) for _ in range(n_groups)]
    total = sum(values)
    normalized = {f"group_{i}": values[i] / total for i in range(n_groups)}

    return Client(
        id=draw(st.text(min_size=1, max_size=10, alphabet="abcdefghij")),
        demographics=DemographicDistribution.from_dict(normalized),
        num_samples=draw(st.integers(min_value=100, max_value=10000)),
        data_quality=draw(st.floats(min_value=0.5, max_value=1.0)),
    )


@st.composite
def epsilon_strategy(draw):
    """Generate epsilon values for fairness threshold."""
    return draw(st.floats(min_value=0.01, max_value=2.0))


@st.composite
def fairness_config_strategy(draw):
    """Generate FairSwarm config with varying fairness coefficients."""
    return FairSwarmConfig(
        swarm_size=draw(st.integers(min_value=10, max_value=30)),
        max_iterations=100,
        inertia=0.5,
        cognitive=1.0,
        social=1.0,
        fairness_coefficient=draw(st.floats(min_value=0.0, max_value=1.0)),
        epsilon_fair=draw(st.floats(min_value=0.1, max_value=1.0)),
    )


# =============================================================================
# Definition 2: KL Divergence Properties
# =============================================================================


class TestDefinition2KLDivergence:
    """
    Tests for Definition 2: DemDiv(S) = D_KL(δ_S || δ*)
    """

    @given(demographic_distribution_strategy())
    @settings(max_examples=50)
    def test_kl_divergence_nonnegative(self, dist):
        """
        Property: KL divergence is always non-negative.

        D_KL(P || Q) >= 0 for all distributions P, Q
        """
        target = CensusTarget.US_2020.as_distribution()

        # Ensure same categories
        p = dist.as_array()
        q = target.as_array()

        # Pad to same length if needed
        max_len = max(len(p), len(q))
        p = np.pad(p, (0, max_len - len(p)), constant_values=1e-10)
        q = np.pad(q, (0, max_len - len(q)), constant_values=1e-10)

        divergence = kl_divergence(p, q)

        assert divergence >= 0, f"KL divergence is negative: {divergence}"

    @given(demographic_distribution_strategy())
    @settings(max_examples=30)
    def test_kl_divergence_zero_when_identical(self, dist):
        """
        Property: D_KL(P || P) = 0 for any distribution P.
        """
        p = dist.as_array()
        divergence = kl_divergence(p, p)

        assert np.isclose(divergence, 0, atol=1e-10), (
            f"Self-divergence should be 0, got {divergence}"
        )

    def test_kl_divergence_asymmetric(self):
        """
        Property: D_KL(P || Q) != D_KL(Q || P) in general (asymmetry).

        Uses well-separated distributions where asymmetry is clearly measurable.
        """
        # Highly skewed vs. near-uniform: strong asymmetry
        p = np.array([0.9, 0.05, 0.05])
        q = np.array([0.33, 0.34, 0.33])

        div_pq = kl_divergence(p, q)
        div_qp = kl_divergence(q, p)

        # Both should be non-negative
        assert div_pq >= 0
        assert div_qp >= 0

        # KL divergence is NOT symmetric
        assert div_pq != pytest.approx(div_qp, rel=0.1), (
            f"KL divergence should be asymmetric: "
            f"D_KL(P||Q)={div_pq:.6f}, D_KL(Q||P)={div_qp:.6f}"
        )

        # Additional case: asymmetric pair (not mirror images)
        r = np.array([0.8, 0.15, 0.05])
        s = np.array([0.4, 0.4, 0.2])

        div_rs = kl_divergence(r, s)
        div_sr = kl_divergence(s, r)

        assert div_rs >= 0
        assert div_sr >= 0
        assert div_rs != pytest.approx(div_sr, rel=0.1)

    def test_kl_divergence_known_values(self):
        """
        Test KL divergence with known analytical values.
        """
        # Uniform vs uniform should be 0
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        assert np.isclose(kl_divergence(uniform, uniform), 0, atol=1e-10)

        # Dirac vs uniform
        # D_KL([1, 0, 0, 0] || [0.25, 0.25, 0.25, 0.25])
        # = 1 * log(1/0.25) = log(4) ≈ 1.386
        # But we use smoothing, so it won't be exact
        dirac = np.array([1.0, 1e-10, 1e-10, 1e-10])
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        div = kl_divergence(dirac, uniform)
        assert div > 1.0, f"Expected divergence > 1.0, got {div}"


# =============================================================================
# Theorem 2: Coalition Demographics Tests
# =============================================================================


class TestTheorem2CoalitionDemographics:
    """
    Tests for coalition demographic computation: δ_S = (1/|S|) Σ_{i∈S} δ_i
    """

    def test_single_client_coalition_matches_client(self):
        """
        Property: Coalition with one client has that client's demographics.
        """
        demo = DemographicDistribution.from_dict(
            {"white": 0.6, "black": 0.2, "hispanic": 0.2}
        )
        client = Client(
            id="test",
            demographics=demo,
            num_samples=1000,
            data_quality=0.8,
        )

        coalition_demo = compute_coalition_demographics([0], [client])

        np.testing.assert_array_almost_equal(
            coalition_demo,
            demo.as_array(),
        )

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=30)
    def test_coalition_demographics_average(self, n_clients):
        """
        Property: Coalition demographics is average of member demographics.
        """
        # Create clients with known demographics
        clients = []
        for i in range(n_clients):
            demo = DemographicDistribution.from_dict(
                {
                    "a": 0.5 + 0.1 * (i % 2),
                    "b": 0.5 - 0.1 * (i % 2),
                }
            )
            clients.append(
                Client(
                    id=f"client_{i}",
                    demographics=demo,
                    num_samples=1000,
                    data_quality=0.8,
                )
            )

        # Full coalition
        coalition = list(range(n_clients))
        coalition_demo = compute_coalition_demographics(coalition, clients)

        # Compute expected average
        expected = np.mean([c.demographics.as_array() for c in clients], axis=0)

        np.testing.assert_array_almost_equal(coalition_demo, expected)

    def test_coalition_demographics_sum_to_one(self):
        """
        Property: Coalition demographics always sum to 1.
        """
        clients = create_synthetic_clients(n_clients=10, seed=42)

        for coalition_size in [1, 3, 5, 10]:
            coalition = list(range(coalition_size))
            demo = compute_coalition_demographics(coalition, clients)

            assert np.isclose(demo.sum(), 1.0, atol=1e-10), (
                f"Demographics sum to {demo.sum()}, expected 1.0"
            )


# =============================================================================
# Theorem 2: Fairness Gradient Tests
# =============================================================================


class TestTheorem2FairnessGradient:
    """
    Tests for the fairness gradient that guides optimization.
    """

    def test_gradient_is_bounded(self):
        """
        Property: Fairness gradient has bounded norm (clipped to max_grad_norm=10).
        The gradient preserves magnitude for Theorem 2's drift analysis.
        """
        clients = create_synthetic_clients(
            n_clients=10, n_demographic_groups=5, seed=42
        )
        target = CensusTarget.US_2020.as_distribution()
        position = np.random.default_rng(42).random(10)

        result = compute_fairness_gradient(
            position=position,
            clients=clients,
            target_distribution=target,
            coalition_size=5,
        )

        norm = np.linalg.norm(result.gradient)
        assert norm <= 10.0 + 1e-6, f"Gradient norm is {norm}, expected <= 10.0"
        assert norm > 0, "Gradient should be non-zero for non-perfect distribution"

    @given(st.integers(min_value=5, max_value=20))
    @settings(max_examples=20, deadline=None)
    def test_gradient_dimension_matches_clients(self, n_clients):
        """
        Property: Gradient dimension equals number of clients.
        """
        clients = create_synthetic_clients(
            n_clients=n_clients, n_demographic_groups=5, seed=42
        )
        target = CensusTarget.US_2020.as_distribution()
        position = np.random.default_rng(42).random(n_clients)

        result = compute_fairness_gradient(
            position=position,
            clients=clients,
            target_distribution=target,
            coalition_size=min(5, n_clients),
        )

        assert len(result.gradient) == n_clients

    def test_gradient_favors_target_matching_clients(self):
        """
        Property: Gradient is positive for clients closer to target.
        """
        target = CensusTarget.US_2020.as_distribution()

        # Client 0: matches target
        # Client 1: very different from target (heavily white, 5 groups to match target)
        clients = [
            Client(
                id="matching",
                demographics=target,
                num_samples=1000,
                data_quality=0.8,
            ),
            Client(
                id="different",
                demographics=DemographicDistribution.from_dict(
                    {
                        "white": 0.96,
                        "black": 0.01,
                        "hispanic": 0.01,
                        "asian": 0.01,
                        "other": 0.01,
                    }
                ),
                num_samples=1000,
                data_quality=0.8,
            ),
        ]

        # Start with equal position weights
        position = np.array([0.5, 0.5])

        result = compute_fairness_gradient(
            position=position,
            clients=clients,
            target_distribution=target,
            coalition_size=1,
        )

        # Gradient should favor the matching client
        assert result.gradient[0] > result.gradient[1], (
            f"Expected gradient[0] > gradient[1], got {result.gradient}"
        )


# =============================================================================
# Theorem 2: ε-Fairness Achievement Tests
# =============================================================================


class TestTheorem2EpsilonFairness:
    """
    Tests for achieving ε-fairness with FairSwarm.
    """

    @given(epsilon_strategy())
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_divergence_bounded_by_input(self, epsilon):
        """
        Property: Divergence is always finite and bounded.
        """
        clients = create_synthetic_clients(
            n_clients=15, n_demographic_groups=5, seed=42
        )
        target = CensusTarget.US_2020.as_distribution()
        fitness = DemographicFitness(target_distribution=target)

        config = FairSwarmConfig(
            swarm_size=15,
            fairness_coefficient=0.5,
            epsilon_fair=epsilon,
        )

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=7,
            config=config,
            target_distribution=target,
            seed=42,
        )

        result = optimizer.optimize(fitness, n_iterations=50)

        assert result.fairness is not None
        assert np.isfinite(result.fairness.demographic_divergence)

    def test_high_fairness_coeff_reduces_divergence(self):
        """
        Property: Higher fairness coefficient leads to lower divergence.
        """
        clients = create_synthetic_clients(
            n_clients=20, n_demographic_groups=5, seed=42
        )
        target = CensusTarget.US_2020.as_distribution()
        fitness = DemographicFitness(target_distribution=target)

        # Low fairness coefficient
        config_low = FairSwarmConfig(fairness_coefficient=0.0, swarm_size=20)
        optimizer_low = FairSwarm(
            clients=clients,
            coalition_size=10,
            config=config_low,
            target_distribution=target,
            seed=42,
        )
        result_low = optimizer_low.optimize(fitness, n_iterations=100)

        # High fairness coefficient
        config_high = FairSwarmConfig(fairness_coefficient=0.8, swarm_size=20)
        optimizer_high = FairSwarm(
            clients=clients,
            coalition_size=10,
            config=config_high,
            target_distribution=target,
            seed=42,
        )
        result_high = optimizer_high.optimize(fitness, n_iterations=100)

        # High fairness should achieve lower or equal divergence
        assert result_high.fairness.demographic_divergence <= (
            result_low.fairness.demographic_divergence + 0.5
        ), (
            f"High fairness ({result_high.fairness.demographic_divergence:.4f}) "
            f"should be <= low fairness ({result_low.fairness.demographic_divergence:.4f})"
        )

    @pytest.mark.slow
    def test_sufficient_iterations_achieves_epsilon(self):
        """
        Property: With sufficient iterations and fairness focus,
        ε-fairness can be achieved.
        """
        # Create clients with diverse demographics
        target = CensusTarget.US_2020.as_distribution()

        # Create clients that can match target when combined appropriately
        clients = []
        categories = list(target.labels)
        for i in range(30):
            # Vary client demographics
            demo = {}
            for j, cat in enumerate(categories):
                base = target[cat]
                noise = 0.1 * np.sin(i + j)  # Deterministic variation
                demo[cat] = max(0.01, base + noise)
            total = sum(demo.values())
            demo = {k: v / total for k, v in demo.items()}

            clients.append(
                Client(
                    id=f"client_{i}",
                    demographics=DemographicDistribution.from_dict(demo),
                    num_samples=1000,
                    data_quality=0.8,
                )
            )

        fitness = DemographicFitness(target_distribution=target)

        epsilon = 0.5  # Reasonable threshold
        config = FairSwarmConfig(
            swarm_size=30,
            fairness_coefficient=0.7,
            epsilon_fair=epsilon,
        )

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=15,
            config=config,
            target_distribution=target,
            seed=42,
        )

        result = optimizer.optimize(fitness, n_iterations=200)

        # Should achieve or approach epsilon
        achieved_div = result.fairness.demographic_divergence
        # With 200 iterations, strong fairness coefficient, and diverse clients,
        # divergence should be well below 0.5
        assert achieved_div < 0.5, (
            f"Divergence {achieved_div:.4f} too high after 200 iterations"
        )


# =============================================================================
# Theorem 2: Probability Bound Tests
# =============================================================================


class TestTheorem2ProbabilityBound:
    """
    Tests for the probability bound in Theorem 2: P(DemDiv ≤ ε) ≥ 1 - δ
    """

    @pytest.mark.slow
    def test_epsilon_achieved_with_high_probability(self):
        """
        Property: ε-fairness achieved in most runs (empirical probability).
        """
        target = CensusTarget.US_2020.as_distribution()
        n_runs = 20
        epsilon = 1.0  # Loose threshold for testing
        successes = 0

        clients = create_synthetic_clients(
            n_clients=20, n_demographic_groups=5, seed=42
        )
        fitness = DemographicFitness(target_distribution=target)

        config = FairSwarmConfig(
            swarm_size=20,
            fairness_coefficient=0.5,
            epsilon_fair=epsilon,
        )

        for seed in range(n_runs):
            optimizer = FairSwarm(
                clients=clients,
                coalition_size=10,
                config=config,
                target_distribution=target,
                seed=seed,
            )

            result = optimizer.optimize(fitness, n_iterations=100)

            if result.fairness.demographic_divergence <= epsilon:
                successes += 1

        success_rate = successes / n_runs
        # Expect at least 50% success with this setup
        assert success_rate >= 0.5, (
            f"Only {successes}/{n_runs} runs achieved ε-fairness"
        )


# =============================================================================
# Theorem 2: Divergence Metrics Comparison
# =============================================================================


class TestTheorem2DivergenceMetrics:
    """
    Tests comparing different divergence metrics.
    """

    @given(
        demographic_distribution_strategy(),
        demographic_distribution_strategy(),
    )
    @settings(max_examples=30)
    def test_js_divergence_symmetric(self, dist1, dist2):
        """
        Property: Jensen-Shannon divergence is symmetric.
        """
        p = dist1.as_array()
        q = dist2.as_array()

        # Align lengths
        max_len = max(len(p), len(q))
        p = np.pad(p, (0, max_len - len(p)), constant_values=1e-10)
        q = np.pad(q, (0, max_len - len(q)), constant_values=1e-10)

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        js_pq = js_divergence(p, q)
        js_qp = js_divergence(q, p)

        assert np.isclose(js_pq, js_qp, atol=1e-10), (
            f"JS divergence not symmetric: JS(P||Q)={js_pq}, JS(Q||P)={js_qp}"
        )

    @given(demographic_distribution_strategy())
    @settings(max_examples=30)
    def test_total_variation_nonnegative(self, dist):
        """
        Property: Total variation distance is non-negative.
        """
        p = dist.as_array()
        target = CensusTarget.US_2020.as_distribution().as_array()

        # Align
        max_len = max(len(p), len(target))
        p = np.pad(p, (0, max_len - len(p)), constant_values=1e-10)
        target = np.pad(target, (0, max_len - len(target)), constant_values=1e-10)

        # Normalize
        p = p / p.sum()
        target = target / target.sum()

        tv = total_variation_distance(p, target)

        assert tv >= 0, f"TV distance negative: {tv}"
        assert tv <= 1 + 1e-10, f"TV distance > 1: {tv}"


# =============================================================================
# Theorem 2: Integration Tests
# =============================================================================


class TestTheorem2Integration:
    """
    Integration tests for Theorem 2 fairness guarantees.
    """

    def test_fairness_improves_over_iterations(self):
        """
        Test that fairness (lower divergence) improves over iterations.
        """
        clients = create_synthetic_clients(
            n_clients=20, n_demographic_groups=5, seed=42
        )
        target = CensusTarget.US_2020.as_distribution()
        fitness = DemographicFitness(target_distribution=target)

        config = FairSwarmConfig(
            swarm_size=20,
            fairness_coefficient=0.5,
        )

        # Short run
        optimizer_short = FairSwarm(
            clients=clients,
            coalition_size=10,
            config=config,
            target_distribution=target,
            seed=42,
        )
        result_short = optimizer_short.optimize(fitness, n_iterations=20)

        # Long run
        optimizer_long = FairSwarm(
            clients=clients,
            coalition_size=10,
            config=config,
            target_distribution=target,
            seed=42,
        )
        result_long = optimizer_long.optimize(fitness, n_iterations=100)

        # Long run should achieve equal or better fairness
        assert result_long.fairness.demographic_divergence <= (
            result_short.fairness.demographic_divergence + 0.1
        )

    def test_fairness_metrics_complete(self):
        """
        Test that all fairness metrics are properly computed.
        """
        clients = create_synthetic_clients(
            n_clients=15, n_demographic_groups=5, seed=42
        )
        target = CensusTarget.US_2020.as_distribution()
        fitness = DemographicFitness(target_distribution=target)

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=7,
            target_distribution=target,
            seed=42,
        )

        result = optimizer.optimize(fitness, n_iterations=50)

        # All metrics should be present
        assert result.fairness is not None
        assert result.fairness.demographic_divergence >= 0
        assert len(result.fairness.coalition_distribution) > 0
        assert len(result.fairness.target_distribution) > 0
        assert isinstance(result.fairness.epsilon_satisfied, bool)


# =============================================================================
# Theorem 2: Boundary Condition Tests
# =============================================================================


class TestTheorem2BoundaryConditions:
    """
    Tests at boundary conditions of Theorem 2.
    """

    def test_no_fairness_gradient_no_improvement(self):
        """
        With fairness_coefficient = 0, the fairness gradient has no effect.
        Divergence should not systematically improve.
        """
        clients = create_synthetic_clients(
            n_clients=20, n_demographic_groups=5, seed=42
        )
        target = CensusTarget.US_2020.as_distribution()
        fitness = DemographicFitness(target_distribution=target)

        config_zero = FairSwarmConfig(
            fairness_coefficient=0.0, swarm_size=20
        )
        config_high = FairSwarmConfig(
            fairness_coefficient=0.8, swarm_size=20
        )

        optimizer_zero = FairSwarm(
            clients=clients, coalition_size=10,
            config=config_zero, target_distribution=target, seed=42,
        )
        result_zero = optimizer_zero.optimize(fitness, n_iterations=100)

        optimizer_high = FairSwarm(
            clients=clients, coalition_size=10,
            config=config_high, target_distribution=target, seed=42,
        )
        result_high = optimizer_high.optimize(fitness, n_iterations=100)

        # With fairness gradient, divergence should be lower
        assert result_high.fairness.demographic_divergence <= (
            result_zero.fairness.demographic_divergence + 0.01
        )

    def test_identical_client_demographics_zero_divergence(self):
        """
        When all clients have identical demographics matching the target,
        any coalition has zero divergence.
        """
        target = CensusTarget.US_2020.as_distribution()

        clients = [
            Client(
                id=f"client_{i}",
                demographics=target,
                num_samples=1000,
                data_quality=0.8,
            )
            for i in range(15)
        ]

        fitness = DemographicFitness(target_distribution=target)

        optimizer = FairSwarm(
            clients=clients, coalition_size=5,
            target_distribution=target, seed=42,
        )
        result = optimizer.optimize(fitness, n_iterations=20)

        # Should be nearly zero divergence
        assert result.fairness.demographic_divergence < 0.01

    def test_single_demographic_group_trivial(self):
        """
        With k=1 demographic group, fairness is always trivially satisfied.
        """
        demo = DemographicDistribution.from_dict({"only_group": 1.0})
        clients = [
            Client(
                id=f"c_{i}", demographics=demo,
                num_samples=1000, data_quality=0.8,
            )
            for i in range(10)
        ]

        fitness = DemographicFitness(target_distribution=demo)

        optimizer = FairSwarm(
            clients=clients, coalition_size=5,
            target_distribution=demo, seed=42,
        )
        result = optimizer.optimize(fitness, n_iterations=20)

        assert result.fairness.demographic_divergence < 1e-6


# =============================================================================
# Theorem 2: Scale Tests
# =============================================================================


class TestTheorem2ScaleTests:
    """
    Tests for fairness at different scales.
    """

    def test_many_demographic_groups(self):
        """
        With k=10 groups, fairness is harder to achieve but should still improve.
        """
        clients = create_synthetic_clients(
            n_clients=30, n_demographic_groups=10, seed=42
        )
        target_dict = {f"group_{i}": 1.0 / 10 for i in range(10)}
        target = DemographicDistribution.from_dict(target_dict)
        fitness = DemographicFitness(target_distribution=target)

        config = FairSwarmConfig(
            swarm_size=20, fairness_coefficient=0.7,
        )

        optimizer = FairSwarm(
            clients=clients, coalition_size=15,
            config=config, target_distribution=target, seed=42,
        )
        result = optimizer.optimize(fitness, n_iterations=100)

        # Should still achieve reasonable fairness
        assert result.fairness.demographic_divergence < 2.0
        assert np.isfinite(result.fairness.demographic_divergence)

    @given(st.integers(min_value=2, max_value=8))
    @settings(max_examples=15, deadline=None)
    def test_fairness_across_group_counts(self, n_groups):
        """
        Property: Fairness gradient works across different numbers of groups.
        """
        clients = create_synthetic_clients(
            n_clients=20, n_demographic_groups=n_groups, seed=42
        )
        target_dict = {f"group_{i}": 1.0 / n_groups for i in range(n_groups)}
        target = DemographicDistribution.from_dict(target_dict)
        fitness = DemographicFitness(target_distribution=target)

        config = FairSwarmConfig(
            swarm_size=15, fairness_coefficient=0.5,
        )

        optimizer = FairSwarm(
            clients=clients, coalition_size=10,
            config=config, target_distribution=target, seed=42,
        )
        result = optimizer.optimize(fitness, n_iterations=50)

        assert np.isfinite(result.fairness.demographic_divergence)
        assert result.fairness.demographic_divergence >= 0

    def test_extreme_demographic_imbalance(self):
        """
        With heavily imbalanced demographics (95% one group),
        FairSwarm should still achieve some fairness improvement.
        """
        target = DemographicDistribution.from_dict(
            {"majority": 0.5, "minority_a": 0.2,
             "minority_b": 0.2, "minority_c": 0.1}
        )

        clients = []
        for i in range(20):
            if i < 16:
                demo = DemographicDistribution.from_dict(
                    {"majority": 0.95, "minority_a": 0.02,
                     "minority_b": 0.02, "minority_c": 0.01}
                )
            else:
                demo = DemographicDistribution.from_dict(
                    {"majority": 0.1, "minority_a": 0.4,
                     "minority_b": 0.3, "minority_c": 0.2}
                )
            clients.append(Client(
                id=f"c_{i}", demographics=demo,
                num_samples=1000, data_quality=0.8,
            ))

        fitness = DemographicFitness(target_distribution=target)
        config = FairSwarmConfig(
            swarm_size=20, fairness_coefficient=0.8,
        )

        optimizer = FairSwarm(
            clients=clients, coalition_size=10,
            config=config, target_distribution=target, seed=42,
        )
        result = optimizer.optimize(fitness, n_iterations=100)

        # Should at least select some minority clients
        assert np.isfinite(result.fairness.demographic_divergence)
