"""
Theorem 1 (Convergence) Validation Tests.

This module uses Hypothesis property-based testing to empirically validate
Theorem 1 from CLAUDE.md:

    Theorem 1 (Convergence): Under the FairSwarm velocity update rule with
    hyperparameters satisfying ω + (c₁ + c₂)/2 < 2, the swarm converges
    to a stationary point with probability 1.

Key Properties Tested:
    1. Stability condition: ω + (c₁+c₂)/2 < 2 ensures bounded behavior
    2. Fitness monotonicity: Global best never decreases
    3. Diversity decay: Swarm diversity decreases over time
    4. Position boundedness: Positions remain in [0, 1]^n after sigmoid
    5. Velocity boundedness: Velocities stay within [-v_max, v_max]

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from fairswarm.algorithms.fairswarm import FairSwarm
from fairswarm.core.client import create_synthetic_clients
from fairswarm.core.config import FairSwarmConfig
from fairswarm.core.particle import Particle
from fairswarm.core.position import sigmoid
from fairswarm.fitness.mock import MockFitness, ConstantFitness


# =============================================================================
# Hypothesis Strategies
# =============================================================================


@st.composite
def stable_config_strategy(draw):
    """
    Generate FairSwarmConfig satisfying Theorem 1 stability condition.

    Constraint: ω + (c₁ + c₂)/2 < 2
    """
    # Generate inertia in valid range [0, 1]
    inertia = draw(st.floats(min_value=0.1, max_value=0.9))

    # Maximum allowed sum for cognitive + social
    # From: ω + (c₁ + c₂)/2 < 2  =>  c₁ + c₂ < 2(2 - ω)
    max_sum = 2 * (2 - inertia) - 0.1  # Small margin for safety

    # Generate cognitive and social that sum to less than max_sum
    cognitive = draw(st.floats(min_value=0.1, max_value=min(2.5, max_sum - 0.1)))
    max_social = max_sum - cognitive
    social = draw(st.floats(min_value=0.1, max_value=max(0.1, max_social)))

    # Verify stability condition
    stability = inertia + (cognitive + social) / 2
    assume(stability < 2.0)

    return FairSwarmConfig(
        swarm_size=draw(st.integers(min_value=5, max_value=20)),
        max_iterations=100,
        inertia=inertia,
        cognitive=cognitive,
        social=social,
        fairness_coefficient=draw(st.floats(min_value=0.0, max_value=0.5)),
        velocity_max=draw(st.floats(min_value=0.5, max_value=2.0)),
    )


@st.composite
def unstable_config_strategy(draw):
    """
    Generate FairSwarmConfig violating Theorem 1 stability condition.

    Constraint: ω + (c₁ + c₂)/2 >= 2
    """
    # Force high values that violate stability
    inertia = draw(st.floats(min_value=0.8, max_value=1.0))
    cognitive = draw(st.floats(min_value=2.0, max_value=3.0))
    social = draw(st.floats(min_value=2.0, max_value=3.0))

    # Verify instability condition
    stability = inertia + (cognitive + social) / 2
    assume(stability >= 2.0)

    return FairSwarmConfig(
        swarm_size=10,
        max_iterations=50,
        inertia=inertia,
        cognitive=cognitive,
        social=social,
        fairness_coefficient=0.0,
        velocity_max=1.0,
    )


@st.composite
def client_count_strategy(draw):
    """Generate valid (n_clients, coalition_size) pairs."""
    n_clients = draw(st.integers(min_value=5, max_value=30))
    coalition_size = draw(st.integers(min_value=1, max_value=n_clients))
    return n_clients, coalition_size


# =============================================================================
# Theorem 1: Stability Condition Tests
# =============================================================================


class TestTheorem1StabilityCondition:
    """
    Tests for Theorem 1 stability condition: ω + (c₁+c₂)/2 < 2
    """

    @given(stable_config_strategy())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_stable_config_produces_bounded_fitness(self, config):
        """
        Property: Stable configurations produce bounded fitness values.

        When ω + (c₁+c₂)/2 < 2, the optimization should not diverge.
        """
        clients = create_synthetic_clients(n_clients=15, seed=42)
        fitness = MockFitness(mode="mean_quality")

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=5,
            config=config,
            seed=42,
        )

        result = optimizer.optimize(fitness, n_iterations=50)

        # Fitness should be finite (not diverged)
        assert np.isfinite(result.fitness), (
            f"Fitness diverged with stable config: "
            f"ω={config.inertia}, c₁={config.cognitive}, c₂={config.social}"
        )

        # All fitness history should be finite
        for f in result.convergence.fitness_history:
            assert np.isfinite(f), "Fitness history contains non-finite values"

    @given(stable_config_strategy())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_stable_config_positions_remain_bounded(self, config):
        """
        Property: Positions remain in [0, 1]^n under stable conditions.

        The sigmoid bounding ensures positions stay valid.
        """
        clients = create_synthetic_clients(n_clients=15, seed=42)
        fitness = MockFitness(mode="mean_quality")

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=5,
            config=config,
            seed=42,
        )

        result = optimizer.optimize(fitness, n_iterations=50)

        # Final position should be in [0, 1]
        assert result.position is not None
        assert np.all(result.position >= 0), "Position has negative values"
        assert np.all(result.position <= 1), "Position exceeds 1"

        # All particles should have bounded positions
        for particle in optimizer.swarm.particles:
            assert np.all(particle.position >= 0)
            assert np.all(particle.position <= 1)

    def test_stability_condition_formula(self):
        """
        Property: The stability formula ω + (c₁+c₂)/2 < 2 is correctly computed.
        """
        # Test cases with known stability: ω + (c₁+c₂)/2 < 2
        stable_configs = [
            (0.5, 1.0, 1.0),  # 0.5 + (1.0 + 1.0)/2 = 0.5 + 1.0 = 1.5 < 2 ✓
            (0.7, 1.2, 1.2),  # 0.7 + (1.2 + 1.2)/2 = 0.7 + 1.2 = 1.9 < 2 ✓
            (0.4, 1.5, 1.5),  # 0.4 + (1.5 + 1.5)/2 = 0.4 + 1.5 = 1.9 < 2 ✓
        ]

        for omega, c1, c2 in stable_configs:
            stability_value = omega + (c1 + c2) / 2
            assert stability_value < 2, (
                f"Config ({omega}, {c1}, {c2}) should be stable but "
                f"stability value = {stability_value}"
            )

            # Also verify via FairSwarmConfig
            config = FairSwarmConfig(inertia=omega, cognitive=c1, social=c2)
            assert config.satisfies_convergence_condition, (
                f"FairSwarmConfig should satisfy convergence condition for "
                f"({omega}, {c1}, {c2})"
            )


# =============================================================================
# Theorem 1: Fitness Monotonicity Tests
# =============================================================================


class TestTheorem1FitnessMonotonicity:
    """
    Tests for fitness monotonicity under Theorem 1.

    The global best fitness should never decrease.
    """

    @given(client_count_strategy())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_global_best_never_decreases(self, client_params):
        """
        Property: Global best fitness is monotonically non-decreasing.
        """
        n_clients, coalition_size = client_params
        clients = create_synthetic_clients(n_clients=n_clients, seed=42)
        fitness = MockFitness(mode="mean_quality")

        config = FairSwarmConfig(
            swarm_size=10,
            inertia=0.5,
            cognitive=1.2,
            social=1.2,
        )

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=coalition_size,
            config=config,
            seed=42,
        )

        result = optimizer.optimize(fitness, n_iterations=50)
        history = result.convergence.fitness_history

        # Check monotonicity
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-10, (
                f"Fitness decreased at iteration {i}: "
                f"{history[i-1]:.6f} -> {history[i]:.6f}"
            )

    @given(st.integers(min_value=10, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_constant_fitness_converges_immediately(self, n_iterations):
        """
        Property: With constant fitness, convergence is immediate.
        """
        clients = create_synthetic_clients(n_clients=10, seed=42)
        fitness = ConstantFitness(value=1.0)

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=5,
            seed=42,
        )

        result = optimizer.optimize(
            fitness,
            n_iterations=n_iterations,
            convergence_window=5,
        )

        # Should detect convergence
        if n_iterations >= 10:
            assert result.convergence.converged, (
                f"Failed to converge with constant fitness after {n_iterations} iterations"
            )


# =============================================================================
# Theorem 1: Diversity Decay Tests
# =============================================================================


class TestTheorem1DiversityDecay:
    """
    Tests for swarm diversity decay (convergence indicator).
    """

    @given(stable_config_strategy())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_diversity_trends_downward(self, config):
        """
        Property: Swarm diversity generally decreases over time.

        This indicates convergence toward a consensus solution.
        """
        clients = create_synthetic_clients(n_clients=20, seed=42)
        fitness = MockFitness(mode="mean_quality")

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=10,
            config=config,
            seed=42,
        )

        result = optimizer.optimize(fitness, n_iterations=100)
        diversity = result.convergence.diversity_history

        if len(diversity) >= 4:
            # Compare first quarter to last quarter
            quarter = len(diversity) // 4
            first_avg = np.mean(diversity[:quarter])
            last_avg = np.mean(diversity[-quarter:])

            # Diversity should decrease or stay similar
            # Allow some tolerance for randomness
            assert last_avg <= first_avg + 0.2, (
                f"Diversity increased significantly: "
                f"first quarter avg = {first_avg:.4f}, "
                f"last quarter avg = {last_avg:.4f}"
            )

    def test_diversity_bounded_zero_to_one(self):
        """
        Property: Diversity is always in [0, 1] range.
        """
        clients = create_synthetic_clients(n_clients=15, seed=42)
        fitness = MockFitness(mode="mean_quality")

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=5,
            seed=42,
        )

        result = optimizer.optimize(fitness, n_iterations=50)

        for d in result.convergence.diversity_history:
            assert 0 <= d <= 1 + 1e-10, f"Diversity {d} out of [0, 1] range"


# =============================================================================
# Theorem 1: Velocity Boundedness Tests
# =============================================================================


class TestTheorem1VelocityBoundedness:
    """
    Tests for velocity boundedness under the clamping rule.
    """

    @given(
        st.floats(min_value=0.5, max_value=3.0),  # velocity_max
        st.integers(min_value=10, max_value=50),  # n_iterations
    )
    @settings(max_examples=30, deadline=None)
    def test_velocities_clamped_to_vmax(self, velocity_max, n_iterations):
        """
        Property: Velocities are always in [-v_max, v_max].
        """
        clients = create_synthetic_clients(n_clients=15, seed=42)
        fitness = MockFitness(mode="mean_quality")

        config = FairSwarmConfig(
            swarm_size=10,
            inertia=0.5,
            cognitive=1.0,
            social=1.0,
            velocity_max=velocity_max,
        )

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=5,
            config=config,
            seed=42,
        )

        optimizer.optimize(fitness, n_iterations=n_iterations)

        # Check all particle velocities
        for particle in optimizer.swarm.particles:
            assert np.all(particle.velocity >= -velocity_max - 1e-10), (
                f"Velocity below -v_max: min={particle.velocity.min()}"
            )
            assert np.all(particle.velocity <= velocity_max + 1e-10), (
                f"Velocity above v_max: max={particle.velocity.max()}"
            )


# =============================================================================
# Theorem 1: Sigmoid Boundedness Tests
# =============================================================================


class TestTheorem1SigmoidBoundedness:
    """
    Tests for sigmoid position bounding.
    """

    @given(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=50))
    def test_sigmoid_bounds_to_zero_one(self, values):
        """
        Property: Sigmoid maps any real values to (0, 1).
        """
        x = np.array(values, dtype=np.float64)
        result = sigmoid(x)

        assert np.all(result >= 0), "Sigmoid output < 0"
        assert np.all(result <= 1), "Sigmoid output > 1"

    @given(st.floats(min_value=-1000, max_value=1000))
    def test_sigmoid_symmetric(self, x):
        """
        Property: σ(x) + σ(-x) = 1 (sigmoid symmetry).
        """
        assume(np.isfinite(x))

        result_pos = sigmoid(np.array([x]))[0]
        result_neg = sigmoid(np.array([-x]))[0]

        assert np.isclose(result_pos + result_neg, 1.0, atol=1e-7), (
            f"Symmetry violated: σ({x}) + σ({-x}) = {result_pos + result_neg}"
        )

    def test_sigmoid_at_zero_is_half(self):
        """
        Property: σ(0) = 0.5
        """
        result = sigmoid(np.array([0.0]))[0]
        assert np.isclose(result, 0.5, atol=1e-10)


# =============================================================================
# Theorem 1: Particle Update Invariants
# =============================================================================


class TestTheorem1ParticleInvariants:
    """
    Tests for particle update invariants.
    """

    @given(st.integers(min_value=5, max_value=30))
    @settings(max_examples=20, deadline=None)
    def test_particle_personal_best_improves_or_stays(self, n_clients):
        """
        Property: Personal best fitness never decreases.
        """
        rng = np.random.default_rng(42)
        particle = Particle.initialize(n_clients=n_clients, rng=rng)

        # Simulate updates
        initial_pbest = particle.p_best_fitness

        # Try updating with worse fitness
        particle.update_personal_best(fitness=initial_pbest - 1.0)
        assert particle.p_best_fitness == initial_pbest, (
            "Personal best decreased with worse fitness"
        )

        # Try updating with better fitness
        better_fitness = initial_pbest + 1.0
        particle.update_personal_best(fitness=better_fitness)
        assert particle.p_best_fitness == better_fitness, (
            "Personal best not updated with better fitness"
        )

    @given(
        st.integers(min_value=5, max_value=20),
        st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_decode_returns_correct_coalition_size(self, n_clients, coalition_size):
        """
        Property: decode() always returns exactly coalition_size indices.
        """
        assume(coalition_size <= n_clients)

        rng = np.random.default_rng(42)
        particle = Particle.initialize(n_clients=n_clients, rng=rng)

        coalition = particle.decode(coalition_size)

        assert len(coalition) == coalition_size, (
            f"Expected {coalition_size} clients, got {len(coalition)}"
        )

        # All indices should be valid and unique
        assert len(set(coalition)) == coalition_size, "Coalition has duplicates"
        for idx in coalition:
            assert 0 <= idx < n_clients, f"Invalid index {idx}"


# =============================================================================
# Theorem 1: Long-Run Convergence Tests
# =============================================================================


class TestTheorem1LongRunConvergence:
    """
    Tests for long-run convergence behavior.
    """

    @pytest.mark.slow
    def test_extended_optimization_converges(self):
        """
        Test that extended optimization eventually converges.
        """
        clients = create_synthetic_clients(n_clients=20, seed=42)
        fitness = MockFitness(mode="mean_quality")

        config = FairSwarmConfig(
            swarm_size=20,
            inertia=0.5,
            cognitive=1.0,
            social=1.0,
        )

        optimizer = FairSwarm(
            clients=clients,
            coalition_size=10,
            config=config,
            seed=42,
        )

        result = optimizer.optimize(
            fitness,
            n_iterations=200,
            convergence_threshold=1e-8,
            convergence_window=30,
        )

        # With enough iterations, should converge
        assert result.convergence.converged or (
            result.convergence.fitness_improvement(window=50) < 0.01
        ), "Failed to converge after 200 iterations"

    def test_multiple_runs_find_similar_solutions(self):
        """
        Test that multiple runs find solutions of similar quality.
        """
        clients = create_synthetic_clients(n_clients=15, seed=42)
        fitness = MockFitness(mode="mean_quality")

        config = FairSwarmConfig(
            swarm_size=15,
            inertia=0.5,
            cognitive=1.0,
            social=1.0,
        )

        results = []
        for seed in [1, 2, 3, 4, 5]:
            optimizer = FairSwarm(
                clients=clients,
                coalition_size=5,
                config=config,
                seed=seed,
            )
            result = optimizer.optimize(fitness, n_iterations=100)
            results.append(result.fitness)

        # All results should be within reasonable range of each other
        fitness_range = max(results) - min(results)
        assert fitness_range < 0.3, (
            f"Too much variance across runs: range = {fitness_range}"
        )
