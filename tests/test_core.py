"""
Unit tests for FairSwarm core module.

Tests for Client, FairSwarmConfig, and core type utilities.

Author: Tenicka Norwood
"""

from __future__ import annotations

import numpy as np
import pytest

from fairswarm.core.client import Client, create_synthetic_clients
from fairswarm.core.config import FairSwarmConfig, get_preset_config
from fairswarm.types import (
    ClientId,
    normalize_to_distribution,
    validate_coalition,
    validate_demographic_vector,
)


class TestClient:
    """Tests for the Client dataclass."""

    def test_client_creation_valid(self, uniform_demographics):
        """Test creating a valid client."""
        client = Client(
            id=ClientId("hospital_01"),
            demographics=uniform_demographics,
            dataset_size=5000,
            communication_cost=0.3,
        )

        assert client.id == "hospital_01"
        assert client.dataset_size == 5000
        assert client.communication_cost == 0.3
        assert client.n_demographic_groups == 4

    def test_client_creation_invalid_demographics(self):
        """Test that invalid demographics raise ValueError."""
        # Demographics don't sum to 1
        with pytest.raises(ValueError, match="probability distribution"):
            Client(
                id=ClientId("bad_client"),
                demographics=np.array([0.5, 0.2, 0.1, 0.1]),  # sums to 0.9
            )

    def test_client_creation_negative_demographics(self):
        """Test that negative demographics raise ValueError."""
        with pytest.raises(ValueError, match="probability distribution"):
            Client(
                id=ClientId("bad_client"),
                demographics=np.array([0.6, 0.3, -0.1, 0.2]),
            )

    def test_client_creation_invalid_dataset_size(self, uniform_demographics):
        """Test that non-positive dataset_size raises ValueError."""
        with pytest.raises(ValueError, match="dataset_size"):
            Client(
                id=ClientId("bad_client"),
                demographics=uniform_demographics,
                dataset_size=0,
            )

    def test_client_creation_invalid_communication_cost(self, uniform_demographics):
        """Test that communication_cost outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="communication_cost"):
            Client(
                id=ClientId("bad_client"),
                demographics=uniform_demographics,
                communication_cost=1.5,
            )

    def test_client_from_dict(self):
        """Test creating a client from a dictionary."""
        data = {
            "id": "hospital_02",
            "demographics": [0.4, 0.3, 0.2, 0.1],
            "dataset_size": 3000,
            "communication_cost": 0.4,
        }
        client = Client.from_dict(data)

        assert client.id == "hospital_02"
        assert client.dataset_size == 3000
        assert np.allclose(client.demographics, [0.4, 0.3, 0.2, 0.1])

    def test_client_immutability(self, single_client):
        """Test that clients are immutable (frozen dataclass)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            single_client.dataset_size = 9999

    def test_demographic_contribution(self, single_client):
        """Test getting contribution for specific demographic group."""
        contrib = single_client.demographic_contribution(0)
        assert contrib == 0.25  # uniform distribution

    def test_demographic_contribution_out_of_range(self, single_client):
        """Test that out-of-range group index raises IndexError."""
        with pytest.raises(IndexError):
            single_client.demographic_contribution(10)


class TestCreateSyntheticClients:
    """Tests for synthetic client generation."""

    def test_creates_correct_count(self):
        """Test that correct number of clients is created."""
        clients = create_synthetic_clients(15, seed=42)
        assert len(clients) == 15

    def test_demographics_are_valid(self):
        """Test that all generated demographics are valid distributions."""
        clients = create_synthetic_clients(20, seed=42)
        for client in clients:
            assert validate_demographic_vector(client.demographics)
            assert np.isclose(np.sum(client.demographics), 1.0)

    def test_reproducibility(self):
        """Test that same seed produces same clients."""
        clients1 = create_synthetic_clients(10, seed=42)
        clients2 = create_synthetic_clients(10, seed=42)

        for c1, c2 in zip(clients1, clients2):
            assert c1.id == c2.id
            assert np.allclose(c1.demographics, c2.demographics)

    def test_different_seeds_produce_different_clients(self):
        """Test that different seeds produce different clients."""
        clients1 = create_synthetic_clients(10, seed=42)
        clients2 = create_synthetic_clients(10, seed=123)

        # Demographics should differ (with very high probability)
        demographics_match = all(
            np.allclose(c1.demographics, c2.demographics)
            for c1, c2 in zip(clients1, clients2)
        )
        assert not demographics_match


class TestFairSwarmConfig:
    """Tests for FairSwarmConfig."""

    def test_default_config_valid(self):
        """Test that default configuration is valid."""
        config = FairSwarmConfig()
        assert config.swarm_size == 30
        assert config.max_iterations == 100
        # Default config has inertia=0.7, cognitive=1.5, social=1.5
        # Convergence metric = 0.7 + (1.5+1.5)/2 = 2.2 >= 2.0
        # So the strict convergence condition is NOT satisfied by default
        assert config.convergence_metric == pytest.approx(2.2)

    def test_convergence_metric_calculation(self):
        """Test convergence metric is computed correctly."""
        config = FairSwarmConfig(inertia=0.5, cognitive=1.0, social=1.0)
        # ω + (c₁ + c₂)/2 = 0.5 + (1.0 + 1.0)/2 = 1.5
        assert config.convergence_metric == 1.5

    def test_invalid_inertia_raises(self):
        """Test that inertia outside (0,1) raises ValueError."""
        with pytest.raises(ValueError, match="inertia"):
            FairSwarmConfig(inertia=1.5)

        with pytest.raises(ValueError, match="inertia"):
            FairSwarmConfig(inertia=0.0)

    def test_invalid_swarm_size_raises(self):
        """Test that swarm_size < 2 raises ValueError."""
        with pytest.raises(ValueError, match="swarm_size"):
            FairSwarmConfig(swarm_size=1)

    def test_invalid_epsilon_fair_raises(self):
        """Test that non-positive epsilon_fair raises ValueError."""
        with pytest.raises(ValueError, match="epsilon_fair"):
            FairSwarmConfig(epsilon_fair=-0.1)

    def test_with_updates(self):
        """Test creating new config with updates."""
        config = FairSwarmConfig(swarm_size=30, seed=42)
        new_config = config.with_updates(swarm_size=50, max_iterations=200)

        assert config.swarm_size == 30  # Original unchanged
        assert new_config.swarm_size == 50
        assert new_config.max_iterations == 200
        assert new_config.seed == 42  # Preserved

    def test_preset_configs(self):
        """Test preset configuration loading."""
        default = get_preset_config("default")
        fast = get_preset_config("fast")
        fair = get_preset_config("fair")

        assert fast.max_iterations < default.max_iterations
        assert fair.fairness_weight > default.fairness_weight

    def test_invalid_preset_raises(self):
        """Test that invalid preset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("nonexistent")


class TestTypeValidation:
    """Tests for type validation utilities."""

    def test_validate_demographic_vector_valid(self):
        """Test validation of valid demographic vectors."""
        assert validate_demographic_vector(np.array([0.5, 0.3, 0.2]))
        assert validate_demographic_vector(np.array([1.0]))
        assert validate_demographic_vector(np.array([0.25, 0.25, 0.25, 0.25]))

    def test_validate_demographic_vector_invalid_sum(self):
        """Test validation fails for incorrect sum."""
        assert not validate_demographic_vector(np.array([0.5, 0.3, 0.1]))
        assert not validate_demographic_vector(np.array([0.5, 0.6]))

    def test_validate_demographic_vector_negative(self):
        """Test validation fails for negative values."""
        assert not validate_demographic_vector(np.array([0.5, 0.6, -0.1]))

    def test_validate_coalition_valid(self):
        """Test validation of valid coalitions."""
        assert validate_coalition([0, 2, 5], n_clients=10)
        assert validate_coalition([0], n_clients=5)

    def test_validate_coalition_duplicates(self):
        """Test validation fails for duplicate indices."""
        assert not validate_coalition([0, 2, 2, 5], n_clients=10)

    def test_validate_coalition_out_of_range(self):
        """Test validation fails for out-of-range indices."""
        assert not validate_coalition([0, 2, 15], n_clients=10)
        assert not validate_coalition([-1, 2], n_clients=10)

    def test_normalize_to_distribution(self):
        """Test normalization to probability distribution."""
        result = normalize_to_distribution([60, 20, 15, 5])
        assert np.allclose(result, [0.6, 0.2, 0.15, 0.05])
        assert np.isclose(np.sum(result), 1.0)

    def test_normalize_to_distribution_negative_raises(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            normalize_to_distribution([60, -20, 15, 5])

    def test_normalize_to_distribution_zero_sum_raises(self):
        """Test that all-zero values raise ValueError."""
        with pytest.raises(ValueError, match="positive sum"):
            normalize_to_distribution([0, 0, 0, 0])


# =============================================================================
# Phase 3: Particle & Swarm Tests
# =============================================================================


from fairswarm.core.particle import Particle
from fairswarm.core.position import (
    coalition_overlap,
    decode_coalition,
    encode_coalition,
    inverse_sigmoid,
    position_similarity,
    sigmoid,
    soft_decode_coalition,
)
from fairswarm.core.swarm import Swarm, SwarmHistory


class TestSigmoid:
    """Tests for sigmoid function."""

    def test_sigmoid_zero_is_half(self):
        """σ(0) = 0.5"""
        result = sigmoid(np.array([0.0]))
        assert result[0] == pytest.approx(0.5)

    def test_sigmoid_bounds_to_zero_one(self):
        """Sigmoid output is in (0, 1)."""
        x = np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
        result = sigmoid(x)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_sigmoid_large_positive(self):
        """σ(large) ≈ 1"""
        result = sigmoid(np.array([100.0]))
        assert result[0] > 0.99

    def test_sigmoid_large_negative(self):
        """σ(-large) ≈ 0"""
        result = sigmoid(np.array([-100.0]))
        assert result[0] < 0.01

    def test_sigmoid_monotonic(self):
        """Sigmoid is monotonically increasing."""
        x = np.array([-2, -1, 0, 1, 2])
        result = sigmoid(x)
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]

    def test_sigmoid_handles_extreme_values(self):
        """Sigmoid handles extreme values without overflow."""
        x = np.array([-1000, 1000])
        result = sigmoid(x)
        assert np.isfinite(result).all()


class TestInverseSigmoid:
    """Tests for inverse sigmoid (logit) function."""

    def test_inverse_sigmoid_roundtrip(self):
        """logit(sigmoid(x)) ≈ x for reasonable x."""
        x = np.array([-2, -1, 0, 1, 2])
        roundtrip = inverse_sigmoid(sigmoid(x))
        assert np.allclose(roundtrip, x, atol=1e-6)

    def test_inverse_sigmoid_half_is_zero(self):
        """logit(0.5) = 0"""
        result = inverse_sigmoid(np.array([0.5]))
        assert result[0] == pytest.approx(0.0)


class TestDecodeCoalition:
    """Tests for decode_coalition (SelectTop) function."""

    def test_decode_selects_top_k(self):
        """SelectTop selects clients with highest position values."""
        position = np.array([0.9, 0.2, 0.7, 0.5, 0.8])
        coalition = decode_coalition(position, coalition_size=3)

        # Top 3: indices 0 (0.9), 4 (0.8), 2 (0.7)
        assert set(coalition) == {0, 4, 2}

    def test_decode_returns_correct_size(self):
        """SelectTop returns exactly m clients."""
        position = np.random.rand(20)
        coalition = decode_coalition(position, coalition_size=5)
        assert len(coalition) == 5

    def test_decode_all_clients(self):
        """SelectTop with m=n returns all clients."""
        position = np.array([0.3, 0.7, 0.5])
        coalition = decode_coalition(position, coalition_size=3)
        assert set(coalition) == {0, 1, 2}

    def test_decode_single_client(self):
        """SelectTop with m=1 returns single best client."""
        position = np.array([0.1, 0.9, 0.5])
        coalition = decode_coalition(position, coalition_size=1)
        assert coalition == [1]

    def test_decode_invalid_coalition_size_raises(self):
        """SelectTop with invalid coalition_size raises ValueError."""
        position = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="cannot exceed"):
            decode_coalition(position, coalition_size=5)

        with pytest.raises(ValueError, match="must be >= 1"):
            decode_coalition(position, coalition_size=0)


class TestEncodeCoalition:
    """Tests for encode_coalition function."""

    def test_encode_produces_high_values_for_selected(self):
        """Encode gives high values to selected clients."""
        position = encode_coalition([0, 2, 4], n_clients=5)

        assert position[0] == 0.9  # selected
        assert position[1] == 0.1  # not selected
        assert position[2] == 0.9  # selected
        assert position[3] == 0.1  # not selected
        assert position[4] == 0.9  # selected

    def test_encode_decode_roundtrip(self):
        """encode then decode recovers original coalition."""
        original = [1, 3, 5]
        position = encode_coalition(original, n_clients=10)
        decoded = decode_coalition(position, coalition_size=3)

        assert set(decoded) == set(original)


class TestSoftDecodeCoalition:
    """Tests for stochastic coalition decoding."""

    def test_soft_decode_returns_correct_size(self):
        """Stochastic decode returns m clients."""
        rng = np.random.default_rng(42)
        position = rng.random(20)
        coalition = soft_decode_coalition(position, coalition_size=5, rng=rng)
        assert len(coalition) == 5

    def test_soft_decode_no_duplicates(self):
        """Stochastic decode has no duplicate clients."""
        rng = np.random.default_rng(42)
        position = rng.random(20)
        coalition = soft_decode_coalition(position, coalition_size=10, rng=rng)
        assert len(coalition) == len(set(coalition))


class TestPositionSimilarity:
    """Tests for position_similarity function."""

    def test_identical_positions_similarity_one(self):
        """Identical positions have similarity 1."""
        p = np.array([0.5, 0.3, 0.8])
        assert position_similarity(p, p) == pytest.approx(1.0)

    def test_orthogonal_positions(self):
        """Orthogonal positions have similarity 0."""
        p1 = np.array([1, 0])
        p2 = np.array([0, 1])
        assert position_similarity(p1, p2) == pytest.approx(0.0)


class TestCoalitionOverlap:
    """Tests for coalition_overlap function."""

    def test_identical_coalitions(self):
        """Identical coalitions have overlap 1."""
        assert coalition_overlap([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

    def test_disjoint_coalitions(self):
        """Disjoint coalitions have overlap 0."""
        assert coalition_overlap([0, 1, 2], [3, 4, 5]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partial overlap computed correctly."""
        # Jaccard: |{1,2}| / |{0,1,2,3}| = 2/4 = 0.5
        overlap = coalition_overlap([0, 1, 2], [1, 2, 3])
        assert overlap == pytest.approx(0.5)


class TestParticle:
    """Tests for Particle class."""

    def test_particle_initialize(self):
        """Test particle initialization."""
        particle = Particle.initialize(n_clients=20, seed=42)

        assert particle.n_clients == 20
        assert len(particle.position) == 20
        assert len(particle.velocity) == 20
        assert particle.p_best_fitness == float("-inf")

    def test_particle_position_in_range(self):
        """Initialized position is in [0, 1]."""
        particle = Particle.initialize(n_clients=50, seed=42)
        assert np.all(particle.position >= 0)
        assert np.all(particle.position <= 1)

    def test_particle_velocity_in_range(self):
        """Initialized velocity is in [-1, 1]."""
        particle = Particle.initialize(n_clients=50, seed=42)
        assert np.all(particle.velocity >= -1)
        assert np.all(particle.velocity <= 1)

    def test_particle_decode(self):
        """Test particle decode method."""
        particle = Particle.initialize(n_clients=10, seed=42)
        coalition = particle.decode(coalition_size=3)

        assert len(coalition) == 3
        assert len(set(coalition)) == 3  # No duplicates

    def test_particle_update_personal_best(self):
        """Test updating personal best."""
        particle = Particle.initialize(n_clients=10, seed=42)

        # First update should succeed (better than -inf)
        updated = particle.update_personal_best(fitness=0.5, coalition=[0, 1, 2])
        assert updated
        assert particle.p_best_fitness == 0.5
        assert particle.p_best_coalition == [0, 1, 2]

        # Lower fitness should not update
        updated = particle.update_personal_best(fitness=0.3)
        assert not updated
        assert particle.p_best_fitness == 0.5

        # Higher fitness should update
        updated = particle.update_personal_best(fitness=0.8)
        assert updated
        assert particle.p_best_fitness == 0.8

    def test_particle_velocity_update(self):
        """Test velocity update method."""
        rng = np.random.default_rng(42)
        particle = Particle.initialize(n_clients=10, rng=rng)
        particle.p_best = particle.position.copy()  # Set p_best

        g_best = rng.random(10)
        fairness_gradient = rng.standard_normal(10) * 0.1

        old_velocity = particle.velocity.copy()

        particle.apply_velocity_update(
            inertia=0.7,
            cognitive=1.5,
            social=1.5,
            fairness_coeff=0.5,
            g_best=g_best,
            fairness_gradient=fairness_gradient,
            velocity_max=4.0,
            rng=rng,
        )

        # Velocity should have changed
        assert not np.allclose(particle.velocity, old_velocity)

        # Velocity should be clamped
        assert np.all(particle.velocity >= -4.0)
        assert np.all(particle.velocity <= 4.0)

    def test_particle_position_update(self):
        """Test position update with sigmoid bounding."""
        particle = Particle.initialize(n_clients=10, seed=42)

        # Set velocity to push position out of bounds
        particle.velocity = np.ones(10) * 10

        particle.apply_position_update()

        # Position should still be in (0, 1) due to sigmoid
        assert np.all(particle.position > 0)
        assert np.all(particle.position < 1)

    def test_particle_initialize_with_bias(self):
        """Test biased particle initialization."""
        particle = Particle.initialize_with_bias(
            n_clients=10,
            bias_indices=[0, 5, 9],
            bias_strength=0.3,
            seed=42,
        )

        # Biased indices should have higher values on average
        biased_mean = np.mean([particle.position[i] for i in [0, 5, 9]])
        other_mean = np.mean([particle.position[i] for i in [1, 2, 3, 4, 6, 7, 8]])
        assert biased_mean > other_mean

    def test_particle_copy(self):
        """Test particle deep copy."""
        original = Particle.initialize(n_clients=10, seed=42)
        original.p_best_fitness = 0.5
        original.p_best_coalition = [0, 1, 2]

        copy = original.copy()

        # Modify original
        original.position[0] = 0.999
        original.p_best_coalition.append(3)

        # Copy should be unchanged
        assert copy.position[0] != 0.999
        assert 3 not in copy.p_best_coalition


class TestSwarm:
    """Tests for Swarm class."""

    def test_swarm_initialize(self):
        """Test swarm initialization."""
        swarm = Swarm.initialize(swarm_size=30, n_clients=20, seed=42)

        assert swarm.size == 30
        assert swarm.n_clients == 20
        assert len(swarm.particles) == 30

    def test_swarm_iteration(self):
        """Test iterating over swarm."""
        swarm = Swarm.initialize(swarm_size=10, n_clients=5, seed=42)

        count = 0
        for particle in swarm:
            assert isinstance(particle, Particle)
            count += 1

        assert count == 10

    def test_swarm_indexing(self):
        """Test swarm indexing."""
        swarm = Swarm.initialize(swarm_size=10, n_clients=5, seed=42)

        particle = swarm[0]
        assert isinstance(particle, Particle)

    def test_swarm_update_global_best(self):
        """Test updating global best."""
        swarm = Swarm.initialize(swarm_size=5, n_clients=10, seed=42)

        # Set varying fitness values
        for i, particle in enumerate(swarm.particles):
            particle.p_best_fitness = float(i)

        swarm.update_global_best(coalition_size=3)

        # Global best should be the highest (index 4, fitness 4.0)
        assert swarm.g_best_fitness == 4.0
        assert swarm.g_best is not None

    def test_swarm_get_global_best_coalition(self):
        """Test getting global best coalition."""
        swarm = Swarm.initialize(swarm_size=5, n_clients=10, seed=42)

        # Set a particle as best
        swarm.particles[0].p_best_fitness = 1.0
        swarm.update_global_best()

        coalition = swarm.get_global_best_coalition(coalition_size=3)
        assert len(coalition) == 3

    def test_swarm_get_global_best_coalition_no_best_raises(self):
        """Test error when no global best exists."""
        swarm = Swarm(particles=[])

        with pytest.raises(ValueError, match="No global best"):
            swarm.get_global_best_coalition(coalition_size=3)

    def test_swarm_diversity(self):
        """Test swarm diversity calculation."""
        swarm = Swarm.initialize(swarm_size=20, n_clients=10, seed=42)

        diversity = swarm.get_diversity()
        assert diversity > 0  # Random initialization should have diversity

    def test_swarm_statistics(self):
        """Test swarm statistics."""
        swarm = Swarm.initialize(swarm_size=10, n_clients=5, seed=42)

        stats = swarm.get_statistics()

        assert stats["size"] == 10
        assert "position_mean" in stats
        assert "velocity_max" in stats

    def test_swarm_inject_diversity(self):
        """Test diversity injection."""
        rng = np.random.default_rng(42)
        swarm = Swarm.initialize(swarm_size=10, n_clients=5, rng=rng)

        # Make all particles similar
        template = swarm.particles[0].position.copy()
        for particle in swarm.particles:
            particle.position = template.copy()
            particle.p_best_fitness = 0.5

        # Diversity should be low
        diversity_before = swarm.get_diversity()

        # Inject diversity
        swarm.inject_diversity(n_particles=5, rng=rng)

        # Diversity should increase
        diversity_after = swarm.get_diversity()
        assert diversity_after > diversity_before

    def test_swarm_with_seed_coalitions(self):
        """Test swarm initialization with seed coalitions."""
        seed_coalitions = [[0, 1, 2], [3, 4, 5]]

        swarm = Swarm.initialize_with_seed_coalitions(
            swarm_size=10,
            n_clients=10,
            seed_coalitions=seed_coalitions,
            seed=42,
        )

        assert swarm.size == 10
        # Some particles should have higher values at seed indices
        seeded_particle = swarm.particles[0]
        seed_mean = np.mean([seeded_particle.position[i] for i in [0, 1, 2]])
        # This should be biased toward higher values
        assert seed_mean > 0.3  # More than random average


class TestSwarmHistory:
    """Tests for SwarmHistory class."""

    def test_history_record(self):
        """Test recording history."""
        history = SwarmHistory()

        history.record(fitness=0.5, fairness=0.03, diversity=0.2, coalition=[0, 1])
        history.record(fitness=0.6, fairness=0.02, diversity=0.15, coalition=[0, 2])

        assert history.n_iterations == 2
        assert len(history.fitness_history) == 2
        assert len(history.coalition_history) == 2

    def test_history_as_arrays(self):
        """Test converting history to arrays."""
        history = SwarmHistory()
        history.record(fitness=0.5, fairness=0.03, diversity=0.2)
        history.record(fitness=0.6, fairness=0.02, diversity=0.15)

        arrays = history.as_arrays()

        assert np.allclose(arrays["fitness"], [0.5, 0.6])
        assert np.allclose(arrays["fairness"], [0.03, 0.02])
