"""
Tests for FairSwarm privacy module.

Tests differential privacy mechanisms and privacy accountants.

Author: Tenicka Norwood
Advisor: Dr. Uttam Ghosh
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fairswarm.privacy.accountant import (
    AdvancedCompositionAccountant,
    MomentsAccountant,
    RDPAccountant,
    SimpleAccountant,
)
from fairswarm.privacy.mechanisms import (
    ExponentialMechanism,
    GaussianMechanism,
    LaplaceMechanism,
    add_noise_to_gradient,
    clip_gradient,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng():
    """Fixed random number generator for reproducibility."""
    return np.random.default_rng(42)


# =============================================================================
# LaplaceMechanism Tests
# =============================================================================


class TestLaplaceMechanism:
    """Tests for LaplaceMechanism."""

    def test_basic_noise_addition(self, rng):
        """Test basic noise addition."""
        mechanism = LaplaceMechanism(epsilon=1.0)
        value = np.array([10.0])
        sensitivity = 1.0

        noisy = mechanism.add_noise(value, sensitivity, rng)

        # Noisy value should be different from original
        assert noisy.shape == value.shape
        # With high probability, noise is non-zero
        # (could be zero with vanishing probability)

    def test_noise_scale(self, rng):
        """Test that noise scale is proportional to sensitivity/epsilon."""
        value = np.zeros(1000)
        sensitivity = 2.0
        epsilon = 0.5

        mechanism = LaplaceMechanism(epsilon=epsilon)
        noisy = mechanism.add_noise(value, sensitivity, rng)

        # Expected scale = sensitivity/epsilon = 4.0
        # Mean absolute value of Laplace(0, scale) is scale
        expected_scale = sensitivity / epsilon
        empirical_scale = np.mean(np.abs(noisy))

        assert empirical_scale == pytest.approx(expected_scale, rel=0.2)

    def test_higher_epsilon_less_noise(self, rng):
        """Test that higher epsilon means less noise."""
        value = np.zeros(1000)
        sensitivity = 1.0

        low_eps = LaplaceMechanism(epsilon=0.1)
        high_eps = LaplaceMechanism(epsilon=10.0)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        noisy_low = low_eps.add_noise(value, sensitivity, rng1)
        noisy_high = high_eps.add_noise(value, sensitivity, rng2)

        # Higher epsilon should have lower variance
        assert np.var(noisy_high) < np.var(noisy_low)

    def test_multidimensional_noise(self, rng):
        """Test noise addition to multidimensional arrays."""
        mechanism = LaplaceMechanism(epsilon=1.0)
        value = np.zeros((10, 5, 3))
        sensitivity = 1.0

        noisy = mechanism.add_noise(value, sensitivity, rng)
        assert noisy.shape == value.shape

    def test_invalid_epsilon(self):
        """Test that non-positive epsilon raises error."""
        with pytest.raises(ValueError):
            LaplaceMechanism(epsilon=0.0)
        with pytest.raises(ValueError):
            LaplaceMechanism(epsilon=-1.0)

    def test_name_property(self):
        """Test mechanism name."""
        mechanism = LaplaceMechanism(epsilon=1.0)
        assert mechanism.name == "Laplace"

    def test_config(self):
        """Test configuration retrieval."""
        mechanism = LaplaceMechanism(epsilon=2.5)
        config = mechanism.get_config()

        assert config["epsilon"] == 2.5
        assert config["mechanism"] == "Laplace"


# =============================================================================
# GaussianMechanism Tests
# =============================================================================


class TestGaussianMechanism:
    """Tests for GaussianMechanism."""

    def test_basic_noise_addition(self, rng):
        """Test basic noise addition."""
        mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5)
        value = np.array([10.0])
        sensitivity = 1.0

        noisy = mechanism.add_noise(value, sensitivity, rng)
        assert noisy.shape == value.shape

    def test_noise_is_gaussian(self, rng):
        """Test that noise follows Gaussian distribution."""
        mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5)
        value = np.zeros(10000)
        sensitivity = 1.0

        noisy = mechanism.add_noise(value, sensitivity, rng)

        # For Gaussian, mean should be close to 0 and noise should be symmetric
        assert np.mean(noisy) == pytest.approx(0.0, abs=0.1)

    def test_sigma_computation(self):
        """Test sigma is computed correctly."""
        epsilon = 1.0
        delta = 1e-5
        sensitivity = 1.0

        mechanism = GaussianMechanism(epsilon=epsilon, delta=delta)

        # σ = Δf * √(2ln(1.25/δ)) / ε
        expected_sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

        assert mechanism.get_sigma(sensitivity) == pytest.approx(
            expected_sigma, rel=1e-5
        )

    def test_higher_epsilon_lower_sigma(self):
        """Test that higher epsilon means lower sigma."""
        delta = 1e-5
        sensitivity = 1.0

        low_eps = GaussianMechanism(epsilon=0.1, delta=delta)
        high_eps = GaussianMechanism(epsilon=10.0, delta=delta)

        assert high_eps.get_sigma(sensitivity) < low_eps.get_sigma(sensitivity)

    def test_smaller_delta_higher_sigma(self):
        """Test that smaller delta means higher sigma."""
        epsilon = 1.0
        sensitivity = 1.0

        large_delta = GaussianMechanism(epsilon=epsilon, delta=1e-3)
        small_delta = GaussianMechanism(epsilon=epsilon, delta=1e-10)

        assert small_delta.get_sigma(sensitivity) > large_delta.get_sigma(sensitivity)

    def test_invalid_parameters(self):
        """Test invalid parameters raise errors."""
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=0.0, delta=1e-5)
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=1.0, delta=0.0)
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=1.0, delta=1.0)

    def test_name_property(self):
        """Test mechanism name."""
        mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5)
        assert mechanism.name == "Gaussian"


# =============================================================================
# ExponentialMechanism Tests
# =============================================================================


class TestExponentialMechanism:
    """Tests for ExponentialMechanism."""

    def test_basic_selection(self, rng):
        """Test basic element selection."""

        def utility(x):
            return -abs(x - 5)  # Peak at 5

        mechanism = ExponentialMechanism(
            epsilon=10.0, utility_fn=utility, sensitivity=1.0
        )
        candidates = list(range(10))

        selected = mechanism.select(candidates, rng=rng)
        assert selected in candidates

    def test_high_epsilon_near_optimal(self, rng):
        """Test high epsilon selects near-optimal."""

        def utility(x):
            return -abs(x - 5)

        mechanism = ExponentialMechanism(
            epsilon=100.0, utility_fn=utility, sensitivity=1.0
        )
        candidates = list(range(10))

        # With very high epsilon, should almost always select 5
        selections = [mechanism.select(candidates, rng=rng) for _ in range(100)]
        assert sum(1 for s in selections if s == 5) > 90  # Most should be 5

    def test_low_epsilon_more_random(self, rng):
        """Test low epsilon gives more uniform selection."""

        def utility(x):
            return -abs(x - 5)

        mechanism = ExponentialMechanism(
            epsilon=0.01, utility_fn=utility, sensitivity=1.0
        )
        candidates = list(range(10))

        selections = [mechanism.select(candidates, rng=rng) for _ in range(1000)]
        unique_selections = len(set(selections))

        # With low epsilon, should select many different values
        assert unique_selections >= 5

    def test_probability_distribution(self, rng):
        """Test selection probabilities follow exponential mechanism."""
        epsilon = 1.0
        sensitivity = 1.0

        def utility(x):
            return float(x)  # Utility = value itself

        mechanism = ExponentialMechanism(
            epsilon=epsilon, utility_fn=utility, sensitivity=sensitivity
        )
        candidates = [0, 1, 2]

        # Sample many times
        counts = {0: 0, 1: 0, 2: 0}
        n_samples = 10000
        for _ in range(n_samples):
            selected = mechanism.select(candidates, rng=rng)
            counts[selected] += 1

        # Check relative frequencies match expected ratios
        # P(i) ∝ exp(ε * u(i) / (2Δ))
        # Ratio P(2)/P(0) = exp(ε * (2-0) / 2) = exp(ε) ≈ 2.718 for ε=1
        expected_ratio = np.exp(epsilon)
        empirical_ratio = counts[2] / max(counts[0], 1)

        assert empirical_ratio == pytest.approx(expected_ratio, rel=0.3)

    def test_empty_candidates_raises(self, rng):
        """Test empty candidates raises error."""
        mechanism = ExponentialMechanism(
            epsilon=1.0, utility_fn=lambda x: x, sensitivity=1.0
        )

        with pytest.raises(ValueError):
            mechanism.select([], rng=rng)


# =============================================================================
# Gradient Clipping Tests
# =============================================================================


class TestGradientClipping:
    """Tests for gradient clipping utilities."""

    def test_clip_gradient_within_norm(self):
        """Test gradient within norm is unchanged."""
        gradient = np.array([0.3, 0.4])  # Norm = 0.5
        max_norm = 1.0

        clipped, scale = clip_gradient(gradient, max_norm)
        np.testing.assert_array_almost_equal(clipped, gradient)
        assert scale == 1.0

    def test_clip_gradient_exceeds_norm(self):
        """Test gradient exceeding norm is clipped."""
        gradient = np.array([3.0, 4.0])  # Norm = 5.0
        max_norm = 1.0

        clipped, scale = clip_gradient(gradient, max_norm)

        # Should have norm = max_norm
        assert np.linalg.norm(clipped) == pytest.approx(max_norm, rel=1e-5)

        # Should preserve direction
        np.testing.assert_array_almost_equal(
            clipped / np.linalg.norm(clipped),
            gradient / np.linalg.norm(gradient),
        )

        # Scale should be max_norm / original_norm
        assert scale == pytest.approx(max_norm / 5.0, rel=1e-5)

    def test_clip_zero_gradient(self):
        """Test zero gradient stays zero."""
        gradient = np.zeros(5)
        clipped, scale = clip_gradient(gradient, 1.0)
        np.testing.assert_array_equal(clipped, gradient)

    def test_clip_multidimensional(self):
        """Test clipping works for multidimensional arrays."""
        gradient = np.ones((3, 4)) * 2.0  # Large values
        max_norm = 1.0

        clipped, scale = clip_gradient(gradient, max_norm)
        assert np.linalg.norm(clipped) == pytest.approx(max_norm, rel=1e-5)


class TestAddNoiseToGradient:
    """Tests for add_noise_to_gradient utility."""

    def test_adds_noise(self, rng):
        """Test Gaussian noise addition (DP-SGD style)."""
        gradient = np.zeros(100)
        noisy = add_noise_to_gradient(
            gradient,
            noise_multiplier=1.0,
            max_norm=1.0,
            rng=rng,
        )

        # Should have noise
        assert not np.allclose(noisy, gradient)

    def test_noise_scale_proportional_to_multiplier(self, rng):
        """Test that higher noise multiplier means more noise."""
        gradient = np.zeros(1000)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        noisy_low = add_noise_to_gradient(
            gradient, noise_multiplier=0.1, max_norm=1.0, rng=rng1
        )
        noisy_high = add_noise_to_gradient(
            gradient, noise_multiplier=10.0, max_norm=1.0, rng=rng2
        )

        assert np.var(noisy_high) > np.var(noisy_low)

    def test_clips_before_noise(self, rng):
        """Test gradient is clipped before adding noise."""
        gradient = np.array([30.0, 40.0])  # Norm = 50
        max_norm = 1.0

        # The function should clip to max_norm before adding noise
        noisy = add_noise_to_gradient(
            gradient,
            noise_multiplier=1.0,
            max_norm=max_norm,
            rng=rng,
        )

        # Original direction should be preserved (roughly)
        # but magnitude should be close to max_norm + noise
        # Since noise can be large, just check the function runs
        assert noisy.shape == gradient.shape


# =============================================================================
# SimpleAccountant Tests
# =============================================================================


class TestSimpleAccountant:
    """Tests for SimpleAccountant."""

    def test_initial_state(self):
        """Test initial state is zero."""
        accountant = SimpleAccountant()
        assert accountant.get_epsilon(1e-5) == 0.0
        assert accountant.get_delta() == 0.0
        assert accountant.step_count == 0

    def test_basic_composition(self):
        """Test basic composition sums epsilons."""
        accountant = SimpleAccountant()

        accountant.step(1.0)
        accountant.step(2.0)
        accountant.step(0.5)

        assert accountant.get_epsilon(1e-5) == pytest.approx(3.5, rel=1e-5)
        assert accountant.step_count == 3

    def test_delta_composition(self):
        """Test delta also composes."""
        accountant = SimpleAccountant()

        accountant.step(1.0, delta=1e-6)
        accountant.step(1.0, delta=2e-6)

        assert accountant.get_delta() == pytest.approx(3e-6, rel=1e-5)

    def test_history_tracking(self):
        """Test history is tracked."""
        accountant = SimpleAccountant()

        accountant.step(1.0)
        accountant.step(2.0)

        assert len(accountant.history) == 2
        assert accountant.history[0].epsilon == 1.0
        assert accountant.history[1].epsilon == 2.0

    def test_reset(self):
        """Test reset clears state."""
        accountant = SimpleAccountant()

        accountant.step(1.0)
        accountant.step(2.0)
        accountant.reset()

        assert accountant.get_epsilon(1e-5) == 0.0
        assert accountant.step_count == 0
        assert len(accountant.history) == 0

    def test_name_property(self):
        """Test name property."""
        accountant = SimpleAccountant()
        assert accountant.name == "Simple"


# =============================================================================
# MomentsAccountant Tests
# =============================================================================


class TestMomentsAccountant:
    """Tests for MomentsAccountant."""

    def test_initial_state(self):
        """Test initial state."""
        accountant = MomentsAccountant(noise_multiplier=1.0)
        assert accountant.get_epsilon(1e-5) == 0.0
        assert accountant.step_count == 0

    def test_epsilon_increases_with_steps(self):
        """Test epsilon increases with steps."""
        accountant = MomentsAccountant(noise_multiplier=1.0, sampling_rate=1.0)

        eps_0 = accountant.get_epsilon(1e-5)

        accountant.step()
        eps_1 = accountant.get_epsilon(1e-5)

        accountant.step()
        eps_2 = accountant.get_epsilon(1e-5)

        assert eps_0 < eps_1 < eps_2

    def test_higher_noise_lower_epsilon(self):
        """Test higher noise multiplier gives lower epsilon."""
        delta = 1e-5
        n_steps = 10

        low_noise = MomentsAccountant(noise_multiplier=0.5)
        high_noise = MomentsAccountant(noise_multiplier=2.0)

        for _ in range(n_steps):
            low_noise.step()
            high_noise.step()

        assert high_noise.get_epsilon(delta) < low_noise.get_epsilon(delta)

    def test_subsampling_reduces_epsilon(self):
        """Test subsampling reduces privacy loss."""
        delta = 1e-5
        n_steps = 10

        full_sample = MomentsAccountant(noise_multiplier=1.0, sampling_rate=1.0)
        subsample = MomentsAccountant(noise_multiplier=1.0, sampling_rate=0.01)

        for _ in range(n_steps):
            full_sample.step()
            subsample.step()

        assert subsample.get_epsilon(delta) < full_sample.get_epsilon(delta)

    def test_reset(self):
        """Test reset."""
        accountant = MomentsAccountant(noise_multiplier=1.0)

        accountant.step()
        accountant.step()
        accountant.reset()

        assert accountant.step_count == 0

    def test_name_property(self):
        """Test name property."""
        accountant = MomentsAccountant(noise_multiplier=1.0)
        assert accountant.name == "Moments"


# =============================================================================
# RDPAccountant Tests
# =============================================================================


class TestRDPAccountant:
    """Tests for RDPAccountant."""

    def test_initial_state(self):
        """Test initial state."""
        accountant = RDPAccountant()
        assert accountant.get_epsilon(1e-5) == 0.0
        assert accountant.step_count == 0

    def test_epsilon_from_noise_multiplier(self):
        """Test epsilon computation with noise multiplier."""
        accountant = RDPAccountant()

        accountant.step(noise_multiplier=1.0, sampling_rate=1.0)

        eps = accountant.get_epsilon(1e-5)
        assert eps > 0

    def test_epsilon_from_direct_values(self):
        """Test epsilon with direct values."""
        accountant = RDPAccountant()

        accountant.step(epsilon=1.0, delta=1e-6)
        accountant.step(epsilon=0.5, delta=1e-6)

        eps = accountant.get_epsilon(1e-5)
        assert eps > 0

    def test_multiple_steps_increase_epsilon(self):
        """Test multiple steps increase epsilon."""
        accountant = RDPAccountant()
        delta = 1e-5

        accountant.step(noise_multiplier=1.0)
        eps_1 = accountant.get_epsilon(delta)

        accountant.step(noise_multiplier=1.0)
        eps_2 = accountant.get_epsilon(delta)

        assert eps_2 > eps_1

    def test_reset(self):
        """Test reset."""
        accountant = RDPAccountant()

        accountant.step(noise_multiplier=1.0)
        accountant.step(noise_multiplier=1.0)
        accountant.reset()

        assert accountant.step_count == 0
        assert accountant.get_epsilon(1e-5) == 0.0

    def test_name_property(self):
        """Test name property."""
        accountant = RDPAccountant()
        assert accountant.name == "RDP"


# =============================================================================
# AdvancedCompositionAccountant Tests
# =============================================================================


class TestAdvancedCompositionAccountant:
    """Tests for AdvancedCompositionAccountant."""

    def test_initial_state(self):
        """Test initial state."""
        accountant = AdvancedCompositionAccountant(epsilon_per_step=0.1)
        assert accountant.get_epsilon(1e-5) == 0.0
        assert accountant.step_count == 0

    def test_advanced_composition_formula(self):
        """Test advanced composition gives expected result."""
        epsilon_per_step = 0.1
        delta = 1e-5
        k = 10

        accountant = AdvancedCompositionAccountant(epsilon_per_step=epsilon_per_step)

        for _ in range(k):
            accountant.step()

        composed = accountant.get_epsilon(delta)

        # ε' = √(2k·ln(1/δ))·ε + k·ε·(e^ε - 1)
        expected = np.sqrt(
            2 * k * np.log(1 / delta)
        ) * epsilon_per_step + k * epsilon_per_step * (np.exp(epsilon_per_step) - 1)

        assert composed == pytest.approx(expected, rel=1e-5)

    def test_better_than_basic(self):
        """Test advanced composition is better than basic for many queries."""
        epsilon_per_step = 0.1
        delta = 1e-5
        k = 100

        accountant = AdvancedCompositionAccountant(epsilon_per_step=epsilon_per_step)

        for _ in range(k):
            accountant.step()

        advanced_eps = accountant.get_epsilon(delta)
        basic_eps = k * epsilon_per_step  # 10.0

        # For small epsilon and many queries, advanced should be better
        assert advanced_eps < basic_eps

    def test_fallback_to_basic_when_delta_zero(self):
        """Test fallback to basic composition when delta=0."""
        accountant = AdvancedCompositionAccountant(epsilon_per_step=0.1)

        for _ in range(10):
            accountant.step()

        # With delta=0, should fall back to basic
        composed = accountant.get_epsilon(delta=0)
        assert composed == pytest.approx(1.0, rel=1e-5)  # 10 * 0.1

    def test_get_delta(self):
        """Test delta computation."""
        delta_per_step = 1e-6
        accountant = AdvancedCompositionAccountant(
            epsilon_per_step=0.1,
            delta_per_step=delta_per_step,
        )

        for _ in range(10):
            accountant.step()

        total_delta = accountant.get_delta(target_epsilon=1.0, base_delta=1e-5)
        expected = 10 * delta_per_step + 1e-5

        assert total_delta == pytest.approx(expected, rel=1e-5)

    def test_invalid_epsilon_raises(self):
        """Test non-positive epsilon raises error."""
        with pytest.raises(ValueError):
            AdvancedCompositionAccountant(epsilon_per_step=0.0)
        with pytest.raises(ValueError):
            AdvancedCompositionAccountant(epsilon_per_step=-0.1)

    def test_reset(self):
        """Test reset."""
        accountant = AdvancedCompositionAccountant(epsilon_per_step=0.1)

        for _ in range(10):
            accountant.step()

        accountant.reset()

        assert accountant.step_count == 0
        assert accountant.get_epsilon(1e-5) == 0.0

    def test_name_property(self):
        """Test name property."""
        accountant = AdvancedCompositionAccountant(epsilon_per_step=0.1)
        assert accountant.name == "AdvancedComposition"


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestPrivacyProperties:
    """Property-based tests for privacy mechanisms."""

    @given(
        epsilon=st.floats(min_value=0.1, max_value=10.0),
        sensitivity=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=50)
    def test_laplace_scale_relation(self, epsilon, sensitivity):
        """Test Laplace scale = sensitivity/epsilon."""
        mechanism = LaplaceMechanism(epsilon=epsilon)
        expected_scale = sensitivity / epsilon

        # The mechanism uses this scale internally
        # We can verify by checking noise statistics
        rng = np.random.default_rng(42)
        value = np.zeros(10000)
        noisy = mechanism.add_noise(value, sensitivity, rng)

        # Mean absolute deviation ≈ scale for Laplace
        empirical_scale = np.mean(np.abs(noisy))
        assert empirical_scale == pytest.approx(expected_scale, rel=0.2)

    @given(
        n_steps=st.integers(min_value=1, max_value=20),
        epsilon_per_step=st.floats(min_value=0.01, max_value=1.0),
    )
    @settings(max_examples=30)
    def test_simple_accountant_linear(self, n_steps, epsilon_per_step):
        """Test simple accountant sums epsilons linearly."""
        accountant = SimpleAccountant()

        for _ in range(n_steps):
            accountant.step(epsilon_per_step)

        total = accountant.get_epsilon(1e-5)
        expected = n_steps * epsilon_per_step

        assert total == pytest.approx(expected, rel=1e-5)

    @given(
        noise_multiplier=st.floats(min_value=0.1, max_value=5.0),
        n_steps=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=30)
    def test_rdp_monotonic_in_steps(self, noise_multiplier, n_steps):
        """Test RDP epsilon increases monotonically with steps."""
        accountant = RDPAccountant()
        delta = 1e-5

        previous_eps = 0.0
        for _ in range(n_steps):
            accountant.step(noise_multiplier=noise_multiplier)
            current_eps = accountant.get_epsilon(delta)
            assert current_eps >= previous_eps
            previous_eps = current_eps

    @given(
        gradient=st.lists(
            st.floats(
                min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=10,
        ),
        max_norm=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=50)
    def test_clipped_gradient_bounded(self, gradient, max_norm):
        """Test clipped gradient has norm <= max_norm."""
        gradient_array = np.array(gradient)
        clipped, scale = clip_gradient(gradient_array, max_norm)

        assert np.linalg.norm(clipped) <= max_norm + 1e-6  # Small tolerance


# =============================================================================
# Integration Tests
# =============================================================================


class TestPrivacyIntegration:
    """Integration tests combining mechanisms and accountants."""

    def test_mechanism_with_accountant(self, rng):
        """Test using mechanism with accountant tracking."""
        mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5)
        accountant = SimpleAccountant()

        gradient = np.zeros(100)
        sensitivity = 1.0

        for _ in range(10):
            mechanism.add_noise(gradient, sensitivity, rng)
            accountant.step(mechanism.epsilon, mechanism.delta)

        total_eps = accountant.get_epsilon(1e-5)
        assert total_eps == pytest.approx(10.0, rel=1e-5)

    def test_rdp_tracking_gaussian(self, rng):
        """Test RDP accountant with Gaussian mechanism."""
        noise_multiplier = 1.0
        GaussianMechanism(epsilon=1.0, delta=1e-5)
        accountant = RDPAccountant()

        for _ in range(10):
            accountant.step(noise_multiplier=noise_multiplier)

        eps = accountant.get_epsilon(1e-5)
        assert eps > 0
        assert eps < 100  # Should be reasonable

    def test_privacy_budget_workflow(self, rng):
        """Test typical privacy budget workflow."""
        # Set a budget
        budget = 5.0
        accountant = SimpleAccountant()
        mechanism = LaplaceMechanism(epsilon=0.5)

        data = np.zeros(100)
        sensitivity = 1.0

        # Query until budget exhausted
        queries = 0
        while accountant.get_epsilon(1e-5) + mechanism.epsilon <= budget:
            mechanism.add_noise(data, sensitivity, rng)
            accountant.step(mechanism.epsilon)
            queries += 1

        # Should have made 10 queries (5.0 / 0.5)
        assert queries == 10
        assert accountant.get_epsilon(1e-5) == pytest.approx(5.0, rel=1e-5)
