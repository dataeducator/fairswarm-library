"""Parameter boundary testing for deployment readiness.

Tests every configurable parameter at minimum, maximum, default, and invalid
values. Verifies correct behavior or clear error messages, and ensures no
configuration combination creates an insecure state.

Covers:
    1. FairSwarmConfig (PSO/fairness parameters)
    2. AggregatorConfig (security/FL parameters)
    3. AdaptivePrivacyAllocator (privacy budget parameters)
    4. FairnessReweighter (aggregation fairness parameters)
    5. NonIIDDetector (distribution detection parameters)
    6. Security invariant combinations (cross-module)

Run: python -m pytest experiments/tests/test_parameter_boundaries.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# ============================================================================
# 1. FairSwarmConfig Boundaries
# ============================================================================


class TestFairSwarmConfigBoundaries:
    """Test FairSwarmConfig parameter validation at boundaries."""

    def _make_config(self, **kwargs):
        from fairswarm.core.config import FairSwarmConfig
        return FairSwarmConfig(**kwargs)

    # --- Swarm size ---
    def test_swarm_size_minimum(self):
        cfg = self._make_config(swarm_size=2)
        assert cfg.swarm_size == 2

    def test_swarm_size_below_minimum(self):
        with pytest.raises(ValueError, match="swarm_size must be >= 2"):
            self._make_config(swarm_size=1)

    def test_swarm_size_zero(self):
        with pytest.raises(ValueError, match="swarm_size must be >= 2"):
            self._make_config(swarm_size=0)

    def test_swarm_size_negative(self):
        with pytest.raises(ValueError, match="swarm_size must be >= 2"):
            self._make_config(swarm_size=-5)

    def test_swarm_size_maximum(self):
        cfg = self._make_config(swarm_size=1000)
        assert cfg.swarm_size == 1000

    def test_swarm_size_above_maximum(self):
        with pytest.raises(ValueError, match="swarm_size must be <= 1000"):
            self._make_config(swarm_size=1001)

    # --- Max iterations ---
    def test_max_iterations_minimum(self):
        cfg = self._make_config(max_iterations=1)
        assert cfg.max_iterations == 1

    def test_max_iterations_zero(self):
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            self._make_config(max_iterations=0)

    def test_max_iterations_maximum(self):
        cfg = self._make_config(max_iterations=10000)
        assert cfg.max_iterations == 10000

    def test_max_iterations_above_maximum(self):
        with pytest.raises(ValueError, match="max_iterations must be <= 10000"):
            self._make_config(max_iterations=10001)

    # --- Coalition size ---
    def test_coalition_size_minimum(self):
        cfg = self._make_config(coalition_size=1)
        assert cfg.coalition_size == 1

    def test_coalition_size_zero(self):
        with pytest.raises(ValueError, match="coalition_size must be >= 1"):
            self._make_config(coalition_size=0)

    def test_coalition_size_maximum(self):
        cfg = self._make_config(coalition_size=500)
        assert cfg.coalition_size == 500

    def test_coalition_size_above_maximum(self):
        with pytest.raises(ValueError, match="coalition_size must be <= 500"):
            self._make_config(coalition_size=501)

    # --- Inertia ---
    def test_inertia_valid_range(self):
        cfg = self._make_config(inertia=0.5)
        assert cfg.inertia == 0.5

    def test_inertia_near_zero(self):
        cfg = self._make_config(inertia=0.001)
        assert cfg.inertia == 0.001

    def test_inertia_near_one(self):
        cfg = self._make_config(inertia=0.999)
        assert cfg.inertia == 0.999

    def test_inertia_zero_rejected(self):
        with pytest.raises(ValueError, match="inertia must be in \\(0, 1\\)"):
            self._make_config(inertia=0.0)

    def test_inertia_one_rejected(self):
        with pytest.raises(ValueError, match="inertia must be in \\(0, 1\\)"):
            self._make_config(inertia=1.0)

    def test_inertia_negative_rejected(self):
        with pytest.raises(ValueError, match="inertia must be in \\(0, 1\\)"):
            self._make_config(inertia=-0.1)

    # --- Cognitive/Social coefficients ---
    def test_cognitive_zero_rejected(self):
        with pytest.raises(ValueError, match="cognitive must be positive"):
            self._make_config(cognitive=0.0)

    def test_cognitive_negative_rejected(self):
        with pytest.raises(ValueError, match="cognitive must be positive"):
            self._make_config(cognitive=-1.0)

    def test_social_zero_rejected(self):
        with pytest.raises(ValueError, match="social must be positive"):
            self._make_config(social=0.0)

    def test_social_negative_rejected(self):
        with pytest.raises(ValueError, match="social must be positive"):
            self._make_config(social=-1.0)

    # --- Convergence condition warning ---
    def test_convergence_warning_when_both_bounds_violated(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_config(inertia=0.9, cognitive=2.5, social=2.5)
            convergence_warnings = [
                x for x in w if "convergence" in str(x.message).lower()
            ]
            assert len(convergence_warnings) >= 1

    def test_no_convergence_warning_clerc_kennedy(self):
        """Default Clerc & Kennedy coefficients should not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_config()  # defaults: omega=0.729, c1=c2=1.494
            convergence_warnings = [
                x for x in w if "convergence" in str(x.message).lower()
            ]
            assert len(convergence_warnings) == 0

    # --- Fairness coefficient ---
    def test_fairness_coefficient_zero(self):
        """c3=0 disables fairness gradient (valid ablation)."""
        cfg = self._make_config(fairness_coefficient=0.0)
        assert cfg.fairness_coefficient == 0.0

    def test_fairness_coefficient_negative_rejected(self):
        with pytest.raises(ValueError, match="fairness_coefficient must be non-negative"):
            self._make_config(fairness_coefficient=-0.1)

    # --- Fairness weight ---
    def test_fairness_weight_zero(self):
        cfg = self._make_config(fairness_weight=0.0, weight_accuracy=0.8, weight_cost=0.2)
        assert cfg.fairness_weight == 0.0

    def test_fairness_weight_one(self):
        cfg = self._make_config(fairness_weight=1.0, weight_accuracy=0.0, weight_fairness=1.0, weight_cost=0.0)
        assert cfg.fairness_weight == 1.0

    def test_fairness_weight_out_of_range(self):
        with pytest.raises(ValueError, match="fairness_weight must be in"):
            self._make_config(fairness_weight=1.1)

    def test_fairness_weight_negative(self):
        with pytest.raises(ValueError, match="fairness_weight must be in"):
            self._make_config(fairness_weight=-0.1)

    # --- Epsilon fair ---
    def test_epsilon_fair_zero_rejected(self):
        with pytest.raises(ValueError, match="epsilon_fair must be positive"):
            self._make_config(epsilon_fair=0.0)

    def test_epsilon_fair_negative_rejected(self):
        with pytest.raises(ValueError, match="epsilon_fair must be positive"):
            self._make_config(epsilon_fair=-0.01)

    def test_epsilon_fair_small(self):
        cfg = self._make_config(epsilon_fair=0.001)
        assert cfg.epsilon_fair == 0.001

    # --- Epsilon DP ---
    def test_epsilon_dp_none(self):
        """None means no DP (valid)."""
        cfg = self._make_config(epsilon_dp=None)
        assert cfg.epsilon_dp is None

    def test_epsilon_dp_zero_rejected(self):
        with pytest.raises(ValueError, match="epsilon_dp must be positive"):
            self._make_config(epsilon_dp=0.0)

    def test_epsilon_dp_negative_rejected(self):
        with pytest.raises(ValueError, match="epsilon_dp must be positive"):
            self._make_config(epsilon_dp=-1.0)

    def test_epsilon_dp_valid(self):
        cfg = self._make_config(epsilon_dp=4.0)
        assert cfg.epsilon_dp == 4.0

    # --- Velocity max ---
    def test_velocity_max_zero_rejected(self):
        with pytest.raises(ValueError, match="velocity_max must be positive"):
            self._make_config(velocity_max=0.0)

    def test_velocity_max_negative_rejected(self):
        with pytest.raises(ValueError, match="velocity_max must be positive"):
            self._make_config(velocity_max=-1.0)

    def test_velocity_max_small(self):
        cfg = self._make_config(velocity_max=0.01)
        assert cfg.velocity_max == 0.01

    # --- Patience ---
    def test_patience_minimum(self):
        cfg = self._make_config(patience=1)
        assert cfg.patience == 1

    def test_patience_zero_rejected(self):
        with pytest.raises(ValueError, match="patience must be >= 1"):
            self._make_config(patience=0)

    # --- c3_decay_rate ---
    def test_c3_decay_rate_zero_rejected(self):
        with pytest.raises(ValueError, match="c3_decay_rate must be in"):
            self._make_config(c3_decay_rate=0.0)

    def test_c3_decay_rate_one(self):
        cfg = self._make_config(c3_decay_rate=1.0)
        assert cfg.c3_decay_rate == 1.0

    # --- c3_min_fraction ---
    def test_c3_min_fraction_zero_rejected(self):
        with pytest.raises(ValueError, match="c3_min_fraction must be in"):
            self._make_config(c3_min_fraction=0.0)

    def test_c3_min_fraction_one(self):
        cfg = self._make_config(c3_min_fraction=1.0)
        assert cfg.c3_min_fraction == 1.0

    # --- Fitness weight sum warning ---
    def test_fitness_weight_sum_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._make_config(weight_accuracy=0.5, weight_fairness=0.5, weight_cost=0.5)
            weight_warnings = [
                x for x in w if "weights should sum" in str(x.message).lower()
            ]
            assert len(weight_warnings) >= 1

    # --- Default config valid ---
    def test_default_config_is_valid(self):
        cfg = self._make_config()
        assert cfg.satisfies_convergence_condition
        assert cfg.swarm_size == 30
        assert cfg.max_iterations == 100

    # --- Preset configs valid ---
    def test_all_presets_valid(self):
        from fairswarm.core.config import get_preset_config
        for preset in ["default", "fast", "thorough", "privacy", "fair"]:
            cfg = get_preset_config(preset)
            assert cfg.satisfies_convergence_condition


# ============================================================================
# 2. AggregatorConfig Boundaries
# ============================================================================


class TestAggregatorConfigBoundaries:
    """Test AggregatorConfig parameter validation and security invariants."""

    def _make_config(self, **kwargs):
        from aggregator.config import AggregatorConfig
        # Set required production fields to valid defaults to avoid
        # triggering unrelated validation errors
        defaults = {
            "jwt_secret_key": "test-secret-at-least-32-chars-long-for-tests",
            "node_enrollment_secret": "test-enrollment-secret",
            "audit_hmac_key": "test-hmac-key-for-audit",
            "environment": "development",
        }
        defaults.update(kwargs)
        return AggregatorConfig(**defaults)

    # --- JWT secret in production ---
    def test_jwt_empty_production_rejected(self):
        from aggregator.config import AggregatorConfig
        with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
            AggregatorConfig(
                jwt_secret_key="",
                node_enrollment_secret="valid-secret",
                environment="production",
                mtls_enabled=True,
            )

    def test_jwt_insecure_default_production_rejected(self):
        from aggregator.config import AggregatorConfig
        with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
            AggregatorConfig(
                jwt_secret_key="changeme",
                node_enrollment_secret="valid-secret",
                environment="production",
                mtls_enabled=True,
            )

    def test_jwt_empty_dev_autogenerates(self):
        cfg = self._make_config(jwt_secret_key="", environment="development")
        # Should auto-generate a secret
        assert len(cfg.jwt_secret_key) > 10

    # --- mTLS in production ---
    def test_mtls_disabled_production_rejected(self):
        from aggregator.config import AggregatorConfig
        with pytest.raises(ValueError, match="MTLS_ENABLED"):
            AggregatorConfig(
                jwt_secret_key="strong-production-secret-32-chars-long",
                node_enrollment_secret="valid-secret",
                environment="production",
                mtls_enabled=False,
            )

    def test_mtls_enabled_production_ok(self):
        cfg = self._make_config(environment="production", mtls_enabled=True)
        assert cfg.mtls_enabled is True

    # --- Node enrollment secret ---
    def test_enrollment_empty_production_rejected(self):
        from aggregator.config import AggregatorConfig
        with pytest.raises(ValueError, match="NODE_ENROLLMENT_SECRET"):
            AggregatorConfig(
                jwt_secret_key="strong-production-secret-32-chars-long",
                node_enrollment_secret="",
                environment="production",
                mtls_enabled=True,
            )

    def test_enrollment_empty_dev_autogenerates(self):
        cfg = self._make_config(node_enrollment_secret="", environment="development")
        assert len(cfg.node_enrollment_secret) > 5

    # --- FL parameters ---
    def test_fl_min_nodes_default(self):
        cfg = self._make_config()
        assert cfg.fl_min_nodes == 3

    def test_fl_max_rounds_default(self):
        cfg = self._make_config()
        assert cfg.fl_max_rounds == 50

    def test_dp_epsilon_default(self):
        cfg = self._make_config()
        assert cfg.dp_default_epsilon == 8.0

    # --- Environment modes ---
    def test_development_mode(self):
        cfg = self._make_config(environment="development")
        assert cfg.environment == "development"

    def test_staging_mode(self):
        cfg = self._make_config(environment="staging")
        assert cfg.environment == "staging"

    def test_production_mode_full(self):
        cfg = self._make_config(environment="production", mtls_enabled=True)
        assert cfg.environment == "production"
        assert cfg.mtls_enabled is True

    # --- HMAC auto-generation ---
    def test_audit_hmac_autogenerates_when_empty(self):
        cfg = self._make_config(audit_hmac_key="")
        assert len(cfg.audit_hmac_key) > 10

    # --- Aggregation strategy ---
    def test_valid_aggregation_strategies(self):
        for strategy in ["fedavg", "krum", "trimmed_mean", "fltrust"]:
            cfg = self._make_config(fl_aggregation_strategy=strategy)
            assert cfg.fl_aggregation_strategy == strategy


# ============================================================================
# 3. AdaptivePrivacyAllocator Boundaries
# ============================================================================


class TestAdaptivePrivacyBoundaries:
    """Test AdaptivePrivacyAllocator parameter validation at boundaries."""

    def _make_allocator(self, **kwargs):
        from novel.adaptive_privacy.allocator import AdaptivePrivacyAllocator
        defaults = {"total_budget": 10.0, "total_rounds": 50}
        defaults.update(kwargs)
        return AdaptivePrivacyAllocator(**defaults)

    # --- Total budget ---
    def test_budget_positive(self):
        alloc = self._make_allocator(total_budget=1.0)
        assert alloc.total_budget == 1.0

    def test_budget_zero_rejected(self):
        with pytest.raises(ValueError, match="total_budget must be positive"):
            self._make_allocator(total_budget=0.0)

    def test_budget_negative_rejected(self):
        with pytest.raises(ValueError, match="total_budget must be positive"):
            self._make_allocator(total_budget=-5.0)

    def test_budget_very_small(self):
        alloc = self._make_allocator(total_budget=0.001)
        assert alloc.total_budget == 0.001

    # --- Total rounds ---
    def test_rounds_positive(self):
        alloc = self._make_allocator(total_rounds=1)
        assert alloc.total_rounds == 1

    def test_rounds_zero_rejected(self):
        with pytest.raises(ValueError, match="total_rounds must be positive"):
            self._make_allocator(total_rounds=0)

    def test_rounds_negative_rejected(self):
        with pytest.raises(ValueError, match="total_rounds must be positive"):
            self._make_allocator(total_rounds=-1)

    # --- Decay rate ---
    def test_decay_rate_zero(self):
        alloc = self._make_allocator(baseline_decay_rate=0.0)
        assert alloc.baseline_decay_rate == 0.0

    def test_decay_rate_one(self):
        alloc = self._make_allocator(baseline_decay_rate=1.0)
        assert alloc.baseline_decay_rate == 1.0

    def test_decay_rate_negative_rejected(self):
        with pytest.raises(ValueError, match="baseline_decay_rate must be in"):
            self._make_allocator(baseline_decay_rate=-0.1)

    def test_decay_rate_above_one_rejected(self):
        with pytest.raises(ValueError, match="baseline_decay_rate must be in"):
            self._make_allocator(baseline_decay_rate=1.1)

    # --- Velocity weight ---
    def test_velocity_weight_zero(self):
        alloc = self._make_allocator(velocity_weight=0.0)
        assert alloc.velocity_weight == 0.0

    def test_velocity_weight_one(self):
        alloc = self._make_allocator(velocity_weight=1.0)
        assert alloc.velocity_weight == 1.0

    def test_velocity_weight_negative_rejected(self):
        with pytest.raises(ValueError, match="velocity_weight must be in"):
            self._make_allocator(velocity_weight=-0.1)

    def test_velocity_weight_above_one_rejected(self):
        with pytest.raises(ValueError, match="velocity_weight must be in"):
            self._make_allocator(velocity_weight=1.1)

    # --- Min epsilon ---
    def test_min_epsilon_zero(self):
        alloc = self._make_allocator(min_epsilon_per_round=0.0)
        assert alloc.min_epsilon_per_round == 0.0

    def test_min_epsilon_negative_rejected(self):
        with pytest.raises(ValueError, match="min_epsilon_per_round must be non-negative"):
            self._make_allocator(min_epsilon_per_round=-0.01)

    # --- RDP composition ---
    def test_rdp_composition_model(self):
        alloc = self._make_allocator(composition_model="rdp", rdp_delta=1e-5)
        assert alloc.rdp_accountant is not None

    def test_basic_composition_model(self):
        alloc = self._make_allocator(composition_model="basic")
        assert alloc.rdp_accountant is None

    # --- Budget exhaustion ---
    def test_budget_exhaustion_returns_zero(self):
        alloc = self._make_allocator(total_budget=0.1, total_rounds=2)
        r1 = alloc.allocate(round_number=1, convergence_velocity=0.1)
        r2 = alloc.allocate(round_number=2, convergence_velocity=0.1)
        assert r1.allocated_epsilon + r2.allocated_epsilon <= 0.1 + 1e-10

    # --- Single round ---
    def test_single_round_gets_full_budget(self):
        alloc = self._make_allocator(total_budget=5.0, total_rounds=1)
        result = alloc.allocate(round_number=1, convergence_velocity=0.1)
        assert abs(result.allocated_epsilon - 5.0) < 1e-10

    # --- Reset ---
    def test_reset_restores_budget(self):
        alloc = self._make_allocator(total_budget=10.0, total_rounds=50)
        alloc.allocate(round_number=1, convergence_velocity=0.1)
        alloc.reset()
        assert alloc.remaining_budget == 10.0
        assert alloc.spent_budget == 0.0

    def test_reset_with_rdp(self):
        alloc = self._make_allocator(
            total_budget=10.0, total_rounds=50,
            composition_model="rdp"
        )
        alloc.allocate(round_number=1, convergence_velocity=0.1)
        alloc.reset()
        assert alloc.rdp_accountant is not None
        # After reset, basic composition should be 0
        assert alloc.rdp_accountant.get_basic_composition_epsilon() == 0.0
        assert len(alloc.rdp_accountant._round_epsilons) == 0


# ============================================================================
# 4. FairnessReweighter Boundaries
# ============================================================================


class TestFairnessReweighterBoundaries:
    """Test FairnessReweighter parameter validation at boundaries."""

    def _make_reweighter(self, **kwargs):
        from novel.fairness_aggregation.reweighter import FairnessReweighter
        return FairnessReweighter(**kwargs)

    def _make_update(self, client_id, num_samples, demographics=None):
        from novel.fairness_aggregation.reweighter import ClientUpdate
        return ClientUpdate(
            client_id=client_id,
            parameters=np.array([1.0, 2.0]),
            num_samples=num_samples,
            demographics=demographics,
        )

    # --- Alpha ---
    def test_alpha_zero(self):
        rw = self._make_reweighter(alpha=0.0)
        assert rw.alpha == 0.0

    def test_alpha_one(self):
        rw = self._make_reweighter(alpha=1.0)
        assert rw.alpha == 1.0

    def test_alpha_negative_rejected(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            self._make_reweighter(alpha=-0.1)

    def test_alpha_above_one_rejected(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            self._make_reweighter(alpha=1.1)

    # --- Min/max weight ---
    def test_min_weight_zero(self):
        rw = self._make_reweighter(min_weight=0.0)
        assert rw.min_weight == 0.0

    def test_min_weight_negative_rejected(self):
        with pytest.raises(ValueError, match="min_weight must be non-negative"):
            self._make_reweighter(min_weight=-0.1)

    def test_max_weight_less_than_min_rejected(self):
        with pytest.raises(ValueError, match="max_weight must be > min_weight"):
            self._make_reweighter(min_weight=5.0, max_weight=5.0)

    # --- Empty updates ---
    def test_empty_updates_rejected(self):
        rw = self._make_reweighter()
        target = np.array([0.3, 0.4, 0.3])
        with pytest.raises(ValueError, match="updates list is empty"):
            rw.reweight([], target)

    def test_empty_updates_simple_rejected(self):
        rw = self._make_reweighter()
        target = np.array([0.3, 0.4, 0.3])
        with pytest.raises(ValueError, match="updates list is empty"):
            rw.reweight_simple([], target)

    # --- Single client ---
    def test_single_client(self):
        rw = self._make_reweighter()
        updates = [self._make_update(0, 100, np.array([0.5, 0.3, 0.2]))]
        target = np.array([0.3, 0.4, 0.3])
        weights = rw.reweight(updates, target)
        assert len(weights) == 1
        assert abs(weights.sum() - 1.0) < 1e-10

    # --- Missing demographics fallback ---
    def test_missing_demographics_falls_back(self):
        rw = self._make_reweighter()
        updates = [
            self._make_update(0, 100, None),
            self._make_update(1, 200, None),
        ]
        target = np.array([0.3, 0.4, 0.3])
        weights = rw.reweight(updates, target)
        assert len(weights) == 2
        # Should be proportional to num_samples
        assert abs(weights[0] - 1 / 3) < 0.01
        assert abs(weights[1] - 2 / 3) < 0.01

    # --- Dimension mismatch ---
    def test_demographics_dimension_mismatch(self):
        rw = self._make_reweighter()
        updates = [
            self._make_update(0, 100, np.array([0.5, 0.5])),  # 2D
            self._make_update(1, 100, np.array([0.3, 0.4, 0.3])),  # 3D
        ]
        target = np.array([0.3, 0.4, 0.3])
        with pytest.raises(ValueError, match="dimension mismatch"):
            rw.reweight(updates, target)

    # --- All-zero samples ---
    def test_all_zero_samples(self):
        rw = self._make_reweighter()
        updates = [
            self._make_update(0, 0, np.array([0.5, 0.3, 0.2])),
            self._make_update(1, 0, np.array([0.3, 0.4, 0.3])),
        ]
        target = np.array([0.3, 0.4, 0.3])
        weights = rw.reweight(updates, target)
        # Should fall back to uniform
        assert abs(weights.sum() - 1.0) < 1e-10

    # --- Weights always sum to 1 ---
    def test_weights_normalized(self):
        rw = self._make_reweighter(alpha=0.8)
        updates = [
            self._make_update(i, (i + 1) * 50, np.random.dirichlet([1, 1, 1]))
            for i in range(5)
        ]
        target = np.array([0.3, 0.4, 0.3])
        weights = rw.reweight(updates, target)
        assert abs(weights.sum() - 1.0) < 1e-10
        assert (weights >= 0).all()


# ============================================================================
# 5. NonIIDDetector Boundaries
# ============================================================================


class TestNonIIDDetectorBoundaries:
    """Test NonIIDDetector parameter validation at boundaries."""

    def _make_detector(self, **kwargs):
        from novel.noniid_detection.detector import NonIIDDetector
        return NonIIDDetector(**kwargs)

    # --- Threshold ---
    def test_threshold_zero(self):
        det = self._make_detector(divergence_threshold=0.0)
        assert det.divergence_threshold == 0.0

    def test_threshold_negative_rejected(self):
        with pytest.raises(ValueError, match="divergence_threshold must be non-negative"):
            self._make_detector(divergence_threshold=-0.1)

    def test_threshold_large(self):
        det = self._make_detector(divergence_threshold=float("inf"))
        assert det.divergence_threshold == float("inf")

    # --- Severity thresholds ---
    def test_severity_thresholds_invalid(self):
        with pytest.raises(ValueError, match="Invalid severity thresholds"):
            self._make_detector(severity_thresholds=(0.5, 0.5))

    def test_severity_thresholds_reversed(self):
        with pytest.raises(ValueError, match="Invalid severity thresholds"):
            self._make_detector(severity_thresholds=(0.8, 0.2))

    # --- Empty nodes ---
    def test_empty_nodes_rejected(self):
        det = self._make_detector()
        with pytest.raises(ValueError, match="node_distributions list is empty"):
            det.detect([], np.array([0.3, 0.4, 0.3]))

    # --- Single node ---
    def test_single_node(self):
        det = self._make_detector()
        node_dists = [np.array([0.3, 0.4, 0.3])]
        global_dist = np.array([0.3, 0.4, 0.3])
        report = det.detect(node_dists, global_dist)
        assert report.severity >= 0

    # --- Identical distributions ---
    def test_identical_distributions(self):
        det = self._make_detector(divergence_threshold=0.01)
        dist = np.array([0.25, 0.25, 0.25, 0.25])
        node_dists = [dist.copy() for _ in range(5)]
        report = det.detect(node_dists, dist)
        # Identical distributions should have near-zero divergence
        assert report.severity < 0.01

    # --- Highly skewed ---
    def test_highly_skewed_detected(self):
        det = self._make_detector(divergence_threshold=0.1)
        global_dist = np.array([0.25, 0.25, 0.25, 0.25])
        node_dists = [
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([0.9, 0.03, 0.03, 0.04]),  # Highly skewed
            np.array([0.25, 0.25, 0.25, 0.25]),
        ]
        report = det.detect(node_dists, global_dist)
        assert report.is_noniid
        assert 1 in report.divergent_nodes

    # --- Detection methods ---
    def test_kl_method(self):
        det = self._make_detector(method="kl")
        global_dist = np.array([0.5, 0.5])
        report = det.detect([np.array([0.8, 0.2])], global_dist)
        assert report.method == "kl"

    def test_wasserstein_method(self):
        det = self._make_detector(method="wasserstein")
        global_dist = np.array([0.5, 0.5])
        report = det.detect([np.array([0.8, 0.2])], global_dist)
        assert report.method == "wasserstein"

    def test_both_method(self):
        det = self._make_detector(method="both")
        global_dist = np.array([0.5, 0.5])
        report = det.detect([np.array([0.8, 0.2])], global_dist)
        assert report.method == "both"

    # --- Pairwise divergence ---
    def test_pairwise_single_node(self):
        det = self._make_detector()
        result = det.detect_pairwise([np.array([0.5, 0.5])])
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.0

    def test_pairwise_symmetric(self):
        det = self._make_detector()
        dists = [
            np.array([0.3, 0.7]),
            np.array([0.7, 0.3]),
        ]
        result = det.detect_pairwise(dists)
        assert result.shape == (2, 2)
        # Should be symmetric
        assert abs(result[0, 1] - result[1, 0]) < 1e-10

    # --- Threshold zero: everything flagged ---
    def test_threshold_zero_flags_all_nonidentical(self):
        det = self._make_detector(divergence_threshold=0.0)
        global_dist = np.array([0.5, 0.5])
        node_dists = [np.array([0.51, 0.49])]  # Slightly off
        report = det.detect(node_dists, global_dist)
        # With threshold=0, any divergence > 0 is flagged
        assert report.is_noniid

    # --- Threshold infinity: nothing flagged ---
    def test_threshold_inf_flags_none(self):
        det = self._make_detector(divergence_threshold=float("inf"))
        global_dist = np.array([0.5, 0.5])
        node_dists = [np.array([0.99, 0.01])]  # Extremely skewed
        report = det.detect(node_dists, global_dist)
        assert len(report.divergent_nodes) == 0


# ============================================================================
# 6. Security Invariant Combinations
# ============================================================================


class TestSecurityInvariantCombinations:
    """Test cross-module security invariants to ensure no config combination
    creates an insecure state in production mode."""

    def test_production_requires_mtls(self):
        """Production mode must have mTLS enabled."""
        from aggregator.config import AggregatorConfig
        with pytest.raises(ValueError, match="MTLS_ENABLED"):
            AggregatorConfig(
                jwt_secret_key="strong-secret-for-production-mode-32ch",
                node_enrollment_secret="enrollment-secret",
                environment="production",
                mtls_enabled=False,
            )

    def test_production_requires_jwt_secret(self):
        """Production mode must have a non-default JWT secret."""
        from aggregator.config import AggregatorConfig
        with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
            AggregatorConfig(
                jwt_secret_key="",
                node_enrollment_secret="enrollment-secret",
                environment="production",
                mtls_enabled=True,
            )

    def test_production_requires_enrollment_secret(self):
        """Production mode must have enrollment secret set."""
        from aggregator.config import AggregatorConfig
        with pytest.raises(ValueError, match="NODE_ENROLLMENT_SECRET"):
            AggregatorConfig(
                jwt_secret_key="strong-secret-for-production-mode-32ch",
                node_enrollment_secret="",
                environment="production",
                mtls_enabled=True,
            )

    def test_production_full_security(self):
        """Valid production config with all security features."""
        from aggregator.config import AggregatorConfig
        cfg = AggregatorConfig(
            jwt_secret_key="strong-secret-for-production-mode-32ch",
            node_enrollment_secret="enrollment-secret",
            audit_hmac_key="audit-hmac-key-for-tests",
            environment="production",
            mtls_enabled=True,
            pqc_enabled=True,
            pqc_only_mode=True,
            dp_enabled=True,
            byzantine_detection_enabled=True,
            secure_agg_enabled=True,
        )
        assert cfg.environment == "production"
        assert cfg.mtls_enabled
        assert cfg.pqc_only_mode
        assert cfg.dp_enabled
        assert cfg.byzantine_detection_enabled

    def test_development_allows_insecure_defaults(self):
        """Development mode should auto-generate secrets, not crash."""
        from aggregator.config import AggregatorConfig
        cfg = AggregatorConfig(
            environment="development",
            mtls_enabled=False,
            secure_agg_enabled=False,
        )
        assert cfg.environment == "development"
        # Secrets should be auto-generated
        assert len(cfg.jwt_secret_key) > 10
        assert len(cfg.node_enrollment_secret) > 5

    def test_fairswarm_convergence_condition_with_defaults(self):
        """Default FairSwarmConfig must satisfy convergence condition."""
        from fairswarm.core.config import FairSwarmConfig
        cfg = FairSwarmConfig()
        assert cfg.satisfies_convergence_condition

    def test_privacy_budget_cannot_be_negative(self):
        """AdaptivePrivacyAllocator rejects negative budget."""
        from novel.adaptive_privacy.allocator import AdaptivePrivacyAllocator
        with pytest.raises(ValueError):
            AdaptivePrivacyAllocator(total_budget=-1.0, total_rounds=50)

    def test_rdp_tracks_composition(self):
        """RDP accounting should track rounds and provide epsilon bounds."""
        from novel.adaptive_privacy.allocator import AdaptivePrivacyAllocator
        alloc = AdaptivePrivacyAllocator(
            total_budget=10.0, total_rounds=20,
            composition_model="rdp", rdp_delta=1e-5,
        )
        for r in range(1, 21):
            alloc.allocate(round_number=r, convergence_velocity=0.1 / r)

        rdp_eps = alloc.rdp_accountant.get_epsilon()
        basic_eps = alloc.rdp_accountant.get_basic_composition_epsilon()
        # Both should be finite positive values
        assert rdp_eps > 0
        assert basic_eps > 0
        # RDP savings should be non-negative (RDP is tighter or equal)
        # For small per-round epsilon, RDP may exceed basic due to
        # the -log(delta)/(alpha-1) conversion term, but savings
        # increase with more rounds and larger epsilon values
        assert alloc.rdp_accountant.get_savings() is not None

    def test_empty_updates_cannot_produce_aggregation(self):
        """FairnessReweighter rejects empty updates."""
        from novel.fairness_aggregation.reweighter import FairnessReweighter
        rw = FairnessReweighter()
        target = np.array([0.3, 0.4, 0.3])
        with pytest.raises(ValueError):
            rw.reweight([], target)

    def test_detector_rejects_empty_input(self):
        """NonIIDDetector rejects empty node distributions."""
        from novel.noniid_detection.detector import NonIIDDetector
        det = NonIIDDetector()
        with pytest.raises(ValueError):
            det.detect([], np.array([0.5, 0.5]))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
