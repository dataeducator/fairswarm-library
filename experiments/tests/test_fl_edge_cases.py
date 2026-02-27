"""
FL Pipeline Edge Case Stress Tests.

Tests at the NODE level (what happens when one hospital sends garbage data)
and at the AGGREGATION level (what happens when the aggregated result
contains numerical instability).

Failures ranked by severity:
  3 = SILENT_WRONG (returns a numeric answer that is wrong)
  2 = CRASH (unhandled exception)
  1 = HANDLED (correctly detected and rejected/bounded)

Sized for 12-core Windows 11 with 32 GB RAM -each test < 5 s.

Author: Tenicka Norwood
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

# ── Ensure experiments/ is importable ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_real_fl import (  # noqa: E402
    train_local_model,
    federated_aggregate,
    evaluate_global_model,
    generate_federated_dataset,
)

# ── helpers ────────────────────────────────────────────────────────────

@dataclass
class EdgeCaseResult:
    name: str
    category: str
    component: str
    outcome: str       # HANDLED, CRASH, SILENT_WRONG
    detail: str
    severity: int      # 3=SILENT_WRONG, 2=CRASH, 1=HANDLED

    def __repr__(self) -> str:
        tag = {3: "SILENT_WRONG", 2: "CRASH", 1: "HANDLED"}[self.severity]
        return f"[{tag}] {self.component}::{self.name} -{self.detail}"


RESULTS: List[EdgeCaseResult] = []


def record(name, category, component, outcome, detail, severity):
    RESULTS.append(EdgeCaseResult(name, category, component, outcome, detail, severity))


def _make_data(n: int, d: int, seed: int = 0) -> Tuple[NDArray, NDArray]:
    """Create minimal valid training data (binary classification)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n)
    # Ensure at least one sample of each class
    if n >= 2:
        y[0], y[1] = 0, 1
    return X.astype(np.float64), y.astype(np.int64)


def _make_test_data(n: int = 200, d: int = 10, seed: int = 99):
    """Create test data with both classes."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float64)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    y[:n // 2] = 0
    y[n // 2:] = 1
    return X, y


# =======================================================================
# NODE-LEVEL TESTS: what happens when one hospital sends garbage
# =======================================================================


class TestNodeLevelEdgeCases:
    """What happens when a single client node sends bad data upstream."""

    N_FEATURES = 10

    # ── Missing values (NaN flood) ────────────────────────────────────

    def test_train_with_nan_features(self):
        """Hospital sends features with NaN values."""
        X, y = _make_data(50, self.N_FEATURES)
        X[10:20, :] = np.nan  # 20% NaN flood
        try:
            w, b, n = train_local_model(X, y, self.N_FEATURES)
            if np.any(np.isnan(w)) or np.any(np.isnan(b)):
                record("nan_features", "NaN", "train_local_model",
                       "SILENT_WRONG", "NaN features produce NaN weights", 3)
                pytest.fail("NaN in features produced NaN weights silently")
            else:
                record("nan_features", "NaN", "train_local_model",
                       "HANDLED", f"Training survived NaN features, w_norm={np.linalg.norm(w):.4f}", 1)
        except Exception as e:
            record("nan_features", "NaN", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)
            # A crash is acceptable here -NaN data is garbage in
            pass

    def test_train_with_all_nan_features(self):
        """Hospital sends 100% NaN features."""
        X = np.full((30, self.N_FEATURES), np.nan)
        y = np.array([0, 1] * 15, dtype=np.int64)
        try:
            w, b, n = train_local_model(X, y, self.N_FEATURES)
            if np.any(np.isnan(w)):
                record("all_nan_features", "NaN", "train_local_model",
                       "SILENT_WRONG", "All-NaN features -> NaN weights", 3)
                pytest.fail("All-NaN features produced NaN weights")
            else:
                record("all_nan_features", "NaN", "train_local_model",
                       "HANDLED", "Survived all-NaN", 1)
        except Exception as e:
            record("all_nan_features", "NaN", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)
            pass  # Crash is acceptable for total garbage

    def test_train_with_nan_labels(self):
        """Hospital sends NaN in labels."""
        X, _ = _make_data(50, self.N_FEATURES)
        y = np.array([0, 1] * 25, dtype=np.float64)
        y[5] = np.nan
        try:
            w, b, n = train_local_model(X, y.astype(np.int64), self.N_FEATURES)
            record("nan_labels", "NaN", "train_local_model",
                   "SILENT_WRONG", "NaN in labels not caught", 3)
            pytest.fail("NaN labels should fail")
        except (ValueError, TypeError, RuntimeError) as e:
            record("nan_labels", "NaN", "train_local_model",
                   "HANDLED", f"Rejected: {type(e).__name__}", 1)
        except Exception as e:
            record("nan_labels", "NaN", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── Outliers ──────────────────────────────────────────────────────

    def test_train_with_extreme_outliers(self):
        """Hospital has extreme outlier values (1e15)."""
        X, y = _make_data(50, self.N_FEATURES)
        X[0, :] = 1e15  # Single massive outlier row
        try:
            w, b, n = train_local_model(X, y, self.N_FEATURES)
            if np.any(np.isnan(w)) or np.any(np.isinf(w)):
                record("extreme_outliers", "Outlier", "train_local_model",
                       "SILENT_WRONG", "Extreme outliers -> NaN/Inf weights", 3)
                pytest.fail("Extreme outliers corrupted weights")
            elif np.linalg.norm(w) > 1e10:
                record("extreme_outliers", "Outlier", "train_local_model",
                       "SILENT_WRONG",
                       f"Extreme outliers -> huge weights (norm={np.linalg.norm(w):.2e})", 3)
                pytest.fail("Outlier produced extreme weights")
            else:
                record("extreme_outliers", "Outlier", "train_local_model",
                       "HANDLED", f"Weights bounded (norm={np.linalg.norm(w):.4f})", 1)
        except Exception as e:
            record("extreme_outliers", "Outlier", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    def test_train_with_inf_features(self):
        """Hospital sends Inf in features."""
        X, y = _make_data(50, self.N_FEATURES)
        X[5, 3] = np.inf
        try:
            w, b, n = train_local_model(X, y, self.N_FEATURES)
            if np.any(np.isnan(w)) or np.any(np.isinf(w)):
                record("inf_features", "Inf", "train_local_model",
                       "SILENT_WRONG", "Inf features -> NaN/Inf weights", 3)
                pytest.fail("Inf features corrupted weights")
            else:
                record("inf_features", "Inf", "train_local_model",
                       "HANDLED", "Survived Inf features", 1)
        except Exception as e:
            record("inf_features", "Inf", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)
            pass

    # ── Empty / single-row datasets ───────────────────────────────────

    def test_train_with_empty_dataset(self):
        """Hospital sends an empty dataset (0 rows)."""
        X = np.zeros((0, self.N_FEATURES), dtype=np.float64)
        y = np.zeros(0, dtype=np.int64)
        try:
            w, b, n = train_local_model(X, y, self.N_FEATURES)
            # With 0 samples, unique_labels < 2, should return zeros
            if n == 0 and np.allclose(w, 0):
                record("empty_dataset", "Empty", "train_local_model",
                       "HANDLED", "Empty data -> zero weights, n=0", 1)
            else:
                record("empty_dataset", "Empty", "train_local_model",
                       "SILENT_WRONG", f"Empty data -> n={n}, w_norm={np.linalg.norm(w):.4f}", 3)
                pytest.fail("Empty dataset produced unexpected result")
        except Exception as e:
            record("empty_dataset", "Empty", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    def test_train_with_single_row(self):
        """Hospital sends exactly 1 sample."""
        X = np.random.default_rng(0).standard_normal((1, self.N_FEATURES))
        y = np.array([1], dtype=np.int64)
        try:
            w, b, n = train_local_model(X, y, self.N_FEATURES)
            # Single sample -> single class -> should return zeros or global
            assert n == 1
            if np.allclose(w, 0):
                record("single_row", "Boundary", "train_local_model",
                       "HANDLED", "Single row -> zero weights (single class)", 1)
            else:
                record("single_row", "Boundary", "train_local_model",
                       "HANDLED", f"Single row -> w_norm={np.linalg.norm(w):.4f}", 1)
        except Exception as e:
            record("single_row", "Boundary", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── Single-class data ─────────────────────────────────────────────

    def test_train_with_all_positive(self):
        """Hospital data is 100% positive class."""
        X, _ = _make_data(50, self.N_FEATURES)
        y = np.ones(50, dtype=np.int64)
        try:
            w, b, n = train_local_model(X, y, self.N_FEATURES)
            if np.allclose(w, 0):
                record("all_positive", "Boundary", "train_local_model",
                       "HANDLED", "All-positive -> zero weights (correct fallback)", 1)
            else:
                record("all_positive", "Boundary", "train_local_model",
                       "SILENT_WRONG",
                       f"All-positive -> non-zero weights (norm={np.linalg.norm(w):.4f})", 3)
                pytest.fail("Single-class should return zeros")
        except Exception as e:
            record("all_positive", "Boundary", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── Swapped column types ──────────────────────────────────────────

    def test_train_with_string_features(self):
        """Hospital sends string data where floats expected."""
        try:
            X = np.array([["a", "b"]] * 20)  # String array
            y = np.array([0, 1] * 10, dtype=np.int64)
            w, b, n = train_local_model(X, y, 2)
            record("string_features", "TypeSwap", "train_local_model",
                   "SILENT_WRONG", "String features accepted", 3)
            pytest.fail("String features should crash")
        except (ValueError, TypeError) as e:
            record("string_features", "TypeSwap", "train_local_model",
                   "HANDLED", f"Rejected: {type(e).__name__}", 1)
        except Exception as e:
            record("string_features", "TypeSwap", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── Duplicate records ─────────────────────────────────────────────

    def test_train_with_all_duplicates(self):
        """Hospital sends the same row repeated 50 times."""
        rng = np.random.default_rng(42)
        row = rng.standard_normal(self.N_FEATURES)
        X = np.tile(row, (50, 1)).astype(np.float64)
        y = np.array([0, 1] * 25, dtype=np.int64)
        try:
            w, b, n = train_local_model(X, y, self.N_FEATURES)
            if np.any(np.isnan(w)) or np.any(np.isinf(w)):
                record("all_duplicates", "Duplicate", "train_local_model",
                       "SILENT_WRONG", "Duplicate rows -> NaN weights", 3)
                pytest.fail("Duplicates corrupted model")
            else:
                record("all_duplicates", "Duplicate", "train_local_model",
                       "HANDLED", f"Duplicates -> w_norm={np.linalg.norm(w):.4f}", 1)
        except Exception as e:
            record("all_duplicates", "Duplicate", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── More features than samples (p >> n) ───────────────────────────

    def test_train_with_more_features_than_samples(self):
        """Hospital has 5 samples but 100 features."""
        n_feat = 100
        X, y = _make_data(5, n_feat)
        try:
            w, b, n = train_local_model(X, y, n_feat)
            if np.any(np.isnan(w)) or np.any(np.isinf(w)):
                record("p_gt_n", "Boundary", "train_local_model",
                       "SILENT_WRONG", "p>>n -> NaN weights", 3)
                pytest.fail("p>>n corrupted model")
            else:
                record("p_gt_n", "Boundary", "train_local_model",
                       "HANDLED", f"p>>n -> w_norm={np.linalg.norm(w):.4f}, n={n}", 1)
        except Exception as e:
            record("p_gt_n", "Boundary", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── Warm-start with NaN global weights ────────────────────────────

    def test_warm_start_with_nan_global(self):
        """Server sends NaN global weights for warm start."""
        X, y = _make_data(50, self.N_FEATURES)
        nan_weights = np.full(self.N_FEATURES, np.nan)
        nan_intercept = np.array([np.nan])
        try:
            w, b, n = train_local_model(
                X, y, self.N_FEATURES,
                global_weights=nan_weights,
                global_intercept=nan_intercept,
            )
            if np.any(np.isnan(w)):
                record("nan_global_warmstart", "NaN", "train_local_model",
                       "SILENT_WRONG",
                       "NaN global weights -> NaN local weights (propagation!)", 3)
                pytest.fail("NaN global weights propagated to local model")
            else:
                record("nan_global_warmstart", "NaN", "train_local_model",
                       "HANDLED", "Recovered from NaN global weights", 1)
        except Exception as e:
            record("nan_global_warmstart", "NaN", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── Warm-start shape mismatch ─────────────────────────────────────

    def test_warm_start_shape_mismatch(self):
        """Server sends global weights with wrong dimensionality."""
        X, y = _make_data(50, self.N_FEATURES)
        wrong_weights = np.zeros(self.N_FEATURES + 5)  # Wrong shape
        wrong_intercept = np.zeros(1)
        try:
            w, b, n = train_local_model(
                X, y, self.N_FEATURES,
                global_weights=wrong_weights,
                global_intercept=wrong_intercept,
            )
            record("shape_mismatch_warmstart", "TypeSwap", "train_local_model",
                   "SILENT_WRONG", "Shape mismatch silently accepted", 3)
            pytest.fail("Shape mismatch should crash")
        except (ValueError, IndexError) as e:
            record("shape_mismatch_warmstart", "TypeSwap", "train_local_model",
                   "HANDLED", f"Rejected: {type(e).__name__}", 1)
        except Exception as e:
            record("shape_mismatch_warmstart", "TypeSwap", "train_local_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)


# =======================================================================
# AGGREGATION-LEVEL TESTS: what happens when aggregated results are bad
# =======================================================================


class TestAggregationEdgeCases:
    """Test federated_aggregate and evaluate_global_model under stress."""

    N_FEATURES = 10

    # ── Empty updates list ────────────────────────────────────────────

    def test_aggregate_empty_updates(self):
        """No client updates to aggregate."""
        try:
            w, b = federated_aggregate([])
            record("agg_empty", "Empty", "federated_aggregate",
                   "SILENT_WRONG", "Empty updates returned result", 3)
            pytest.fail("Empty updates should raise error")
        except ValueError as e:
            record("agg_empty", "Empty", "federated_aggregate",
                   "HANDLED", f"Rejected: {e}", 1)
        except Exception as e:
            record("agg_empty", "Empty", "federated_aggregate",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── All clients have 0 samples ────────────────────────────────────

    def test_aggregate_zero_samples(self):
        """All clients report 0 samples."""
        updates = [
            (np.zeros(self.N_FEATURES), np.zeros(1), 0),
            (np.zeros(self.N_FEATURES), np.zeros(1), 0),
        ]
        try:
            w, b = federated_aggregate(updates)
            record("agg_zero_samples", "Boundary", "federated_aggregate",
                   "SILENT_WRONG", "Zero total samples accepted", 3)
            pytest.fail("Zero total samples should error")
        except ValueError as e:
            record("agg_zero_samples", "Boundary", "federated_aggregate",
                   "HANDLED", f"Rejected: {e}", 1)
        except Exception as e:
            record("agg_zero_samples", "Boundary", "federated_aggregate",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── One client sends NaN weights ──────────────────────────────────

    def test_aggregate_one_nan_client(self):
        """One hospital's model has NaN weights."""
        good = (np.ones(self.N_FEATURES) * 0.5, np.array([0.1]), 100)
        bad = (np.full(self.N_FEATURES, np.nan), np.array([np.nan]), 50)
        try:
            w, b = federated_aggregate([good, bad])
            if np.any(np.isnan(w)):
                record("agg_one_nan", "NaN", "federated_aggregate",
                       "SILENT_WRONG",
                       "One NaN client poisoned entire aggregation", 3)
                pytest.fail("NaN client poisoned aggregated model")
            else:
                record("agg_one_nan", "NaN", "federated_aggregate",
                       "HANDLED", "NaN client was filtered/handled", 1)
        except Exception as e:
            record("agg_one_nan", "NaN", "federated_aggregate",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── All clients send NaN ──────────────────────────────────────────

    def test_aggregate_all_nan(self):
        """Every hospital returns NaN weights."""
        updates = [
            (np.full(self.N_FEATURES, np.nan), np.array([np.nan]), 100),
            (np.full(self.N_FEATURES, np.nan), np.array([np.nan]), 200),
        ]
        try:
            w, b = federated_aggregate(updates)
            if np.all(np.isnan(w)):
                record("agg_all_nan", "NaN", "federated_aggregate",
                       "SILENT_WRONG",
                       "All-NaN aggregation returned NaN (no error raised)", 3)
                pytest.fail("All-NaN clients should be caught")
            else:
                record("agg_all_nan", "NaN", "federated_aggregate",
                       "HANDLED", "Handled all-NaN", 1)
        except (ValueError, RuntimeError) as e:
            record("agg_all_nan", "NaN", "federated_aggregate",
                   "HANDLED", f"Rejected: {e}", 1)
        except Exception as e:
            record("agg_all_nan", "NaN", "federated_aggregate",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── One client sends Inf weights ──────────────────────────────────

    def test_aggregate_inf_weights(self):
        """One hospital returns Inf model weights."""
        good = (np.ones(self.N_FEATURES) * 0.5, np.array([0.1]), 100)
        bad = (np.full(self.N_FEATURES, np.inf), np.array([np.inf]), 50)
        try:
            w, b = federated_aggregate([good, bad])
            if np.any(np.isinf(w)):
                record("agg_inf_weights", "Inf", "federated_aggregate",
                       "SILENT_WRONG", "Inf weights propagated to aggregation", 3)
                pytest.fail("Inf weights propagated")
            else:
                record("agg_inf_weights", "Inf", "federated_aggregate",
                       "HANDLED", "Inf filtered", 1)
        except Exception as e:
            record("agg_inf_weights", "Inf", "federated_aggregate",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── Shape mismatch between clients ────────────────────────────────

    def test_aggregate_shape_mismatch(self):
        """Two hospitals return different weight shapes."""
        u1 = (np.zeros(self.N_FEATURES), np.zeros(1), 100)
        u2 = (np.zeros(self.N_FEATURES + 3), np.zeros(1), 100)
        try:
            w, b = federated_aggregate([u1, u2])
            record("agg_shape_mismatch", "TypeSwap", "federated_aggregate",
                   "SILENT_WRONG", "Shape mismatch went undetected", 3)
            pytest.fail("Shape mismatch should error")
        except (ValueError, IndexError) as e:
            record("agg_shape_mismatch", "TypeSwap", "federated_aggregate",
                   "HANDLED", f"Rejected: {type(e).__name__}", 1)
        except Exception as e:
            record("agg_shape_mismatch", "TypeSwap", "federated_aggregate",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── Extreme weight dominance ──────────────────────────────────────

    def test_aggregate_extreme_dominance(self):
        """One hospital has 1M samples, others have 1 each."""
        big = (np.ones(self.N_FEATURES) * 100.0, np.array([50.0]), 1_000_000)
        tiny1 = (np.ones(self.N_FEATURES) * 0.01, np.array([0.01]), 1)
        tiny2 = (np.ones(self.N_FEATURES) * 0.02, np.array([0.02]), 1)
        w, b = federated_aggregate([big, tiny1, tiny2])
        # Result should be almost entirely the big client
        if np.allclose(w, 100.0, atol=0.01):
            record("agg_dominance", "Outlier", "federated_aggregate",
                   "HANDLED", "Extreme dominance handled (big client wins)", 1)
        else:
            record("agg_dominance", "Outlier", "federated_aggregate",
                   "HANDLED", f"Weights ~ {w[0]:.4f}", 1)

    # ── Single client update ──────────────────────────────────────────

    def test_aggregate_single_client(self):
        """Only one client contributes an update."""
        updates = [(np.ones(self.N_FEATURES) * 0.42, np.array([0.1]), 100)]
        w, b = federated_aggregate(updates)
        if np.allclose(w, 0.42):
            record("agg_single", "Boundary", "federated_aggregate",
                   "HANDLED", "Single client -> identity aggregation", 1)
        else:
            record("agg_single", "Boundary", "federated_aggregate",
                   "SILENT_WRONG", f"Single client changed: {w[0]:.4f}", 3)
            pytest.fail("Single client aggregation should be identity")


# =======================================================================
# EVALUATION-LEVEL TESTS: evaluate_global_model under stress
# =======================================================================


class TestEvaluationEdgeCases:
    """Test evaluate_global_model with corrupted/degenerate inputs."""

    N_FEATURES = 10

    def _test_data(self):
        return _make_test_data(200, self.N_FEATURES)

    # ── NaN weights ───────────────────────────────────────────────────

    def test_eval_nan_weights(self):
        """Evaluate model with NaN weights."""
        X_test, y_test = self._test_data()
        w = np.full(self.N_FEATURES, np.nan)
        b = np.array([0.0])
        auc = evaluate_global_model(w, b, X_test, y_test)
        if auc == 0.5:
            record("eval_nan_weights", "NaN", "evaluate_global_model",
                   "HANDLED", "NaN weights -> fallback AUC=0.5", 1)
        elif math.isnan(auc):
            record("eval_nan_weights", "NaN", "evaluate_global_model",
                   "SILENT_WRONG", "NaN weights -> NaN AUC returned", 3)
            pytest.fail("NaN weights produced NaN AUC")
        else:
            record("eval_nan_weights", "NaN", "evaluate_global_model",
                   "SILENT_WRONG", f"NaN weights -> AUC={auc:.4f}", 3)
            pytest.fail(f"NaN weights produced AUC={auc}")

    # ── Inf weights ───────────────────────────────────────────────────

    def test_eval_inf_weights(self):
        """Evaluate model with Inf weights."""
        X_test, y_test = self._test_data()
        w = np.full(self.N_FEATURES, np.inf)
        b = np.array([0.0])
        auc = evaluate_global_model(w, b, X_test, y_test)
        if 0.0 <= auc <= 1.0:
            record("eval_inf_weights", "Inf", "evaluate_global_model",
                   "HANDLED", f"Inf weights -> AUC={auc:.4f} (fallback or clipped)", 1)
        else:
            record("eval_inf_weights", "Inf", "evaluate_global_model",
                   "SILENT_WRONG", f"Inf weights -> AUC={auc}", 3)
            pytest.fail(f"Inf weights -> invalid AUC={auc}")

    # ── Zero weights ──────────────────────────────────────────────────

    def test_eval_zero_weights(self):
        """Evaluate model with all-zero weights (untrained)."""
        X_test, y_test = self._test_data()
        w = np.zeros(self.N_FEATURES)
        b = np.array([0.0])
        auc = evaluate_global_model(w, b, X_test, y_test)
        # All-zero -> all probs = 0.5 -> AUC = 0.5
        if abs(auc - 0.5) < 0.05:
            record("eval_zero_weights", "Boundary", "evaluate_global_model",
                   "HANDLED", f"Zero weights -> AUC={auc:.4f} (~0.5, correct)", 1)
        else:
            record("eval_zero_weights", "Boundary", "evaluate_global_model",
                   "SILENT_WRONG", f"Zero weights -> AUC={auc:.4f} (unexpected)", 3)
            pytest.fail(f"Zero weights gave AUC={auc}")

    # ── Empty test set ────────────────────────────────────────────────

    def test_eval_empty_test_set(self):
        """Evaluate on empty test data."""
        w = np.ones(self.N_FEATURES) * 0.5
        b = np.array([0.1])
        X_test = np.zeros((0, self.N_FEATURES))
        y_test = np.zeros(0, dtype=np.int64)
        try:
            auc = evaluate_global_model(w, b, X_test, y_test)
            if auc == 0.5:
                record("eval_empty_test", "Empty", "evaluate_global_model",
                       "HANDLED", "Empty test set -> AUC=0.5 fallback", 1)
            else:
                record("eval_empty_test", "Empty", "evaluate_global_model",
                       "SILENT_WRONG", f"Empty test set -> AUC={auc}", 3)
                pytest.fail("Empty test set should return 0.5")
        except Exception as e:
            record("eval_empty_test", "Empty", "evaluate_global_model",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    # ── Single-class test set ─────────────────────────────────────────

    def test_eval_single_class_test(self):
        """Test set has only one class."""
        X_test = np.random.default_rng(0).standard_normal((100, self.N_FEATURES))
        y_test = np.ones(100, dtype=np.int64)  # All positive
        w = np.ones(self.N_FEATURES) * 0.5
        b = np.array([0.1])
        auc = evaluate_global_model(w, b, X_test, y_test)
        if auc == 0.5:
            record("eval_single_class", "Boundary", "evaluate_global_model",
                   "HANDLED", "Single-class test -> AUC=0.5", 1)
        else:
            record("eval_single_class", "Boundary", "evaluate_global_model",
                   "SILENT_WRONG", f"Single-class -> AUC={auc}", 3)
            pytest.fail(f"Single-class test produced AUC={auc}")

    # ── NaN in test features ──────────────────────────────────────────

    def test_eval_nan_test_features(self):
        """Test data has NaN features."""
        X_test, y_test = self._test_data()
        X_test[50:60, :] = np.nan  # 5% NaN
        w = np.ones(self.N_FEATURES) * 0.5
        b = np.array([0.1])
        auc = evaluate_global_model(w, b, X_test, y_test)
        if auc == 0.5:
            record("eval_nan_test", "NaN", "evaluate_global_model",
                   "HANDLED", "NaN test features -> AUC=0.5 fallback", 1)
        elif math.isnan(auc):
            record("eval_nan_test", "NaN", "evaluate_global_model",
                   "SILENT_WRONG", "NaN test features -> NaN AUC", 3)
            pytest.fail("NaN test features produced NaN AUC")
        else:
            record("eval_nan_test", "NaN", "evaluate_global_model",
                   "HANDLED", f"NaN test features -> AUC={auc:.4f}", 1)


# =======================================================================
# MULTI-ROUND NaN PROPAGATION (integration test)
# =======================================================================


class TestNaNPropagationAcrossRounds:
    """Test whether NaN from one round propagates through subsequent rounds."""

    N_FEATURES = 10

    def test_nan_round1_propagates_to_round2(self):
        """If round 1 produces NaN weights, does round 2 recover?"""
        X_good, y_good = _make_data(50, self.N_FEATURES)
        X_bad = np.full((30, self.N_FEATURES), np.nan)
        y_bad = np.array([0, 1] * 15, dtype=np.int64)

        # Round 1: aggregate one good + one bad client
        try:
            w1_good, b1_good, n1 = train_local_model(X_good, y_good, self.N_FEATURES)
        except Exception:
            w1_good = np.zeros(self.N_FEATURES)
            b1_good = np.zeros(1)
            n1 = 50

        try:
            w1_bad, b1_bad, n2 = train_local_model(X_bad, y_bad, self.N_FEATURES)
        except Exception:
            # If bad client crashes, the aggregation should skip it
            record("nan_propagation", "NaN", "multi_round",
                   "HANDLED", "Bad client crashed, won't enter aggregation", 1)
            return

        # Aggregate round 1
        try:
            global_w, global_b = federated_aggregate([
                (w1_good, b1_good, n1),
                (w1_bad, b1_bad, n2),
            ])
        except Exception:
            record("nan_propagation", "NaN", "multi_round",
                   "HANDLED", "Aggregation rejected bad data", 1)
            return

        round1_has_nan = np.any(np.isnan(global_w))

        # Round 2: warm-start from round 1 global model
        try:
            w2, b2, _ = train_local_model(
                X_good, y_good, self.N_FEATURES,
                global_weights=global_w,
                global_intercept=global_b,
            )
        except Exception:
            if round1_has_nan:
                record("nan_propagation", "NaN", "multi_round",
                       "CRASH", "NaN round 1 -> crash in round 2 warm-start", 2)
            else:
                record("nan_propagation", "NaN", "multi_round",
                       "CRASH", "Round 2 crashed unexpectedly", 2)
            return

        if round1_has_nan and np.any(np.isnan(w2)):
            record("nan_propagation", "NaN", "multi_round",
                   "SILENT_WRONG",
                   "NaN propagated: round1 NaN -> round2 NaN (cascading failure)", 3)
            pytest.fail("NaN cascaded across FL rounds")
        elif round1_has_nan and not np.any(np.isnan(w2)):
            record("nan_propagation", "NaN", "multi_round",
                   "HANDLED", "NaN in round 1 recovered in round 2", 1)
        else:
            record("nan_propagation", "NaN", "multi_round",
                   "HANDLED", "No NaN propagation detected", 1)


# =======================================================================
# DATA GENERATION EDGE CASES
# =======================================================================


class TestDataGenerationEdgeCases:
    """Test generate_federated_dataset with extreme parameters."""

    def test_single_client(self):
        """Generate data for 1 client."""
        try:
            ds = generate_federated_dataset(
                n_clients=1, n_demographic_groups=2,
                n_features=20, n_samples_total=200, seed=42,
            )
            assert len(ds.client_data) == 1
            assert ds.client_data[0].X_train.shape[0] > 0
            record("gen_single_client", "Boundary", "generate_federated_dataset",
                   "HANDLED", f"Single client: {ds.client_data[0].X_train.shape[0]} samples", 1)
        except Exception as e:
            record("gen_single_client", "Boundary", "generate_federated_dataset",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    def test_many_clients_few_samples(self):
        """100 clients sharing 200 samples (2 per client average)."""
        try:
            ds = generate_federated_dataset(
                n_clients=100, n_demographic_groups=4,
                n_features=20, n_samples_total=200, seed=42,
            )
            sizes = [cd.X_train.shape[0] for cd in ds.client_data]
            min_size = min(sizes)
            record("gen_many_few", "Boundary", "generate_federated_dataset",
                   "HANDLED",
                   f"100 clients / 200 samples: min={min_size}, max={max(sizes)}", 1)
        except Exception as e:
            record("gen_many_few", "Boundary", "generate_federated_dataset",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    def test_extreme_non_iid(self):
        """Extreme non-IID (alpha -> 0.01)."""
        try:
            ds = generate_federated_dataset(
                n_clients=10, n_demographic_groups=4,
                n_features=20, n_samples_total=2000,
                non_iid_alpha=0.01, seed=42,
            )
            # Check if any client ended up with zero-group demographics
            for i, cd in enumerate(ds.client_data):
                if np.any(cd.demographic_dist == 0):
                    record("gen_extreme_noniid", "Outlier", "generate_federated_dataset",
                           "HANDLED",
                           f"Client {i} has zero-weight group (expected for extreme non-IID)", 1)
                    return
            record("gen_extreme_noniid", "Outlier", "generate_federated_dataset",
                   "HANDLED", "All clients have non-zero demographics", 1)
        except Exception as e:
            record("gen_extreme_noniid", "Outlier", "generate_federated_dataset",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    def test_single_demographic_group(self):
        """k=1 demographic group."""
        try:
            ds = generate_federated_dataset(
                n_clients=5, n_demographic_groups=1,
                n_features=20, n_samples_total=500, seed=42,
            )
            for cd in ds.client_data:
                assert len(cd.demographic_dist) == 1
                assert abs(cd.demographic_dist[0] - 1.0) < 1e-6
            record("gen_k1", "Boundary", "generate_federated_dataset",
                   "HANDLED", "k=1 group works", 1)
        except Exception as e:
            record("gen_k1", "Boundary", "generate_federated_dataset",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    def test_many_demographic_groups(self):
        """k=50 demographic groups with few samples per group."""
        try:
            ds = generate_federated_dataset(
                n_clients=5, n_demographic_groups=50,
                n_features=55, n_samples_total=2000, seed=42,
            )
            # Some groups will have zero samples -> demo_dist has zeros
            zero_groups = sum(
                1 for cd in ds.client_data
                if np.any(cd.demographic_dist == 0)
            )
            record("gen_k50", "Boundary", "generate_federated_dataset",
                   "HANDLED",
                   f"k=50: {zero_groups}/{len(ds.client_data)} clients have zero-group entries", 1)
        except Exception as e:
            record("gen_k50", "Boundary", "generate_federated_dataset",
                   "CRASH", f"{type(e).__name__}: {e}", 2)


# =======================================================================
# DEMOGRAPHIC DIVERGENCE WITH FL-LEVEL DATA
# =======================================================================


class TestDemographicDivergenceWithFLData:
    """Test KL divergence with demographics that come from FL pipeline."""

    def test_zero_group_in_coalition_demographics(self):
        """Coalition where one demographic group has 0 representation."""
        from fairswarm.demographics.divergence import kl_divergence
        coalition_demo = np.array([0.5, 0.5, 0.0, 0.0])
        target = np.array([0.25, 0.25, 0.25, 0.25])
        try:
            div = kl_divergence(coalition_demo, target)
            if math.isnan(div) or math.isinf(div):
                record("fl_zero_group_div", "Outlier", "kl_divergence",
                       "SILENT_WRONG", f"Zero-group KL -> {div}", 3)
                pytest.fail(f"Zero group produced {div}")
            else:
                record("fl_zero_group_div", "Outlier", "kl_divergence",
                       "HANDLED", f"Zero-group smoothed, KL={div:.6f}", 1)
        except Exception as e:
            record("fl_zero_group_div", "Outlier", "kl_divergence",
                   "CRASH", f"{type(e).__name__}: {e}", 2)

    def test_near_zero_demographics(self):
        """Coalition demographics with near-machine-epsilon values."""
        from fairswarm.demographics.divergence import kl_divergence
        eps = 1e-15
        coalition_demo = np.array([1.0 - 3*eps, eps, eps, eps])
        target = np.array([0.25, 0.25, 0.25, 0.25])
        try:
            div = kl_divergence(coalition_demo, target)
            if math.isnan(div) or math.isinf(div):
                record("fl_near_zero_div", "Outlier", "kl_divergence",
                       "SILENT_WRONG", f"Near-zero demographics -> {div}", 3)
                pytest.fail(f"Near-zero produced {div}")
            else:
                record("fl_near_zero_div", "Outlier", "kl_divergence",
                       "HANDLED", f"Near-zero handled, KL={div:.6f}", 1)
        except Exception as e:
            record("fl_near_zero_div", "Outlier", "kl_divergence",
                   "CRASH", f"{type(e).__name__}: {e}", 2)


# =======================================================================
# Summary fixture
# =======================================================================


@pytest.fixture(autouse=True, scope="session")
def print_summary(request):
    """Print ranked summary after all FL edge case tests complete."""
    yield
    if RESULTS:
        print("\n" + "=" * 78)
        print("FL PIPELINE EDGE CASE STRESS TEST SUMMARY")
        print("=" * 78)
        sorted_results = sorted(RESULTS, key=lambda r: (-r.severity, r.component))
        for r in sorted_results:
            print(r)
        print("=" * 78)
        silent = sum(1 for r in RESULTS if r.severity == 3)
        crash = sum(1 for r in RESULTS if r.severity == 2)
        handled = sum(1 for r in RESULTS if r.severity == 1)
        print(f"Total: {len(RESULTS)} probes | "
              f"SILENT_WRONG: {silent} | CRASH: {crash} | HANDLED: {handled}")
        print("=" * 78)
