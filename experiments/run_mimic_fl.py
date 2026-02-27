"""
MIMIC-III Federated Learning Experiment with FairSwarm.

Runs real federated logistic regression training on the MIMIC-III ICU
mortality cohort (13,169 patients, 20 FL clients, 5 demographic groups)
using FairSwarm for coalition selection alongside baseline methods.

Produces results for the MIMIC-III Clinical Validation section of the paper.

Data Access:
    MIMIC-III is a restricted-access dataset. To run this experiment:
    1. Complete CITI training and sign the PhysioNet Data Use Agreement
       at https://physionet.org/content/mimiciii/
    2. Run the cohort extraction pipeline to produce
       data/processed/mimic_fl_cohort.parquet
    3. Do NOT commit MIMIC data to version control (blocked by .gitignore)

Usage:
    python -m experiments.run_mimic_fl [--n_trials 5] [--n_rounds 30] [--seed 42]

Author: Tenicka Norwood
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# FairSwarm imports
from fairswarm import Client, FairSwarm, FairSwarmConfig
from fairswarm.demographics.distribution import DemographicDistribution
from fairswarm.demographics.divergence import kl_divergence
from fairswarm.fitness import AccuracyFairnessFitness
from experiments.baselines import GreedySelection, RandomSelection


# Constants


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "mimic_fl"

DEMOGRAPHIC_GROUPS = ("white", "black", "hispanic", "asian", "other")
DEMOGRAPHIC_COLS = [f"race_{g}" for g in DEMOGRAPHIC_GROUPS]

COALITION_SIZE = 8  # Select 8 of 20 clients per round
N_LOCAL_EPOCHS = 5  # Local training epochs per client per round



# Data Loading



def load_mimic_data() -> tuple[pd.DataFrame, dict]:
    """Load MIMIC-III FL cohort and metadata."""
    cohort_path = DATA_DIR / "mimic_fl_cohort.parquet"
    meta_path = DATA_DIR / "mimic_fl_meta.json"

    if not cohort_path.exists():
        raise FileNotFoundError(f"MIMIC cohort not found: {cohort_path}")

    df = pd.read_parquet(cohort_path)
    with open(meta_path) as f:
        meta = json.load(f)

    return df, meta


def build_clients(df: pd.DataFrame, meta: dict) -> list[Client]:
    """Build FairSwarm Client objects from MIMIC-III data."""
    feature_cols = meta["feature_cols"]
    clients = []

    for client_id in sorted(df["fl_client_id"].unique()):
        subset = df[df["fl_client_id"] == client_id]
        n_samples = len(subset)

        # Compute aggregate demographic distribution for this client
        demo_counts = {
            group: int(subset[f"race_{group}"].sum())
            for group in DEMOGRAPHIC_GROUPS
        }
        total = sum(demo_counts.values())
        if total == 0:
            continue

        demo_array = np.array(
            [demo_counts[g] / total for g in DEMOGRAPHIC_GROUPS],
            dtype=np.float64,
        )

        client = Client(
            id=client_id,
            demographics=demo_array,
            dataset_size=n_samples,
            communication_cost=0.5,
            data_quality=1.0,
            metadata={"demo_counts": demo_counts},
        )
        clients.append(client)

    return clients



# Federated Learning Training



@dataclass
class FLResult:
    """Result of a single federated learning run."""

    auc_roc: float
    auprc: float
    precision: float
    recall: float
    f1: float
    per_group_tpr: dict[str, float]
    per_group_fpr: dict[str, float]
    confusion_matrix: dict[str, int]
    per_group_cm: dict[str, dict[str, int | float]]


def fedavg_train(
    df: pd.DataFrame,
    meta: dict,
    selected_client_ids: list[str],
    n_rounds: int = 30,
    n_local_epochs: int = N_LOCAL_EPOCHS,
    seed: int = 42,
) -> FLResult:
    """
    Run FedAvg training with logistic regression.

    Each selected client trains locally for n_local_epochs, then
    model parameters are averaged (weighted by dataset size).
    """
    rng = np.random.default_rng(seed)
    feature_cols = meta["feature_cols"]

    # Prepare client data splits
    client_data = {}
    for cid in selected_client_ids:
        subset = df[df["fl_client_id"] == cid]
        X = subset[feature_cols].values.astype(np.float64)
        y = subset["mortality"].values.astype(np.float64)
        client_data[cid] = (X, y)

    # Prepare global test set (all data from ALL clients, not just selected)
    X_all = df[feature_cols].values.astype(np.float64)
    y_all = df["mortality"].values.astype(np.float64)
    race_all = {
        g: df[f"race_{g}"].values.astype(bool) for g in DEMOGRAPHIC_GROUPS
    }

    # Standardize features using training data statistics
    scaler = StandardScaler()
    X_train_all = np.vstack([X for X, _ in client_data.values()])
    scaler.fit(X_train_all)
    X_all_scaled = scaler.transform(X_all)
    client_data_scaled = {
        cid: (scaler.transform(X), y) for cid, (X, y) in client_data.items()
    }

    # Initialize global model parameters
    n_features = len(feature_cols)
    global_coef = rng.standard_normal(n_features) * 0.01
    global_intercept = 0.0

    # FedAvg rounds
    for round_idx in range(n_rounds):
        local_coefs = []
        local_intercepts = []
        local_weights = []

        for cid in selected_client_ids:
            X_local, y_local = client_data_scaled[cid]
            n_local = len(y_local)
            if n_local < 2 or len(np.unique(y_local)) < 2:
                continue

            # Initialize local model with global parameters
            model = LogisticRegression(
                max_iter=n_local_epochs * 50,
                warm_start=True,
                solver="lbfgs",
                random_state=seed + round_idx,
                C=1.0,
            )
            # Set initial weights
            model.classes_ = np.array([0.0, 1.0])
            model.coef_ = global_coef.reshape(1, -1).copy()
            model.intercept_ = np.array([global_intercept])

            # Local training
            model.fit(X_local, y_local)

            local_coefs.append(model.coef_[0])
            local_intercepts.append(model.intercept_[0])
            local_weights.append(n_local)

        if not local_coefs:
            continue

        # Weighted average (FedAvg)
        total_weight = sum(local_weights)
        global_coef = sum(
            w * c for w, c in zip(local_weights, local_coefs)
        ) / total_weight
        global_intercept = sum(
            w * b for w, b in zip(local_weights, local_intercepts)
        ) / total_weight

    # Final evaluation on global test set
    final_model = LogisticRegression()
    final_model.classes_ = np.array([0.0, 1.0])
    final_model.coef_ = global_coef.reshape(1, -1)
    final_model.intercept_ = np.array([global_intercept])

    y_prob = final_model.predict_proba(X_all_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    auc_roc = roc_auc_score(y_all, y_prob)

    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    auprc = average_precision_score(y_all, y_prob)
    precision = precision_score(y_all, y_pred, zero_division=0)
    recall = recall_score(y_all, y_pred, zero_division=0)
    f1 = f1_score(y_all, y_pred, zero_division=0)

    cm = confusion_matrix(y_all, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Per-group metrics
    per_group_tpr = {}
    per_group_fpr = {}
    per_group_cm = {}

    for group in DEMOGRAPHIC_GROUPS:
        mask = race_all[group]
        y_g = y_all[mask]
        yp_g = y_pred[mask]

        if len(y_g) == 0 or len(np.unique(y_g)) < 2:
            per_group_tpr[group] = 0.0
            per_group_fpr[group] = 0.0
            per_group_cm[group] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "tpr": 0.0, "fpr": 0.0}
            continue

        cm_g = confusion_matrix(y_g, yp_g, labels=[0, 1])
        tn_g, fp_g, fn_g, tp_g = cm_g.ravel()

        tpr_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
        fpr_g = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0.0

        per_group_tpr[group] = round(tpr_g, 4)
        per_group_fpr[group] = round(fpr_g, 4)
        per_group_cm[group] = {
            "tp": int(tp_g),
            "fp": int(fp_g),
            "fn": int(fn_g),
            "tn": int(tn_g),
            "tpr": round(tpr_g, 4),
            "fpr": round(fpr_g, 4),
        }

    return FLResult(
        auc_roc=round(auc_roc, 4),
        auprc=round(auprc, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        per_group_tpr=per_group_tpr,
        per_group_fpr=per_group_fpr,
        confusion_matrix={"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        per_group_cm=per_group_cm,
    )



# Selection Methods



def compute_coalition_demdiv(
    selected_ids: list[str],
    clients: list[Client],
    target: DemographicDistribution,
) -> float:
    """Compute DemDiv for a coalition specified by client IDs."""
    id_to_idx = {c.id: i for i, c in enumerate(clients)}
    indices = [id_to_idx[cid] for cid in selected_ids]

    if not indices:
        return float("inf")

    demo_vecs = [np.asarray(clients[i].demographics) for i in indices]
    coalition_demo = np.mean(demo_vecs, axis=0)
    return float(kl_divergence(coalition_demo, target.as_array()))


def compute_equalized_odds_gap(per_group_tpr: dict[str, float]) -> float:
    """Compute equalized odds gap (max TPR - min TPR)."""
    tprs = [v for v in per_group_tpr.values() if v > 0]
    if len(tprs) < 2:
        return 0.0
    return max(tprs) - min(tprs)


def select_fairswarm(
    clients: list[Client],
    target: DemographicDistribution,
    df: pd.DataFrame,
    meta: dict,
    n_rounds: int,
    seed: int,
) -> list[str]:
    """Select coalition using FairSwarm PSO."""
    config = FairSwarmConfig(
        swarm_size=50,
        max_iterations=100,
        fairness_coefficient=0.5,
        fairness_weight=0.3,
        adaptive_fairness=True,
        epsilon_fair=0.01,
    )

    fitness = AccuracyFairnessFitness(
        target_distribution=target,
        fairness_weight=0.3,
    )

    optimizer = FairSwarm(
        clients=clients,
        coalition_size=COALITION_SIZE,
        config=config,
        target_distribution=target,
        seed=seed,
    )

    result = optimizer.optimize(fitness_fn=fitness, n_iterations=100)
    return [clients[i].id for i in result.coalition]


def select_random(
    clients: list[Client],
    target: DemographicDistribution,
    seed: int,
) -> list[str]:
    """Select coalition randomly (best of 100 trials)."""
    fitness = AccuracyFairnessFitness(
        target_distribution=target,
        fairness_weight=0.3,
    )
    selector = RandomSelection(clients, COALITION_SIZE, seed=seed)
    coalition, _ = selector.select(fitness, n_trials=100)
    return [clients[i].id for i in coalition]


def select_greedy_fair(
    clients: list[Client],
    target: DemographicDistribution,
) -> list[str]:
    """Select coalition using greedy fairness-only optimization."""
    from fairswarm.fitness import DemographicFitness

    fitness = DemographicFitness(target_distribution=target, divergence_weight=1.0)
    selector = GreedySelection(clients, COALITION_SIZE)
    coalition, _ = selector.select(fitness)
    return [clients[i].id for i in coalition]


def select_greedy_acc_fair(
    clients: list[Client],
    target: DemographicDistribution,
) -> list[str]:
    """Select coalition using greedy accuracy+fairness optimization."""
    fitness = AccuracyFairnessFitness(
        target_distribution=target,
        fairness_weight=0.3,
    )
    selector = GreedySelection(clients, COALITION_SIZE)
    coalition, _ = selector.select(fitness)
    return [clients[i].id for i in coalition]


def select_round_robin(
    clients: list[Client],
) -> list[str]:
    """Select coalition via round-robin (first 8 in sorted order)."""
    return [c.id for c in clients[:COALITION_SIZE]]


def select_all_clients(clients: list[Client]) -> list[str]:
    """FedAvg baseline: use all clients."""
    return [c.id for c in clients]



# Experiment Runner



def run_experiment(
    n_trials: int = 5,
    n_rounds: int = 30,
    seed: int = 42,
) -> dict:
    """Run the full MIMIC-III FL experiment."""
    print("=" * 70)
    print("MIMIC-III Federated Learning Experiment with FairSwarm")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading MIMIC-III cohort...")
    df, meta = load_mimic_data()
    print(f"  Loaded {len(df)} patients, {df['fl_client_id'].nunique()} clients")
    print(f"  Features: {len(meta['feature_cols'])}, Groups: {len(DEMOGRAPHIC_GROUPS)}")

    # Build clients
    print("\n[2/4] Building FL clients...")
    clients = build_clients(df, meta)
    print(f"  Created {len(clients)} clients")

    # Compute target distribution (population-level demographics)
    pop_counts = {
        g: int(df[f"race_{g}"].sum()) for g in DEMOGRAPHIC_GROUPS
    }
    target = DemographicDistribution.from_counts(pop_counts)
    print(f"  Target distribution: {dict(zip(DEMOGRAPHIC_GROUPS, [f'{v:.3f}' for v in target.as_array()]))}")

    # Define methods
    methods = {
        "fairswarm": lambda s: select_fairswarm(clients, target, df, meta, n_rounds, s),
        "random": lambda s: select_random(clients, target, s),
        "round_robin": lambda _: select_round_robin(clients),
        "all_clients": lambda _: select_all_clients(clients),
        "greedy_fair": lambda _: select_greedy_fair(clients, target),
        "greedy_acc_fair": lambda _: select_greedy_acc_fair(clients, target),
    }

    # Run trials
    print(f"\n[3/4] Running {n_trials} trials x {len(methods)} methods x {n_rounds} rounds...")
    results: dict[str, list[dict]] = {name: [] for name in methods}

    for trial in range(n_trials):
        trial_seed = seed + trial
        print(f"\n  Trial {trial + 1}/{n_trials} (seed={trial_seed})")

        for method_name, select_fn in methods.items():
            t0 = time.time()

            # Select coalition
            selected_ids = select_fn(trial_seed)
            demdiv = compute_coalition_demdiv(selected_ids, clients, target)

            # Train FL model
            fl_result = fedavg_train(
                df, meta, selected_ids, n_rounds=n_rounds,
                n_local_epochs=N_LOCAL_EPOCHS, seed=trial_seed,
            )

            eq_odds_gap = compute_equalized_odds_gap(fl_result.per_group_tpr)
            elapsed = time.time() - t0

            results[method_name].append({
                "trial": trial,
                "seed": trial_seed,
                "selected_clients": selected_ids,
                "n_selected": len(selected_ids),
                "auc_roc": fl_result.auc_roc,
                "auprc": fl_result.auprc,
                "precision": fl_result.precision,
                "recall": fl_result.recall,
                "f1": fl_result.f1,
                "demdiv": demdiv,
                "eqodds_gap": eq_odds_gap,
                "per_group_tpr": fl_result.per_group_tpr,
                "per_group_fpr": fl_result.per_group_fpr,
                "confusion_matrix": fl_result.confusion_matrix,
                "per_group_cm": fl_result.per_group_cm,
                "elapsed_seconds": round(elapsed, 2),
            })

            print(
                f"    {method_name:<18} AUC={fl_result.auc_roc:.4f}  "
                f"DemDiv={demdiv:.6f}  EqOdds={eq_odds_gap:.4f}  "
                f"({elapsed:.1f}s)"
            )

    # Aggregate results
    print(f"\n[4/4] Aggregating results...")
    summary = {}

    for method_name, trial_results in results.items():
        aucs = [r["auc_roc"] for r in trial_results]
        auprcs = [r["auprc"] for r in trial_results]
        demdevs = [r["demdiv"] for r in trial_results]
        eqodds = [r["eqodds_gap"] for r in trial_results]
        f1s = [r["f1"] for r in trial_results]

        def ci95(vals):
            m = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            return [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)]

        # Aggregate per-group TPR across trials
        all_group_tpr = {}
        for g in DEMOGRAPHIC_GROUPS:
            tprs = [r["per_group_tpr"].get(g, 0.0) for r in trial_results]
            all_group_tpr[g] = round(float(np.mean(tprs)), 4)

        # Use last trial's confusion matrix as representative
        last_cm = trial_results[-1]["confusion_matrix"]
        last_pg_cm = trial_results[-1]["per_group_cm"]

        summary[method_name] = {
            "n_trials": len(trial_results),
            "auc_mean": round(float(np.mean(aucs)), 4),
            "auc_std": round(float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0, 4),
            "auc_ci95": ci95(aucs),
            "auprc_mean": round(float(np.mean(auprcs)), 4),
            "auprc_std": round(float(np.std(auprcs, ddof=1)) if len(auprcs) > 1 else 0.0, 4),
            "precision_mean": round(float(np.mean([r["precision"] for r in trial_results])), 4),
            "recall_mean": round(float(np.mean([r["recall"] for r in trial_results])), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0, 4),
            "demdiv_mean": round(float(np.mean(demdevs)), 4),
            "demdiv_std": round(float(np.std(demdevs, ddof=1)) if len(demdevs) > 1 else 0.0, 4),
            "demdiv_ci95": ci95(demdevs),
            "eqodds_mean": round(float(np.mean(eqodds)), 4),
            "eqodds_std": round(float(np.std(eqodds, ddof=1)) if len(eqodds) > 1 else 0.0, 4),
            "eqodds_ci95": ci95(eqodds),
            "per_group_tpr": all_group_tpr,
            "confusion_matrix": last_cm,
            "per_group_confusion_matrices": last_pg_cm,
        }

    # Add metadata
    output = {
        **summary,
        "_meta": {
            "n_clients": len(clients),
            "coalition_size": COALITION_SIZE,
            "n_rounds": n_rounds,
            "n_trials": n_trials,
            "n_local_epochs": N_LOCAL_EPOCHS,
            "n_patients": len(df),
            "n_features": len(meta["feature_cols"]),
            "demographic_groups": list(DEMOGRAPHIC_GROUPS),
            "target_distribution": {
                g: round(float(v), 4)
                for g, v in zip(DEMOGRAPHIC_GROUPS, target.as_array())
            },
            "search_space": int(
                np.prod(range(len(clients) - COALITION_SIZE + 1, len(clients) + 1))
                // np.prod(range(1, COALITION_SIZE + 1))
            ),
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<18} {'AUC-ROC':>10} {'DemDiv':>10} {'EqOdds':>10} {'F1':>10}")
    print("-" * 60)
    for name, s in summary.items():
        print(
            f"{name:<18} {s['auc_mean']:>10.4f} {s['demdiv_mean']:>10.4f} "
            f"{s['eqodds_mean']:>10.4f} {s['f1_mean']:>10.4f}"
        )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"mimic_fl_results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Also save as the canonical experiment_results.json
    canonical_path = DATA_DIR / "experiment_results.json"
    with open(canonical_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Canonical results updated: {canonical_path}")

    return output



# Main



def main():
    parser = argparse.ArgumentParser(
        description="MIMIC-III FL experiment with FairSwarm"
    )
    parser.add_argument("--n_trials", type=int, default=5, help="Number of trials")
    parser.add_argument("--n_rounds", type=int, default=30, help="FL rounds per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_experiment(
        n_trials=args.n_trials,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
