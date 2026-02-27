"""Empirical validation of Theorem 1 convergence rate.

Estimates the effective spectral radius from FairSwarm fitness trajectories
and compares against the theoretical prediction from the transition matrix.

Usage:
    python experiments/validate_convergence_rate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure project root importable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "fairswarm" / "src"))


def theoretical_spectral_radius(omega: float, c1: float, c2: float) -> float:
    """Compute spectral radius of PSO transition matrix.

    The characteristic polynomial is:
        lambda^2 - (1 + omega - phi)*lambda + omega = 0
    where phi = (c1 + c2) / 2.

    Returns rho(A) = max|eigenvalue|.
    """
    phi = (c1 + c2) / 2.0
    a = 1.0
    b = -(1.0 + omega - phi)
    c = omega
    discriminant = b**2 - 4 * a * c

    if discriminant >= 0:
        lam1 = (-b + np.sqrt(discriminant)) / (2 * a)
        lam2 = (-b - np.sqrt(discriminant)) / (2 * a)
        return max(abs(lam1), abs(lam2))
    else:
        real_part = -b / (2 * a)
        imag_part = np.sqrt(-discriminant) / (2 * a)
        return np.sqrt(real_part**2 + imag_part**2)


def estimate_empirical_spectral_radius(fitness_history: list[float]) -> float:
    """Estimate effective convergence rate from fitness trajectory.

    Fits log|f(t) - f*| ~ t * log(rho) via linear regression.
    Returns rho_hat in (0, 1) if converging, or > 1 if diverging.
    """
    if len(fitness_history) < 10:
        return float("nan")

    f_star = fitness_history[-1]
    residuals = [abs(f - f_star) + 1e-12 for f in fitness_history[:-5]]
    log_residuals = np.log(residuals)
    t = np.arange(len(log_residuals), dtype=np.float64)

    if len(t) < 5:
        return float("nan")

    # Linear regression: log|r(t)| = intercept + t * log(rho)
    slope, _ = np.polyfit(t, log_residuals, 1)
    rho_empirical = np.exp(slope)
    return float(np.clip(rho_empirical, 0.0, 2.0))


def run_validation(n_trials: int = 30) -> dict:
    """Run convergence rate validation across parameter configurations."""
    from fairswarm.algorithms.fairswarm import FairSwarm
    from fairswarm.core.config import FairSwarmConfig
    from fairswarm.fitness.mock import MockFitness

    configs = [
        {"omega": 0.3, "c1": 0.5, "c2": 0.5},
        {"omega": 0.5, "c1": 1.0, "c2": 1.0},
        {"omega": 0.7, "c1": 1.0, "c2": 1.0},
        {"omega": 0.9, "c1": 0.5, "c2": 0.5},
        # Violating configurations
        {"omega": 0.9, "c1": 2.0, "c2": 2.0},
    ]

    results = []
    for params in configs:
        omega, c1, c2 = params["omega"], params["c1"], params["c2"]
        rho_theory = theoretical_spectral_radius(omega, c1, c2)
        satisfies = omega + (c1 + c2) / 2 < 2.0

        empirical_rhos = []
        for trial in range(n_trials):
            config = FairSwarmConfig(
                swarm_size=20,
                max_iterations=100,
                inertia=omega,
                cognitive=c1,
                social=c2,
                fairness_coefficient=0.5,
                seed=trial,
            )
            n_clients = 20
            coalition_size = 10
            demographics = np.random.default_rng(trial).dirichlet(
                [1.0] * 4, size=n_clients
            )
            fitness_fn = MockFitness(n_clients=n_clients)

            optimizer = FairSwarm(
                config=config,
                n_clients=n_clients,
                coalition_size=coalition_size,
                demographics=demographics,
            )
            result = optimizer.optimize(fitness_fn)

            if hasattr(result, "fitness_history") and result.fitness_history:
                rho_emp = estimate_empirical_spectral_radius(result.fitness_history)
                if not np.isnan(rho_emp):
                    empirical_rhos.append(rho_emp)

        entry = {
            "omega": omega,
            "c1": c1,
            "c2": c2,
            "metric": omega + (c1 + c2) / 2,
            "satisfies_theorem": satisfies,
            "rho_theoretical": rho_theory,
            "rho_empirical_mean": float(np.mean(empirical_rhos))
            if empirical_rhos
            else float("nan"),
            "rho_empirical_std": float(np.std(empirical_rhos))
            if empirical_rhos
            else float("nan"),
            "n_valid_trials": len(empirical_rhos),
        }
        results.append(entry)

        status = "SATISFIES" if satisfies else "VIOLATES"
        print(
            f"  omega={omega}, c1={c1}, c2={c2} [{status}]: "
            f"rho_theory={rho_theory:.3f}, "
            f"rho_empirical={entry['rho_empirical_mean']:.3f} "
            f"+/- {entry['rho_empirical_std']:.3f} "
            f"({entry['n_valid_trials']} trials)"
        )

    return {"configs": results}


def main():
    print("=" * 60)
    print("  Theorem 1 Convergence Rate Validation")
    print("=" * 60)

    results = run_validation(n_trials=30)

    print("\n" + "-" * 60)
    print("  Summary")
    print("-" * 60)

    satisfying = [r for r in results["configs"] if r["satisfies_theorem"]]
    violating = [r for r in results["configs"] if not r["satisfies_theorem"]]

    if satisfying:
        rhos = [r["rho_empirical_mean"] for r in satisfying if not np.isnan(r["rho_empirical_mean"])]
        print(f"  Satisfying configs: rho_empirical = {np.mean(rhos):.3f} +/- {np.std(rhos):.3f}")
        theory_range = [r["rho_theoretical"] for r in satisfying]
        print(f"  Theoretical range: [{min(theory_range):.3f}, {max(theory_range):.3f}]")

    if violating:
        rhos = [r["rho_empirical_mean"] for r in violating if not np.isnan(r["rho_empirical_mean"])]
        if rhos:
            print(f"  Violating configs: rho_empirical = {np.mean(rhos):.3f} +/- {np.std(rhos):.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
