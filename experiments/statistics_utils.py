"""
Statistical Utilities for FairSwarm Experiments.

Provides confidence intervals, hypothesis tests, and statistical summaries
for publication-quality experimental results.

Author: Tenicka Norwood
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import List, Union

import numpy as np
from scipy import stats


def get_git_hash() -> str:
    """Get current git commit hash for provenance tracking in experiment results."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class ConfidenceInterval:
    """Represents a confidence interval."""

    mean: float
    lower: float
    upper: float
    confidence: float = 0.95
    n: int = 0

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}] ({int(self.confidence * 100)}% CI, n={self.n})"

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "confidence": self.confidence,
            "n": self.n,
        }

    def latex(self, precision: int = 3) -> str:
        """Format for LaTeX."""
        fmt = f"{{:.{precision}f}}"
        return f"${fmt.format(self.mean)}$ $[{fmt.format(self.lower)}, {fmt.format(self.upper)}]$"


def mean_ci(
    data: Union[List[float], np.ndarray],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """
    Compute confidence interval for the mean using t-distribution.

    Args:
        data: Sample data
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        ConfidenceInterval with mean and bounds
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]  # Remove NaN values

    n = len(data)
    if n < 2:
        mean = float(data[0]) if n == 1 else float("nan")
        return ConfidenceInterval(
            mean=mean, lower=mean, upper=mean, confidence=confidence, n=n
        )

    mean = float(np.mean(data))
    se = float(stats.sem(data))
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_critical * se

    return ConfidenceInterval(
        mean=mean,
        lower=mean - margin,
        upper=mean + margin,
        confidence=confidence,
        n=n,
    )


def proportion_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
    method: str = "wilson",
) -> ConfidenceInterval:
    """
    Compute confidence interval for a proportion (binomial).

    Uses Wilson score interval which performs well for all sample sizes.

    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level
        method: "wilson" (recommended) or "normal"

    Returns:
        ConfidenceInterval with proportion and bounds
    """
    if total == 0:
        return ConfidenceInterval(
            mean=0.0, lower=0.0, upper=1.0, confidence=confidence, n=0
        )

    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)

    if method == "wilson":
        # Wilson score interval
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = (
            z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
        ) / denominator

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
    else:
        # Normal approximation (less accurate for small n or extreme p)
        se = np.sqrt(p * (1 - p) / total)
        margin = z * se
        lower = max(0.0, p - margin)
        upper = min(1.0, p + margin)

    return ConfidenceInterval(
        mean=p,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n=total,
    )


def std_ci(
    data: Union[List[float], np.ndarray],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """
    Compute confidence interval for standard deviation using chi-squared distribution.

    Args:
        data: Sample data
        confidence: Confidence level

    Returns:
        ConfidenceInterval for standard deviation
    """
    data = np.asarray(data)
    n = len(data)

    if n < 2:
        return ConfidenceInterval(
            mean=0.0, lower=0.0, upper=0.0, confidence=confidence, n=n
        )

    std = float(np.std(data, ddof=1))
    alpha = 1 - confidence

    chi2_lower = stats.chi2.ppf(1 - alpha / 2, n - 1)
    chi2_upper = stats.chi2.ppf(alpha / 2, n - 1)

    lower = std * np.sqrt((n - 1) / chi2_lower)
    upper = std * np.sqrt((n - 1) / chi2_upper)

    return ConfidenceInterval(
        mean=std,
        lower=lower,
        upper=upper,
        confidence=confidence,
        n=n,
    )


def compare_means(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    confidence: float = 0.95,
    n_comparisons: int = 1,
) -> dict:
    """
    Compare two groups using Welch's t-test with Bonferroni correction.

    Args:
        group1: First group data
        group2: Second group data
        confidence: Confidence level
        n_comparisons: Number of simultaneous comparisons for Bonferroni
            correction. When comparing against multiple baselines, set this
            to the number of baselines to control family-wise error rate.

    Returns:
        Dictionary with test results including corrected p-value
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    # Remove NaN
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    # Cohen's d effect size
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    # CI for difference in means
    diff = mean1 - mean2
    se_diff = np.sqrt(std1**2 / n1 + std2**2 / n2)
    df = (std1**2 / n1 + std2**2 / n2) ** 2 / (
        (std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1)
    )
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    margin = t_crit * se_diff

    # Bonferroni correction: multiply p-value by number of comparisons
    p_corrected = min(1.0, float(p_value) * n_comparisons)
    alpha = 1 - confidence

    return {
        "group1_mean": float(mean1),
        "group2_mean": float(mean2),
        "difference": float(diff),
        "ci_lower": float(diff - margin),
        "ci_upper": float(diff + margin),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "p_value_corrected": p_corrected,
        "n_comparisons": n_comparisons,
        "cohens_d": float(cohens_d),
        "significant": p_corrected < alpha,
        "n1": n1,
        "n2": n2,
    }


def statistical_summary(
    data: Union[List[float], np.ndarray],
    confidence: float = 0.95,
    name: str = "metric",
) -> dict:
    """
    Compute comprehensive statistical summary with CIs.

    Args:
        data: Sample data
        confidence: Confidence level
        name: Name of the metric

    Returns:
        Dictionary with full summary
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    n = len(data)

    if n == 0:
        return {"name": name, "n": 0, "error": "No valid data"}

    mean_interval = mean_ci(data, confidence)
    std_interval = std_ci(data, confidence)

    return {
        "name": name,
        "n": n,
        "mean": float(np.mean(data)),
        "mean_ci": mean_interval.to_dict(),
        "std": float(np.std(data, ddof=1)) if n > 1 else 0.0,
        "std_ci": std_interval.to_dict(),
        "median": float(np.median(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
        "confidence_level": confidence,
    }


def format_ci_latex(
    ci: ConfidenceInterval,
    precision: int = 3,
    percentage: bool = False,
) -> str:
    """
    Format confidence interval for LaTeX tables.

    Args:
        ci: Confidence interval
        precision: Decimal places
        percentage: If True, format as percentage

    Returns:
        LaTeX-formatted string
    """
    if percentage:
        return f"${ci.mean * 100:.{precision - 2}f}\\%$ $[{ci.lower * 100:.{precision - 2}f}, {ci.upper * 100:.{precision - 2}f}]$"
    return f"${ci.mean:.{precision}f}$ $[{ci.lower:.{precision}f}, {ci.upper:.{precision}f}]$"


# Effect size interpretation
def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


if __name__ == "__main__":
    # Example usage
    print("Statistical Utilities Demo")
    print("=" * 50)

    # Sample data
    np.random.seed(42)
    data = np.random.normal(0.8, 0.1, 30)

    # Mean CI
    ci = mean_ci(data)
    print(f"\nMean with 95% CI: {ci}")

    # Proportion CI
    prop_ci = proportion_ci(successes=27, total=30)
    print(f"Proportion with 95% CI: {prop_ci}")

    # Compare two groups
    group1 = np.random.normal(0.8, 0.1, 20)
    group2 = np.random.normal(0.7, 0.15, 25)

    comparison = compare_means(group1, group2)
    print("\nGroup comparison:")
    print(
        f"  Difference: {comparison['difference']:.3f} "
        f"[{comparison['ci_lower']:.3f}, {comparison['ci_upper']:.3f}]"
    )
    print(f"  p-value: {comparison['p_value']:.4f}")
    print(
        f"  Cohen's d: {comparison['cohens_d']:.2f} ({interpret_cohens_d(comparison['cohens_d'])})"
    )

    # Full summary
    summary = statistical_summary(data, name="accuracy")
    print(f"\nFull summary for {summary['name']}:")
    print(f"  n={summary['n']}, mean={summary['mean']:.3f}, std={summary['std']:.3f}")
    print(
        f"  95% CI: [{summary['mean_ci']['ci_lower']:.3f}, {summary['mean_ci']['ci_upper']:.3f}]"
    )
