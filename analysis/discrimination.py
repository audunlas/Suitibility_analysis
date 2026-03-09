"""Q8: Score distribution and discriminative power.

- Score distribution: count and percent per level
- Eta-squared: between-group variance / total variance
- Per-location: fraction of time at max score
- Always-suitable regions (signal ceiling problem)
"""

import pandas as pd
import numpy as np

from suitability.model import SuitabilityModel


def analyze_discrimination(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
) -> dict:
    """Assess how well the score discriminates between observations."""
    composite = model.compute_composite_score(df)
    cases = df[cases_col]
    max_score = len(model.components)

    results = {}

    # --- Score distribution ---
    distribution = []
    for level in range(max_score + 1):
        n = int((composite == level).sum())
        pct = _safe(n / len(composite) * 100)
        level_cases = cases[composite == level]
        distribution.append({
            "score": level,
            "n": n,
            "pct": pct,
            "mean_cases": _safe(level_cases.mean()),
            "median_cases": _safe(level_cases.median()),
        })
    results["distribution"] = distribution

    # --- Eta-squared ---
    # Between-group variance / total variance of cases
    grand_mean = cases.mean()
    total_ss = ((cases - grand_mean) ** 2).sum()

    between_ss = 0.0
    for level in range(max_score + 1):
        group = cases[composite == level]
        if len(group) > 0:
            between_ss += len(group) * (group.mean() - grand_mean) ** 2

    eta_squared = _safe(between_ss / total_ss) if total_ss > 0 else None
    results["eta_squared"] = eta_squared

    # --- Per-location max-score fraction ---
    if location_col in df.columns:
        loc_stats = []
        for location, group in df.groupby(location_col):
            loc_composite = composite.loc[group.index]
            n = len(group)
            n_max = int((loc_composite == max_score).sum())
            pct_max = _safe(n_max / n * 100) if n > 0 else None
            always_max = n_max == n

            loc_stats.append({
                "location": str(location),
                "n": n,
                "n_at_max": n_max,
                "pct_at_max": pct_max,
                "always_max": always_max,
                "mean_score": _safe(loc_composite.mean()),
                "mean_cases": _safe(group[cases_col].mean()),
            })

        # Sort by pct_at_max descending
        loc_stats.sort(key=lambda x: x["pct_at_max"] or 0, reverse=True)
        results["per_location"] = loc_stats

        always_max_locs = [s["location"] for s in loc_stats if s["always_max"]]
        results["n_always_max"] = len(always_max_locs)
        results["always_max_locations"] = always_max_locs

        # Fraction of all observations at max score
        results["pct_observations_at_max"] = _safe(
            (composite == max_score).sum() / len(composite) * 100
        )

    return results


def _safe(val) -> float | None:
    if val is None:
        return None
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
