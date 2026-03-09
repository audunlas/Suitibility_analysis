"""Q2: Within-location temporal analysis using the annual rolling approach.

For each location, computes rolling 12-month windows of (months_suitable,
total_annual_cases) and correlates them. This answers:

  "Within a given district, does annual suitability exposure predict
   annual disease burden as the window advances through time?"

This avoids the seasonal co-occurrence problem of month-to-month comparisons
(where both suitability and cases rise in transmission season together).
A within-location temporal correlation using annual windows tests whether
year-to-year or multi-year variation in suitability tracks variation in burden.
"""

import numpy as np
from scipy import stats
import pandas as pd

from suitability.model import SuitabilityModel
from analysis.annual import compute_window_data


def analyze_within_region(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
    window: int = 12,
) -> dict:
    """Per-location correlation of rolling annual suitability vs. annual cases.

    For each location, extracts its rolling window observations and computes
    Spearman r between months_suitable and total_cases within that location
    across time. This tests temporal tracking within districts.

    Note: meaningful within-location correlations require enough windows
    (i.e., enough months of data beyond the initial window). Locations
    with few windows will be flagged.

    Returns:
        - per_location: per-location statistics and within-location Spearman r
        - summary: mean/median r across locations with enough data
    """
    win_df = compute_window_data(df, model, cases_col, location_col, time_col, window)

    results = {"window": window, "per_location": {}, "summary": {}}

    if win_df.empty:
        results["note"] = f"No complete {window}-month windows available"
        return results

    all_rs = []

    for location, group in win_df.groupby("location"):
        loc_result = {"n_windows": len(group)}

        loc_result["mean_months_suitable"] = _safe(group["months_suitable"].mean())
        loc_result["mean_total_cases"] = _safe(group["total_cases"].mean())

        # Within-location temporal correlation: does annual suitability
        # track annual burden over time within this district?
        if len(group) >= 3 and group["months_suitable"].nunique() > 1:
            r, p = stats.spearmanr(group["months_suitable"], group["total_cases"])
            loc_result["spearman_r"] = _safe(r)
            loc_result["spearman_p"] = _safe(p)
            if r is not None and not np.isnan(r):
                all_rs.append(r)
        else:
            loc_result["spearman_r"] = None
            loc_result["spearman_p"] = None
            if len(group) < 3:
                loc_result["note"] = f"Too few windows ({len(group)}) for within-location correlation"
            else:
                loc_result["note"] = "No variation in months_suitable within this location"

        results["per_location"][str(location)] = loc_result

    # Summary across locations
    valid_rs = [r for r in all_rs if r is not None and not np.isnan(r)]
    if valid_rs:
        results["summary"]["n_locations_with_correlation"] = len(valid_rs)
        results["summary"]["mean_within_location_r"] = _safe(np.mean(valid_rs))
        results["summary"]["median_within_location_r"] = _safe(np.median(valid_rs))
        results["summary"]["min_within_location_r"] = _safe(np.min(valid_rs))
        results["summary"]["max_within_location_r"] = _safe(np.max(valid_rs))

    no_variation = [
        loc for loc, res in results["per_location"].items()
        if res.get("spearman_r") is None and "variation" in res.get("note", "")
    ]
    results["summary"]["n_locations_no_variation"] = len(no_variation)

    return results


def _safe(val) -> float | None:
    if val is None:
        return None
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
