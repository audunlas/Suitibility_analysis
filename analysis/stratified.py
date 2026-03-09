import pandas as pd
import numpy as np
from scipy import stats

from suitability.model import SuitabilityModel
from analysis.annual import compute_window_data


def analyze_by_location(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
    window: int = 12,
) -> dict:
    """Per-location Spearman r between rolling months_suitable and total annual cases.

    For each location, extracts its rolling window observations and reports the
    within-location temporal Spearman r (do locations with more suitable months
    accumulate more annual cases as time advances?), plus the mean months_suitable
    and mean total annual cases.

    This is the per-location breakdown of the primary analysis. Cross-location
    comparison of means tells you: do some districts experience more annually-suitable
    months AND more annual disease burden?
    """
    win_df = compute_window_data(df, model, cases_col, location_col, time_col, window)
    results = {}

    for location, group in win_df.groupby("location"):
        loc_result = {
            "n_windows": len(group),
            "mean_months_suitable": _safe(group["months_suitable"].mean()),
            "mean_fraction_suitable": _safe(group["fraction_suitable"].mean()),
            "mean_total_cases": _safe(group["total_cases"].mean()),
        }

        # Within-location temporal correlation
        if len(group) >= 3 and group["months_suitable"].nunique() > 1:
            r, p = stats.spearmanr(group["months_suitable"], group["total_cases"])
            loc_result["spearman_r"] = _safe(r)
            loc_result["spearman_p"] = _safe(p)
        else:
            loc_result["spearman_r"] = None
            loc_result["spearman_p"] = None
            if len(group) < 3:
                loc_result["note"] = f"Too few windows ({len(group)}) for within-location correlation"

        results[str(location)] = loc_result

    return results


def analyze_by_month(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    time_col: str = "time_period",
) -> dict:
    """Per-calendar-month cross-sectional Spearman r across locations.

    For each calendar month (Jan, Feb, ...), correlates suitability score vs.
    cases across all locations sharing that month. Controls for seasonal timing
    (all observations are at the same point in the annual cycle), so this is
    less affected by the seasonal co-occurrence artifact than pooled monthly analysis.

    NORMATIVE DECISION (ND-C): Calendar months are grouped ignoring year — all
    Januaries across all years are pooled together, and likewise for other months.
    This is a cross-sectional design: it tests whether the score discriminates
    between locations within a given season, not whether suitability tracks disease
    burden over time. A January 2018 observation is treated as exchangeable with a
    January 2019 observation from the same location. This controls for seasonality
    but does not account for inter-annual trend (e.g., year-to-year climate shifts).
    """
    composite = model.compute_composite_score(df)
    months = pd.to_datetime(df[time_col]).dt.month
    results = {}

    for month, group in df.groupby(months):
        month_composite = composite.loc[group.index]
        month_cases = group[cases_col]

        month_result = {"n": len(group), "month": int(month)}

        # Require variation in both score and cases; if either is constant,
        # Spearman r is undefined (all ranks tied → NaN).
        if len(group) >= 3 and month_composite.nunique() > 1 and month_cases.nunique() > 1:
            r, p = stats.spearmanr(month_composite, month_cases)
            month_result["spearman_r"] = _safe(r)
            month_result["spearman_p"] = _safe(p)
        else:
            month_result["spearman_r"] = None
            month_result["spearman_p"] = None
            month_result["note"] = "Too few observations or no variance in score or cases"

        month_result["mean_cases"] = _safe(month_cases.mean())
        month_result["mean_score"] = _safe(month_composite.mean())

        results[str(month)] = month_result

    return results


def _safe(val) -> float | None:
    if val is None:
        return None
    import numpy as np
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
