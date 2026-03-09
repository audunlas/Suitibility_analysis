"""Q3: Cross-sectional (spatial) analysis.

For each time point, correlate suitability score vs. cases across locations.
This answers: at a given month, do higher-suitability districts have more cases?
"""

import pandas as pd
import numpy as np
from scipy import stats

from suitability.model import SuitabilityModel


def analyze_cross_sectional(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
) -> dict:
    """Per-time-point spatial correlation between score and cases.

    For each unique time_period, compute Spearman r across locations.
    Also computes annual summaries: total cases vs. months suitable per year.
    """
    composite = model.compute_composite_score(df)
    df = df.copy()
    df["_score"] = composite

    results = {"per_time_period": {}, "summary": {}}

    all_rs = []

    for tp, group in df.groupby(time_col):
        tp_result = {"n_locations": len(group)}

        scores = group["_score"]
        cases = group[cases_col]

        if len(group) >= 3 and scores.nunique() > 1:
            r, p = stats.spearmanr(scores, cases)
            tp_result["spearman_r"] = _safe(r)
            tp_result["spearman_p"] = _safe(p)
            if r is not None and not np.isnan(r):
                all_rs.append(r)
        else:
            tp_result["spearman_r"] = None
            tp_result["spearman_p"] = None

        tp_result["mean_score"] = _safe(scores.mean())
        tp_result["mean_cases"] = _safe(cases.mean())

        results["per_time_period"][str(tp)] = tp_result

    # Summary across time points
    valid_rs = [r for r in all_rs if r is not None and not np.isnan(r)]
    if valid_rs:
        results["summary"]["n_time_periods"] = len(valid_rs)
        results["summary"]["mean_cross_sectional_r"] = _safe(np.mean(valid_rs))
        results["summary"]["median_cross_sectional_r"] = _safe(np.median(valid_rs))
        results["summary"]["min_cross_sectional_r"] = _safe(np.min(valid_rs))
        results["summary"]["max_cross_sectional_r"] = _safe(np.max(valid_rs))
        results["summary"]["pct_positive"] = _safe(
            np.mean([r > 0 for r in valid_rs])
        )
        results["summary"]["pct_significant"] = _safe(
            np.mean([
                results["per_time_period"][tp]["spearman_p"] is not None
                and results["per_time_period"][tp]["spearman_p"] < 0.05
                for tp in results["per_time_period"]
                if results["per_time_period"][tp]["spearman_r"] is not None
            ])
        )

    # Annual summary: total cases per location-year vs. months suitable
    df["_year"] = pd.to_datetime(df[time_col]).dt.year
    max_score = len(model.components)
    annual = df.groupby([location_col, "_year"]).agg(
        total_cases=(cases_col, "sum"),
        months_suitable=("_score", lambda s: (s == max_score).sum()),
        mean_score=("_score", "mean"),
    ).reset_index()

    if len(annual) >= 3 and annual["mean_score"].nunique() > 1:
        r, p = stats.spearmanr(annual["mean_score"], annual["total_cases"])
        results["summary"]["annual_score_vs_cases_r"] = _safe(r)
        results["summary"]["annual_score_vs_cases_p"] = _safe(p)

    if len(annual) >= 3 and annual["months_suitable"].nunique() > 1:
        r, p = stats.spearmanr(annual["months_suitable"], annual["total_cases"])
        results["summary"]["months_max_vs_total_cases_r"] = _safe(r)
        results["summary"]["months_max_vs_total_cases_p"] = _safe(p)

    return results


def _safe(val) -> float | None:
    if val is None:
        return None
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
