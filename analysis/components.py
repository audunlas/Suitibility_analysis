"""Q5: Per-component deep analysis.

- Bin each continuous variable into ranges, compute mean cases per bin
  (detect nonlinearity in the climate-cases relationship).
- Component redundancy: correlation matrix between binary component scores.
- Leave-one-out: compare composite with/without each component.
"""

import pandas as pd
import numpy as np
from scipy import stats

from suitability.model import SuitabilityModel
from analysis.annual import compute_window_data


def analyze_binned_response(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    n_bins: int = 10,
) -> dict:
    """Bin each continuous climate variable and compute mean cases per bin.

    Helps detect nonlinearity: if the threshold captures a real breakpoint,
    mean cases should jump near the threshold value.
    """
    results = {}

    for comp in model.components:
        col = comp.column
        if col not in df.columns:
            continue

        values = df[col].dropna()
        cases = df.loc[values.index, cases_col]

        # NORMATIVE DECISION (ND-E): Bins are quantile-based (qcut), not equal-width (cut).
        # qcut ensures roughly equal sample sizes per bin, which stabilises estimates
        # in sparse regions of the climate variable's range. Trade-off: bin edges do
        # not align with the threshold boundaries defined in the model, making it
        # harder to visually confirm whether the threshold captures a breakpoint.
        # Alternative: equal-width bins (pd.cut) would align better with thresholds
        # but would produce empty or near-empty bins where data is sparse.
        # Falls back to equal-width if qcut fails (e.g., too many duplicate values).
        try:
            bins = pd.qcut(values, q=n_bins, duplicates="drop")
        except ValueError:
            bins = pd.cut(values, bins=n_bins, duplicates="drop")

        bin_stats = []
        for bin_label, group_idx in bins.groupby(bins, observed=False).groups.items():
            bin_cases = cases.loc[group_idx]
            bin_values = values.loc[group_idx]
            bin_stats.append({
                "bin": str(bin_label),
                "bin_mid": _safe(bin_values.mean()),
                "n": len(bin_cases),
                "mean_cases": _safe(bin_cases.mean()),
                "median_cases": _safe(bin_cases.median()),
                "std_cases": _safe(bin_cases.std()),
            })

        # Sort by bin midpoint
        bin_stats.sort(key=lambda x: x["bin_mid"] if x["bin_mid"] is not None else 0)

        results[comp.name] = {
            "column": col,
            "threshold_min": comp.min_value,
            "threshold_max": comp.max_value,
            "bins": bin_stats,
        }

    return results


def analyze_redundancy(
    df: pd.DataFrame,
    model: SuitabilityModel,
) -> dict:
    """Correlation matrix between binary component scores.

    High correlation means components are redundant — they fire together.
    """
    component_scores = model.compute_component_scores(df)
    names = list(component_scores.columns)

    # Phi coefficient (Pearson on binary variables)
    corr_matrix = component_scores.corr().to_dict()

    # Percent agreement (both met or both not met)
    agreement = {}
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                agree = (component_scores[name_i] == component_scores[name_j]).mean()
                agreement[f"{name_i}_vs_{name_j}"] = _safe(agree)

    # Joint frequency: how often each pair is both met
    joint_met = {}
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                both = ((component_scores[name_i] == 1) & (component_scores[name_j] == 1)).mean()
                joint_met[f"{name_i}_and_{name_j}"] = _safe(both)

    return {
        "correlation_matrix": corr_matrix,
        "pct_agreement": agreement,
        "joint_met_frequency": joint_met,
    }


def analyze_leave_one_out(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
    window: int = 12,
) -> dict:
    """Compare full model vs. model without each component, using annual rolling approach.

    For each model variant (full and each leave-one-out), computes rolling window
    data (months_suitable, total_annual_cases) and correlates them. This is
    consistent with the primary analysis method.

    "months_suitable" for a reduced model = months where the reduced score reaches
    its maximum (i.e., all remaining thresholds met).

    If removing a component doesn't change the annual correlation, that component
    may be redundant or uninformative for predicting annual disease burden.
    """
    # Full model
    win_df_full = compute_window_data(df, model, cases_col, location_col, time_col, window)

    full_r, full_p = (None, None)
    if len(win_df_full) >= 3 and win_df_full["months_suitable"].nunique() > 1:
        full_r, full_p = stats.spearmanr(win_df_full["months_suitable"], win_df_full["total_cases"])

    results = {
        "full_model": {
            "n_components": len(model.components),
            "spearman_r": _safe(full_r),
            "spearman_p": _safe(full_p),
            "n_windows": len(win_df_full),
        },
        "without": {},
    }

    for comp in model.components:
        # Build a reduced model without this component
        remaining_components = [c for c in model.components if c.name != comp.name]
        reduced_model = SuitabilityModel(name=model.name, components=remaining_components)

        win_df_reduced = compute_window_data(
            df, reduced_model, cases_col, location_col, time_col, window
        )

        if len(win_df_reduced) >= 3 and win_df_reduced["months_suitable"].nunique() > 1:
            r, p = stats.spearmanr(win_df_reduced["months_suitable"], win_df_reduced["total_cases"])
            fr = _safe(full_r)
            rr = _safe(r)
            results["without"][comp.name] = {
                "spearman_r": rr,
                "spearman_p": _safe(p),
                # delta_r is undefined (None) if either correlation is unavailable,
                # rather than silently substituting 0 for a missing full_r.
                "delta_r": _safe(fr - rr) if fr is not None and rr is not None else None,
                "n_windows": len(win_df_reduced),
            }
        else:
            results["without"][comp.name] = {
                "spearman_r": None,
                "spearman_p": None,
                "delta_r": None,
                "n_windows": len(win_df_reduced),
                "note": "No variance in months_suitable for reduced model",
            }

    return results


def _safe(val) -> float | None:
    if val is None:
        return None
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
