"""Q7: Threshold sensitivity analysis using the annual rolling approach.

For each component, sweeps threshold values and tracks the Spearman r between
months_suitable (at that threshold) and total annual cases in the same 12-month
window. This is consistent with the primary analysis method.

Answers: how sensitive is the annual correlation to the specific threshold choices?
Are the Uganda DHIS2 thresholds empirically optimal for predicting annual burden?
"""

import pandas as pd
import numpy as np
from scipy import stats

from suitability.model import SuitabilityModel
from suitability.thresholds import ThresholdComponent
from analysis.annual import compute_window_data


def sweep_thresholds(
    df: pd.DataFrame,
    model: SuitabilityModel,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
    window: int = 12,
) -> dict:
    """Sweep each component's thresholds and track annual Spearman r.

    For each threshold combination, computes the rolling window data using the
    modified model, then correlates months_suitable vs total_cases. This ensures
    threshold selection is evaluated against the same metric as the primary analysis.

    Sweep ranges are derived from the actual column data distribution (percentiles),
    so they are automatically unit-correct regardless of whether the data uses
    mm/month, mm/day, or other scales.
    """
    sweep_config = _default_sweep_config(model, df)
    results = {}

    for comp in model.components:
        if comp.name not in sweep_config:
            continue

        config = sweep_config[comp.name]
        comp_results = {
            "component": comp.name,
            "column": comp.column,
            "original_min": comp.min_value,
            "original_max": comp.max_value,
            "sweeps": [],
        }

        min_values = config.get("min_values", [comp.min_value])
        max_values = config.get("max_values", [comp.max_value])

        for min_val in min_values:
            for max_val in max_values:
                modified_model = _modify_component(model, comp.name, min_val, max_val)
                win_df = compute_window_data(
                    df, modified_model, cases_col, location_col, time_col, window
                )

                if len(win_df) >= 3 and win_df["months_suitable"].nunique() > 1:
                    r, p = stats.spearmanr(win_df["months_suitable"], win_df["total_cases"])
                    sweep_result = {
                        "min_value": min_val,
                        "max_value": max_val,
                        "spearman_r": _safe(r),
                        "spearman_p": _safe(p),
                        "n_windows": len(win_df),
                    }
                else:
                    sweep_result = {
                        "min_value": min_val,
                        "max_value": max_val,
                        "spearman_r": None,
                        "spearman_p": None,
                        "n_windows": len(win_df) if not win_df.empty else 0,
                    }

                comp_results["sweeps"].append(sweep_result)

        # Find best threshold combination (highest positive r)
        valid = [s for s in comp_results["sweeps"] if s["spearman_r"] is not None]
        if valid:
            best = max(valid, key=lambda s: s["spearman_r"])
            comp_results["best_min"] = best["min_value"]
            comp_results["best_max"] = best["max_value"]
            comp_results["best_r"] = best["spearman_r"]
            comp_results["original_r"] = next(
                (s["spearman_r"] for s in valid
                 if s["min_value"] == comp.min_value and s["max_value"] == comp.max_value),
                None,
            )

        results[comp.name] = comp_results

    return results


def _default_sweep_config(model: SuitabilityModel, df: pd.DataFrame) -> dict:
    """Derive sweep ranges from the actual column data distribution.

    Uses percentiles of each column so the ranges are automatically unit-correct
    (e.g. mm/day vs mm/month, different humidity distributions across countries).
    5 values for min thresholds and 5 for max thresholds, giving 25 combinations
    per component (or 5 for one-sided thresholds).
    """
    config = {}
    for comp in model.components:
        col = df[comp.column].dropna()
        quantiles = [0.05, 0.15, 0.25, 0.35, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
        p = {q: col.quantile(q) for q in quantiles}

        if comp.name == "temperature":
            # NORMATIVE DECISION: Sweep centered on the original threshold values
            # (±4°C in 2°C steps), so the original threshold always falls in the
            # middle cell of the heatmap grid. The previous approach used data-distribution
            # percentiles, which placed the original threshold at the edge of the sweep
            # range (the red reference box appeared in a corner).
            # Alternative: keep percentile approach but widen the range so the original
            # threshold is never in the outer 20% of the grid.
            if comp.min_value is not None and comp.max_value is not None:
                step = 2.0
                mins = _round_series(
                    np.linspace(comp.min_value - 2 * step, comp.min_value + 2 * step, 5), col
                )
                maxs = _round_series(
                    np.linspace(comp.max_value - 2 * step, comp.max_value + 2 * step, 5), col
                )
            else:
                mins = _round_series(np.linspace(p[0.05], p[0.50], 5), col)
                maxs = _round_series(np.linspace(p[0.60], col.max() + 5, 5), col)
            # Always include original thresholds (already at centre when min/max are set above)
            if comp.min_value is not None:
                mins = sorted(set(mins + [float(round(comp.min_value, 1))]))
            if comp.max_value is not None:
                maxs = sorted(set(maxs + [float(round(comp.max_value, 1))]))
            config["temperature"] = {"min_values": mins, "max_values": maxs}

        elif comp.name == "precipitation":
            # One-sided (min only): sweep from near-zero to p80.
            # NORMATIVE DECISION: Sweep range derived from data distribution, not
            # hardcoded mm/month values. This ensures correct scaling regardless
            # of whether data is in mm/day, mm/month, or other units.
            mins = _round_series(np.linspace(p[0.05], p[0.80], 9), col)
            if comp.min_value is not None:
                mins = sorted(set(mins + [float(round(comp.min_value, 2))]))
            config["precipitation"] = {
                "min_values": mins,
                "max_values": [comp.max_value],
            }

        elif comp.name == "humidity":
            # Fixed sweep ranges: min threshold 42–58, max threshold 72–88.
            # Gaps between ranges ensure min < max for all combinations.
            # Current thresholds [50, 80] are centred within each range.
            mins = _round_series(np.linspace(42, 58, 5), col)
            maxs = _round_series(np.linspace(72, 88, 5), col)
            if comp.min_value is not None:
                mins = sorted(set(mins + [float(round(comp.min_value, 1))]))
            if comp.max_value is not None:
                maxs = sorted(set(maxs + [float(round(comp.max_value, 1))]))
            config["humidity"] = {"min_values": mins, "max_values": maxs}

    return config


def _round_series(values: np.ndarray, col: pd.Series) -> list:
    """Round sweep values sensibly based on the column's scale."""
    scale = col.max() - col.min()
    if scale > 50:
        decimals = 0
    elif scale > 5:
        decimals = 1
    else:
        decimals = 2
    return sorted(set(float(round(v, decimals)) for v in values))


def _modify_component(
    model: SuitabilityModel,
    comp_name: str,
    new_min: float | None,
    new_max: float | None,
) -> SuitabilityModel:
    """Return a copy of the model with one component's thresholds changed."""
    new_components = []
    for comp in model.components:
        if comp.name == comp_name:
            new_components.append(ThresholdComponent(
                name=comp.name,
                column=comp.column,
                min_value=new_min,
                max_value=new_max,
            ))
        else:
            new_components.append(comp)
    return SuitabilityModel(name=model.name, components=new_components)


def _safe(val) -> float | None:
    if val is None:
        return None
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
