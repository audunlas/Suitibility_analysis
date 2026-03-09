"""CHAP train entry point.

Runs correlation analysis between suitability scores and observed disease cases.
Saves model config and analysis results.
"""

import sys
import os

import joblib
import pandas as pd

from configs.lao_malaria import LAO_MALARIA
from suitability.model import SuitabilityModel
from visualize import main as run_visualize
from analysis.correlation import CorrelationAnalysis
from analysis.stratified import analyze_by_location, analyze_by_month
from analysis.lag import run_lag_sweep
from analysis.temporal import analyze_within_region
from analysis.spatial import analyze_cross_sectional
from analysis.components import (
    analyze_binned_response,
    analyze_redundancy,
    analyze_leave_one_out,
)
from analysis.sensitivity import sweep_thresholds
from analysis.discrimination import analyze_discrimination
from analysis.annual import analyze_months_suitable
from analysis.report import (
    print_summary,
    print_stratified_summary,
    print_lag_summary,
    print_temporal_summary,
    print_spatial_summary,
    print_components_summary,
    print_sensitivity_summary,
    print_discrimination_summary,
    print_annual_summary,
    save_json,
    save_csv_tables,
)


def train(train_data_path: str, model_output_path: str, model=None, missing_strategy: str = "drop"):
    """Run suitability correlation analysis and save model config.

    Args:
        train_data_path: Path to the input CSV.
        model_output_path: Path to save the serialized model config.
        model: SuitabilityModel to use (defaults to LAO_MALARIA).
        missing_strategy: How to handle NaN disease_cases in rolling windows.
            "drop"      — keep 12-month calendar windows; exclude NaN months from
                          counts; drop windows with ≤3 valid months. Recommended.
            "fill_zero" — fill NaN disease_cases with 0 before analysis.
    """
    if model is None:
        model = LAO_MALARIA

    df = pd.read_csv(train_data_path)

    # Always exclude locations where ALL disease_cases are NaN — no usable outcome.
    if "disease_cases" in df.columns and "location" in df.columns:
        all_nan_locs = (
            df.groupby("location")["disease_cases"]
            .apply(lambda s: s.isna().all())
        )
        all_nan_locs = all_nan_locs[all_nan_locs].index.tolist()
        if all_nan_locs:
            print(f"NOTE: Excluding locations with no disease data: {all_nan_locs}")
            df = df[~df["location"].isin(all_nan_locs)].reset_index(drop=True)

    # Handle remaining NaN disease_cases according to missing_strategy.
    if missing_strategy == "fill_zero":
        # NORMATIVE DECISION: Fill NaN disease_cases with 0.
        # Treats missing surveillance reports as zero observed cases, keeping
        # windows contiguous. Alternative: "drop" strategy (default) which uses
        # calendar-based windows and excludes NaN months from counts.
        if "disease_cases" in df.columns:
            n_missing = df["disease_cases"].isna().sum()
            if n_missing > 0:
                print(f"NOTE: Filling {n_missing} missing disease_cases with 0 "
                      f"({n_missing / len(df) * 100:.1f}% of remaining data)")
                df["disease_cases"] = df["disease_cases"].fillna(0)
    else:
        # "drop" strategy: NaN months are excluded per window inside compute_window_data.
        n_missing = df["disease_cases"].isna().sum() if "disease_cases" in df.columns else 0
        if n_missing > 0:
            print(f"NOTE: {n_missing} missing disease_cases will be excluded within "
                  f"each rolling window (missing_strategy='drop')")

    # Check required columns
    missing = [c for c in model.required_columns() if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    if "disease_cases" not in df.columns:
        print("ERROR: Missing 'disease_cases' column in training data")
        sys.exit(1)

    # Determine primary outcome: incidence (cases/population) if possible,
    # otherwise raw disease cases. Suitability scores predict risk (rate),
    # not absolute counts, so incidence is the appropriate metric for
    # cross-location and pooled analyses.
    if "population" in df.columns and (df["population"] > 0).all():
        df["incidence"] = df["disease_cases"] / df["population"] * 1000
        outcome_col = "incidence"
        print("Using incidence (cases per 1,000 population) as primary outcome\n")
    else:
        outcome_col = "disease_cases"
        print("WARNING: No population data — using raw disease cases.\n"
              "Cross-location comparisons will be confounded by population size.\n")

    # Base output directory
    output_dir = os.path.dirname(model_output_path)
    analysis_dir = os.path.join(output_dir, "analysis")

    # --- PRIMARY ANALYSIS: Months at max suitability vs. total annual cases ---
    # Standard method: for each rolling 12-month window, count months where score
    # reached maximum, and compare to total cases in the same window. This avoids
    # the circular problem of comparing current-month suitability to current-month
    # cases (both co-occur in transmission season by construction).
    annual_results = None
    if "time_period" in df.columns and "location" in df.columns:
        annual_results = analyze_months_suitable(df, model, cases_col=outcome_col)
        annual_dir = os.path.join(analysis_dir, "annual")
        save_json(annual_results, os.path.join(annual_dir, "annual_mean.json"))
        print_annual_summary(annual_results)

    # --- Q1: Monthly correlation analysis (supplementary) ---
    # Compares current-month suitability score to current-month outcome, pooled
    # across all location-months. Useful for understanding temporal co-variation
    # but subject to the seasonal co-occurrence artifact described above.
    analysis = CorrelationAnalysis(model)
    results = analysis.run(df, cases_col=outcome_col)
    results["outcome_metric"] = outcome_col

    # Stratified analyses (by location, by month)
    by_location = None
    by_month = None
    if "location" in df.columns and "time_period" in df.columns:
        by_location = analyze_by_location(
            df, model, cases_col=outcome_col,
            location_col="location", time_col="time_period",
        )
        results["by_location"] = by_location
    if "time_period" in df.columns:
        by_month = analyze_by_month(df, model, cases_col=outcome_col)
        results["by_month"] = by_month

    # Output directory for overall results (used at end of pipeline for full save)
    overall_dir = os.path.join(analysis_dir, "overall")

    # Print Q1
    print_summary(results)
    print_stratified_summary(by_location, by_month)

    # --- Q2: Temporal (within-region) analysis ---
    # Per-location rolling annual analysis: does annual suitability exposure
    # track annual disease burden within each district over time?
    temporal_results = None
    if "time_period" in df.columns and "location" in df.columns:
        temporal_results = analyze_within_region(
            df, model, cases_col=outcome_col,
            location_col="location", time_col="time_period",
        )
        temporal_dir = os.path.join(analysis_dir, "temporal")
        save_json(temporal_results, os.path.join(temporal_dir, "within_region.json"))
        print_temporal_summary(temporal_results)

    # --- Q3: Spatial (cross-sectional) analysis ---
    spatial_results = None
    if "time_period" in df.columns and "location" in df.columns:
        spatial_results = analyze_cross_sectional(df, model, cases_col=outcome_col)
        spatial_dir = os.path.join(analysis_dir, "spatial")
        save_json(spatial_results, os.path.join(spatial_dir, "cross_sectional.json"))
        print_spatial_summary(spatial_results)

    # --- Q4: Lag analysis ---
    lag_results = None
    if "time_period" in df.columns and "location" in df.columns:
        lag_results = run_lag_sweep(df, model, max_lag=3, cases_col=outcome_col)
        results["lag_analysis"] = {
            "summary": lag_results["summary"],
            "best_lag": lag_results["best_lag"],
            "component_by_lag": lag_results["component_by_lag"],
            "continuous_by_lag": lag_results["continuous_by_lag"],
        }
        lag_dir = os.path.join(analysis_dir, "lag")
        save_json(lag_results["summary"], os.path.join(lag_dir, "lag_summary.json"))
        print_lag_summary(lag_results)

    # --- Q5: Component deep analysis ---
    binned_results = analyze_binned_response(df, model, cases_col=outcome_col)
    redundancy_results = analyze_redundancy(df, model)
    leave_one_out_results = analyze_leave_one_out(
        df, model, cases_col=outcome_col,
        location_col="location", time_col="time_period",
    )

    components_dir = os.path.join(analysis_dir, "components")
    save_json(binned_results, os.path.join(components_dir, "binned_analysis.json"))
    save_json(redundancy_results, os.path.join(components_dir, "redundancy.json"))
    save_json(leave_one_out_results, os.path.join(components_dir, "leave_one_out.json"))
    print_components_summary(binned_results, redundancy_results, leave_one_out_results)

    # --- Q7: Threshold sensitivity ---
    sensitivity_results = sweep_thresholds(
        df, model, cases_col=outcome_col,
        location_col="location", time_col="time_period",
    )
    sensitivity_dir = os.path.join(analysis_dir, "sensitivity")
    save_json(sensitivity_results, os.path.join(sensitivity_dir, "threshold_sweep.json"))
    print_sensitivity_summary(sensitivity_results)

    # --- Q8: Score discrimination ---
    discrimination_results = analyze_discrimination(df, model, cases_col=outcome_col)
    discrimination_dir = os.path.join(analysis_dir, "discrimination")
    save_json(discrimination_results, os.path.join(discrimination_dir, "score_distribution.json"))
    print_discrimination_summary(discrimination_results)

    # Save the full combined results JSON to overall/ for visualize.py.
    # This is the only save — the earlier partial save (before lag analysis) was removed
    # because it was always overwritten here and was missing lag_analysis data.
    save_json(results, os.path.join(overall_dir, "correlation_results.json"))
    save_csv_tables(results, overall_dir)

    # Save model config
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model.to_dict(), model_output_path)
    print(f"\nModel config saved to {model_output_path}")

    # Generate plots and overview HTML into the same output directory
    plot_dir = os.path.join(output_dir, "plots")
    run_visualize(
        base_analysis_dir=analysis_dir,
        base_plot_dir=plot_dir,
        train_data_path=train_data_path,
        model=model,
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train.py <train_data_csv> <model_output_path> [model_config.json] [missing_strategy]")
        print("  model_config.json:  path to JSON model config (default: configs/lao_malaria.json)")
        print("  missing_strategy:   'drop' (default) or 'fill_zero'")
        sys.exit(1)
    model = None
    missing_strategy = "drop"
    if len(sys.argv) > 3:
        import json
        with open(sys.argv[3]) as f:
            model = SuitabilityModel.from_dict(json.load(f))
    if len(sys.argv) > 4:
        missing_strategy = sys.argv[4]
    train(sys.argv[1], sys.argv[2], model=model, missing_strategy=missing_strategy)
