"""CHAP train entry point.

Runs correlation analysis between suitability scores and observed disease cases.
Saves model config and analysis results.
"""

import sys
import os

import joblib

from configs.lao_malaria import LAO_MALARIA
from suitability.model import SuitabilityModel
from visualize import main as run_visualize
from analysis.pipeline import load_and_clean, run_all_analyses
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

    df, outcome_col = load_and_clean(train_data_path, model, missing_strategy)

    output_dir = os.path.dirname(model_output_path)
    analysis_dir = os.path.join(output_dir, "analysis")

    all_results = run_all_analyses(df, model, outcome_col)

    results = all_results["results"]
    annual_results = all_results["annual_results"]
    temporal_results = all_results["temporal_results"]
    lag_results = all_results["lag_results"]
    binned_results = all_results["binned_results"]
    redundancy_results = all_results["redundancy_results"]
    leave_one_out_results = all_results["leave_one_out_results"]
    sensitivity_results = all_results["sensitivity_results"]
    discrimination_results = all_results["discrimination_results"]

    by_location = results.get("by_location")
    by_month = results.get("by_month")

    # Print summaries
    if annual_results is not None:
        print_annual_summary(annual_results)
    print_summary(results)
    print_stratified_summary(by_location, by_month)
    if temporal_results is not None:
        print_temporal_summary(temporal_results)
    if all_results["spatial_results"] is not None:
        print_spatial_summary(all_results["spatial_results"])
    if lag_results is not None:
        print_lag_summary(lag_results)
    print_components_summary(binned_results, redundancy_results, leave_one_out_results)
    print_sensitivity_summary(sensitivity_results)
    print_discrimination_summary(discrimination_results)

    # Save JSONs
    overall_dir = os.path.join(analysis_dir, "overall")
    save_json(results, os.path.join(overall_dir, "correlation_results.json"))
    save_csv_tables(results, overall_dir)

    if annual_results is not None:
        save_json(annual_results, os.path.join(analysis_dir, "annual", "annual_mean.json"))
    if temporal_results is not None:
        save_json(temporal_results, os.path.join(analysis_dir, "temporal", "within_region.json"))
    if all_results["spatial_results"] is not None:
        save_json(all_results["spatial_results"], os.path.join(analysis_dir, "spatial", "cross_sectional.json"))
    if lag_results is not None:
        save_json(lag_results["summary"], os.path.join(analysis_dir, "lag", "lag_summary.json"))
    save_json(binned_results, os.path.join(analysis_dir, "components", "binned_analysis.json"))
    save_json(redundancy_results, os.path.join(analysis_dir, "components", "redundancy.json"))
    save_json(leave_one_out_results, os.path.join(analysis_dir, "components", "leave_one_out.json"))
    save_json(sensitivity_results, os.path.join(analysis_dir, "sensitivity", "threshold_sweep.json"))
    save_json(discrimination_results, os.path.join(analysis_dir, "discrimination", "score_distribution.json"))

    # Save model config
    model_dir = os.path.dirname(model_output_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model.to_dict(), model_output_path)
    print(f"\nModel config saved to {model_output_path}")

    # Generate plots and overview HTML
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
