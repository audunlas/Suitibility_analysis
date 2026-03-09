import json
import os
import csv


def print_summary(results: dict):
    """Print human-readable summary to console."""
    print("=" * 70)
    print(f"SUITABILITY CORRELATION ANALYSIS: {results.get('model_name', '?')}")
    print(f"Observations: {results.get('n_observations', '?')}")
    print("=" * 70)

    # Overall correlation
    overall = results.get("overall_correlation", {})
    print("\n--- Overall Correlation (composite score vs. cases) ---")
    if overall.get("spearman_r") is not None:
        print(f"  Spearman r = {overall['spearman_r']:.4f}  (p = {_fmt(overall.get('spearman_p'))})")
        print(f"  Pearson  r = {_fmt(overall.get('pearson_r'))}  (p = {_fmt(overall.get('pearson_p'))})")
        print(f"  R²         = {_fmt(overall.get('r_squared'))}")
    else:
        print(f"  {overall.get('note', 'N/A')}")

    # Component analysis
    comp = results.get("component_analysis", {})
    print("\n--- Component Analysis ---")
    for name, stats in comp.items():
        print(f"\n  {name}:")
        print(f"    Met: n={stats['n_met']}, mean cases={_fmt(stats['mean_cases_met'])}, median={_fmt(stats['median_cases_met'])}")
        print(f"    Not met: n={stats['n_not_met']}, mean cases={_fmt(stats['mean_cases_not_met'])}, median={_fmt(stats['median_cases_not_met'])}")
        if stats.get("point_biserial_r") is not None:
            print(f"    Point-biserial r = {stats['point_biserial_r']:.4f}  (p = {stats['point_biserial_p']:.4f})")
            print(f"    Mann-Whitney U = {stats['mann_whitney_u']:.1f}  (p = {stats['mann_whitney_p']:.4f})")
        elif "note" in stats:
            print(f"    {stats['note']}")

    # Composite score levels
    composite = results.get("composite_analysis", {})
    level_stats = composite.get("level_stats", {})
    print("\n--- Composite Score Levels ---")
    for level in sorted(level_stats.keys(), key=int):
        s = level_stats[level]
        print(f"  Score {level}: n={s['n']}, mean={_fmt(s['mean'])}, median={_fmt(s['median'])}, std={_fmt(s['std'])}")
    if "kruskal_wallis_h" in composite:
        h = composite['kruskal_wallis_h']
        p = composite['kruskal_wallis_p']
        h_str = f"{h:.4f}" if h is not None else "N/A"
        p_str = f"{p:.4f}" if p is not None else "N/A"
        print(f"  Kruskal-Wallis H = {h_str}  (p = {p_str})")

    # Continuous variables
    cont = results.get("continuous_variables", {})
    if cont:
        print("\n--- Continuous Variable Correlations (raw values vs. cases) ---")
        for col, stats in cont.items():
            if stats.get("spearman_r") is not None:
                print(f"  {col}: Spearman r = {stats['spearman_r']:.4f} (p = {_fmt(stats.get('spearman_p'))}), "
                      f"Pearson r = {_fmt(stats.get('pearson_r'))} (p = {_fmt(stats.get('pearson_p'))})")

    # Incidence analysis
    if "incidence_analysis" in results:
        inc = results["incidence_analysis"]
        inc_corr = inc.get("composite_correlation", {})
        print("\n--- Incidence Analysis (cases/population) ---")
        if inc_corr.get("spearman_r") is not None:
            print(f"  Spearman r = {inc_corr['spearman_r']:.4f}  (p = {_fmt(inc_corr.get('spearman_p'))})")

    print("\n" + "=" * 70)


def print_stratified_summary(
    by_location: dict | None = None, by_month: dict | None = None
):
    """Print stratified analysis summary."""
    if by_location:
        print("\n--- By Location (annual rolling windows) ---")
        for loc, s in sorted(by_location.items()):
            r = _fmt(s.get("spearman_r"))
            p = _fmt(s.get("spearman_p"))
            months = _fmt(s.get("mean_months_suitable"))
            cases = _fmt(s.get("mean_total_cases"))
            n_win = s.get("n_windows", 0)
            print(f"  {loc}: n_windows={n_win}, Spearman r={r}, p={p}, "
                  f"mean months suitable={months}, mean annual cases={cases}")

    if by_month:
        print("\n--- By Month ---")
        for month_key in sorted(by_month.keys(), key=lambda x: int(x)):
            stats = by_month[month_key]
            r = _fmt(stats.get("spearman_r"))
            p = _fmt(stats.get("spearman_p"))
            print(f"  Month {stats.get('month', month_key)}: n={stats['n']}, Spearman r={r}, p={p}, "
                  f"mean cases={_fmt(stats.get('mean_cases'))}, mean score={_fmt(stats.get('mean_score'))}")


def print_lag_summary(lag_results: dict):
    """Print lag analysis summary."""
    summary = lag_results.get("summary", [])
    best = lag_results.get("best_lag", 0)

    print("\n--- Lag Analysis (composite score vs. cases) ---")
    print(f"  {'Lag':>4}  {'n':>4}  {'Spearman r':>11}  {'p-value':>10}  {'R²':>8}")
    for row in summary:
        r = _fmt(row.get("spearman_r"))
        p = _fmt(row.get("spearman_p"))
        r2 = _fmt(row.get("r_squared"))
        marker = " <-- best" if row["lag"] == best else ""
        print(f"  {row['lag']:>4}  {row['n']:>4}  {r:>11}  {p:>10}  {r2:>8}{marker}")

    # Incidence if available
    has_inc = any(row.get("incidence_spearman_r") is not None for row in summary)
    if has_inc:
        print(f"\n  Incidence (cases/population):")
        print(f"  {'Lag':>4}  {'Spearman r':>11}  {'p-value':>10}")
        for row in summary:
            r = _fmt(row.get("incidence_spearman_r"))
            p = _fmt(row.get("incidence_spearman_p"))
            print(f"  {row['lag']:>4}  {r:>11}  {p:>10}")

    # Component breakdown
    comp_by_lag = lag_results.get("component_by_lag", {})
    if comp_by_lag:
        print(f"\n  Per-component point-biserial r by lag:")
        for comp_name, rows in comp_by_lag.items():
            vals = "  ".join(
                f"lag {r['lag']}={_fmt(r.get('point_biserial_r'))}"
                for r in rows
            )
            print(f"    {comp_name}: {vals}")

    print(f"\n  Best lag: {best} month(s)")


def print_temporal_summary(temporal_results: dict):
    """Print Q2 within-region temporal analysis summary (annual rolling approach)."""
    window = temporal_results.get("window", 12)
    print(f"\n--- Q2: Within-Region Temporal Analysis ({window}-month rolling windows) ---")

    if "note" in temporal_results:
        print(f"  {temporal_results['note']}")
        return

    summary = temporal_results.get("summary", {})
    if summary:
        n = summary.get("n_locations_with_correlation", 0)
        print(f"  Locations with enough windows for within-location correlation: {n}")
        print(f"  Mean within-location r:   {_fmt(summary.get('mean_within_location_r'))}")
        print(f"  Median within-location r: {_fmt(summary.get('median_within_location_r'))}")
        print(f"  Range: {_fmt(summary.get('min_within_location_r'))} to {_fmt(summary.get('max_within_location_r'))}")
        print(f"  Locations with no months_suitable variation: {summary.get('n_locations_no_variation', 0)}")

    per_loc = temporal_results.get("per_location", {})
    if per_loc:
        print(f"\n  Per-location:")
        for loc in sorted(per_loc.keys()):
            s = per_loc[loc]
            r = _fmt(s.get("spearman_r"))
            p = _fmt(s.get("spearman_p"))
            months = _fmt(s.get("mean_months_suitable"))
            cases = _fmt(s.get("mean_total_cases"))
            n_win = s.get("n_windows", 0)
            note = f" [{s['note']}]" if "note" in s else ""
            print(f"    {loc}: r={r}, p={p}, mean months suitable={months}/{window}, "
                  f"mean annual cases={cases}, n_windows={n_win}{note}")


def print_spatial_summary(spatial_results: dict):
    """Print Q3 cross-sectional spatial analysis summary."""
    print("\n--- Q3: Cross-Sectional Spatial Analysis ---")

    summary = spatial_results.get("summary", {})
    if summary:
        print(f"  Time periods analyzed:      {summary.get('n_time_periods', 0)}")
        print(f"  Mean cross-sectional r:     {_fmt(summary.get('mean_cross_sectional_r'))}")
        print(f"  Median cross-sectional r:   {_fmt(summary.get('median_cross_sectional_r'))}")
        print(f"  Range: {_fmt(summary.get('min_cross_sectional_r'))} to {_fmt(summary.get('max_cross_sectional_r'))}")
        print(f"  Pct positive:               {_fmt(summary.get('pct_positive'))}")
        print(f"  Pct significant (p<0.05):   {_fmt(summary.get('pct_significant'))}")

        if summary.get("annual_score_vs_cases_r") is not None:
            print(f"  Annual mean score vs. cases: r={_fmt(summary.get('annual_score_vs_cases_r'))}, "
                  f"p={_fmt(summary.get('annual_score_vs_cases_p'))}")


def print_components_summary(
    binned: dict, redundancy: dict, leave_one_out: dict
):
    """Print Q5 component analysis summary."""
    print("\n--- Q5: Component Deep Analysis ---")

    # Binned response
    print("\n  Binned climate-cases response:")
    for comp_name, comp_data in binned.items():
        bins = comp_data.get("bins", [])
        print(f"    {comp_name} ({comp_data.get('column', '?')}):")
        for b in bins:
            print(f"      {b['bin']}: n={b['n']}, mean cases={_fmt(b.get('mean_cases'))}")

    # Redundancy
    agreement = redundancy.get("pct_agreement", {})
    if agreement:
        print("\n  Component agreement (% same value):")
        for pair, pct in agreement.items():
            print(f"    {pair}: {_fmt(pct)}")

    # Leave-one-out
    full = leave_one_out.get("full_model", {})
    print(f"\n  Leave-one-out (full model r = {_fmt(full.get('spearman_r'))}):")
    for comp_name, stats in leave_one_out.get("without", {}).items():
        r = _fmt(stats.get("spearman_r"))
        delta = _fmt(stats.get("delta_r"))
        print(f"    Without {comp_name}: r={r}, delta_r={delta}")


def print_sensitivity_summary(sensitivity_results: dict):
    """Print Q7 threshold sensitivity summary."""
    print("\n--- Q7: Threshold Sensitivity ---")

    for comp_name, comp_data in sensitivity_results.items():
        original_r = _fmt(comp_data.get("original_r"))
        best_r = _fmt(comp_data.get("best_r"))
        best_min = comp_data.get("best_min")
        best_max = comp_data.get("best_max")
        orig_min = comp_data.get("original_min")
        orig_max = comp_data.get("original_max")

        print(f"\n  {comp_name}:")
        print(f"    Original thresholds: [{orig_min}, {orig_max}] -> r={original_r}")
        print(f"    Best thresholds:     [{best_min}, {best_max}] -> r={best_r}")
        print(f"    Sweep results ({len(comp_data.get('sweeps', []))} combinations):")

        # Show top 5
        sweeps = comp_data.get("sweeps", [])
        valid = [s for s in sweeps if s.get("spearman_r") is not None]
        valid.sort(key=lambda s: abs(s["spearman_r"]), reverse=True)
        for s in valid[:5]:
            print(f"      [{s['min_value']}, {s['max_value']}]: r={_fmt(s['spearman_r'])}")


def print_discrimination_summary(discrimination_results: dict):
    """Print Q8 score distribution and discrimination summary."""
    print("\n--- Q8: Score Distribution & Discrimination ---")

    dist = discrimination_results.get("distribution", [])
    if dist:
        print("\n  Score distribution:")
        for d in dist:
            print(f"    Score {d['score']}: n={d['n']} ({_fmt(d.get('pct'))}%), "
                  f"mean cases={_fmt(d.get('mean_cases'))}")

    eta = discrimination_results.get("eta_squared")
    print(f"\n  Eta-squared (between-group / total variance): {_fmt(eta)}")

    pct_max = discrimination_results.get("pct_observations_at_max")
    n_always = discrimination_results.get("n_always_max", 0)
    print(f"  Observations at max score: {_fmt(pct_max)}%")
    print(f"  Locations always at max:   {n_always}")

    always_locs = discrimination_results.get("always_max_locations", [])
    if always_locs:
        print(f"    {', '.join(always_locs)}")


def print_annual_summary(annual_results: dict):
    """Print primary analysis: months suitable vs. total annual cases."""
    window = annual_results.get("window", 12)
    max_score = annual_results.get("max_score", "?")
    print(f"\n--- PRIMARY ANALYSIS: Months at Max Suitability (last {window}) vs. Total Cases (same {window} months) ---")
    print(f"  (Predictor: months where score == {max_score}; Outcome: summed cases over same window)")

    if "note" in annual_results:
        print(f"  {annual_results['note']}")
        return

    n = annual_results.get("n_observations", 0)
    print(f"  Location-window observations (full {window}-month windows): {n}")

    corr = annual_results.get("correlation", {})
    if corr.get("spearman_r") is not None:
        print(f"  Pooled Spearman r = {_fmt(corr['spearman_r'])}, p = {_fmt(corr['spearman_p'])}")

    per_loc = annual_results.get("per_location", {})
    if per_loc:
        print(f"\n  Per-location (mean months suitable / mean total annual cases):")
        for loc, s in sorted(per_loc.items()):
            months = _fmt(s.get("mean_months_suitable"))
            frac = _fmt(s.get("mean_fraction_suitable"))
            cases = _fmt(s.get("mean_total_cases"))
            n_win = s.get("n_windows", 0)
            print(f"    {loc}: {months}/{window} months suitable ({frac} fraction), "
                  f"mean annual cases={cases}, n_windows={n_win}")


def save_json(results: dict, path: str):
    """Save full results as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def save_csv_tables(results: dict, output_dir: str):
    """Save per-score-level and per-component tables as CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Score level table
    level_stats = results.get("composite_analysis", {}).get("level_stats", {})
    if level_stats:
        path = os.path.join(output_dir, "score_levels.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["score", "n", "mean_cases", "median_cases", "std_cases"])
            for level in sorted(level_stats.keys(), key=int):
                s = level_stats[level]
                writer.writerow([level, s["n"], s["mean"], s["median"], s["std"]])
        print(f"Score levels saved to {path}")

    # Component table
    comp = results.get("component_analysis", {})
    if comp:
        path = os.path.join(output_dir, "component_analysis.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "component", "n_met", "n_not_met",
                "mean_cases_met", "mean_cases_not_met",
                "point_biserial_r", "point_biserial_p",
                "mann_whitney_u", "mann_whitney_p",
            ])
            for name, s in comp.items():
                writer.writerow([
                    name, s["n_met"], s["n_not_met"],
                    s["mean_cases_met"], s["mean_cases_not_met"],
                    s.get("point_biserial_r"), s.get("point_biserial_p"),
                    s.get("mann_whitney_u"), s.get("mann_whitney_p"),
                ])
        print(f"Component analysis saved to {path}")


def _fmt(val) -> str:
    """Format a value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)
