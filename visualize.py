"""Generate plots from suitability correlation analysis results.

Reads analysis output from output/analysis/ and training data from input/.
Writes plots to output/plots/ organized in subfolders by analysis type.

Usage:
    python visualize.py
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from configs.dhis2_uganda_malaria import DHIS2_UGANDA_MALARIA

BASE_ANALYSIS_DIR = "output/analysis"
BASE_PLOT_DIR = "output/plots"
TRAIN_DATA = "input/trainData.csv"


def _fmt_p(p: float) -> str:
    """Format a p-value: scientific notation when very small, 3 d.p. otherwise."""
    if p is None:
        return "n/a"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def load_json(path: str) -> dict | None:
    """Load a JSON file, returning None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_train_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_DATA)


# ===========================================================================
# Overall plots (Q1) — output/plots/overall/
# ===========================================================================

def plot_cases_by_score(df: pd.DataFrame, composite: pd.Series, plot_dir: str,
                        outcome_col: str = "disease_cases", outcome_label: str = "Disease Cases",
                        model=None):
    """Box plot of outcome at each composite suitability score level."""
    if model is None:
        model = DHIS2_UGANDA_MALARIA
    fig, ax = plt.subplots(figsize=(6, 5))

    levels = sorted(composite.unique())
    data = [df.loc[composite == level, outcome_col] for level in levels]

    bp = ax.boxplot(data, tick_labels=[str(int(l)) for l in levels], patch_artist=True,
                    widths=0.5, medianprops=dict(color="black", linewidth=1.5))

    n_components = len(model.components)
    box_cmap = plt.cm.RdYlGn
    colors = [box_cmap(i / n_components) for i in range(len(levels))]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, level in enumerate(levels):
        y = df.loc[composite == level, outcome_col]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.5, color="black", s=20, zorder=3)

    ax.set_xlabel("Composite Suitability Score")
    ax.set_ylabel(outcome_label)
    ax.set_title(f"{outcome_label} by Suitability Score Level")

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "cases_by_score.png"), dpi=150)
    plt.close(fig)
    print("  overall/cases_by_score.png")


def plot_component_comparison(df: pd.DataFrame, component_scores: pd.DataFrame, plot_dir: str,
                              outcome_col: str = "disease_cases", outcome_label: str = "Disease Cases"):
    """Grouped bar chart: mean outcome when each threshold is met vs. not met."""
    components = list(component_scores.columns)
    met_means = []
    not_met_means = []
    met_sems = []
    not_met_sems = []

    for comp in components:
        binary = component_scores[comp]
        met = df.loc[binary == 1, outcome_col]
        not_met = df.loc[binary == 0, outcome_col]
        met_means.append(met.mean())
        not_met_means.append(not_met.mean())
        met_sems.append(met.sem())
        not_met_sems.append(not_met.sem())

    x = np.arange(len(components))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x - width / 2, not_met_means, width, yerr=not_met_sems,
           label="Threshold NOT met", color="#d9534f", alpha=0.8, capsize=4)
    ax.bar(x + width / 2, met_means, width, yerr=met_sems,
           label="Threshold met", color="#5cb85c", alpha=0.8, capsize=4)

    ax.set_ylabel(f"Mean {outcome_label}")
    ax.set_title(f"Mean {outcome_label}: Threshold Met vs. Not Met")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in components])
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "component_comparison.png"), dpi=150)
    plt.close(fig)
    print("  overall/component_comparison.png")


def plot_climate_vs_cases(df: pd.DataFrame, plot_dir: str,
                          outcome_col: str = "disease_cases", outcome_label: str = "Disease Cases",
                          model=None):
    """Scatter plots of each climate variable vs. outcome, with threshold zones."""
    if model is None:
        model = DHIS2_UGANDA_MALARIA

    fig, axes = plt.subplots(1, len(model.components), figsize=(5 * len(model.components), 5))
    if len(model.components) == 1:
        axes = [axes]

    for ax, comp in zip(axes, model.components):
        x = df[comp.column]
        y = df[outcome_col]
        ax.scatter(x, y, alpha=0.6, edgecolors="black", linewidth=0.5, s=40)

        xlim = ax.get_xlim()
        lo = comp.min_value if comp.min_value is not None else xlim[0]
        hi = comp.max_value if comp.max_value is not None else xlim[1]
        ax.axvspan(lo, hi, alpha=0.12, color="green", label="Suitable range")

        if comp.min_value is not None:
            ax.axvline(comp.min_value, color="green", linestyle="--", linewidth=1, alpha=0.7)
        if comp.max_value is not None:
            ax.axvline(comp.max_value, color="green", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_xlabel(comp.column.replace("_", " ").title())
        ax.set_ylabel(outcome_label)
        ax.set_title(f"{comp.name.capitalize()} vs. {outcome_label}")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "climate_vs_cases.png"), dpi=150)
    plt.close(fig)
    print("  overall/climate_vs_cases.png")


def plot_time_series(df: pd.DataFrame, composite: pd.Series, plot_dir: str,
                     outcome_col: str = "disease_cases", outcome_label: str = "Disease Cases",
                     model=None):
    """Per-location time series: outcome (bars) overlaid with suitability score (line)."""
    if model is None:
        model = DHIS2_UGANDA_MALARIA
    locations = df["location"].unique()
    n_locs = len(locations)
    fig, axes = plt.subplots(n_locs, 1, figsize=(10, 3.5 * n_locs), sharex=True)
    if n_locs == 1:
        axes = [axes]

    for ax, loc in zip(axes, sorted(locations)):
        mask = df["location"] == loc
        loc_df = df.loc[mask].copy()
        loc_df["time"] = pd.to_datetime(loc_df["time_period"])
        loc_df = loc_df.sort_values("time")
        loc_composite = composite.loc[mask].loc[loc_df.index]

        x = range(len(loc_df))
        labels = loc_df["time_period"].values

        ax.bar(x, loc_df[outcome_col], color="steelblue", alpha=0.7, label=outcome_label)

        ax2 = ax.twinx()
        ax2.plot(x, loc_composite.values, color="black", marker="o", markersize=4,
                 linewidth=1.5, label="Suitability score")
        ax2.set_ylabel("Score")
        ax2.set_ylim(-0.2, len(model.components) + 0.5)
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        ax.set_ylabel(outcome_label)
        ax.set_title(loc)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.suptitle("Monthly Cases and Suitability Score by Location", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "time_series.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  overall/time_series.png")


def plot_location_correlations(results: dict, plot_dir: str):
    """Bar chart of within-location annual Spearman r per location."""
    by_loc = results.get("by_location", {})
    if not by_loc:
        return

    locations = sorted(by_loc.keys())
    rs = [by_loc[loc].get("spearman_r") for loc in locations]
    ps = [by_loc[loc].get("spearman_p") for loc in locations]
    mean_months = [by_loc[loc].get("mean_months_suitable") for loc in locations]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    colors = ["#337ab7"] * len(locations)
    ax1.bar(locations, [m or 0 for m in mean_months], color=colors, alpha=0.8)
    ax1.set_ylabel("Mean Months at Max Suitability (per 12-month window)")
    ax1.set_title("Mean Annual Suitability Exposure by Province")

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "location_correlations.png"), dpi=150)
    plt.close(fig)
    print("  overall/location_correlations.png")


def plot_correlation_comparison(results: dict, plot_dir: str, outcome_label: str = "incidence"):
    """Compare Spearman r: composite score vs. raw continuous variables."""
    overall = results.get("overall_correlation", {})
    continuous = results.get("continuous_variables", {})

    names = ["Composite\nscore"]
    rs = [overall.get("spearman_r", 0)]

    for col, stats in continuous.items():
        label = col.replace("_", "\n")
        names.append(label)
        rs.append(stats.get("spearman_r", 0))

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#337ab7"] + ["#5bc0de"] * len(continuous)
    ax.barh(names, rs, color=colors, alpha=0.8, height=0.5)

    ax.set_xlabel(f"Spearman r (vs. {outcome_label})")
    ax.set_title("Thresholded Score vs. Raw Variables")
    ax.set_xlim(0, 1)

    for i, r in enumerate(rs):
        ax.text(r + 0.02, i, f"{r:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "correlation_comparison.png"), dpi=150)
    plt.close(fig)
    print("  overall/correlation_comparison.png")


def plot_threshold_frequency(component_scores: pd.DataFrame, plot_dir: str, model=None):
    """Bar chart: percentage of province-months each threshold is met."""
    if model is None:
        model = DHIS2_UGANDA_MALARIA

    names = [comp.name.capitalize() for comp in model.components]
    pcts = [component_scores[comp.name].mean() * 100 for comp in model.components]

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#337ab7"] * len(names)
    bars = ax.bar(names, pcts, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", fontsize=10)

    ax.set_ylabel("% of Province-Months Threshold is Met")
    ax.set_title("Threshold Hit Rate by Component")
    ax.set_ylim(0, 110)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="50%")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "threshold_frequency.png"), dpi=150)
    plt.close(fig)
    print("  overall/threshold_frequency.png")


def plot_score_heatmap(df: pd.DataFrame, composite: pd.Series, plot_dir: str, model=None):
    """Heatmap of modal suitability score by location and month."""
    from matplotlib.colors import BoundaryNorm

    if model is None:
        model = DHIS2_UGANDA_MALARIA
    df = df.copy()
    df["_score"] = composite
    df["_month"] = pd.to_datetime(df["time_period"]).dt.month

    # Use mode (most common score) so cells are integers → exactly 4 discrete colours
    pivot = df.pivot_table(values="_score", index="location", columns="_month",
                           aggfunc=lambda x: x.mode().iloc[0])

    n_levels = len(model.components) + 1  # 0, 1, 2, 3
    cmap = plt.cm.get_cmap("RdYlGn", n_levels)
    bounds = np.arange(-0.5, n_levels, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Month")
    ax.set_title("Suitability Score by Location and Month (modal score)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=10,
                    color="white" if val < 1 else "black")

    fig.colorbar(im, ax=ax, label="Score", shrink=0.8, ticks=range(n_levels))
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "score_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  overall/score_heatmap.png")


# ===========================================================================
# Temporal plots (Q2) — output/plots/temporal/
# ===========================================================================

def plot_within_location_temporal(temporal_results: dict, annual_results: dict, plot_dir: str,
                                   outcome_label: str = "Disease Cases"):
    """Q2: Ranked dot plot of within-province Spearman r values.

    Each dot is one province's Spearman r between rolling months_suitable and
    rolling total cases over time. Sorted by r; zero line and mean r annotated.
    Significance markers show p < 0.05 / 0.01 / 0.001.
    """
    per_loc = temporal_results.get("per_location", {})
    if not per_loc:
        return

    # Collect and sort by r
    rows = []
    for loc, stats in per_loc.items():
        r = stats.get("spearman_r")
        p = stats.get("spearman_p")
        if r is not None:
            rows.append((loc, r, p))
    if not rows:
        return
    rows.sort(key=lambda x: x[1])

    locs = [r[0] for r in rows]
    rs   = [r[1] for r in rows]
    ps   = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(6, max(4, len(locs) * 0.35)))

    colors = ["#d9534f" if r < 0 else "#337ab7" for r in rs]
    ax.scatter(rs, range(len(locs)), color=colors, s=80, zorder=4)
    ax.hlines(range(len(locs)), 0, rs, color=colors, linewidth=1.2, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8)

    # Significance markers
    for i, (r, p) in enumerate(zip(rs, ps)):
        if p is not None:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if sig:
                x_off = r + 0.015 if r >= 0 else r - 0.015
                ha = "left" if r >= 0 else "right"
                ax.text(x_off, i, sig, va="center", ha=ha, fontsize=9, color="black")

    mean_r = temporal_results.get("summary", {}).get("mean_within_location_r")
    if mean_r is not None:
        ax.axvline(mean_r, color="gray", linewidth=1.2, linestyle="--",
                   label=f"Mean r = {mean_r:.3f}")
        ax.legend(fontsize=9, loc="lower right")

    ax.set_yticks(range(len(locs)))
    ax.set_yticklabels(locs, fontsize=9)
    ax.set_xlabel("Within-Province Spearman r (months_suitable vs. annual cases)")
    ax.set_title("Q2: Within-Province Temporal Correlation\n(* p<0.05  ** p<0.01  *** p<0.001)")

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "within_location_temporal.png"), dpi=150)
    plt.close(fig)
    print("  temporal/within_location_temporal.png")


# ===========================================================================
# Spatial plots (Q3) — output/plots/spatial/
# ===========================================================================

def plot_score_vs_cases_by_location(annual_results: dict, plot_dir: str,
                                     outcome_label: str = "Disease Cases"):
    """Q3: Mean months_suitable vs. mean annual incidence per province (spatial/between-province).

    Each point is one province averaged over all 12-month rolling windows. Shows whether
    provinces that experience more fully-suitable months on average also have more annual
    disease burden. Uses the same x-axis concept as Q2 (months_suitable) for direct comparison.
    """
    from scipy import stats as sp_stats

    per_loc = annual_results.get("per_location", {})
    if len(per_loc) < 2:
        return

    locations = sorted(per_loc.keys())
    mean_months = [per_loc[loc].get("mean_months_suitable") for loc in locations]
    mean_cases = [per_loc[loc].get("mean_total_cases") for loc in locations]

    valid = [(loc, m, c) for loc, m, c in zip(locations, mean_months, mean_cases)
             if m is not None and c is not None]
    if len(valid) < 2:
        return

    locs, xs, ys = zip(*valid)
    xs, ys = list(xs), list(ys)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(xs, ys, s=100, color="#337ab7", edgecolors="black", linewidth=0.8, zorder=3)
    for loc, x, y in zip(locs, xs, ys):
        ax.annotate(loc, xy=(x, y), xytext=(8, 4), textcoords="offset points", fontsize=9)

    if len(xs) >= 3:
        slope, intercept, _, _, _ = sp_stats.linregress(xs, ys)
        x_line = np.linspace(min(xs), max(xs), 50)
        ax.plot(x_line, slope * x_line + intercept, "--", color="gray", linewidth=1.5)

        spear_r, spear_p = sp_stats.spearmanr(xs, ys)
        ax.text(0.02, 0.98, f"Spearman r = {spear_r:.3f}\np = {_fmt_p(spear_p)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Mean Months at Max Suitability per Year")
    ax.set_ylabel(f"Mean Annual {outcome_label}")
    ax.set_title(f"Q3: Do Higher-Suitability Provinces Have Higher Annual {outcome_label}?")

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "score_vs_cases_by_location.png"), dpi=150)
    plt.close(fig)
    print("  spatial/score_vs_cases_by_location.png")


# ===========================================================================
# Lag plots (Q4) — output/plots/lag/
# ===========================================================================

def plot_lag_correlation(results: dict, plot_dir: str, outcome_label: str = "incidence"):
    """Line plot of Spearman r vs. lag months."""
    lag_data = results.get("lag_analysis", {})
    summary = lag_data.get("summary", [])
    if not summary:
        return

    lags = [row["lag"] for row in summary]
    rs = [row.get("spearman_r") for row in summary]
    best = lag_data.get("best_lag", 0)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(lags, rs, "o-", color="#337ab7", linewidth=2, markersize=8, label=outcome_label.capitalize())

    best_r = next((row.get("spearman_r") for row in summary if row["lag"] == best), None)
    if best_r is not None:
        ax.axvline(best, color="gray", linestyle=":", alpha=0.5)
        ax.annotate(f"Best: lag {best}", xy=(best, best_r),
                    xytext=(best + 0.3, best_r - 0.05),
                    fontsize=9, color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1))

    for lag, r in zip(lags, rs):
        if r is not None:
            ax.annotate(f"{r:.3f}", xy=(lag, r), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Spearman r")
    ax.set_title(f"Suitability-{outcome_label.capitalize()} Correlation by Lag")
    ax.set_xticks(lags)
    ax.set_xticklabels([str(l) for l in lags])
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "lag_correlation.png"), dpi=150)
    plt.close(fig)
    print("  lag/lag_correlation.png")


def plot_component_lag(results: dict, plot_dir: str):
    """Line plot of point-biserial r per component across lags."""
    lag_data = results.get("lag_analysis", {})
    comp_by_lag = lag_data.get("component_by_lag", {})
    if not comp_by_lag:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    colors = ["#d9534f", "#337ab7", "#5cb85c", "#f0ad4e"]
    for i, (comp_name, rows) in enumerate(comp_by_lag.items()):
        lags = [r["lag"] for r in rows]
        rs = [r.get("point_biserial_r") for r in rows]
        color = colors[i % len(colors)]
        ax.plot(lags, rs, "o-", color=color, linewidth=2, markersize=7, label=comp_name.capitalize())

        for lag, r in zip(lags, rs):
            if r is not None:
                ax.annotate(f"{r:.3f}", xy=(lag, r), xytext=(0, 10),
                            textcoords="offset points", ha="center", fontsize=8, color=color)

    summary = lag_data.get("summary", [])
    if summary:
        lags = [row["lag"] for row in summary]
        rs = [row.get("spearman_r") for row in summary]
        ax.plot(lags, rs, "D--", color="black", linewidth=1.5, markersize=6,
                alpha=0.6, label="Composite (Spearman)")

    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Correlation with incidence")
    ax.set_title("Component Correlations by Lag")
    ax.set_xticks(lags)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "component_lag.png"), dpi=150)
    plt.close(fig)
    print("  lag/component_lag.png")


def plot_continuous_vs_composite_lag(results: dict, plot_dir: str):
    """Compare raw variable correlations with composite score across lags."""
    lag_data = results.get("lag_analysis", {})
    summary = lag_data.get("summary", [])
    cont_by_lag = lag_data.get("continuous_by_lag", {})
    if not summary or not cont_by_lag:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    lags = [row["lag"] for row in summary]
    composite_rs = [row.get("spearman_r") for row in summary]
    ax.plot(lags, composite_rs, "D-", color="#337ab7", linewidth=2.5, markersize=8,
            label="Composite score", zorder=3)

    lighter_colors = ["#5bc0de", "#f0ad4e", "#d9534f"]
    for i, (col, rows) in enumerate(cont_by_lag.items()):
        col_lags = [r["lag"] for r in rows]
        col_rs = [r.get("spearman_r") for r in rows]
        label = col.replace("_", " ")
        color = lighter_colors[i % len(lighter_colors)]
        ax.plot(col_lags, col_rs, "o--", color=color, linewidth=1.5, markersize=6, label=label)

    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Spearman r (vs. incidence)")
    ax.set_title("Composite Score vs. Raw Variables Across Lags")
    ax.set_xticks(lags)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "continuous_vs_composite_lag.png"), dpi=150)
    plt.close(fig)
    print("  lag/continuous_vs_composite_lag.png")


# ===========================================================================
# Component plots (Q5) — output/plots/components/
# ===========================================================================

def plot_binned_cases(binned_results: dict, plot_dir: str, outcome_label: str = "Disease Cases"):
    """Bar chart of mean outcome per climate variable bin."""
    n_comps = len(binned_results)
    if n_comps == 0:
        return

    fig, axes = plt.subplots(1, n_comps, figsize=(5 * n_comps, 5))
    if n_comps == 1:
        axes = [axes]

    for ax, (comp_name, comp_data) in zip(axes, binned_results.items()):
        bins = comp_data.get("bins", [])
        if not bins:
            continue

        mids = [b["bin_mid"] for b in bins]
        means = [b["mean_cases"] if b["mean_cases"] is not None else 0 for b in bins]
        ns = [b["n"] for b in bins]

        ax.bar(range(len(bins)), means, color="#337ab7", alpha=0.8)

        # Threshold lines
        thresh_min = comp_data.get("threshold_min")
        thresh_max = comp_data.get("threshold_max")

        for thresh, label in [(thresh_min, "min"), (thresh_max, "max")]:
            if thresh is not None:
                for j, m in enumerate(mids):
                    if m is not None and m >= thresh:
                        ax.axvline(j - 0.5, color="red", linestyle="--", linewidth=1.5,
                                   label=f"Threshold {label}={thresh}")
                        break

        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([f"{m:.0f}" if m is not None else "?" for m in mids],
                           rotation=45, ha="right", fontsize=8)

        for j, n in enumerate(ns):
            ax.text(j, means[j] + max(means) * 0.02, f"n={n}", ha="center", fontsize=7, color="gray")

        ax.set_xlabel(comp_data.get("column", comp_name).replace("_", " ").title())
        ax.set_ylabel(f"Mean {outcome_label}")
        ax.set_title(f"{comp_name.capitalize()}: {outcome_label} by Climate Bin")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "binned_cases.png"), dpi=150)
    plt.close(fig)
    print("  components/binned_cases.png")


def plot_redundancy_heatmap(redundancy_results: dict, plot_dir: str):
    """Heatmap of component correlation matrix."""
    corr = redundancy_results.get("correlation_matrix", {})
    if not corr:
        return

    names = list(corr.keys())
    matrix = np.array([[corr[r].get(c, 0) for c in names] for r in names])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.capitalize() for n in names], rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.capitalize() for n in names])

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if abs(matrix[i, j]) > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="Phi coefficient", shrink=0.8)
    ax.set_title("Component Redundancy (Binary Correlation)")

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "redundancy_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  components/redundancy_heatmap.png")


def plot_leave_one_out(leave_one_out_results: dict, plot_dir: str):
    """Bar chart comparing full model vs. leave-one-out Spearman r."""
    full = leave_one_out_results.get("full_model", {})
    without = leave_one_out_results.get("without", {})
    if not without:
        return

    full_r = full.get("spearman_r", 0) or 0
    names = ["Full model"] + [f"Without\n{n}" for n in without.keys()]
    rs = [full_r] + [s.get("spearman_r", 0) or 0 for s in without.values()]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#337ab7"] + ["#d9534f"] * len(without)
    ax.bar(names, rs, color=colors, alpha=0.8)

    for i, r in enumerate(rs):
        ax.text(i, r + 0.01, f"{r:.3f}", ha="center", fontsize=9)

    ax.set_ylabel("Spearman r")
    ax.set_title("Q5: Leave-One-Out Component Comparison")
    ax.axhline(full_r, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "leave_one_out.png"), dpi=150)
    plt.close(fig)
    print("  components/leave_one_out.png")


# ===========================================================================
# Sensitivity plots (Q7) — output/plots/sensitivity/
# ===========================================================================

def plot_threshold_sweep(sensitivity_results: dict, plot_dir: str):
    """Heatmap or line plot for each component's threshold sweep."""
    for comp_name, comp_data in sensitivity_results.items():
        sweeps = comp_data.get("sweeps", [])
        if not sweeps:
            continue

        min_vals = sorted(set(s["min_value"] for s in sweeps if s["min_value"] is not None))
        max_vals = sorted(set(s["max_value"] for s in sweeps if s["max_value"] is not None))

        if len(min_vals) > 1 and len(max_vals) > 1:
            _plot_sweep_heatmap(comp_name, comp_data, sweeps, min_vals, max_vals, plot_dir)
        elif len(min_vals) > 1:
            _plot_sweep_line(comp_name, comp_data, sweeps, "min_value", min_vals, plot_dir)
        elif len(max_vals) > 1:
            _plot_sweep_line(comp_name, comp_data, sweeps, "max_value", max_vals, plot_dir)


def _plot_sweep_heatmap(comp_name, comp_data, sweeps, min_vals, max_vals, plot_dir):
    """2D heatmap for threshold sweep with both min and max varying."""
    matrix = np.full((len(min_vals), len(max_vals)), np.nan)
    for s in sweeps:
        if s["min_value"] in min_vals and s["max_value"] in max_vals:
            i = min_vals.index(s["min_value"])
            j = max_vals.index(s["max_value"])
            matrix[i, j] = s["spearman_r"] if s["spearman_r"] is not None else np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=np.nanmin(matrix) if not np.all(np.isnan(matrix)) else 0,
                   vmax=np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1)

    ax.set_xticks(range(len(max_vals)))
    ax.set_xticklabels([str(v) for v in max_vals])
    ax.set_yticks(range(len(min_vals)))
    ax.set_yticklabels([str(v) for v in min_vals])
    ax.set_xlabel("Max threshold")
    ax.set_ylabel("Min threshold")
    ax.set_title(f"Q7: {comp_name.capitalize()} Threshold Sensitivity")

    for i in range(len(min_vals)):
        for j in range(len(max_vals)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    orig_min = comp_data.get("original_min")
    orig_max = comp_data.get("original_max")
    if orig_min in min_vals and orig_max in max_vals:
        i = min_vals.index(orig_min)
        j = max_vals.index(orig_max)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="red", linewidth=2.5))

    fig.colorbar(im, ax=ax, label="Spearman r", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"sweep_{comp_name}.png"), dpi=150)
    plt.close(fig)
    print(f"  sensitivity/sweep_{comp_name}.png")


def _plot_sweep_line(comp_name, comp_data, sweeps, vary_key, vals, plot_dir):
    """1D line plot for threshold sweep with only one dimension varying."""
    rs = []
    for v in vals:
        match = [s for s in sweeps if s[vary_key] == v]
        r = match[0]["spearman_r"] if match and match[0]["spearman_r"] is not None else None
        rs.append(r)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(vals, rs, "o-", color="#337ab7", linewidth=2, markersize=8)

    orig_val = comp_data.get(f"original_{vary_key.split('_')[0]}")
    if orig_val in vals:
        idx = vals.index(orig_val)
        if rs[idx] is not None:
            ax.plot(orig_val, rs[idx], "s", color="red", markersize=12, zorder=5,
                    label=f"Original ({orig_val})")
            ax.legend(fontsize=9)

    for v, r in zip(vals, rs):
        if r is not None:
            ax.annotate(f"{r:.3f}", xy=(v, r), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=8)

    label = "Min threshold" if "min" in vary_key else "Max threshold"
    ax.set_xlabel(label)
    ax.set_ylabel("Spearman r")
    ax.set_title(f"Q7: {comp_name.capitalize()} Threshold Sensitivity")

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"sweep_{comp_name}.png"), dpi=150)
    plt.close(fig)
    print(f"  sensitivity/sweep_{comp_name}.png")


# ===========================================================================
# Discrimination plots (Q8) — output/plots/discrimination/
# ===========================================================================

def plot_score_distribution(discrimination_results: dict, plot_dir: str,
                            outcome_label: str = "Disease Cases", model=None):
    """Bar chart of score distribution with mean outcome overlay."""
    if model is None:
        model = DHIS2_UGANDA_MALARIA
    dist = discrimination_results.get("distribution", [])
    if not dist:
        return

    scores = [d["score"] for d in dist]
    pcts = [d.get("pct", 0) or 0 for d in dist]
    mean_cases = [d.get("mean_cases", 0) or 0 for d in dist]

    fig, ax = plt.subplots(figsize=(6, 5))

    n_components = len(model.components)
    score_cmap = plt.cm.RdYlGn
    colors = [score_cmap(s / n_components) for s in scores]

    bars = ax.bar(scores, pcts, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    for bar, pct, n in zip(bars, pcts, [d["n"] for d in dist]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%\n(n={n})", ha="center", fontsize=9)

    ax.set_xlabel("Composite Suitability Score")
    ax.set_ylabel("% of Observations")
    ax.set_title("Q8: Score Distribution")

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "score_distribution.png"), dpi=150)
    plt.close(fig)
    print("  discrimination/score_distribution.png")


def plot_location_max_score(discrimination_results: dict, plot_dir: str):
    """Bar chart of per-location % at max score."""
    per_loc = discrimination_results.get("per_location", [])
    if not per_loc:
        return

    locations = [d["location"] for d in per_loc]
    pcts = [d.get("pct_at_max", 0) or 0 for d in per_loc]

    fig, ax = plt.subplots(figsize=(max(6, len(locations) * 0.6), 5))
    colors = ["#d9534f" if p == 100 else "#f0ad4e" if p >= 80 else "#337ab7" for p in pcts]
    ax.bar(range(len(locations)), pcts, color=colors, alpha=0.8)

    ax.set_ylabel("% of Months at Max Score")
    ax.set_title("Q8: Signal Ceiling by Location")
    ax.set_xticks(range(len(locations)))
    ax.set_xticklabels(locations, rotation=45, ha="right", fontsize=9)
    ax.axhline(100, color="red", linestyle="--", linewidth=1, alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "location_max_score.png"), dpi=150)
    plt.close(fig)
    print("  discrimination/location_max_score.png")


# ===========================================================================
# Annual plots — output/plots/annual/
# ===========================================================================

def _compute_rolling_climate_means(
    df: pd.DataFrame,
    model,
    window: int = 12,
    cases_col: str = "disease_cases",
    location_col: str = "location",
    time_col: str = "time_period",
) -> pd.DataFrame:
    """Compute rolling window means for each climate column.

    Mirrors compute_window_data() from analysis/annual.py but computes the
    mean of each climate column over each window instead of threshold counts.

    Returns DataFrame with columns: location, time_period, total_cases,
    and {col}_mean for each component column. Empty if no complete windows.
    """
    climate_cols = [comp.column for comp in model.components]

    df = df.copy()
    df["_dt"] = pd.to_datetime(df[time_col])
    df = df.sort_values([location_col, "_dt"])

    rows = []
    for location, group in df.groupby(location_col):
        group = group.sort_values("_dt")
        rolling_cases = group[cases_col].rolling(window=window, min_periods=window).sum()
        rolling_means = {
            col: group[col].rolling(window=window, min_periods=window).mean()
            for col in climate_cols
        }
        for idx in rolling_cases.index:
            total = rolling_cases.loc[idx]
            means_ok = all(pd.notna(rolling_means[col].loc[idx]) for col in climate_cols)
            if pd.notna(total) and means_ok:
                row = {
                    "location": str(location),
                    "time_period": str(group.loc[idx, time_col]),
                    "total_cases": float(total),
                }
                for col in climate_cols:
                    row[f"{col}_mean"] = float(rolling_means[col].loc[idx])
                rows.append(row)

    if rows:
        return pd.DataFrame(rows)
    cols = ["location", "time_period", "total_cases"] + [f"{col}_mean" for col in climate_cols]
    return pd.DataFrame(columns=cols)


def plot_annual_climate_vs_cases(roll_df: pd.DataFrame, model, plot_dir: str,
                                  outcome_label: str = "Disease Cases"):
    """3-panel scatter: 12-month mean climate value vs. total annual incidence.

    Each point is a (location, window) pair. Complements the primary scatter
    which uses thresholded months_suitable — this shows the un-thresholded
    continuous climate signal at annual scale.
    """
    from scipy import stats as sp_stats

    n_comps = len(model.components)
    if len(roll_df) < 3:
        return

    fig, axes = plt.subplots(1, n_comps, figsize=(5 * n_comps, 5))
    if n_comps == 1:
        axes = [axes]

    locations = sorted(roll_df["location"].unique())
    loc_colors = plt.cm.Set2(np.linspace(0, 1, max(len(locations), 3)))
    loc_color_map = {loc: loc_colors[i] for i, loc in enumerate(locations)}

    for ax, comp in zip(axes, model.components):
        mean_col = f"{comp.column}_mean"
        if mean_col not in roll_df.columns:
            continue

        x = roll_df[mean_col]
        y = roll_df["total_cases"]

        for loc in locations:
            mask = roll_df["location"] == loc
            ax.scatter(x[mask], y[mask], color=loc_color_map[loc], alpha=0.5,
                       s=25, edgecolors="none", zorder=3)

        # Green shaded threshold zone
        xlim = ax.get_xlim()
        lo = comp.min_value if comp.min_value is not None else xlim[0]
        hi = comp.max_value if comp.max_value is not None else xlim[1]
        ax.axvspan(lo, hi, alpha=0.12, color="green", label="Threshold range")
        if comp.min_value is not None:
            ax.axvline(comp.min_value, color="green", linestyle="--", linewidth=1, alpha=0.7)
        if comp.max_value is not None:
            ax.axvline(comp.max_value, color="green", linestyle="--", linewidth=1, alpha=0.7)

        valid = x.notna() & y.notna()
        if valid.sum() >= 3:
            r, p = sp_stats.spearmanr(x[valid], y[valid])
            ax.text(0.02, 0.98, f"Spearman r = {r:.3f}\np = {_fmt_p(p)}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel(f"12-Month Mean {comp.column.replace('_', ' ').title()}")
        ax.set_ylabel(f"Total Annual {outcome_label}")
        ax.set_title(f"{comp.name.capitalize()}: Annual Mean vs. Annual Burden")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "annual_climate_vs_cases.png"), dpi=150)
    plt.close(fig)
    print("  annual/annual_climate_vs_cases.png")


def plot_annual_correlation_comparison(roll_df: pd.DataFrame, model, annual_results: dict,
                                        plot_dir: str):
    """Bar chart: annual Spearman r for months_suitable vs. each 12-month mean climate column.

    Directly tests whether thresholding loses information at the annual scale —
    same question as the monthly correlation comparison but for the primary analysis level.
    """
    from scipy import stats as sp_stats

    months_r = annual_results.get("correlation", {}).get("spearman_r", None)

    names = ["months_suitable"]
    rs = [months_r if months_r is not None else 0]
    colors = ["#337ab7"]

    y = roll_df["total_cases"]
    for comp in model.components:
        mean_col = f"{comp.column}_mean"
        if mean_col not in roll_df.columns:
            continue
        x = roll_df[mean_col]
        valid = x.notna() & y.notna()
        if valid.sum() >= 3:
            r, _ = sp_stats.spearmanr(x[valid], y[valid])
        else:
            r = 0.0
        names.append(comp.column.replace("_", "\n") + "\n(12m mean)")
        rs.append(float(r))
        colors.append("#5bc0de")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(names, rs, color=colors, alpha=0.8, height=0.5)
    ax.set_xlabel("Annual Spearman r (vs. total annual cases)")
    ax.set_title("Annual Correlation: Thresholded vs. Continuous\n(12-month rolling windows)")
    ax.axvline(0, color="black", linewidth=0.5)

    for i, r in enumerate(rs):
        x_off = r + 0.005 if r >= 0 else r - 0.005
        ha = "left" if r >= 0 else "right"
        ax.text(x_off, i, f"{r:.3f}", va="center", fontsize=9, ha=ha)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "annual_correlation_comparison.png"), dpi=150)
    plt.close(fig)
    print("  annual/annual_correlation_comparison.png")


def plot_months_suitable_vs_cases(annual_results: dict, plot_dir: str,
                                   outcome_label: str = "Disease Cases"):
    """Bar chart: how many (location, window) pairs had each count of suitable months.

    Aggregates over all regions and time — shows the distribution of annual
    suitability exposure across the full dataset.
    """
    window_data = annual_results.get("window_data", [])
    if len(window_data) < 3:
        return

    window = annual_results.get("window", 12)
    max_score = annual_results.get("max_score", "?")

    months = [int(d["months_suitable"]) for d in window_data]
    counts = {m: months.count(m) for m in range(0, max(months) + 1)}
    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]
    pcts = [100 * y / len(months) for y in ys]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(xs, pcts, color="#337ab7", alpha=0.8, edgecolor="black", linewidth=0.5)

    for bar, pct, n in zip(bars, pcts, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", fontsize=9)

    ax.set_xlabel(f"Months at Max Suitability (score == {max_score}) in {window}-Month Window")
    ax.set_ylabel("% of Location-Windows")
    ax.set_title("Distribution of Annual Suitability Exposure\n(aggregated over all regions and time)")
    ax.set_xticks(xs)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "months_suitable_vs_cases.png"), dpi=150)
    plt.close(fig)
    print("  annual/months_suitable_vs_cases.png")


def plot_annual_time_series(annual_results: dict, plot_dir: str,
                             outcome_label: str = "Disease Cases"):
    """Time series: months suitable per window and total annual cases, per location."""
    window_data = annual_results.get("window_data", [])
    if not window_data:
        return

    window = annual_results.get("window", 12)
    win_df = pd.DataFrame(window_data)
    win_df["_dt"] = pd.to_datetime(win_df["time_period"])
    locations = sorted(win_df["location"].unique())
    n_locs = len(locations)

    fig, axes = plt.subplots(n_locs, 1, figsize=(10, 3.5 * n_locs), sharex=True)
    if n_locs == 1:
        axes = [axes]

    for ax, loc in zip(axes, locations):
        loc_df = win_df[win_df["location"] == loc].sort_values("_dt")

        ax.bar(range(len(loc_df)), loc_df["total_cases"], color="#337ab7", alpha=0.6,
               label=f"Total cases ({window}-month sum)")
        ax.set_ylabel(f"Total {outcome_label}", color="#337ab7")

        ax2 = ax.twinx()
        ax2.plot(range(len(loc_df)), loc_df["months_suitable"], "o-",
                 color="#d9534f", linewidth=2, markersize=4, label="Months suitable")
        ax2.set_ylabel("Months at Max Suitability", color="#d9534f")
        ax2.set_ylim(-0.5, window + 0.5)

        ax.set_title(loc)
        ax.set_xticks(range(len(loc_df)))
        ax.set_xticklabels(loc_df["time_period"].values, rotation=45, ha="right", fontsize=8)

    fig.suptitle(
        f"Months at Max Suitability vs. Total Cases (rolling {window}-month window)",
        fontsize=13, y=1.01
    )
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "annual_time_series.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  annual/annual_time_series.png")


# ===========================================================================
# Main
# ===========================================================================
def main(base_analysis_dir: str = BASE_ANALYSIS_DIR,
         base_plot_dir: str = BASE_PLOT_DIR,
         train_data_path: str = TRAIN_DATA,
         model=None):
    if model is None:
        model = DHIS2_UGANDA_MALARIA

    # Create all plot subdirectories
    plot_dirs = {}
    for subdir in ["overall", "temporal", "spatial", "lag", "components", "sensitivity", "discrimination", "annual"]:
        d = os.path.join(base_plot_dir, subdir)
        os.makedirs(d, exist_ok=True)
        plot_dirs[subdir] = d

    # Load shared data
    results = load_json(os.path.join(base_analysis_dir, "overall", "correlation_results.json"))
    df = pd.read_csv(train_data_path)
    # Mirror train.py NaN handling: exclude all-NaN locations, fill remaining with 0
    if "disease_cases" in df.columns and "location" in df.columns:
        all_nan_locs = (
            df.groupby("location")["disease_cases"]
            .apply(lambda s: s.isna().all())
        )
        all_nan_locs = all_nan_locs[all_nan_locs].index.tolist()
        if all_nan_locs:
            df = df[~df["location"].isin(all_nan_locs)].reset_index(drop=True)
    if "disease_cases" in df.columns:
        df["disease_cases"] = df["disease_cases"].fillna(0)
    composite = model.compute_composite_score(df)
    component_scores = model.compute_component_scores(df)

    # Determine outcome column — match what train.py used
    if "population" in df.columns and (df["population"] > 0).all():
        df["incidence"] = df["disease_cases"] / df["population"] * 1000
        outcome_col = "incidence"
        outcome_label = "Incidence (per 1,000)"
    else:
        outcome_col = "disease_cases"
        outcome_label = "Disease Cases"

    print(f"Generating plots in {base_plot_dir}/ (outcome: {outcome_col})\n")

    # Load annual results early — needed by Q2, Q3, and annual section
    annual_results = load_json(os.path.join(base_analysis_dir, "annual", "annual_mean.json"))

    # --- Q1: Overall plots ---
    if results:
        print("Overall (Q1):")
        plot_threshold_frequency(component_scores, plot_dirs["overall"], model)
        plot_time_series(df, composite, plot_dirs["overall"], outcome_col, outcome_label, model)
        plot_location_correlations(results, plot_dirs["overall"])
        plot_score_heatmap(df, composite, plot_dirs["overall"], model)

    # --- Q2: Temporal plots ---
    temporal_results = load_json(os.path.join(base_analysis_dir, "temporal", "within_region.json"))
    if temporal_results and annual_results:
        print("\nTemporal (Q2):")
        plot_within_location_temporal(temporal_results, annual_results, plot_dirs["temporal"], outcome_label)

    # --- Q3: Spatial plots ---
    if annual_results:
        print("\nSpatial (Q3):")
        plot_score_vs_cases_by_location(annual_results, plot_dirs["spatial"], outcome_label)

    # --- Q4: Lag plots ---
    if results and results.get("lag_analysis"):
        print("\nLag (Q4):")
        plot_lag_correlation(results, plot_dirs["lag"], outcome_label)
        plot_component_lag(results, plot_dirs["lag"])
        plot_continuous_vs_composite_lag(results, plot_dirs["lag"])

    # --- Q5: Component plots ---
    leave_one_out_results = load_json(os.path.join(base_analysis_dir, "components", "leave_one_out.json"))

    if leave_one_out_results:
        print("\nComponents (Q5):")
        plot_leave_one_out(leave_one_out_results, plot_dirs["components"])

    # --- Q7: Sensitivity plots ---
    sensitivity_results = load_json(os.path.join(base_analysis_dir, "sensitivity", "threshold_sweep.json"))
    if sensitivity_results:
        print("\nSensitivity (Q7):")
        plot_threshold_sweep(sensitivity_results, plot_dirs["sensitivity"])

    # --- Q8: Discrimination plots ---
    discrimination_results = load_json(os.path.join(base_analysis_dir, "discrimination", "score_distribution.json"))
    if discrimination_results:
        print("\nDiscrimination (Q8):")
        plot_score_distribution(discrimination_results, plot_dirs["discrimination"], outcome_label, model)

    # --- Primary analysis: months suitable vs. annual cases ---
    if annual_results and "window_data" in annual_results and annual_results["window_data"]:
        print("\nPrimary analysis (annual):")
        plot_months_suitable_vs_cases(annual_results, plot_dirs["annual"], outcome_label)
        roll_df = _compute_rolling_climate_means(
            df, model, window=annual_results.get("window", 12),
            cases_col=outcome_col, location_col="location", time_col="time_period",
        )
        if len(roll_df) >= 3:
            plot_annual_correlation_comparison(roll_df, model, annual_results, plot_dirs["annual"])

    # --- Generate overview HTML ---
    html_dir = os.path.dirname(base_plot_dir)
    generate_overview_html(html_dir, model_name=model.name if hasattr(model, "name") else "Suitability Model")
    print("  overview.html")

    print("\nDone.")


def generate_overview_html(output_dir: str, model_name: str = "Suitability Model"):
    """Generate overview.html in output_dir with generic descriptions of all plots."""
    # Relative path prefix from HTML file to plots directory
    P = "plots"

    def card(img_rel, title, tags, paras, method=None):
        tag_html = "".join(f'<span class="tag tag-{cls}">{label}</span>' for cls, label in tags)
        p_html = "".join(f"<p>{t}</p>" for t in paras)
        m_html = f'<div class="method">{method}</div>' if method else ""
        return f"""
  <div class="plot-card">
    <img src="{P}/{img_rel}" alt="{title}" onclick="openLightbox(this)">
    <div class="plot-info">
      <h3>{title}</h3>
      {tag_html}
      {p_html}
      {m_html}
    </div>
  </div>
"""

    css = """
  body { font-family: Georgia, serif; max-width: 1400px; margin: 0 auto; padding: 24px; background: #f8f8f6; color: #222; }
  h1 { font-size: 1.8em; border-bottom: 3px solid #444; padding-bottom: 8px; }
  h2 { font-size: 1.3em; margin-top: 48px; padding: 8px 14px; background: #2c5f8a; color: white; border-radius: 4px; }
  h3 { font-size: 1.05em; margin: 0 0 6px 0; color: #2c5f8a; }
  .section-intro { background: #e8f0f7; border-left: 4px solid #2c5f8a; padding: 10px 16px; margin-bottom: 20px; font-size: 0.93em; }
  .plot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin-bottom: 12px; }
  .plot-grid.single { grid-template-columns: 1fr; }
  .plot-grid.triple { grid-template-columns: 1fr 1fr 1fr; }
  .plot-card { background: white; border: 1px solid #ccc; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.07); }
  .plot-card img { width: 100%; display: block; cursor: zoom-in; }
  .plot-info { padding: 12px 14px; }
  .plot-info p { margin: 4px 0; font-size: 0.88em; line-height: 1.5; }
  .tag { display: inline-block; font-size: 0.75em; font-weight: bold; padding: 1px 7px; border-radius: 3px; margin-right: 4px; margin-bottom: 4px; }
  .tag-monthly { background: #fff3cd; color: #856404; border: 1px solid #ffc107; }
  .tag-annual { background: #d1e7dd; color: #0a5a2e; border: 1px solid #28a745; }
  .tag-raw { background: #f8d7da; color: #721c24; border: 1px solid #dc3545; }
  .method { background: #f0f4ff; border-left: 3px solid #6c8ebf; padding: 5px 10px; margin-top: 6px; font-size: 0.87em; }
  .toc { background: #f0f4ff; padding: 12px 20px; border-radius: 6px; margin-bottom: 32px; font-size: 0.9em; }
  .toc a { color: #2c5f8a; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
  .toc li { margin: 3px 0; }
  hr { border: none; border-top: 1px solid #ddd; margin: 32px 0; }
  #lightbox { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.85); z-index:999; align-items:center; justify-content:center; }
  #lightbox.active { display:flex; }
  #lightbox img { max-width:92%; max-height:92%; border: 3px solid white; border-radius: 4px; }
  #lightbox-close { position:absolute; top:18px; right:28px; color:white; font-size:2em; cursor:pointer; }
"""

    js = """
function openLightbox(img) {
  document.getElementById('lightbox-img').src = img.src;
  document.getElementById('lightbox').classList.add('active');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('active');
}
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeLightbox();
});
"""

    annual_cards = (
        card("annual/months_suitable_vs_cases.png", "months_suitable_vs_cases.png",
             [("annual", "12-MONTH ROLLING")],
             ["<strong>X-axis:</strong> Number of months in the 12-month rolling window where all thresholds were simultaneously met (score == max). Each bar represents one integer value (0, 1, 2, …).",
              "<strong>Y-axis:</strong> Percentage of all (location, window-end) pairs that had that many suitable months.",
              "<strong>Purpose:</strong> Shows the distribution of annual suitability exposure across all location-windows — how often is the location fully suitable for 0, 1, 2, … months per year?"])
        +
        card("annual/annual_correlation_comparison.png", "annual_correlation_comparison.png",
             [("annual", "12-MONTH ROLLING"), ("raw", "RAW CLIMATE")],
             ["<strong>Structure:</strong> Horizontal bar chart — one bar per predictor. Blue bar = months_suitable (thresholded count, from primary analysis). Light blue bars = 12-month mean of each raw (un-thresholded) climate variable over the same rolling windows.",
              "<strong>All correlations</strong> are Spearman r vs. total annual incidence across the same (location, window) pairs.",
              "<strong>Purpose:</strong> Tests whether thresholding loses information at the annual scale. If raw climate means have higher r than months_suitable, the threshold-and-count approach discards predictive signal."])
    )

    overall_cards = (
        card("overall/threshold_frequency.png", "threshold_frequency.png",
             [("monthly", "CURRENT MONTH")],
             ["<strong>Structure:</strong> Bar chart — one bar per component threshold. Height = percentage of all province-months in which that threshold condition is met. Dashed reference line at 50%.",
              "<strong>Purpose:</strong> Shows how selective each threshold is. A threshold met in &gt;80% of months adds little discrimination; one met in &lt;20% is highly restrictive and will rarely allow the score to reach its maximum."])
        +
        card("overall/location_correlations.png", "location_correlations.png",
             [("annual", "12-MONTH ROLLING")],
             ["<strong>Structure:</strong> Bar chart — one bar per province. Height = mean number of months at max suitability per 12-month rolling window for that province.",
              "<strong>Purpose:</strong> Shows which provinces experience more vs. fewer fully-suitable months on average, giving a geographic profile of annual suitability exposure."])
        +
        card("overall/score_heatmap.png", "score_heatmap.png",
             [("monthly", "CURRENT MONTH")],
             ["<strong>Structure:</strong> Heatmap — rows = provinces, columns = calendar months (1–12), cells = modal composite suitability score for that province-month combination. Colour: green = high score (all thresholds met), red = low score (no thresholds met). Exactly 4 discrete colour levels.",
              "<strong>Purpose:</strong> Reveals the seasonal pattern of the model — when does the model fire across provinces, and is there consistent cross-province behaviour in particular months?"])
    )

    ts_card = card("overall/time_series.png", "time_series.png",
         [("monthly", "CURRENT MONTH")],
         ["<strong>Structure:</strong> One panel per province. X-axis = calendar time. Bars (steelblue) = monthly incidence. Line (right axis) = composite suitability score (0 to max).",
          "<strong>Purpose:</strong> Diagnostic — shows how the model score and observed incidence co-vary within each province over time. Not a formal correlation test."])

    temporal_card = card("temporal/within_location_temporal.png", "within_location_temporal.png",
         [("annual", "12-MONTH ROLLING")],
         ["<strong>Structure:</strong> Ranked lollipop dot plot — one dot per province, sorted by within-location Spearman r (highest to lowest). X-axis = Spearman r between rolling months_suitable and rolling total incidence within that province. Dashed vertical line at zero; dashed horizontal line at mean r across all provinces. Significance stars (* p&lt;0.05, ** p&lt;0.01) shown next to each dot.",
          "<strong>Purpose:</strong> Answers Q2: within a given province, does year-to-year variation in annual suitability exposure track year-to-year variation in annual disease burden? Each province's r indicates whether more suitable months in a window tends to coincide with more cases in that province.",
          "<strong>Method:</strong> Same 12-month rolling windows as primary analysis. Within-province Spearman r computed separately for each province across its ~145 rolling windows."])

    spatial_card = card("spatial/score_vs_cases_by_location.png", "score_vs_cases_by_location.png",
         [("annual", "12-MONTH ROLLING")],
         ["<strong>Structure:</strong> Scatter — one point per province. X-axis = mean months at max suitability per 12-month window (averaged over all windows for that province). Y-axis = mean annual incidence (sum of monthly incidence over each window, then averaged). OLS trend line. Spearman r annotated.",
          "<strong>Purpose:</strong> Answers Q3: do provinces that experience more fully-suitable months on average also have higher annual disease burden? This is the between-province (spatial) counterpart to Q2.",
          "<strong>Method note:</strong> Uses the same x-axis concept as Q2 (months_suitable) so Q2 and Q3 are directly comparable — within-province vs. between-province variation along the same predictor."])

    lag_cards = (
        card("lag/lag_correlation.png", "lag_correlation.png",
             [("monthly", "CURRENT MONTH")],
             ["<strong>Structure:</strong> Line plot of composite score Spearman r vs. lag (0–3 months). Each point shows the correlation when climate conditions at month T−lag are compared to disease cases at month T. Values annotated at each lag.",
              "<strong>Purpose:</strong> Identifies the temporal offset that best aligns the suitability signal with observed disease burden, accounting for mosquito development time, pathogen incubation, and reporting delay."])
        +
        card("lag/component_lag.png", "component_lag.png",
             [("monthly", "CURRENT MONTH")],
             ["<strong>Structure:</strong> Line plot of per-component point-biserial r vs. lag (0–3 months), plus composite Spearman r (black dashed). Each component's contribution to the lagged signal is shown separately.",
              "<strong>Purpose:</strong> Shows whether individual components have different optimal lags — e.g., temperature may peak at a different lag than precipitation."])
        +
        card("lag/continuous_vs_composite_lag.png", "continuous_vs_composite_lag.png",
             [("monthly", "CURRENT MONTH"), ("raw", "RAW CLIMATE")],
             ["<strong>Structure:</strong> Line plot comparing composite score Spearman r (blue solid) to raw continuous climate variable Spearman r (dashed lines) across lags 0–3.",
              "<strong>Purpose:</strong> Tests whether thresholding loses information at the monthly level across all lags. If raw continuous variables consistently outperform the thresholded composite, the threshold-and-count approach discards predictive signal at every lag."])
    )

    loo_card = card("components/leave_one_out.png", "leave_one_out.png",
         [("annual", "12-MONTH ROLLING")],
         ["<strong>Structure:</strong> Bar chart — full model annual Spearman r (blue) compared to the r obtained when each component is removed in turn (red bars). Uses the same 12-month rolling window primary analysis for all bars.",
          "<strong>Purpose:</strong> Identifies which components help vs. hurt the primary annual correlation. A red bar higher than the blue bar means removing that component <em>improves</em> the correlation — that component is a suppressor in this dataset."])

    sens_cards = (
        card("sensitivity/sweep_temperature.png", "sweep_temperature.png",
             [("annual", "12-MONTH ROLLING")],
             ["<strong>Structure:</strong> 2D heatmap — rows = minimum temperature threshold, columns = maximum temperature threshold. Each cell = Spearman r of the full 12-month rolling analysis run with those thresholds. Red box = original thresholds. Sweep ranges derived from the data distribution.",
              "<strong>Purpose:</strong> Shows how sensitive the primary annual correlation is to the specific temperature threshold values, and whether the original thresholds are empirically optimal for this dataset."])
        +
        card("sensitivity/sweep_precipitation.png", "sweep_precipitation.png",
             [("annual", "12-MONTH ROLLING")],
             ["<strong>Structure:</strong> 1D line plot — x-axis = minimum rainfall threshold (one-sided: no upper cap), y-axis = Spearman r. Sweep range derived from the actual data distribution so values are unit-correct regardless of mm/day vs. mm/month. Original threshold marked.",
              "<strong>Purpose:</strong> Shows how the annual correlation changes as the rainfall threshold is raised or lowered. Because precipitation is one-sided (lower bound only), a single sweep suffices."])
        +
        card("sensitivity/sweep_humidity.png", "sweep_humidity.png",
             [("annual", "12-MONTH ROLLING")],
             ["<strong>Structure:</strong> 2D heatmap — rows = minimum humidity threshold, columns = maximum humidity threshold. Each cell = Spearman r. Red box = original thresholds. Sweep ranges chosen to centre the original thresholds in the middle of the grid (±8 pp around each).",
              "<strong>Purpose:</strong> Shows how sensitive the annual correlation is to the humidity threshold window. Particularly important because the humidity distribution differs substantially between the dataset the thresholds were calibrated on (Uganda) and the target dataset."])
    )

    disc_card = card("discrimination/score_distribution.png", "score_distribution.png",
         [("monthly", "CURRENT MONTH")],
         ["<strong>Structure:</strong> Bar chart — one bar per composite score level (0 to max). Height = percentage of all province-months at that score. Each bar labelled with percentage and count.",
          "<strong>Purpose:</strong> Shows whether the score discriminates or whether most observations collapse into a single category. A model where most observations land in a middle score cannot usefully separate high from low risk months.",
         ],
         method="Eta-squared (ratio of between-group to total variance in incidence) quantifies how much of the incidence variance is explained by score level.")

    toc = """
<div class="toc">
  <strong>Contents</strong>
  <ul>
    <li><a href="#primary">Primary Analysis (Annual)</a> — months_suitable vs. annual cases</li>
    <li><a href="#overall">Q1 — Overall Diagnostic / Descriptive</a></li>
    <li><a href="#temporal">Q2 — Within-Location Temporal</a></li>
    <li><a href="#spatial">Q3 — Cross-Sectional Spatial</a></li>
    <li><a href="#lag">Q4 — Lag Analysis</a></li>
    <li><a href="#components">Q5 — Component Analysis</a></li>
    <li><a href="#sensitivity">Q7 — Threshold Sensitivity</a></li>
    <li><a href="#discrimination">Q8 — Score Discrimination</a></li>
  </ul>
</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{model_name} — Plot Overview</title>
<style>{css}</style>
</head>
<body>

<div id="lightbox" onclick="closeLightbox()">
  <span id="lightbox-close">&#10005;</span>
  <img id="lightbox-img" src="" alt="">
</div>

<h1>{model_name} — Plot Overview</h1>
<p style="color:#555; font-size:0.9em;">
  <span class="tag tag-annual">12-MONTH ROLLING</span> uses rolling window aggregation &nbsp;
  <span class="tag tag-monthly">CURRENT MONTH</span> uses raw month-by-month values &nbsp;
  <span class="tag tag-raw">RAW CLIMATE</span> uses continuous (non-thresholded) climate values
</p>

{toc}

<!-- ===================================================================== -->
<h2 id="primary">Primary Analysis — Annual Suitability Exposure vs. Annual Disease Burden</h2>
<div class="section-intro">
  <strong>Core question:</strong> Do districts/years with more fully-suitable months accumulate more annual disease burden?<br>
  <strong>Method:</strong> For each location, a 12-month rolling window advances through the time series. Each window produces one observation: predictor = count of months where composite score == max (all thresholds met); outcome = sum of monthly incidence over the same 12 months. Spearman r pooled across all location-window pairs. This is the <em>primary</em> comparison because it avoids the seasonal co-occurrence artifact (comparing current-month score to current-month cases would find a positive association by construction, since both peak in the transmission season).
</div>

<div class="plot-grid">
{annual_cards}
</div>

<!-- ===================================================================== -->
<h2 id="overall">Q1 — Overall Diagnostic / Descriptive</h2>
<div class="section-intro">
  <strong>Purpose:</strong> Descriptive and diagnostic plots showing model structure — when the model fires, how scores are distributed across locations and months, and the within-location rolling correlation profile. These are not current-month correlation tests; they reveal the model's seasonal behaviour and geographic variation in suitability exposure.
</div>

<div class="plot-grid triple">
{overall_cards}
</div>

<div class="plot-grid single">
{ts_card}
</div>

<!-- ===================================================================== -->
<h2 id="temporal">Q2 — Within-Location Temporal Analysis</h2>
<div class="section-intro">
  <strong>Core question:</strong> Within a given location, does year-to-year variation in annual suitability exposure predict year-to-year variation in annual disease burden?<br>
  <strong>Method:</strong> Same 12-month rolling windows as primary analysis. Within-province Spearman r computed separately for each location. A positive r means more suitable months in a window tends to coincide with more cases in that location.
</div>

<div class="plot-grid single">
{temporal_card}
</div>

<!-- ===================================================================== -->
<h2 id="spatial">Q3 — Cross-Sectional Spatial Analysis</h2>
<div class="section-intro">
  <strong>Core question:</strong> Do locations that experience more fully-suitable months on average also have more annual disease burden?<br>
  <strong>Method:</strong> One point per location — mean months_suitable per rolling window vs. mean annual incidence, averaged over the full observation period. Uses the same x-axis concept as Q2 (months_suitable) so the two questions are directly comparable: within-location (Q2) vs. between-location (Q3).
</div>

<div class="plot-grid single">
{spatial_card}
</div>

<!-- ===================================================================== -->
<h2 id="lag">Q4 — Lag Analysis</h2>
<div class="section-intro">
  <strong>Core question:</strong> What lag between climate conditions and reported cases maximises the correlation?<br>
  <strong>Method:</strong> For lag N, climate variables from month T−N are matched with disease cases from month T. Lags 0–3 months tested. Best lag = highest signed Spearman r, reflecting the directional hypothesis that more suitability predicts more cases.
</div>

<div class="plot-grid triple">
{lag_cards}
</div>

<!-- ===================================================================== -->
<h2 id="components">Q5 — Component Contribution</h2>
<div class="section-intro">
  <strong>Core question:</strong> What does each component contribute to the primary annual correlation?<br>
  <strong>Method:</strong> Leave-one-out using the 12-month rolling window approach — re-runs the full primary analysis with each component removed in turn. A component is a suppressor if removing it improves the correlation.
</div>

<div class="plot-grid single">
{loo_card}
</div>

<!-- ===================================================================== -->
<h2 id="sensitivity">Q7 — Threshold Sensitivity</h2>
<div class="section-intro">
  <strong>Core question:</strong> How sensitive is the primary annual correlation to the specific threshold values? Are the thresholds empirically optimal for this dataset?<br>
  <strong>Method:</strong> For each component, min and max threshold values are swept across a grid. For each combination, the full 12-month rolling window analysis is re-run and the Spearman r recorded. Sweep ranges are derived from the actual data distribution so they are unit-correct.
</div>

<div class="plot-grid triple">
{sens_cards}
</div>

<!-- ===================================================================== -->
<h2 id="discrimination">Q8 — Score Distribution and Discrimination</h2>
<div class="section-intro">
  <strong>Core question:</strong> Does the composite score discriminate between high- and low-burden observations? Is there a ceiling or floor problem?<br>
  <strong>Method:</strong> Descriptive — counts observations at each score level, reports percentage at each level, and eta-squared effect size (between-group variance in incidence explained by score level).
</div>

<div class="plot-grid">
{disc_card}
</div>

<hr>
<p style="color:#888; font-size:0.8em; text-align:center;">Auto-generated by visualize.py — Model: {model_name}</p>

<script>{js}</script>

</body>
</html>
"""

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "overview.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
