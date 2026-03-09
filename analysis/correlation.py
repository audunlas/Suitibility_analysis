import numpy as np
import pandas as pd
from scipy import stats

from suitability.model import SuitabilityModel


class CorrelationAnalysis:
    """Core analysis: how well do suitability scores correlate with disease cases?"""

    def __init__(self, model: SuitabilityModel):
        self.model = model

    def run(
        self,
        df: pd.DataFrame,
        cases_col: str = "disease_cases",
        population_col: str = "population",
        location_col: str = "location",
        time_col: str = "time_period",
    ) -> dict:
        """Run all analyses. Returns nested dict of results."""
        component_scores = self.model.compute_component_scores(df)
        composite = self.model.compute_composite_score(df)

        results = {}
        results["model_name"] = self.model.name
        results["n_observations"] = len(df)
        results["component_analysis"] = self._analyze_components(
            df, component_scores, cases_col
        )
        results["composite_analysis"] = self._analyze_composite(
            df, composite, cases_col
        )
        results["overall_correlation"] = self._overall_correlation(
            composite, df[cases_col]
        )
        results["continuous_variables"] = self._analyze_continuous(df, cases_col)

        # Only compute secondary incidence analysis when using raw cases
        if population_col in df.columns and cases_col == "disease_cases":
            # Scale to cases per 1,000 population, consistent with train.py
            incidence = df[cases_col] / df[population_col] * 1000
            results["incidence_analysis"] = {
                "composite_correlation": self._overall_correlation(
                    composite, incidence
                ),
                "component_analysis": self._analyze_components(
                    df.assign(_incidence=incidence),
                    component_scores,
                    "_incidence",
                ),
            }

        return results

    def _analyze_components(
        self, df: pd.DataFrame, component_scores: pd.DataFrame, cases_col: str
    ) -> dict:
        """Per-component: point-biserial correlation, Mann-Whitney U, group stats."""
        results = {}
        cases = df[cases_col]

        for comp_name in component_scores.columns:
            binary = component_scores[comp_name]
            met = cases[binary == 1]
            not_met = cases[binary == 0]

            comp_result = {
                "n_met": int(binary.sum()),
                "n_not_met": int((binary == 0).sum()),
            }

            # Group statistics
            comp_result["mean_cases_met"] = _safe_float(met.mean())
            comp_result["mean_cases_not_met"] = _safe_float(not_met.mean())
            comp_result["median_cases_met"] = _safe_float(met.median())
            comp_result["median_cases_not_met"] = _safe_float(not_met.median())

            # Point-biserial correlation
            if len(met) > 0 and len(not_met) > 0:
                r, p = stats.pointbiserialr(binary, cases)
                comp_result["point_biserial_r"] = _safe_float(r)
                comp_result["point_biserial_p"] = _safe_float(p)

                # Mann-Whitney U
                u_stat, u_p = stats.mannwhitneyu(
                    met, not_met, alternative="two-sided"
                )
                comp_result["mann_whitney_u"] = _safe_float(u_stat)
                comp_result["mann_whitney_p"] = _safe_float(u_p)
            else:
                comp_result["point_biserial_r"] = None
                comp_result["point_biserial_p"] = None
                comp_result["mann_whitney_u"] = None
                comp_result["mann_whitney_p"] = None
                comp_result["note"] = "All observations in one group — no comparison possible"

            results[comp_name] = comp_result

        return results

    def _analyze_composite(
        self, df: pd.DataFrame, composite: pd.Series, cases_col: str
    ) -> dict:
        """Per score level: group stats. Kruskal-Wallis across levels."""
        cases = df[cases_col]
        levels = sorted(composite.unique())
        level_stats = {}

        groups = []
        for level in levels:
            group = cases[composite == level]
            groups.append(group)
            level_stats[int(level)] = {
                "n": len(group),
                "mean": _safe_float(group.mean()),
                "median": _safe_float(group.median()),
                "std": _safe_float(group.std()),
            }

        result = {"level_stats": level_stats}

        # Kruskal-Wallis across all levels (if at least 2 non-empty groups)
        non_empty = [g for g in groups if len(g) > 0]
        if len(non_empty) >= 2:
            h_stat, h_p = stats.kruskal(*non_empty)
            result["kruskal_wallis_h"] = _safe_float(h_stat)
            result["kruskal_wallis_p"] = _safe_float(h_p)

        # Pairwise Mann-Whitney between adjacent levels
        pairwise = []
        for i in range(len(levels) - 1):
            g1 = cases[composite == levels[i]]
            g2 = cases[composite == levels[i + 1]]
            if len(g1) > 0 and len(g2) > 0:
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                pairwise.append({
                    "levels": [int(levels[i]), int(levels[i + 1])],
                    "u_statistic": _safe_float(u_stat),
                    "p_value": _safe_float(u_p),
                })
        result["pairwise_adjacent"] = pairwise

        return result

    def _overall_correlation(
        self, scores: pd.Series, cases: pd.Series
    ) -> dict:
        """Spearman, Pearson, and R² between score and cases."""
        result = {}

        if len(scores) < 3:
            result["note"] = "Too few observations for correlation"
            return result

        spearman_r, spearman_p = stats.spearmanr(scores, cases)
        result["spearman_r"] = _safe_float(spearman_r)
        result["spearman_p"] = _safe_float(spearman_p)

        pearson_r, pearson_p = stats.pearsonr(scores, cases)
        result["pearson_r"] = _safe_float(pearson_r)
        result["pearson_p"] = _safe_float(pearson_p)
        result["r_squared"] = _safe_float(pearson_r**2)

        return result

    def _analyze_continuous(self, df: pd.DataFrame, cases_col: str) -> dict:
        """Correlation between raw climate variables and cases."""
        results = {}
        cases = df[cases_col]

        for component in self.model.components:
            col = component.column
            if col not in df.columns:
                continue

            values = df[col]
            if len(values) < 3:
                results[col] = {"note": "Too few observations"}
                continue

            spearman_r, spearman_p = stats.spearmanr(values, cases)
            pearson_r, pearson_p = stats.pearsonr(values, cases)

            results[col] = {
                "spearman_r": _safe_float(spearman_r),
                "spearman_p": _safe_float(spearman_p),
                "pearson_r": _safe_float(pearson_r),
                "pearson_p": _safe_float(pearson_p),
            }

        return results


def _safe_float(val) -> float | None:
    """Convert to float, returning None for NaN/inf."""
    if val is None:
        return None
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return None
    return val
