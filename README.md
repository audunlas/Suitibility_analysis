# Suitability Correlation Analysis

An empirical analysis of climate-based malaria suitability models — and an example of how to run such an analysis through [CHAP](https://github.com/dhis2/chap-core), giving direct access to DHIS2 data.

The central question: do knowledge-based suitability thresholds (derived without disease case data) actually correlate with observed malaria incidence? The analysis applies the Uganda DHIS2 malaria suitability thresholds (temperature 20–30°C, rainfall ≥100 mm/month, humidity 50–80%) to Lao PDR admin-1 monthly data (1998–2010) across 17 provinces. Pre-generated results are in `output/lao/` — open `output/lao/overview.html` to browse all plots.

---

## Code structure

```
suitability/
    model.py          — SuitabilityModel: computes composite score from components
    thresholds.py     — ThresholdComponent: single binary criterion (min/max bounds)

configs/
    lao_malaria.py    — Lao PDR model (Uganda thresholds, unit-adjusted)
    lao_malaria.json  — Same config as JSON, used by the CHAP entry point
    dhis2_uganda_malaria.py — Original Uganda model for reference

analysis/
    annual.py         — Rolling 12-month window analysis (primary method)
    temporal.py       — Within-location rolling correlation
    spatial.py        — Cross-sectional analysis
    lag.py            — Lag sweep (0–3 months)
    components.py     — Leave-one-out, redundancy
    sensitivity.py    — Threshold sweep
    discrimination.py — Score distribution and discrimination
    correlation.py    — Monthly correlation (supplementary)
    stratified.py     — Stratified by location and month
    report.py         — JSON/CSV output helpers

train.py              — CHAP train entry point: runs all analyses + visualization
predict.py            — CHAP predict stub: outputs placeholder predictions
visualize.py          — Generates all plots and overview.html
isolated_run_lao.py   — Run the Lao analysis directly without CHAP
```

`SuitabilityModel` is a pure scoring function — it takes a DataFrame and returns a composite score. All analysis modules take the model and data as inputs and are independent of each other. `train.py` orchestrates the full pipeline and calls `visualize.py` at the end to generate plots and the HTML report.

Model configurations are serialisable to/from JSON (`model.to_dict()` / `SuitabilityModel.from_dict()`), which is what enables the CHAP integration without hardcoding config choices in `train.py`.

---

## Running directly

```bash
uv run python isolated_run_lao.py
```

Runs the full pipeline on `data/chap_LAO_admin1_monthly.csv` and writes all outputs to `output/lao/`. Requires Python ≥ 3.10; `uv` handles environment setup automatically.

---

## CHAP integration

This model follows the `MLproject` convention used by CHAP external models. CHAP (Climate Health Analytics Platform) provides direct integration with DHIS2 — meaning you can point this analysis at any DHIS2 instance and run it on real programme data without manual data extraction. CHAP fetches the required climate covariates automatically based on the `required_covariates` in `MLproject`.

One note: this analysis requires **relative humidity**, which is not always included in CHAP's default covariate set. Verify availability before running on a new dataset.

CHAP uses two entry points defined in `MLproject`:

**`train`** — runs the full retrospective correlation analysis and generates all plots and the HTML report into CHAP's run directory.

**`predict`** — this is not a forecasting model, so `predict.py` outputs a placeholder CSV to satisfy the CHAP interface. The real output is from `train`.

The model configuration is passed as a JSON file path, so it can be swapped without touching the code:

```bash
chap eval \
  --model-name /path/to/suitability_analysis \
  --dataset-csv your_data.csv \
  --output-file results.nc \
  --model_config configs/your_country.json
```

To adapt to a new dataset, create a JSON config following the format in `configs/lao_malaria.json` with column names matching your data. `min_value` or `max_value` can be `null` for one-sided thresholds.

---

## Data

`data/chap_LAO_admin1_monthly.csv` — Lao PDR admin-1 monthly data, 1998–2010, 17 provinces. Columns: `time_period`, `location`, `disease_cases`, `population`, `mean_temperature`, `rainfall`, `mean_relative_humidity`.
