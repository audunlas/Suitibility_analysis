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
    lao_malaria.py        — Lao PDR model (Uganda thresholds, unit-adjusted), local column names
    lao_malaria.json      — Same config as JSON, for direct/isolated runs (uses mean_relative_humidity)
    chap_malaria.json     — Same thresholds, CHAP-compatible column names (uses humidity)
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
report.py             — CHAP report entry point: generates a PDF from any dataset
visualize.py          — Generates all plots, overview.html, and PDF reports
analysis/pipeline.py  — Shared load/clean and analysis logic used by train.py and report.py
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

This model is fully CHAP-compatible. CHAP (Climate Health Analytics Platform) provides direct integration with DHIS2 — meaning you can point this analysis at any DHIS2 instance and run it on real programme data without manual data extraction. CHAP fetches the required climate covariates (`rainfall`, `mean_temperature`, `humidity`) automatically based on `required_covariates` in `MLproject`.

One note: this analysis requires **relative humidity**, which is not always included in CHAP's default covariate set. Verify availability before running on a new dataset.

Three entry points are defined in `MLproject`:

**`train`** — runs the full retrospective correlation analysis and generates all plots and the HTML report.

**`predict`** — this is not a forecasting model, so `predict.py` outputs a placeholder CSV to satisfy the CHAP interface. The real output is from `train` and `report`.

**`report`** — trains the model on the supplied data and produces a self-contained multi-page PDF summarising all analyses. This is the primary output for sharing results.

### Generating a PDF report

```bash
chap report \
  /path/to/suitability_analysis \
  your_data.csv \
  report.pdf
```

Where `your_data.csv` must have columns: `time_period`, `location`, `disease_cases`, `rainfall`, `mean_temperature`, `humidity`.

### Backtesting

```bash
chap eval \
  --model-name /path/to/suitability_analysis \
  --dataset-csv your_data.csv \
  --output-file results.nc
```

Note: evaluation metrics from `chap eval` are not meaningful for this model since `predict.py` is a stub. Use `chap report` to assess model behaviour.

### Adapting to a new country

Create a JSON config following the format in `configs/chap_malaria.json` with thresholds appropriate for your setting. Column names must match CHAP's standardized names (`humidity`, `rainfall`, `mean_temperature`). Pass it via `--model-configuration-yaml` or set it as the default in `MLproject`.

---

## Data

`data/chap_LAO_admin1_monthly.csv` — Lao PDR admin-1 monthly data, 1998–2010, 17 provinces. Columns: `time_period`, `location`, `disease_cases`, `population`, `mean_temperature`, `rainfall`, `mean_relative_humidity`.
