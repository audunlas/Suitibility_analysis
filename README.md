# Suitability Correlation Analysis

Empirical analysis of climate-based malaria suitability models: do knowledge-based thresholds (derived without disease case data) actually correlate with observed malaria incidence?

The analysis applies the Uganda DHIS2 malaria suitability thresholds (temperature 20–30°C, rainfall ≥100 mm/month, humidity 50–80%) to Lao PDR admin-1 monthly data (1998–2010) across 17 provinces.

---

## Code structure

```
suitability/
    model.py          — SuitabilityModel: computes composite score from components
    thresholds.py     — ThresholdComponent: single binary criterion (min/max bounds)

configs/
    chap_malaria.json — Model config with CHAP-compatible column names

analysis/
    pipeline.py       — Data loading, cleaning, and orchestration of all analyses
    annual.py         — Rolling 12-month window analysis (primary method)
    temporal.py       — Within-location rolling correlation
    spatial.py        — Cross-sectional analysis
    lag.py            — Lag sweep (0–3 months)
    components.py     — Leave-one-out and redundancy analysis
    sensitivity.py    — Threshold sweep
    discrimination.py — Score distribution and discrimination
    correlation.py    — Monthly correlation
    stratified.py     — Stratified by location and month

report.py             — CHAP entry point: runs all analyses and generates a PDF
visualize.py          — All plot functions and PDF generation
```

---

## Generating a report

```bash
chap report \
  --model-path /path/to/suitability_analysis \
  --dataset-csv your_data.csv \
  --out-file report.pdf
```

`your_data.csv` must have columns: `time_period`, `location`, `disease_cases`, `rainfall`, `mean_temperature`, `humidity`.

CHAP fetches the required covariates automatically from DHIS2 based on `required_covariates` in `MLproject`. Note that relative humidity is not always included in CHAP's default covariate set — verify availability before running on a new dataset.

---

## Adapting to a new country

Create a JSON config following the format in `configs/chap_malaria.json` with thresholds appropriate for your setting. Column names must match CHAP's standardized names (`humidity`, `rainfall`, `mean_temperature`).

---

## Data

`data/chap_LAO_admin1_monthly_chap.csv` — Lao PDR admin-1 monthly data, 1998–2010, 17 provinces. Columns: `time_period`, `location`, `disease_cases`, `population`, `mean_temperature`, `rainfall`, `humidity`.
