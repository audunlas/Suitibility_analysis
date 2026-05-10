# Suitability Correlation Analysis

Empirical analysis of climate-based malaria suitability models: do knowledge-based thresholds (derived without disease case data) actually correlate with observed malaria incidence?

The analysis applies Uganda DHIS2 malaria suitability thresholds (temperature 20–30°C, rainfall ≥100 mm/month, humidity 50–80%) to Lao PDR admin-1 monthly data (1998–2010) across 17 provinces.

## Structure

```
suitability/       — SuitabilityModel and ThresholdComponent
configs/           — Model config (chap_malaria.json)
analysis/          — Analysis modules (correlation, lag, sensitivity, etc.)
report.py          — CHAP report entry point
visualize.py       — Plot functions and PDF generation
```

## CHAP integration

The model is CHAP-compatible and can be run via `chap report` on any dataset with columns `time_period`, `location`, `disease_cases`, `rainfall`, `mean_temperature`, `humidity`. CHAP fetches climate covariates from DHIS2 automatically based on `required_covariates` in `MLproject`.

Note: relative humidity is not always included in CHAP's default covariate set.
