from suitability.thresholds import ThresholdComponent
from suitability.model import SuitabilityModel

# NORMATIVE DECISION: Using the same threshold values as the Uganda DHIS2 malaria model
# (temperature 20–30°C, rainfall ≥100mm/month equivalent, humidity 60–80%).
# Rationale: allows direct comparison of model performance across countries and
# tests whether the Uganda operational thresholds generalize to Laos.
# Alternatives: Lao-specific thresholds from entomological literature.

# NORMATIVE DECISION: Rainfall threshold converted from mm/month to mm/day.
# Uganda DHIS2 threshold: 100mm/month ÷ 30.44 days/month = 3.28mm/day.
# The Lao dataset provides mean daily rainfall (mm/day) per month, not monthly totals.
# Confirm or redirect.

LAO_MALARIA = SuitabilityModel(
    name="Lao PDR Malaria Suitability (Uganda DHIS2 thresholds, unit-adjusted)",
    components=[
        ThresholdComponent(
            name="temperature",
            column="mean_temperature",
            min_value=20.0,
            max_value=30.0,
        ),
        ThresholdComponent(
            name="precipitation",
            column="rainfall",
            min_value=3.28,   # ≡ 100mm/month ÷ 30.44 days/month
            max_value=None,
        ),
        ThresholdComponent(
            name="humidity",
            column="mean_relative_humidity",
            min_value=50.0,
            max_value=80.0,
        ),
    ],
)
