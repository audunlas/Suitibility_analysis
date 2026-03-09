from suitability.thresholds import ThresholdComponent
from suitability.model import SuitabilityModel

DHIS2_UGANDA_MALARIA = SuitabilityModel(
    name="DHIS2 Uganda Malaria Suitability",
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
            min_value=100.0,
            max_value=None,
        ),
        ThresholdComponent(
            name="humidity",
            column="humidity",
            min_value=50.0,
            max_value=80.0,
        ),
    ],
)
