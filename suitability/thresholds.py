from dataclasses import dataclass
import pandas as pd


@dataclass
class ThresholdComponent:
    """A single binary threshold criterion for suitability scoring.

    Evaluates whether values fall within [min_value, max_value].
    Either bound can be None for one-sided thresholds.
    """

    name: str
    column: str
    min_value: float | None = None
    max_value: float | None = None

    def evaluate(self, series: pd.Series) -> pd.Series:
        """Returns int Series: 1 where threshold is met, 0 otherwise.

        NORMATIVE DECISION (ND-A): Threshold bounds are inclusive (>= and <=).
        A value exactly equal to min_value or max_value is counted as "suitable."
        This follows the convention in the underlying literature (e.g., temperature
        range [18, 32] means 18°C and 32°C are both suitable). Alternative: strict
        inequality (> and <) would exclude boundary values. The choice matters only
        for observations that land exactly on a threshold, which is uncommon in
        continuous climate data but possible after rounding or aggregation.
        """
        if self.min_value is not None and self.max_value is not None:
            result = (series >= self.min_value) & (series <= self.max_value)
        elif self.min_value is not None:
            result = series >= self.min_value
        elif self.max_value is not None:
            result = series <= self.max_value
        else:
            # No bounds set — always suitable
            result = pd.Series(1, index=series.index)
        return result.astype(int)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "column": self.column,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThresholdComponent":
        return cls(**d)
