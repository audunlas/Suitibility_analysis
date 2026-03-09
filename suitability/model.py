from dataclasses import dataclass, field
import pandas as pd

from .thresholds import ThresholdComponent


@dataclass
class SuitabilityModel:
    """A composite suitability model: sum of binary threshold components."""

    name: str
    components: list[ThresholdComponent] = field(default_factory=list)

    def compute_component_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a DataFrame with one binary column per component."""
        scores = pd.DataFrame(index=df.index)
        for component in self.components:
            scores[component.name] = component.evaluate(df[component.column])
        return scores

    def compute_composite_score(self, df: pd.DataFrame) -> pd.Series:
        """Returns the sum of all component scores."""
        component_scores = self.compute_component_scores(df)
        return component_scores.sum(axis=1)

    def required_columns(self) -> list[str]:
        """Column names needed from the data."""
        return [c.column for c in self.components]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "components": [c.to_dict() for c in self.components],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SuitabilityModel":
        components = [ThresholdComponent.from_dict(c) for c in d["components"]]
        return cls(name=d["name"], components=components)
