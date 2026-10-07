from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class MetricResult:
    name: str
    value: float
    description: str
    metadata: dict[str, Any] | None = None

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __add__(self, other: Any) -> float:
        if isinstance(other, MetricResult):
            return self.value + other.value
        return self.value + float(other)

    def __radd__(self, other: Any) -> float:
        if isinstance(other, MetricResult):
            return other.value + self.value
        return float(other) + self.value

    def __sub__(self, other: Any) -> float:
        if isinstance(other, MetricResult):
            return self.value - other.value
        return self.value - float(other)

    def __rsub__(self, other: Any) -> float:
        if isinstance(other, MetricResult):
            return other.value - self.value
        return float(other) - self.value

    def __truediv__(self, other: Any) -> float:
        return self.value / float(other)

    def __rtruediv__(self, other: Any) -> float:
        return float(other) / self.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "description": self.description,
            "metadata": self.metadata or {},
        }


class BaseMetric(ABC):
    """Base class for all evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the metric measures."""
        pass

    @abstractmethod
    def compute(self, *args, **kwargs) -> MetricResult:
        """Compute the metric value."""
        pass
