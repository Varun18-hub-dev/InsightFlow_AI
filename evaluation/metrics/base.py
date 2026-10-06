from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class MetricResult:
    name: str
    value: float
    description: str
    metadata: Optional[Dict[str, Any]] = None

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
