import abc
from typing import List, Dict, Any, Generic, TypeVar

from ..types import Span, Triplet


class Adapter(abc.ABC):
    @abc.abstractmethod
    def adapt(self, spans: List[Dict[str, object]]) -> List[Triplet]:
        """Convert raw spans into training triplets."""
        ...
