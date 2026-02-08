import abc
from typing import List, Any
from ..types import Triplet, AlgorithmMetrics


class Algorithm(abc.ABC):
    @abc.abstractmethod
    def optimize(self, triplets: List[Triplet]) -> AlgorithmMetrics:
        """Perform a policy update using the given triplets."""
        ...
