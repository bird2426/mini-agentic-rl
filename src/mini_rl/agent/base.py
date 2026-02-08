import abc
from typing import Dict, Any, Generic, TypeVar
from ..types import Rollout, Attempt


class LitAgent(abc.ABC):
    @abc.abstractmethod
    def rollout(self, rollout: Rollout, attempt: Attempt) -> Dict[str, Any]:
        """
        Execute a single rollout attempt.
        Must return a dictionary payload (e.g., input prompt, response, tokens).
        """
        ...
