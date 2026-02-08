import abc
from typing import Generic, TypeVar, Optional, Dict, Any
from ..types import Rollout, Attempt, Span
from ..store.base import LightningStore
from ..agent.base import LitAgent

T_Agent = TypeVar("T_Agent", bound=LitAgent)


class Runner(abc.ABC, Generic[T_Agent]):
    def __init__(self, store: LightningStore, agent: T_Agent):
        self.store = store
        self.agent = agent

    @abc.abstractmethod
    def run_step(self) -> bool:
        """
        Execute a single step: dequeue rollout, run agent, record spans.
        Returns True if a rollout was processed, False if queue was empty.
        """
        ...

    @abc.abstractmethod
    def run_until_empty(self) -> None:
        """Run continuously until the store is empty."""
        ...
