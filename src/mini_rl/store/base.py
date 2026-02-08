import abc
from typing import List, Optional, Tuple, Dict, Any
from ..types import Rollout, Attempt, Span, ResourceSnapshot


class LightningStore(abc.ABC):
    @abc.abstractmethod
    def add_resources(self, resources: Dict[str, Any]) -> ResourceSnapshot:
        """Register a new resource snapshot."""
        ...

    @abc.abstractmethod
    def get_latest_resources(self) -> ResourceSnapshot:
        """Get the most recent resource snapshot."""
        ...

    @abc.abstractmethod
    def get_resources_by_id(self, resources_id: str) -> ResourceSnapshot:
        """Get a specific resource snapshot by ID."""
        ...

    @abc.abstractmethod
    def enqueue_rollout(
        self, input_payload: Dict[str, Any], mode: str = "train"
    ) -> Rollout:
        """Add a rollout request to the queue."""
        ...

    @abc.abstractmethod
    def dequeue_rollout(self) -> Optional[Tuple[Rollout, Attempt]]:
        """Pop the next rollout request and create a new attempt."""
        ...

    @abc.abstractmethod
    def update_attempt(self, rollout_id: str, attempt_id: str, status: str) -> None:
        """Update the status of an attempt (and potentially the rollout)."""
        ...

    @abc.abstractmethod
    def add_span(self, span: Span) -> None:
        """Record an execution span."""
        ...

    @abc.abstractmethod
    def next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        """Get the next sequence ID for a span in an attempt."""
        ...

    @abc.abstractmethod
    def get_rollout_by_id(self, rollout_id: str) -> Rollout:
        """Retrieve a rollout by its ID."""
        ...

    @abc.abstractmethod
    def query_rollouts(
        self, status_in: Optional[Tuple[str, ...]] = None
    ) -> List[Rollout]:
        """Query rollouts matching specific statuses."""
        ...

    @abc.abstractmethod
    def query_spans(self, rollout_id: str) -> List[Span]:
        """Retrieve all spans for a given rollout."""
        ...
