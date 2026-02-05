from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.core.types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    NamedResources,
    Rollout,
    RolloutMode,
    RolloutStatus,
    Span,
)


class LightningStore:
    """Contract for a store coordinating rollouts, attempts, spans, and resources."""

    async def enqueue_rollout(
        self,
        input: Any,
        mode: RolloutMode | None = None,
        resources_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Rollout:
        raise NotImplementedError()

    async def dequeue_rollout(self, worker_id: Optional[str] = None) -> Optional[AttemptedRollout]:
        raise NotImplementedError()

    async def add_span(self, span: Span) -> Optional[Span]:
        raise NotImplementedError()

    async def add_many_spans(self, spans: Sequence[Span]) -> Sequence[Span]:
        raise NotImplementedError()

    async def get_spans(self, rollout_id: str, attempt_id: Optional[str] = None) -> List[Span]:
        raise NotImplementedError()

    async def add_resources(self, resources: Dict[str, Any]) -> NamedResources:
        raise NotImplementedError()

    async def get_resources_by_id(self, resources_id: str) -> Optional[NamedResources]:
        raise NotImplementedError()

    async def get_latest_resources(self) -> Optional[NamedResources]:
        raise NotImplementedError()

    async def update_attempt(self, rollout_id: str, attempt_id: str, status: AttemptStatus) -> Attempt:
        raise NotImplementedError()

    async def update_rollout(self, rollout_id: str, status: RolloutStatus) -> Rollout:
        raise NotImplementedError()

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        raise NotImplementedError()

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        raise NotImplementedError()

    async def query_attempts(self, rollout_id: str) -> Sequence[Attempt]:
        raise NotImplementedError()

    async def wait_for_rollouts(self, rollout_ids: List[str]):
        raise NotImplementedError()

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        raise NotImplementedError()
