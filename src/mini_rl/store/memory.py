import uuid
import time
from collections import defaultdict, deque
from copy import deepcopy
from typing import List, Optional, Tuple, Dict, Any

from ..types import Rollout, Attempt, Span, ResourceSnapshot
from .base import LightningStore


class InMemoryStore(LightningStore):
    def __init__(self) -> None:
        self._queue: deque[str] = deque()
        self._rollouts: Dict[str, Rollout] = {}
        self._spans: Dict[str, List[Span]] = defaultdict(list)
        self._resources: Dict[str, ResourceSnapshot] = {}
        self._latest_resources_id: Optional[str] = None
        self._span_seq: Dict[Tuple[str, str], int] = defaultdict(int)

    def add_resources(self, resources: Dict[str, Any]) -> ResourceSnapshot:
        resources_id = str(uuid.uuid4())
        snapshot = ResourceSnapshot(
            resources_id=resources_id,
            resources=deepcopy(resources),
            created_at=time.time(),
        )
        self._resources[resources_id] = snapshot
        self._latest_resources_id = resources_id
        return snapshot

    def get_latest_resources(self) -> ResourceSnapshot:
        if self._latest_resources_id is None:
            raise ValueError("No resources in store")
        return self._resources[self._latest_resources_id]

    def get_resources_by_id(self, resources_id: str) -> ResourceSnapshot:
        if resources_id not in self._resources:
            raise ValueError(f"Resource ID {resources_id} not found")
        return self._resources[resources_id]

    def enqueue_rollout(
        self, input_payload: Dict[str, Any], mode: str = "train"
    ) -> Rollout:

        if self._latest_resources_id is None:
            raise ValueError(
                "No resources available. Add resources before enqueuing rollouts."
            )

        latest = self.get_latest_resources()
        rollout_id = str(uuid.uuid4())
        rollout = Rollout(
            rollout_id=rollout_id,
            input=deepcopy(input_payload),
            mode="train" if mode == "train" else "eval",
            resources_id=latest.resources_id,
            status="queued",
        )
        self._rollouts[rollout_id] = rollout
        self._queue.append(rollout_id)
        return rollout

    def dequeue_rollout(self) -> Optional[Tuple[Rollout, Attempt]]:
        if not self._queue:
            return None
        rollout_id = self._queue.popleft()
        rollout = self._rollouts[rollout_id]
        attempt = Attempt(
            attempt_id=str(uuid.uuid4()),
            rollout_id=rollout_id,
            sequence_id=len(rollout.attempts) + 1,
            status="preparing",
        )
        rollout.attempts.append(attempt)
        rollout.status = "preparing"
        return rollout, attempt

    def update_attempt(self, rollout_id: str, attempt_id: str, status: str) -> None:
        rollout = self._rollouts[rollout_id]
        for attempt in rollout.attempts:
            if attempt.attempt_id == attempt_id:
                attempt.status = status  # type: ignore[assignment]
                break
        rollout.status = "succeeded" if status == "succeeded" else "failed"

    def add_span(self, span: Span) -> None:
        self._spans[span.rollout_id].append(span)

    def next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        key = (rollout_id, attempt_id)
        self._span_seq[key] += 1
        return self._span_seq[key]

    def get_rollout_by_id(self, rollout_id: str) -> Rollout:
        return self._rollouts[rollout_id]

    def query_rollouts(
        self, status_in: Optional[Tuple[str, ...]] = None
    ) -> List[Rollout]:
        values = list(self._rollouts.values())
        if status_in is None:
            return values
        return [r for r in values if r.status in status_in]

    def query_spans(self, rollout_id: str) -> List[Span]:
        return list(self._spans.get(rollout_id, []))
