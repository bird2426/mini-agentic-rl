from __future__ import annotations
import time
import uuid
import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
from src.core.types import (
    Rollout,
    Attempt,
    AttemptedRollout,
    RolloutStatus,
    AttemptStatus,
    Span,
    NamedResources,
    RolloutMode,
)
from src.store.base import LightningStore

logger = logging.getLogger(__name__)

class InMemoryLightningStore(LightningStore):
    """Minimal in-memory store for coordinating rollouts."""
    
    def __init__(self):
        self._rollouts: Dict[str, Rollout] = {}
        self._attempts: Dict[str, List[Attempt]] = {}
        self._spans: Dict[str, List[Span]] = {} # rollout_id -> spans
        self._resources: List[NamedResources] = []
        self._resources_by_id: Dict[str, NamedResources] = {}
        self._queue: List[str] = [] # List of rollout_ids
        self._span_sequences: Dict[Tuple[str, str], int] = {}
        self._lock = asyncio.Lock()

    async def add_resources(self, resources: Dict[str, Any]) -> NamedResources:
        async with self._lock:
            res = NamedResources(
                resources=resources,
                resources_id=str(uuid.uuid4()),
                timestamp=time.time()
            )
            self._resources.append(res)
            self._resources_by_id[res.resources_id] = res
            return res

    async def get_resources_by_id(self, resources_id: str) -> Optional[NamedResources]:
        async with self._lock:
            return self._resources_by_id.get(resources_id)

    async def get_latest_resources(self) -> Optional[NamedResources]:
        async with self._lock:
            return self._resources[-1] if self._resources else None

    async def enqueue_rollout(
        self,
        input: Any,
        mode: RolloutMode | None = None,
        resources_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Rollout:
        async with self._lock:
            rollout_id = str(uuid.uuid4())
            rollout = Rollout(
                rollout_id=rollout_id,
                input=input,
                mode=mode or "train",
                status="queuing",
                metadata=metadata or {},
                resources_id=resources_id,
            )
            self._rollouts[rollout_id] = rollout
            self._queue.append(rollout_id)
            return rollout

    async def dequeue_rollout(self, worker_id: Optional[str] = None) -> Optional[AttemptedRollout]:
        async with self._lock:
            if not self._queue:
                return None
            
            rollout_id = self._queue.pop(0)
            rollout = self._rollouts[rollout_id]
            rollout.status = "preparing"
            
            attempt_id = str(uuid.uuid4())
            attempt = Attempt(
                rollout_id=rollout_id,
                attempt_id=attempt_id,
                sequence_id=len(self._attempts.get(rollout_id, [])) + 1,
                worker_id=worker_id,
                status="preparing"
            )
            
            if rollout_id not in self._attempts:
                self._attempts[rollout_id] = []
            self._attempts[rollout_id].append(attempt)
            
            return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

    async def add_span(self, span: Span) -> Optional[Span]:
        async with self._lock:
            if span.sequence_id is None:
                key = (span.rollout_id, span.attempt_id)
                current = self._span_sequences.get(key, 0) + 1
                self._span_sequences[key] = current
                span.sequence_id = current
            rid = span.rollout_id
            if rid not in self._spans:
                self._spans[rid] = []
            self._spans[rid].append(span)
            
            rollout = self._rollouts[rid]
            if rollout.status == "preparing":
                rollout.status = "running"
            
            for attempt in self._attempts.get(rid, []):
                if attempt.attempt_id == span.attempt_id:
                    if attempt.status == "preparing":
                        attempt.status = "running"

            return span

    async def add_many_spans(self, spans: Sequence[Span]) -> Sequence[Span]:
        stored: List[Span] = []
        for span in spans:
            stored_span = await self.add_span(span)
            if stored_span is not None:
                stored.append(stored_span)
        return stored

    async def update_attempt(self, rollout_id: str, attempt_id: str, status: AttemptStatus) -> Attempt:
        async with self._lock:
            updated_attempt: Optional[Attempt] = None
            for attempt in self._attempts.get(rollout_id, []):
                if attempt.attempt_id == attempt_id:
                    attempt.status = status
                    attempt.end_time = time.time() if status in ["succeeded", "failed"] else None
                    updated_attempt = attempt

            rollout = self._rollouts[rollout_id]
            if status == "succeeded":
                rollout.status = "succeeded"
                rollout.end_time = time.time()
            elif status == "failed":
                rollout.status = "failed"
                rollout.end_time = time.time()
            elif status == "cancelled":
                rollout.status = "cancelled"
                rollout.end_time = time.time()

            if updated_attempt is None:
                raise ValueError(f"Attempt not found: {attempt_id}")
            return updated_attempt

    async def update_attempt_status(self, rollout_id: str, attempt_id: str, status: AttemptStatus):
        await self.update_attempt(rollout_id, attempt_id, status)

    async def update_rollout(self, rollout_id: str, status: RolloutStatus) -> Rollout:
        async with self._lock:
            rollout = self._rollouts[rollout_id]
            rollout.status = status
            if status in ["succeeded", "failed", "cancelled"]:
                rollout.end_time = time.time()
            return rollout

    async def get_spans(self, rollout_id: str, attempt_id: Optional[str] = None) -> List[Span]:
        async with self._lock:
            spans = self._spans.get(rollout_id, [])
            if attempt_id is None:
                return list(spans)
            return [span for span in spans if span.attempt_id == attempt_id]

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        async with self._lock:
            return self._rollouts.get(rollout_id)

    async def query_attempts(self, rollout_id: str) -> Sequence[Attempt]:
        async with self._lock:
            return list(self._attempts.get(rollout_id, []))

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        async with self._lock:
            attempts = self._attempts.get(rollout_id, [])
            return attempts[-1] if attempts else None

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        async with self._lock:
            key = (rollout_id, attempt_id)
            current = self._span_sequences.get(key, 0) + 1
            self._span_sequences[key] = current
            return current

    async def wait_for_rollouts(self, rollout_ids: List[str]):
        """Wait until all specified rollouts are finished."""
        while True:
            async with self._lock:
                finished = all(self._rollouts[rid].status in ["succeeded", "failed", "cancelled"] for rid in rollout_ids)
                if finished:
                    return [self._rollouts[rid] for rid in rollout_ids]
            await asyncio.sleep(0.1)


InMemoryStore = InMemoryLightningStore
