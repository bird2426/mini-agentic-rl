from __future__ import annotations
import time
import uuid
import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
from src.core.types import (
    Rollout, Attempt, AttemptedRollout, RolloutStatus,
    AttemptStatus, Span, NamedResources, RolloutMode, RolloutConfig
)

logger = logging.getLogger(__name__)

class InMemoryStore:
    """Minimal in-memory store for coordinating rollouts."""
    
    def __init__(self):
        self._rollouts: Dict[str, Rollout] = {}
        self._attempts: Dict[str, List[Attempt]] = {}
        self._spans: Dict[str, List[Span]] = {} # rollout_id -> spans
        self._resources: List[NamedResources] = []
        self._queue: List[str] = [] # List of rollout_ids
        self._lock = asyncio.Lock()

    async def add_resources(self, resources: Dict[str, Any]) -> NamedResources:
        async with self._lock:
            res = NamedResources(
                resources=resources,
                resources_id=str(uuid.uuid4()),
                timestamp=time.time()
            )
            self._resources.append(res)
            return res

    async def get_latest_resources(self) -> Optional[NamedResources]:
        async with self._lock:
            return self._resources[-1] if self._resources else None

    async def enqueue_rollout(self, task_input: Any, mode: RolloutMode = "train") -> Rollout:
        async with self._lock:
            rollout_id = str(uuid.uuid4())
            rollout = Rollout(
                rollout_id=rollout_id,
                input=task_input,
                mode=mode,
                status="queuing"
            )
            self._rollouts[rollout_id] = rollout
            self._queue.append(rollout_id)
            return rollout

    async def dequeue_rollout(self, worker_id: str) -> Optional[AttemptedRollout]:
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

    async def add_span(self, span: Span):
        async with self._lock:
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

    async def update_attempt_status(self, rollout_id: str, attempt_id: str, status: AttemptStatus):
        async with self._lock:
            for attempt in self._attempts.get(rollout_id, []):
                if attempt.attempt_id == attempt_id:
                    attempt.status = status
                    attempt.end_time = time.time() if status in ["succeeded", "failed"] else None
            
            rollout = self._rollouts[rollout_id]
            if status == "succeeded":
                rollout.status = "succeeded"
                rollout.end_time = time.time()
            elif status == "failed":
                rollout.status = "failed"
                rollout.end_time = time.time()

    async def get_spans(self, rollout_id: str) -> List[Span]:
        async with self._lock:
            return self._spans.get(rollout_id, [])

    async def wait_for_rollouts(self, rollout_ids: List[str]):
        """Wait until all specified rollouts are finished."""
        while True:
            async with self._lock:
                finished = all(self._rollouts[rid].status in ["succeeded", "failed", "cancelled"] for rid in rollout_ids)
                if finished:
                    return [self._rollouts[rid] for rid in rollout_ids]
            await asyncio.sleep(0.1)
