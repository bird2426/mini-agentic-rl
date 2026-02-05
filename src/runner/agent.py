from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

from src.core.agent import LitAgent
from src.core.types import Span
from src.runner.base import Runner
from src.store.base import LightningStore

logger = logging.getLogger(__name__)


class LitAgentRunner(Runner):
    """Worker that executes agent rollouts by polling the store."""

    def __init__(self, tracer: Optional[object] = None, poll_interval: float = 0.1):
        self._agent: Optional[LitAgent] = None
        self._store: Optional[LightningStore] = None
        self._running = False
        self._poll_interval = poll_interval
        self._tracer = tracer
        self.worker_id: Optional[str] = None

    def init(self, agent: LitAgent, *, hooks=None, **kwargs) -> None:
        self._agent = agent
        self._agent.set_runner(self)

    def init_worker(self, worker_id: str, store: LightningStore, **kwargs) -> None:
        self.worker_id = worker_id
        self._store = store

    def get_store(self) -> LightningStore:
        if self._store is None:
            raise ValueError("Store not initialized. Call init_worker() first.")
        return self._store

    def get_agent(self) -> LitAgent:
        if self._agent is None:
            raise ValueError("Agent not initialized. Call init() first.")
        return self._agent

    async def run_once(self) -> bool:
        store = self.get_store()
        agent = self.get_agent()

        rollout_attempt = await store.dequeue_rollout(self.worker_id)
        if not rollout_attempt:
            return False

        resources = None
        if rollout_attempt.resources_id:
            resources = await store.get_resources_by_id(rollout_attempt.resources_id)
        if resources is None:
            resources = await store.get_latest_resources()
        if not resources:
            await store.update_attempt(rollout_attempt.rollout_id, rollout_attempt.attempt.attempt_id, "failed")
            return True

        try:
            if agent.is_async():
                result = await agent.rollout_async(rollout_attempt.input, resources, rollout_attempt)
            else:
                result = agent.rollout(rollout_attempt.input, resources, rollout_attempt)

            if isinstance(result, (float, int)):
                now = time.time()
                span = Span(
                    trace_id=str(uuid.uuid4()).replace("-", ""),
                    span_id=str(uuid.uuid4()).replace("-", "")[:16],
                    name="reward",
                    start_time=now,
                    end_time=now,
                    attributes={"value": float(result)},
                    rollout_id=rollout_attempt.rollout_id,
                    attempt_id=rollout_attempt.attempt.attempt_id,
                )
                await store.add_span(span)
            elif isinstance(result, list):
                spans = [item for item in result if isinstance(item, Span)]
                if spans:
                    await store.add_many_spans(spans)

            await store.update_attempt(rollout_attempt.rollout_id, rollout_attempt.attempt.attempt_id, "succeeded")
        except Exception as e:
            logger.exception("Error during rollout %s: %s", rollout_attempt.rollout_id, e)
            await store.update_attempt(rollout_attempt.rollout_id, rollout_attempt.attempt.attempt_id, "failed")

        return True

    async def iter(self) -> None:
        self._running = True
        while self._running:
            did_work = await self.run_once()
            if not did_work:
                await asyncio.sleep(self._poll_interval)

    async def start(self):
        await self.iter()

    def stop(self):
        self._running = False
