from __future__ import annotations
import asyncio
import logging
import time
import uuid
from typing import Optional
from src.core.agent import LitAgent
from src.store.memory import InMemoryStore
from src.core.types import AttemptedRollout, Span

logger = logging.getLogger(__name__)

class AgentRunner:
    """Worker that executes agent rollouts by polling the store."""
    
    def __init__(self, runner_id: str, store: InMemoryStore, agent: LitAgent):
        self.runner_id = runner_id
        self.store = store
        self.agent = agent
        self.agent.set_runner(self)
        self._running = False

    async def run_once(self) -> bool:
        rollout_attempt = await self.store.dequeue_rollout(self.runner_id)
        if not rollout_attempt:
            return False

        resources = await self.store.get_latest_resources()
        if not resources:
            await self.store.update_attempt_status(rollout_attempt.rollout_id, rollout_attempt.attempt.attempt_id, "failed")
            return True

        try:
            if self.agent.is_async():
                result = await self.agent.rollout_async(rollout_attempt.input, resources, rollout_attempt)
            else:
                result = self.agent.rollout(rollout_attempt.input, resources, rollout_attempt)

            if isinstance(result, float) or isinstance(result, int):
                span = Span(
                    trace_id=str(uuid.uuid4()).replace("-", ""),
                    span_id=str(uuid.uuid4()).replace("-", "")[:16],
                    name="reward",
                    start_time=time.time(),
                    end_time=time.time(),
                    attributes={"value": float(result)},
                    rollout_id=rollout_attempt.rollout_id,
                    attempt_id=rollout_attempt.attempt.attempt_id
                )
                await self.store.add_span(span)
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, Span):
                        await self.store.add_span(item)

            await self.store.update_attempt_status(rollout_attempt.rollout_id, rollout_attempt.attempt.attempt_id, "succeeded")
        except Exception as e:
            logger.exception(f"Error during rollout {rollout_attempt.rollout_id}: {e}")
            await self.store.update_attempt_status(rollout_attempt.rollout_id, rollout_attempt.attempt.attempt_id, "failed")
        
        return True

    async def start(self):
        """Start the runner loop."""
        self._running = True
        while self._running:
            did_work = await self.run_once()
            if not did_work:
                await asyncio.sleep(0.1)

    def stop(self):
        """Stop the runner loop."""
        self._running = False
