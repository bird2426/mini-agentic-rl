import time
from typing import Dict, Any
from .base import Runner, T_Agent
from ..types import Span


class LocalRunner(Runner[T_Agent]):
    def run_step(self) -> bool:
        item = self.store.dequeue_rollout()
        if item is None:
            return False

        rollout, attempt = item
        self.store.update_attempt(rollout.rollout_id, attempt.attempt_id, "running")

        try:
            # Resolve resources if needed (not used in mini-rl v1, but good for structure)
            # resources = self.store.get_resources_by_id(rollout.resources_id)

            # Execute agent logic
            result = self.agent.rollout(rollout, attempt)

            # Record success span
            seq = self.store.next_span_sequence_id(
                rollout.rollout_id, attempt.attempt_id
            )
            self.store.add_span(
                Span(
                    rollout_id=rollout.rollout_id,
                    attempt_id=attempt.attempt_id,
                    sequence_id=seq,
                    name="agent.rollout",
                    payload=result,
                    timestamp=time.time(),
                )
            )
            self.store.update_attempt(
                rollout.rollout_id, attempt.attempt_id, "succeeded"
            )

        except Exception as exc:
            # Record failure span
            seq = self.store.next_span_sequence_id(
                rollout.rollout_id, attempt.attempt_id
            )
            self.store.add_span(
                Span(
                    rollout_id=rollout.rollout_id,
                    attempt_id=attempt.attempt_id,
                    sequence_id=seq,
                    name="agent.error",
                    payload={"error": str(exc)},
                    timestamp=time.time(),
                )
            )
            self.store.update_attempt(rollout.rollout_id, attempt.attempt_id, "failed")

        return True

    def run_until_empty(self) -> None:
        while self.run_step():
            pass
