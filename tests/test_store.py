import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.types import Span
from src.store.memory import InMemoryLightningStore


class StoreLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_span_sets_sequence_and_running(self):
        store = InMemoryLightningStore()
        rollout = await store.enqueue_rollout("hi")
        attempted = await store.dequeue_rollout(worker_id="worker-1")
        self.assertIsNotNone(attempted)
        assert attempted is not None

        span = Span(
            trace_id="trace-1",
            span_id="span-1",
            name="generation",
            start_time=1.0,
            end_time=2.0,
            attributes={},
            rollout_id=attempted.rollout_id,
            attempt_id=attempted.attempt.attempt_id,
        )

        await store.add_span(span)
        spans = await store.get_spans(rollout.rollout_id)
        self.assertEqual(spans[0].sequence_id, 1)

        updated_rollout = await store.get_rollout_by_id(rollout.rollout_id)
        latest_attempt = await store.get_latest_attempt(rollout.rollout_id)
        self.assertIsNotNone(updated_rollout)
        self.assertIsNotNone(latest_attempt)
        assert updated_rollout is not None
        assert latest_attempt is not None
        self.assertEqual(updated_rollout.status, "running")
        self.assertEqual(latest_attempt.status, "running")


if __name__ == "__main__":
    unittest.main()
