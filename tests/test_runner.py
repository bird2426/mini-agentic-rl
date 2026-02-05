import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.agent import LitAgent
from src.runner.agent import LitAgentRunner
from src.store.memory import InMemoryLightningStore


class DummyAgent(LitAgent):
    def rollout(self, task_input, resources, rollout):
        return 0.5


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_reward_emits_span_and_marks_success(self):
        store = InMemoryLightningStore()
        await store.add_resources({})

        rollout = await store.enqueue_rollout("hello")
        runner = LitAgentRunner(tracer=None)
        runner.init(agent=DummyAgent())
        runner.init_worker(worker_id="worker-1", store=store)

        did_work = await runner.run_once()
        self.assertTrue(did_work)

        spans = await store.get_spans(rollout.rollout_id)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "reward")
        self.assertEqual(spans[0].attributes.get("value"), 0.5)

        updated_rollout = await store.get_rollout_by_id(rollout.rollout_id)
        self.assertIsNotNone(updated_rollout)
        assert updated_rollout is not None
        self.assertEqual(updated_rollout.status, "succeeded")


if __name__ == "__main__":
    unittest.main()
