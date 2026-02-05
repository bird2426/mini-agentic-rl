import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tracer.base import Tracer


class TracerTests(unittest.TestCase):
    def test_nested_spans_record_parent(self):
        tracer = Tracer(rollout_id="rollout-1", attempt_id="attempt-1", trace_id="trace-1")

        with tracer.span("root"):
            with tracer.span("child"):
                pass

        spans = tracer.get_spans()
        self.assertEqual(len(spans), 2)

        root = next(span for span in spans if span.parent_id is None)
        child = next(span for span in spans if span.parent_id == root.span_id)
        self.assertIsNone(root.parent_id)
        self.assertEqual(child.parent_id, root.span_id)


if __name__ == "__main__":
    unittest.main()
