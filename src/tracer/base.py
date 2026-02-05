import contextlib
import time
import uuid
from typing import Any, Dict, List, Optional

from src.core.types import Span


class Tracer:
    """Minimal tracer that captures spans in memory."""

    def __init__(self, rollout_id: str, attempt_id: str, trace_id: Optional[str] = None):
        self.rollout_id = rollout_id
        self.attempt_id = attempt_id
        self.trace_id = trace_id or str(uuid.uuid4()).replace("-", "")
        self._spans: List[Span] = []
        self._stack: List[str] = []

    @contextlib.contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        span_id = str(uuid.uuid4()).replace("-", "")[:16]
        parent_id = self._stack[-1] if self._stack else None
        start_time = time.time()
        attrs = attributes or {}

        self._stack.append(span_id)
        try:
            yield attrs
        finally:
            end_time = time.time()
            self._stack.pop()
            span = Span(
                trace_id=self.trace_id,
                span_id=span_id,
                parent_id=parent_id,
                name=name,
                start_time=start_time,
                end_time=end_time,
                attributes=attrs,
                rollout_id=self.rollout_id,
                attempt_id=self.attempt_id,
            )
            self._spans.append(span)

    def get_spans(self) -> List[Span]:
        return list(self._spans)
