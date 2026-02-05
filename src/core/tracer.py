import time
import uuid
import contextlib
from typing import Optional, Dict, Any, List
from .types import Span

class Tracer:
    """Minimal tracer inspired by agent-lighting's OTEL integration."""
    
    def __init__(self, rollout_id: str, attempt_id: str):
        self.rollout_id = rollout_id
        self.attempt_id = attempt_id
        self.trace_id = str(uuid.uuid4()).replace("-", "")
        self._spans: List[Span] = []

    @contextlib.contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        span_id = str(uuid.uuid4()).replace("-", "")[:16]
        start_time = time.time()
        attrs = attributes or {}
        
        try:
            yield attrs
        finally:
            end_time = time.time()
            span = Span(
                trace_id=self.trace_id,
                span_id=span_id,
                name=name,
                start_time=start_time,
                end_time=end_time,
                attributes=attrs,
                rollout_id=self.rollout_id,
                attempt_id=self.attempt_id
            )
            self._spans.append(span)

    def get_spans(self) -> List[Span]:
        return self._spans
