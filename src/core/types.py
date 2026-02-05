from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
import time

# Core Status Types
RolloutStatus = Literal["queuing", "preparing", "running", "failed", "succeeded", "cancelled"]
AttemptStatus = Literal["preparing", "running", "failed", "succeeded", "timeout", "cancelled"]
RolloutMode = Literal["train", "val", "test"]

class Span(BaseModel):
    """Strictly aligned with OpenTelemetry span structure in agent-lighting."""
    trace_id: str
    span_id: str
    parent_id: Optional[str] = None
    name: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)

    # Metadata for store coordination
    rollout_id: str
    attempt_id: str
    sequence_id: Optional[int] = None

class Triplet(BaseModel):
    """The standard training unit in agent-lighting."""
    prompt: Any
    response: Any
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RolloutConfig(BaseModel):
    max_attempts: int = 1
    timeout_seconds: Optional[float] = None

class Attempt(BaseModel):
    rollout_id: str
    attempt_id: str
    sequence_id: int
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    status: AttemptStatus = "preparing"
    worker_id: Optional[str] = None

class Rollout(BaseModel):
    rollout_id: str
    input: Any
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    mode: RolloutMode = "train"
    status: RolloutStatus = "queuing"
    config: RolloutConfig = Field(default_factory=RolloutConfig)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    resources_id: Optional[str] = None

class AttemptedRollout(Rollout):
    attempt: Attempt

class ResourcesUpdate(BaseModel):
    """Immutable snapshot for models, tokenizers, etc."""
    resources: Dict[str, Any]
    resources_id: str
    timestamp: float = Field(default_factory=time.time)


class NamedResources(ResourcesUpdate):
    """Alias for backward compatibility."""
    pass
