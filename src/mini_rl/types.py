from dataclasses import dataclass, field
from typing import Any, Literal, Dict, List

RolloutStatus = Literal["queued", "preparing", "running", "succeeded", "failed"]
AttemptStatus = Literal["preparing", "running", "succeeded", "failed"]


@dataclass
class Attempt:
    attempt_id: str
    rollout_id: str
    sequence_id: int
    status: AttemptStatus
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rollout:
    rollout_id: str
    input: Dict[str, Any]
    mode: Literal["train", "eval"]
    resources_id: str
    status: RolloutStatus
    attempts: List[Attempt] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    rollout_id: str
    attempt_id: str
    sequence_id: int
    name: str
    payload: Dict[str, Any]
    timestamp: float = 0.0


@dataclass
class ResourceSnapshot:
    resources_id: str
    resources: Dict[str, Any]
    created_at: float = 0.0


@dataclass
class Trajectory:
    """A processed sequence of spans ready for training."""

    rollout_id: str
    attempt_id: str
    steps: List[Dict[str, Any]]
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmMetrics:
    loss: float
    avg_reward: float
    exact_match: float
    trajectories: int


@dataclass
class Triplet:
    prompt: str
    response: str
    reward: float
    token_ids: List[int]
    loss_mask: List[int]
    group_id: str
