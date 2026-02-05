from __future__ import annotations

from typing import Any, Optional, Sequence

from src.core.agent import LitAgent
from src.store.base import LightningStore


class Runner:
    """Runner interface for executing agent rollouts."""

    def init(self, agent: LitAgent, *, hooks: Optional[Sequence[Any]] = None, **kwargs: Any) -> None:
        raise NotImplementedError()

    def init_worker(self, worker_id: str, store: LightningStore, **kwargs: Any) -> None:
        raise NotImplementedError()

    async def iter(self) -> None:
        raise NotImplementedError()
