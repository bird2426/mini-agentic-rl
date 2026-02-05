from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from .types import Rollout, NamedResources, Span

class LitAgent:
    """Base class for agents, following agent-lighting pattern."""
    
    def __init__(self):
        self._runner = None

    def set_runner(self, runner):
        self._runner = runner

    def rollout(self, task_input: Any, resources: NamedResources, rollout: Rollout) -> Union[float, List[Span], None]:
        """Execute a rollout. Returns a reward (float), a list of spans, or None."""
        raise NotImplementedError("Subclasses must implement rollout")

    async def rollout_async(self, task_input: Any, resources: NamedResources, rollout: Rollout) -> Union[float, List[Span], None]:
        """Execute a rollout asynchronously."""
        return self.rollout(task_input, resources, rollout)

    def is_async(self) -> bool:
        """Check if the agent is async."""
        return self.__class__.rollout_async != LitAgent.rollout_async
