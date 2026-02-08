from typing import List, Dict, Any, Callable

from .base import Adapter
from ..types import Triplet


class TraceAdapter(Adapter):
    def __init__(
        self,
        reward_fn: Callable[[str, str, bool], float],
        require_tool_call: bool = False,
    ):
        self.reward_fn = reward_fn
        self.require_tool_call = require_tool_call

    def adapt(self, spans: List[Dict[str, object]]) -> List[Triplet]:
        triplets: List[Triplet] = []
        for span in spans:
            payload = span.get("payload", {})
            if not isinstance(payload, dict):
                continue

            # Filter for successful rollouts
            if "response" not in payload or "ground_truth" not in payload:
                continue

            response = str(payload["response"])
            ground_truth = str(payload["ground_truth"])

            # Use the injected reward function
            reward = self.reward_fn(
                response,
                ground_truth,
                self.require_tool_call,
            )

            triplets.append(
                Triplet(
                    prompt=str(payload.get("prompt", "")),
                    response=response,
                    reward=reward,
                    token_ids=list(payload.get("tokens", [])),
                    loss_mask=list(payload.get("loss_mask", [])),
                    group_id=str(payload.get("group_id", "default")),
                )
            )
        return triplets
