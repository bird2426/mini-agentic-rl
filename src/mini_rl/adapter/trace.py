from typing import List, Dict, Any

from .base import Adapter
from ..types import Triplet
from ..reward import score_gsm8k_response


class TraceAdapter(Adapter):
    def __init__(self, require_tool_call: bool = False):
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

            reward = score_gsm8k_response(
                response=response,
                ground_truth=ground_truth,
                require_tool_call=self.require_tool_call,
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
