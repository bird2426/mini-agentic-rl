from typing import List, Optional
from .types import Span, Triplet

class TraceToTripletAdapter:
    def adapt(self, spans: List[Span]) -> List[Triplet]:
        triplets = []
        reward = 0.0
        
        # First, extract common reward for all triplets in this trace
        for span in spans:
            if span.name == "reward":
                reward = span.attributes.get("value", 0.0)
        
        # Convert generation spans to Triplets
        for span in spans:
            if span.name == "generation":
                triplet = Triplet(
                    prompt=span.attributes.get("messages", []),
                    response=span.attributes.get("output", ""),
                    reward=reward
                )
                triplets.append(triplet)
        
        return triplets
