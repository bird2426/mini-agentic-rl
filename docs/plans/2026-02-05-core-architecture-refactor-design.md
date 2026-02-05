# Core Architecture Alignment Design

## Goals
- Align the core control-plane concepts with agent-lighting: Store, Runner, Trainer, and Span/Tracer.
- Keep current GRPO/HF training behavior intact while introducing extensible interfaces.
- Enable agent-no-change usage: existing agents keep returning reward/spans without any API changes.

## Non-Goals
- Reproduce agent-lighting execution strategies (client/server, multiprocessing).
- Require OpenTelemetry dependencies or external tracing backends in phase one.

## Proposed Structure
Introduce explicit interfaces that mirror agent-lighting semantics while staying minimal:
- `store/base.py`: `LightningStore` contract with rollout/attempt lifecycle, resources, and span ingestion.
- `store/memory.py`: `InMemoryLightningStore` implementing FIFO queueing and sequence-id ordering.
- `runner/base.py`: `Runner` interface with `init`, `iter`, and `step` lifecycle.
- `runner/agent.py`: `LitAgentRunner` that polls the store, executes agent rollouts, emits reward spans, and updates attempt status.
- `tracer/base.py`: lightweight tracer + context manager for span creation; no external deps.
- `trainer/trainer.py`: minimal orchestrator to wire Algorithm/Runner/Store together.

## Data Model Changes
- `Span` gains `sequence_id` and standardized core fields (trace_id/span_id/parent_id/name/time/attributes).
- `Rollout` gains `resources_id` to bind execution to a resource snapshot.
- Store allocates monotonically increasing `sequence_id` per attempt for ordering.

## Execution Flow
1) Algorithm enqueues rollouts and registers resources in the store.
2) Runner dequeues an `AttemptedRollout`, fetches latest resources, and executes agent rollout.
3) Runner converts rewards to spans and persists spans via the store.
4) Store updates attempt/rollout statuses and provides ordered spans for training.

## Testing Strategy
- Unit tests for store lifecycle transitions and sequence-id allocation.
- Runner test for reward-to-span emission and attempt status updates.
- Tracer test for basic span creation and context management.
