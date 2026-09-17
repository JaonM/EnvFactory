# Sandbox Design and Implementation Agent Task

You are the Code Agent responsible for developing one RL sandbox task. Read `task.json` first. This document is the single source of truth for both phases: phase 1 designs the solution in `spec.md`; phase 2 implements the solution from `spec.md`. If another instruction conflicts with this document, follow this document and preserve the task's success semantics.

The outer workflow tells you which phase is active. In phase 1, write only `spec.md`; do not implement application code, tests, tools, HTTP handlers, or Docker files. In phase 2, read the existing `spec.md` and implement the complete sandbox; do not regenerate the design unless an implementation constraint requires a clearly documented correction.

`spec.md` is the formal solution and implementation contract for phase 2. It must be detailed enough that another Code Agent can implement the sandbox without inventing business rules. Use the exact section headings below, in order. Every section must contain concrete, task-specific content; do not merely restate the input JSON.

## 1. Task understanding and scope

Restate the task, task type, complexity, uncertainty, expected interaction style, success conditions, failure conditions, and out-of-scope behavior. Explain the boundary between semantic milestones in `environment.action` and the atomic operations needed to implement them.

## 2. Requirement decomposition and implementation logic

Decompose the task into implementable requirements. For each requirement specify: preconditions, inputs, business logic, reads, writes, time cost, LLM tools, corresponding Trainer actions, success result, rejection/invalid-action result, failure behavior, and acceptance tests. Include a dependency graph that explicitly identifies serial, parallel, and mixed execution; do not assume one high-level action equals one tool.

## 3. Data model and persistence design

Define entities, value objects, relationships, field types, visibility (observable/hidden), defaults, and invariants. Prefer a concrete SQLite design unless the task clearly requires another local store: provide tables, columns, primary keys, foreign keys, indexes, uniqueness constraints, JSON fields, migrations, and transaction boundaries. Explain how environment state, hidden truth, provenance/evidence, Trainer actions, LLM tool calls, user messages/responses, timestamps, rewards, errors, reset, replay, idempotency, crash recovery, concurrency, and per-task isolation are persisted. Never persist API keys.

## 4. Data simulation plan

Design realistic task-specific seed data from the models above. Define distributions, quantities, edge cases, missing/ambiguous/contradictory records, hidden ground truth, deterministic seed/replay, validation, and provenance. If an external LLM is useful, define an adapter, prompt/output schema, validation, timeout, retry, fallback, and audit record. Runtime data simulation must use the externally supplied LLM configuration, never the Code Agent's credentials.

## 5. LLM user simulator and interaction scripts

Design an independent user simulator, not the Code Agent. Provide multiple deterministic or seeded scripts/personas, including cooperative, incomplete, ambiguous, contradictory, delayed, correction, refusal, and adversarial cases where relevant. Define branching conditions, turn limits, termination, exact message protocol, `ask_user` input/output, state updates, persistence, and timing. The simulator may call an external LLM through the configured adapter; the Code Agent only designs and implements this mechanism and must not act as the runtime user.

## 6. Real-time simulation model

Use real wall-clock timestamps and durations, not an abstract integer time counter. Specify simulation start time, timezone, `current_time`, action start/end timestamps, waiting/deadline behavior, ordering for concurrent actions, replay, and tests without unnecessary sleeping. State the conversion explicitly, for example `simulated_now = start_time + elapsed_simulated_seconds`, and explain whether elapsed time is accelerated or wall-clock based. Every time-consuming action must have a duration and observable effect.

## 7. LLM tools, Trainer actions, and action-chain design

Define LLM-facing tools using the standard function schema: `{"type":"function","function":{"name":"...","description":"...","parameters":{"type":"object","properties":{},"required":[]}}}`.

Tools must expose only inputs the LLM can know or provide; do not put hidden truth, predicted answers as required inputs, or internal `role=` annotations into the public schema. Define Trainer-facing atomic actions separately, with exact parameters, preconditions, effects, time cost, observation changes, persistence writes, and errors. Each LLM tool maps to exactly one Trainer action, and each tool call causes the Trainer to execute that corresponding action. Represent high-level action plans as serial/parallel/mixed chains. `llm_generate` is only an internal plan marker meaning the LLM can reply from context; it is not an LLM tool, Trainer action, mapping, or endpoint.

## 8. External LLM configuration and Docker injection

Identify every runtime feature requiring an external model (user simulation, data simulation, optional evaluation). Define provider/base URL/model/timeout/retry/configuration and the exact environment variables or Docker `-e`/`--env-file` startup arguments. Keys must not be baked into images, committed, returned in observations, or written to logs/database. Define behavior when credentials are absent, including deterministic local fallback or an explicit disabled mode. Make clear that Code Agent credentials are never used by the running sandbox.

## 9. Reward function and Trainer-facing evaluation

Translate every metric into executable reward logic. Define step, terminal, and trajectory scope; formula, weights, positive rewards, penalties, once-only rules, clipping/normalization, invalid actions, hidden-state evaluation, terminal success/failure, and deterministic replay. Specify the reward response shape and how evaluation avoids leaking hidden truth to the Agent.

## 10. Trainer API and environment state transitions

Design all Trainer-facing interfaces: health, reset, observation, tool/action discovery, one action execution, user turn, reward/status, replay/export, and shutdown if needed. For each give method/path or protocol, request, response, status/error codes, authentication boundary, idempotency, concurrency, and persistence behavior. Define the state transition equations explicitly, at minimum:

`S_(t+1) = F(S_t, A_t, P_t)`

`T_(t+1) = G(T_t, A_t)`

`O_(t+1) = H(S_(t+1))`

`R_t = Q(S_t, A_t, S_(t+1))`

`D_t = terminal(S_(t+1))`.

Expand these symbols into this task's observable state, hidden state, persistent records, action effects, time effects, and termination rules. Explain exactly how every Trainer action can change environment variables and persisted data.

## 11. Acceptance and verification plan

Define an executable acceptance plan covering schema validation, unit tests, state-transition tests, invalid/negative actions, hidden-state leakage, user scripts, external-LLM mocks, real-time timestamps/durations, concurrency/isolation, reward calculations, API contract, reset/replay determinism, Docker build/startup, health checks, and an end-to-end trajectory. Specify expected evidence and the condition for creating the final `OK` file.

## Design decisions and unresolved assumptions

List task-specific decisions, alternatives rejected, assumptions, risks, and questions that phase 2 must resolve without changing the task's success semantics.

## Phase 2 implementation requirements

When phase 2 is active, implement the design as executable, task-specific code rather than a generic wrapper or a minimal demo. Treat every requirement, transition equation, data table, action chain, API, reward rule, time rule, user script, and acceptance item in `spec.md` as an implementation obligation. Do not omit, replace, or simplify a requirement merely because a smaller implementation passes a smoke test. Use SQLite or the persistence design selected in `spec.md`; keep runtime data under `data/`; implement deterministic reset and replay; encode preconditions, state transitions, hidden truth, user scripts, rewards, invalid-action behavior, and task termination in code and tests. Implement `ask_user` when interaction is required and keep runtime user/data simulation behind an explicit external-LLM adapter with deterministic fallback. The Code Agent must not use its own credentials as runtime simulation credentials.

Generate `tools.json` with standard LLM function tools, atomic Trainer actions, one-to-one mappings, and complete `task_action_plans`. Both `llm_tools[*].function.parameters` and `trainer_actions[*].parameters` must be complete JSON Schema objects, never a shorthand map of parameter names. For example: `{"type":"object","properties":{"prompt":{"type":"string"}},"required":[]}`. Every Trainer Action object must contain all of these fields: `name`, `action_type`, `description`, `parameters`, `transport`, `request`, `returns`, and `visibility`; the action name must be in `name`, never in a field named `type`. A valid Trainer Action looks like: `{"name":"request_clothing_info","action_type":"atomic","description":"请求衣物信息","parameters":{"type":"object","properties":{"prompt":{"type":"string"}},"required":[]},"transport":"HTTP POST /v1/step","request":{"type":"request_clothing_info"},"returns":"observation and reward","visibility":"trainer"}`. Each `task_action_plans` entry must be an object with `task_action` and a non-empty `steps` array; each step must be a tool pair or the exact `{"type":"llm_generate"}` marker. Do not use a string array such as `{"default":["request_clothing_info"]}`. `ask_user` is a Trainer-to-user-simulator control interface, not an LLM tool; it must not appear in `mappings` or `task_action_plans`. `llm_generate` is only a plan marker and is not a tool, Trainer action, mapping, or endpoint. Do not expose hidden state or internal validation metadata in LLM schemas.

Provide only Docker artifacts: `Dockerfile`, `docker_build.sh`, and `docker_run.sh`; do not create `Containerfile` or Apple Container files. Use `docker.m.daocloud.io/library/python:3.14-slim` by default, validate base-image reachability before building, and use a compatible mirror fallback when necessary. The service must expose at least `/health`, `/v1/reset`, `/v1/observation`, `/v1/actions`, `/v1/step`, `/v1/ask_user`, and `/v1/reward` when HTTP is appropriate for the design. Add and run comprehensive acceptance tests, and create an executable `acceptance.sh` that executes those tests and critical end-to-end/API checks, propagates failures, and is not a placeholder such as `true`, `:`, or an echo-only script. Create a non-empty `IMPLEMENTATION_REPORT.md` mapping every major spec section and requirement to implementation files and test evidence. Do not create `OK`; the outer workflow creates it after implementation and validation.

Before finishing phase 2, validate `tools.json` against the exact object shapes above, verify every required task action has a plan, verify every LLM tool has exactly one mapping, and run `acceptance.sh`. If any check fails, fix the implementation and rerun it; do not merely report the failure as completed. When phase 1 is active, do not create or modify any file other than `spec.md`, do not call an external LLM, and do not create Docker files. When phase 2 is active, implement all required files in the task directory and do not ask for approval or interactive input.
