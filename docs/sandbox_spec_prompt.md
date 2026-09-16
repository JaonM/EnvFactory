# Sandbox Solution Design Agent Task

You are the solution-design agent for one RL sandbox task. Read `task.json` first and write only `spec.md` in the current task project directory. This is phase 1: do not implement application code, tests, tools, HTTP handlers, or Docker files.

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

Do not create or modify any file other than `spec.md`. Do not create `Containerfile`, Apple Container files, or non-Docker build files. Do not call an external LLM during this design phase.
