# Sandbox Design and Implementation Agent Task

You are the Code Agent responsible for developing one RL sandbox task. Read `task.json` first. This document is the single source of truth for both phases: phase 1 designs the solution in `spec.md`; phase 2 implements the solution from `spec.md`. If another instruction conflicts with this document, follow this document and preserve the task's success semantics.

The outer workflow tells you which phase is active. In phase 1, write `spec.md` and the machine-readable `action_plan.json`; do not implement application code, tests, tools, HTTP handlers, or Docker files. In phase 2, read the existing `spec.md` and `action_plan.json` and implement the complete sandbox; do not regenerate the design unless an implementation constraint requires a clearly documented correction.

`spec.md` is the formal solution and implementation contract for phase 2. It must be detailed enough that another Code Agent can implement the sandbox without inventing business rules. Use the exact section headings below, in order. Every section must contain concrete, task-specific content; do not merely restate the input JSON.

## 1. Task understanding and scope

Restate the task, task type, complexity, uncertainty, expected interaction style, success conditions, failure conditions, and out-of-scope behavior. Explain the boundary between semantic milestones in `environment.action` and the atomic operations needed to implement them.

## 2. Requirement decomposition and implementation logic

Decompose the task into implementable requirements. For each requirement specify: preconditions, inputs, business logic, reads, writes, time cost, LLM tools, corresponding Trainer actions, success result, rejection/invalid-action result, failure behavior, and acceptance tests. Before listing tools, analyze every `environment.action` as a semantic milestone: classify it as atomic or composite, explain why, and break composite actions into observable atomic steps. For every step specify its inputs, outputs, state/persistence effects, time cost, and whether it is (a) a public LLM tool call, (b) its one corresponding Trainer action, (c) a deterministic environment-internal operation, or (d) direct `llm_generate`. Only deterministic validation, normalization, transaction, persistence, and hidden evaluation may remain environment-internal. Any step that retrieves task knowledge, queries business data, extracts information, makes an interpretation/decision, or supplies context needed by the LLM must be modeled as a public LLM tool and its one-to-one Trainer action. For example, a material summary must separately query material features and care instructions (parallel when independent), then use `llm_generate`; it must not hide both queries inside one `summarize_material_care` action. Include an explicit dependency graph that identifies serial, parallel, and mixed execution, including fan-out and join points. A high-level action may remain directly mapped to one LLM tool and one Trainer action when the analysis proves it is truly atomic; direct copying of all action names without this analysis is invalid.

## 3. Data model and persistence design

Define entities, value objects, relationships, field types, visibility (observable/hidden), defaults, and invariants. Prefer a concrete SQLite design unless the task clearly requires another local store: provide tables, columns, primary keys, foreign keys, indexes, uniqueness constraints, JSON fields, migrations, and transaction boundaries. Explain how environment state, hidden truth, provenance/evidence, Trainer actions, LLM tool calls, user messages/responses, timestamps, rewards, errors, reset, replay, idempotency, crash recovery, concurrency, and per-task isolation are persisted. Never persist API keys.

## 4. Data simulation plan

Design realistic task-specific seed data from the models above. Define distributions, quantities, edge cases, missing/ambiguous/contradictory records, hidden ground truth, deterministic seed/replay, validation, and provenance. If an external LLM is useful, define an adapter, prompt/output schema, validation, timeout, retry, fallback, and audit record. Runtime data simulation must use the externally supplied LLM configuration, never the Code Agent's credentials.

## 5. LLM user simulator and interaction scripts

Design an independent user simulator, not the Code Agent. Provide multiple deterministic or seeded scripts/personas, including cooperative, incomplete, ambiguous, contradictory, delayed, correction, refusal, and adversarial cases where relevant. Define branching conditions, turn limits, termination, exact message protocol, `ask_user` input/output, state updates, persistence, and timing. The simulator may call an external LLM through the configured adapter; the Code Agent only designs and implements this mechanism and must not act as the runtime user.

## 6. Real-time simulation model

Use real wall-clock timestamps and durations, not an abstract integer time counter. Specify simulation start time, timezone, `current_time`, action start/end timestamps, waiting/deadline behavior, ordering for concurrent actions, replay, and tests without unnecessary sleeping. State the conversion explicitly, for example `simulated_now = start_time + elapsed_simulated_seconds`, and explain whether elapsed time is accelerated or wall-clock based. Every time-consuming action must have a duration and observable effect.

## 7. LLM tools, Trainer actions, and action-chain design

Define LLM-facing tools using the standard function schema: `{"type":"function","function":{"name":"...","description":"...","parameters":{"type":"object","properties":{},"required":[]}}}`. Generate the tool list from the action decomposition, not by copying the task action list. Include an action-decomposition table or graph in this section with columns `task_action`, `atomic_step`, `step_type`, `llm_tool`, `trainer_action`, `depends_on`, `parallel_group`, `inputs`, `outputs`, and `state_effects`. In that table, explicitly mark whether each step is public or internal and justify every internal step. A task action that is proven atomic may use its own name as the tool name; a composite action must expose each required LLM-facing atomic tool separately and may end with `llm_generate` without a tool call. If a composite action has independent lookup/query steps, those steps must be separate public tools with a parallel group; a single high-level wrapper tool is not an acceptable substitute.

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

Generate `action_plan.json` before `tools.json`. It must be a JSON object with an `actions` array; each action has `task_action`, `classification` (`atomic` or `composite`), and `steps`. Each step has `kind` (`public_llm_tool`, `internal`, or `llm_generate`); a public step has a unique `tool_name`, while `llm_generate` has no tool name. For every task action, explain why each step is public or internal and record dependencies, parallel groups, inputs, outputs, and state effects. Generate `tools.json` from the public steps in `action_plan.json`. The outer build script validates only `llm_tools[*]` and its standard Function Tool schema, plus that the public tool names declared by `action_plan.json` are exactly represented in `llm_tools`. `trainer_actions`, `mappings`, and `task_action_plans` are implementation metadata: choose a coherent representation and keep them consistent with the implementation, but do not rely on an outer format validator for them. `llm_tools[*].function.parameters` must be a complete JSON Schema object, never a shorthand map of parameter names. For example: `{"type":"object","properties":{"prompt":{"type":"string"}},"required":[]}`. `ask_user` is a Trainer-to-user-simulator control interface, not an LLM tool; it must not appear in `mappings`. `llm_generate` is only a plan marker and is not a tool, Trainer action, mapping, or endpoint. Do not expose hidden state or internal validation metadata in LLM schemas.

Provide only Docker artifacts: `Dockerfile`, `docker_build.sh`, and `docker_run.sh`; do not create `Containerfile` or Apple Container files. Use `docker.m.daocloud.io/library/python:3.14-slim` by default, validate base-image reachability before building, and use a compatible mirror fallback when necessary. The service must expose at least `/health`, `/v1/reset`, `/v1/observation`, `/v1/actions`, `/v1/step`, `/v1/ask_user`, and `/v1/reward` when HTTP is appropriate for the design. Add and run comprehensive acceptance tests, and create an executable `acceptance.sh` that executes those tests and critical end-to-end/API checks, propagates failures, and is not a placeholder such as `true`, `:`, or an echo-only script. The HTTP acceptance must capture server stderr, wait for `/health` readiness with a bounded retry loop, and distinguish application startup errors from an execution environment that forbids local TCP binding. If local TCP binding fails specifically with a permission error, run equivalent in-process API/handler acceptance and report that fallback; do not treat connection refused alone as a successful fallback. Create a non-empty `IMPLEMENTATION_REPORT.md` mapping every major spec section and requirement to implementation files and test evidence. Do not create `OK`; the outer workflow creates it after implementation and validation.

Before finishing phase 2, validate `action_plan.json`, validate `llm_tools` against the standard Function Tool schema, verify that the public tools exactly match the plan, verify the implemented action-chain behavior, and run `acceptance.sh`. If any check fails, fix the implementation and rerun it; do not merely report the failure as completed. When phase 1 is active, do not create or modify files other than `spec.md` and `action_plan.json`, do not call an external LLM, and do not create Docker files. When phase 2 is active, implement all required files in the task directory and do not ask for approval or interactive input.
