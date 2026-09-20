# RL Sandbox Design and Implementation Contract

You are the Code Agent developing one RL sandbox task. Read `task.json` first. This document is the sole contract for the staged workflow:

- Phase 1 writes only `spec.md`.
- Phase 2 reads `spec.md` and implements the complete sandbox in one Code Agent run, including tools, Trainer actions, tests, acceptance, and Docker artifacts.

There is no action-plan phase, development-topology phase, or per-node Code Agent phase. `spec.md` is the single design handoff between the two phases and must contain the complete tool/action semantics needed for implementation.

The specification must be task-specific and executable. Do not merely restate the input or build a generic demo. Preserve the task's success and failure semantics.

The outer workflow may provide a read-only `BUILD_CONTRACT.json`. It is an
authoritative machine-readable contract generated from the task input, not a
Code Agent proposal. Every required obligation in that file has a stable
`id`; `spec.md` must preserve and reference those IDs. The outer workflow
validates coverage independently.

## Global action and tool rules

1. Analyze every `environment.action` as a semantic milestone before designing tools. `classification` has exactly two allowed values:

   - `atomic`: from the Agent's perspective, the action has one observable decision/input boundary and one corresponding environment operation. The sandbox may still perform multiple hidden implementation steps—validation, normalization, database transactions, media preprocessing, OCR, model calls, persistence, and hidden-truth evaluation—but those steps remain inside the one Trainer action and do not make the classification `composite`.
   - `composite`: from the Agent's perspective, the action requires two or more independently observable operations or decisions, whose outputs are consumed by later operations. It must be decomposed into multiple `public_llm_tool` and/or terminal `llm_generate` steps, with dependencies, serial/parallel groups, inputs, outputs, state effects, persistence effects, time cost, and rationale. Independent lookups should be separate steps; fan-out and join points must be explicit.

   Use this decision test: if the Agent supplies one request/evidence boundary and the sandbox returns the result of that operation, use `atomic`; if the Agent must choose or invoke multiple separate operations, or one operation's public output is required as another operation's input, use `composite`. Do not classify an action based only on its internal implementation complexity, external-model usage, synchronous/asynchronous behavior, or whether it eventually produces text. Never invent values such as `composite_internal_execution`, `direct_terminal_generation`, or `internal`; internal processing is described in the action's implementation fields, not in `classification`.

   Generic examples: `request_user_evidence` is usually `atomic` because it is one interaction request; `recognize_from_submitted_evidence` may be `atomic` when one evidence-to-result tool call is the Agent boundary even though the sandbox performs OCR and model inference internally; `prepare_report` is `composite` when it requires separate data lookups followed by a final response, for example `lookup_facts` and `lookup_constraints` in parallel, then `llm_generate` as the terminal step. These are decision examples, not fixed tool names for every task.
2. The `spec.md` action-decomposition section is authoritative. It must describe each task action's Agent-visible atomic operations, tool name, OpenAI Function parameters, corresponding Trainer action, dependencies, inputs, outputs, effects, and terminal direct-generation steps. There is no separate `action_plan.json` artifact. Each public tool has one canonical `tool_name`; the shared `ask_user` bridge may serve multiple interaction actions, while unrelated tools must not be duplicated. `llm_generate` is an internal design step with no tool name and is never emitted as an LLM tool or Trainer action.
3. Every atomic operation the Agent must choose or perform is a public LLM Tool and has exactly one corresponding Trainer action. Every LLM tool call causes the Trainer to execute that action. Validation, normalization, transactions, persistence, and hidden evaluation are implementation details inside the corresponding public action, not separate plan steps.
4. Any operation that retrieves knowledge or business data, extracts information, interprets evidence, makes a decision, or supplies context needed by the LLM must be a public LLM Tool. Independent lookups must be separate tools and may share a parallel group; do not hide them in a high-level wrapper.
5. `ask_user` is both an LLM Tool and a Trainer action, but it is not the user simulator. It is the single public bridge for all user information, clarification, confirmation, correction, and other user interaction. The LLM decides when to call it and passes the interaction request as a JSON object. Each successful `ask_user` call must trigger exactly one User Simulator turn: the Trainer validates and persists the request, delivers it to the independent LLM user simulator, persists the generated user message, and writes that message into the next observation/context returned to the Agent. `reset`, timers, observation reads, and other Trainer actions must never trigger the User Simulator automatically. `ask_user` must appear in `llm_tools`, `trainer_actions`, and `mappings`.
6. `llm_generate` means the LLM replies directly from the current context. It is not an LLM Tool, Trainer action, mapping, or endpoint.
7. LLM tools use the standard Function Tool shape:

   `{"type":"function","function":{"name":"...","description":"...","parameters":{"type":"object","properties":{},"required":[]}}}`

   `function.description` and every declared parameter's `description` must be non-empty natural-language strings, including nested object properties and object items inside arrays. Expose only values the LLM can know or provide. Never expose hidden truth, credentials, hidden-state fields, predicted answers as required inputs, or internal `role=` annotations.

   ID provenance is strict: an ID-valued LLM-tool parameter may only be
   copied from a prior public tool result or the current redacted observation
   that explicitly exposes that public reference. It must never be inferred
   from or copied out of a User Simulator message. User Simulator output is
   natural-language user content plus registered attachments, not an ID
   namespace.

   For confirmation actions, a user's confirmation is an environment event
   produced by `ask_user` and persisted by the sandbox. Do not expose
   internal acceptance record IDs, message IDs, evidence IDs, database keys,
   or fields such as `acceptance_evidence_id` / `acceptance_message_id` as
   required LLM-tool inputs. The confirmation action must resolve and validate
   the latest valid confirmation for the pending proposal internally. The
   Agent may provide a public proposal or candidate reference when needed to
   select the pending operation, but those references must come from tool
   results/observations. It must not submit an ID proving that the user
   accepted.

8. If a task contains media, generate it programmatically as task data. Do not call OCR or a multimodal model, and do not create a separate media-reference or media-content evaluation. The Agent is evaluated only on producing a valid tool name and parameters, correct process transitions, and task-goal completion. Never replace a required tool parameter with hidden truth or hard-coded task answers.

9. Evidence-recognition tools receive only observable evidence references/content and selectors such as `entity_id` or `item_id`. They must produce the interpretation, extraction, classification, verification result, confidence, or uncertainty. Never require `claim`, `answer`, `predicted_*`, `*_prediction`, `material`, or other answer fields as required inputs to an action whose purpose is to recognize or infer that answer. A claim may appear only as a sandbox result or as an explicitly separate user-provided claim being verified.

10. If a task requires independent knowledge or business-data lookups before a final response, model each Agent-selectable lookup as its own `public_llm_tool`, with separate outputs and dependencies. `llm_generate` is the final direct response step; it has no tool or Trainer action and must be the last step of its action chain. Do not add a high-level `submit_summary`, `summarize_*`, or equivalent public tool after `llm_generate` merely to submit the generated text. A submission tool is allowed only when the original task explicitly requires a separate externally observable submission operation.

## 1. Task understanding and scope

In `spec.md`, describe the task, type, complexity, uncertainty, expected interaction style, success/failure conditions, out-of-scope behavior, and the boundary between semantic task milestones and atomic implementation operations.
For every required `BUILD_CONTRACT.json` obligation, include its literal ID
and explain the design that satisfies it. Do not remove, weaken, or redefine
an outer obligation.

## 2. Requirement decomposition and implementation logic

For every requirement and task action, specify preconditions, public inputs, business logic, reads, writes, time cost, LLM tools, corresponding Trainer actions, success result, invalid/rejected result, failure behavior, and acceptance tests. Do not describe task-environment state variables in the environment-generation stage; derive executable state transitions later from the complete business records and action matrix. For media-based actions, specify only the generated-code execution/persistence boundary, tool-parameter handling, process transitions, and task-goal evaluation; do not add media recognition or separate media-reference evaluation. These implementation steps must remain inside the corresponding Trainer Action execution. Include the complete action-decomposition table or graph with:

`task_action`, `atomic_step`, `step_type`, `llm_tool`, `trainer_action`, `depends_on`, `parallel_group`, `inputs`, `outputs`, `state_effects`.

Each action and step must also include `obligation_ids` for the contract
obligations it implements. The union of these IDs must cover every required
obligation. The Code Agent must implement this design directly; it must not
invent a second plan format or silently omit a tool because it is inconvenient.

Show fan-out, joins, and mixed serial/parallel execution. A task action may map directly to one tool only when the analysis proves it is atomic; copying all task action names without decomposition is invalid.

## 3. Data model and persistence

Define entities, relationships, types, visibility, defaults, and invariants. Prefer SQLite unless the task requires another store. Specify tables, keys, foreign keys, indexes, uniqueness, JSON fields, migrations, transaction boundaries, per-task isolation, crash recovery, concurrency, reset, replay, idempotency, timestamps, state, hidden truth, evidence, actions, tool calls, user messages, rewards, and errors. Before the Agent can act, the sandbox must generate or load an authoritative ground-truth record from the task data model and deterministic seed, assign it a version/hash, and keep it immutable for that episode. Persist the user model, generated behavior script, `user_known_facts`, `user_beliefs`, conversation memory, and user-state transitions with provenance and visibility flags; keep ground truth separate from user knowledge, Agent claims, observations, and model outputs, and never expose it through observation. Never persist API keys.

## 4. Data simulation

Read `artifacts.data_manifest` and the files it references: `data_document.md`, one schema JSON per table, and one rows JSONL file per table. The manifest is the persistence handoff for the Code Agent and keeps `task.json` compact. The document describes each business entity/table, atomic fields, types, visibility, primary/foreign keys, constraints, initialization order, relationships, and persistence requirements. Initialize the database from every non-empty rows JSONL file; do not replace these files with a state/data-model proposal. Tables may be generated and loaded independently, but cross-table foreign-key and business-consistency checks must run before the sandbox is considered ready. If the task requires media, read `artifacts.media_generation`, install only its declared dependencies, run its declared Python entrypoint during sandbox build/startup as specified, and persist only the task-relevant generated media files. Do not add binding tables, truth references, media-evidence mappings, OCR, or multimodal recognition solely because media exists. The Agent is evaluated through tool schema, process transitions, and task-goal completion. If an external LLM is used for user simulation or data proposal, define its adapter, prompt/output schema, timeout, retry, fallback, and configuration. Runtime simulation uses only externally supplied model credentials, never Code Agent credentials.

## 5. LLM user simulator

Design an independent, stateful user simulator, separate from `ask_user` and from the Code Agent. Read `artifacts.user_simulation_manifest` and its separate profile/script/session files when available; do not expect these large artifacts to be embedded in `task.json`. Initialize a user model containing persona, goals, constraints, knowledge level, `user_known_facts`, uncertain `user_beliefs`, memory, patience, trust, urgency, and willingness to share. `user_known_facts` are user-visible facts selected by the sandbox from simulated task data; `user_beliefs` are derived/noisy user interpretations and may be wrong. The Agent cannot write either domain. First write a script-generation prompt that describes the task and asks an external OpenAI-compatible LLM to return a validated JSON behavior script containing persona, goals, constraints, branching rules, and turn behaviors; the script constrains behavior and does not replace the state model.

At runtime, use this exact pipeline: only a successful Agent `ask_user` call starts one simulator turn; the Trainer persists the JSON request; a behavior planner selects one response mode using the user model, conversation memory, request quality, and seeded randomness; a fact/attachment selector chooses only facts and registered attachment IDs allowed by `user_known_facts`/`user_beliefs`; an external LLM renders one natural-language response and decides whether the user behavior includes an attachment; a validator checks facts, attachment ownership, `entity_id` binding, safety, persona, and output schema; then the Trainer persists the response and attachment references, updates user state, and includes the response plus attachments in the next observation/context. The simulator output schema must include `message`, `attachments` (an array of registered attachment objects with `attachment_id`, `entity_id`, and `kind`), and `done`; use an empty array when the user does not provide media. The simulator must emit exactly one user message per `ask_user` call. No simulator turn may occur without `ask_user`, and reset, timers, observation reads, or other actions must not invoke it. `ask_user` must not fabricate the user message or attachment. On external-LLM failure, use the validated behavior plan and deterministic seeded rendering fallback, including a pre-registered attachment when the fallback behavior requires media. The simulator must not read hidden truth, credentials, or evaluation data, and must not act as the Code Agent.

## 6. Real-time simulation

Use real wall-clock timestamps and durations, not an abstract integer clock. Define start time, timezone, `current_time`, action start/end, waiting/deadlines, concurrent ordering, replay, and accelerated versus wall-clock behavior. State the time equation, for example `simulated_now = start_time + elapsed_simulated_seconds`. Every time-consuming action has a duration and observable effect; tests must avoid unnecessary sleeping.

## 7. Tools, Trainer actions, and action chain

The spec must contain the complete Agent-visible tool/action design before
implementation. For each public tool and Trainer action, specify the exact
OpenAI Function schema, preconditions, effects, time cost, observation
changes, persistence writes, return structure, and errors. The Code Agent
must generate the root `tools.json` from this section and keep its
`llm_tools`, `trainer_actions`, and one-to-one mappings consistent with the
implementation. Do not create `action_plan.json` or
`development_plan.json`. The Trainer only submits the action request and
receives the action result, observation, and reward; the sandbox performs all
action-internal media processing and evaluation.

## 8. External LLM configuration and Docker injection

Identify runtime features requiring an external model, including user-script generation, user-message simulation, data simulation, and optional trajectory-quality evaluation. Media handling must remain deterministic and programmatic; no media-recognition model or media API key is part of the sandbox runtime. The sandbox action executor, not the RL Trainer, owns any runtime model calls. Use the common OpenAI-compatible Chat Completions contract (`POST {base_url}/chat/completions`, Bearer API key, JSON `messages`) and define the script-generation and response-generation prompts, output schemas, provider/base URL/model, timeout, retries, environment variables (`SANDBOX_LLM_API_KEY`, `SANDBOX_LLM_BASE_URL`, `SANDBOX_LLM_MODEL`, `SANDBOX_LLM_TIMEOUT_SECONDS`, `SANDBOX_LLM_MAX_RETRIES`), Docker `-e`/`--env-file` usage, missing-key behavior, deterministic fallback or disabled mode. Never bake keys into images, observations, logs, or databases; Code Agent credentials must never be runtime credentials.

## 9. Reward and evaluation

Translate every metric into executable step, terminal, and trajectory logic. Keep only process metrics strongly related to task completion, plus outcome metrics for goal completion or measurable business-data changes. If a task has no key tool actions or can be completed by direct generation, process metrics may be empty; if it has multiple key actions, do not impose an artificial count limit. Observation metrics must not depend on task-generation artifacts such as `user_profiles`, `user_scripts`, or `dialogue_sessions`; `evaluation_inputs` may reference only runtime inputs such as the actual `conversation`, `public_observation`, `available_tools`, `tool_call`, `tool_results`, `business_data`, `final_document`, and `terminal_observation`. Each key process metric is `hybrid` and uses the compact contract `target_action`, `evaluation_inputs`, `criteria`, and `condition=llm_expected_tool_call_exact_match`: the sandbox evaluator calls the externally supplied LLM to generate the expected tool name and arguments from the current runtime context, then performs a deterministic canonical comparison with the Agent's actual tool call. Do not embed a full LLM prompt or expected-call schema in the task metric, and do not let the evaluator LLM directly assign the final process score. Tool selection/parameter mismatch is a process result, not a penalty. Define formulas, weights, rewards, penalties, clipping/normalization, once-only rules, invalid actions, hidden-state evaluation, terminal success/failure, deterministic replay, response shape, and hidden-truth leakage prevention. Evaluate only valid tool name/parameters, process-interface correctness, environment state transitions, and task-goal completion. Media content and media-reference validity are not separate evaluation targets. For `ask_user`, evaluate the Agent's interaction request and its effect on task progress; do not reward or penalize the content of a user-simulator message as if it were an Agent action.

## 10. Trainer API and state transitions

Design health, reset, observation, tool/action discovery, action execution, user-interaction request delivery, reward/status, replay/export, and optional shutdown interfaces. For each define method/path or protocol, request/response, status/error codes, authentication boundary, idempotency, concurrency, and persistence behavior. The user-interaction interface must distinguish the Agent's `ask_user` request from the simulator's generated user message: one successful `ask_user` call produces one simulator response, which must be persisted before it is returned in observation/context. No other interface or background process may invoke the simulator.

Define and expand these equations for the task:

`S_(t+1) = F(S_t, A_t, P_t)`
`T_(t+1) = G(T_t, A_t)`
`O_(t+1) = H(S_(t+1))`
`R_t = Q(S_t, A_t, S_(t+1))`
`D_t = terminal(S_(t+1))`

Explain observable state, hidden state, persistent records, action effects, time effects, termination, and how each Trainer action changes variables and storage.

## 11. Acceptance and verification

Define executable tests for the standard tool schema, action chains, state transitions, invalid actions, hidden-state leakage, user scripts, external-LLM mocks, media-generation code dependencies/entrypoint execution, generated media persistence, timestamps/durations, concurrency/isolation, rewards, API contracts, reset/replay determinism, Docker build/startup, health checks, and an end-to-end trajectory. Assert that no OCR/multimodal call or binding-generation path is required and that valid tool parameters, process interfaces, state transitions, and task-goal completion are evaluated. Do not add tests whose only purpose is media-reference or media-recognition correctness. Specify evidence and the condition for the outer workflow to create `OK`.

## Design decisions and unresolved assumptions

Record task-specific decisions, rejected alternatives, assumptions, risks, and unresolved questions. Phase 2 must resolve implementation details without changing task success semantics.

## Phase 1 requirements

Read `task.json` and the read-only `BUILD_CONTRACT.json`, then write only
`spec.md`. Do not create implementation code, tests, tools, HTTP handlers,
Docker files, or any plan/topology artifact. The spec must be detailed enough
to implement the complete sandbox: task-specific business data and
persistence, generated media when needed, user simulator, complete atomic
action/tool decomposition, Trainer actions, observations, rewards, APIs,
external-model boundaries, and acceptance tests. Include every required
obligation ID literally and explain how the design satisfies it.

## Phase 2 requirements

Read `spec.md`, `task.json`, and `BUILD_CONTRACT.json`, then implement the
complete sandbox in one Code Agent run. Generate all production code and
delivery artifacts directly from the spec, including business-data loading,
persistence, user simulator, Trainer action executor, observation/reward
interfaces, root `tools.json` in standard OpenAI Function Tool format, tests,
`acceptance.sh`, `IMPLEMENTATION_REPORT.md`, `Dockerfile`,
`docker_build.sh`, and `docker_run.sh`. Implement every action/tool described
in the spec and preserve the contract semantics. Do not create
`action_plan.json`, `development_plan.json`, topology scripts, child specs,
or per-node plans. Do not stop at a demo or a design report.

The outer workflow may retry this same Phase 2 Code Agent with the captured
stderr, test output, acceptance output, or missing-file list. On retry, fix
the reported defect in the existing sandbox and finish the complete build;
do not replace the spec or weaken the contract.

Provide only Docker artifacts: `Dockerfile`, `docker_build.sh`, and `docker_run.sh`; never create `Containerfile` or Apple Container files. Default the base image to `docker.m.daocloud.io/library/python:3.14-slim`, validate reachability before building, and use a compatible mirror fallback when needed. When HTTP is appropriate, expose at least `/health`, `/v1/reset`, `/v1/observation`, `/v1/actions`, `/v1/step`, `/v1/ask_user`, and `/v1/reward`.

Create a real executable `acceptance.sh` and non-empty `IMPLEMENTATION_REPORT.md`. Acceptance must capture server stderr, use bounded health retries, distinguish startup errors from an environment permission failure on local TCP binding, and use equivalent in-process checks only for the latter. When `BUILD_CONTRACT.json` declares a non-empty capability `required_trace`, acceptance must write JSONL `runtime_trace.jsonl` at the sandbox root; each event must include `event` and may include `capability_id` and `action_id`. The outer workflow validates the required event order independently. Do not create `OK`; the outer workflow creates it after successful phases, file checks, acceptance, trace validation, and optional Docker build.

Before finishing Phase 2, run the acceptance plan, including tool-schema
validation, public-tool/Trainer-action correspondence, persistence checks,
reward checks, and an end-to-end trajectory. Fix implementation failures
rather than merely reporting them. Do not ask for approval or interactive
input during any phase.
