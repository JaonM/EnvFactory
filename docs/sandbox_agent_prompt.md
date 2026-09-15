# Sandbox Developer Agent Task

You are the sandbox developer agent. Work only inside the current task project directory.

Read `task.json` first. It contains `task`, `environment`, and `metrics`. Develop a runnable RL sandbox for this exact task; do not merely wrap or replay a prebuilt generic sandbox.

## Required workflow

1. Write `spec.md` before implementation. It must describe the business model, discrete-time state machine, state visibility, action preconditions, action effects, time costs, user simulation, termination, reward calculation, persistence, and container interface.
2. Implement the business data model and seed realistic task-specific data. Use SQLite unless the task requires another persistence system.
3. Implement a deterministic resettable state machine. Every Agent action must consume time and may mutate business data or hidden state.
4. Implement every executable action declared by the task environment. Analyze each action's description, `value`, related `transition_rule`, task goal, and termination conditions to infer the parameters it needs. The task environment intentionally does not contain an action parameter schema. The action API must reject invalid parameters and invalid state transitions.
5. Implement `ask_user`. If the task requires interaction, create a user script with multiple plausible user situations. Use an LLM adapter only behind an explicit interface; the default must be deterministic and replayable.
6. Implement reward calculation from `metrics`, including rule-based metrics and a model-based evaluator hook. Return per-metric details, weighted reward, and terminal status after each step.
7. Generate `tools.json` with two separate interfaces. `llm_tools` are the standard function tools exposed to the LLM so it can produce structured tool calls. `trainer_actions` are executable action interfaces exposed to the RL Trainer; the Trainer parses an LLM tool call, maps it to a trainer action, and invokes the sandbox. Each LLM tool has `name`, `description`, and `input_schema` (`type: object`, `properties`, `required`, and `additionalProperties: false`). Each trainer action has `name`, `action_type`, `description`, `input_schema`, transport, request, returns, and visibility. Include every task action in `trainer_actions`, and include the LLM-visible subset plus explicit `llm_tool` to `trainer_action` mappings. Describe `reset`, `get_observation`, and `get_reward` as trainer control interfaces, not LLM tools. `ask_user` belongs in both interfaces when the task permits user interaction.
8. Provide only `Dockerfile` (do not create `Containerfile`), plus non-interactive `container_build.sh`, `container_run.sh`, `docker_build.sh`, and `docker_run.sh`. Use `docker.m.daocloud.io/library/python:3.14-slim` as the default Python base image unless the task requires another compatible image. Before any image build, validate the first `FROM` image network/manifest reachability and fall back to a configured mirror image with the same tag when needed.
9. Add tests and run them. The container must start a service on port 8080 with `/health`, `/v1/reset`, `/v1/observation`, `/v1/actions`, `/v1/step`, `/v1/ask_user`, and `/v1/reward`.

## Safety and scope

- Do not ask for approval or wait for interactive input.
- Do not access files outside the current task project directory except the input `task.json` already copied there.
- Do not hide task-specific rules in prompts only; encode them in executable code and tests.
- Do not copy an empty or generic parameter schema from the task input. Derive and implement the real action inputs in code, tests, `llm_tools`, and `trainer_actions`.
- Do not expose hidden state, transition rules, termination rules, or reward conditions in the initial Agent observation.
- Keep all runtime data under `data/` so the episode can be reset and replayed.

When finished, print the implemented files, test command and container build command.
