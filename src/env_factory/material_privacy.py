"""Policy-visible trajectory boundary and leakage audit."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .data_governance import scan_payloads


FORBIDDEN_POLICY_KEYS = {
    "acceptance_contract",
    "expected_answer",
    "future_user_turns",
    "ground_truth",
    "hidden_truth",
    "trainer_api_key",
    "trainer_authentication",
}


def _normal_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _forbidden_key_paths(value: Any, path: str = "$") -> Iterable[dict[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _normal_key(key) in FORBIDDEN_POLICY_KEYS:
                yield {"kind": _normal_key(key), "path": child_path}
            yield from _forbidden_key_paths(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _forbidden_key_paths(child, f"{path}[{index}]")


def policy_visible_payloads(rollout: Mapping[str, Any]) -> dict[str, Any]:
    payloads = {}
    for episode_index, episode in enumerate(rollout.get("episodes", [])):
        if not isinstance(episode, Mapping):
            continue
        for transition_index, transition in enumerate(episode.get("transitions", [])):
            if not isinstance(transition, Mapping):
                continue
            prefix = f"episode_{episode_index}.transition_{transition_index}"
            for name in (
                "agent_input", "assistant_output", "observation", "action",
                "result", "next_observation",
            ):
                payloads[f"{prefix}.{name}"] = transition.get(name)
    return payloads


def audit_rollout_privacy(rollout: Mapping[str, Any]) -> dict[str, Any]:
    payloads = policy_visible_payloads(rollout)
    scan = scan_payloads(payloads)
    forbidden = [
        finding
        for label, payload in payloads.items()
        for finding in _forbidden_key_paths(payload, f"$.{label}")
    ]
    marker = rollout.get("material_visibility_version") == "1.0"
    return {
        "version": "1.0",
        "eligible_for_policy_training_export": (
            marker
            and not scan["credential_findings"]
            and not scan["pii_findings"]
            and not forbidden
        ),
        "material_visibility_version": rollout.get("material_visibility_version"),
        "policy_visible_fields": [
            "agent_input", "assistant_output", "observation", "action",
            "result", "next_observation",
        ],
        "trainer_only_fields": [
            "trajectory", "replay", "initial_state", "final_state",
            "trainer_metadata", "user_simulator_outcome",
        ],
        "credential_findings": scan["credential_findings"],
        "pii_findings": scan["pii_findings"],
        "forbidden_key_findings": forbidden,
    }
