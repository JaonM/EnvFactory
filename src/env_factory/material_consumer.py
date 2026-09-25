"""Canonical machine-readable contract for downstream RL material consumers."""

from __future__ import annotations

from typing import Any

from .trajectory_schema import POLICY_TRANSITION_FIELDS


BUNDLE_MANIFEST = "bundle_manifest.json"
TRANSITIONS_FILE = "transitions.jsonl"
CERTIFICATION_FILE = "certification.json"
DATASET_CARD_FILE = "dataset_card.json"
CONSUMER_CONTRACT_FILE = "consumer_contract.json"
BUNDLE_SIGNATURE_FILE = "bundle_manifest.sig"
BUNDLE_VERSION = "6.0"
DATASET_SPLITS = ("train", "validation", "test")


def assign_dataset_splits(items: list[dict[str, Any]]) -> dict[str, str]:
    """Deterministically stratify task identities; never split one task's episodes."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item.get("category", "unknown")), []).append(item)
    assignments = {}
    for members in groups.values():
        ordered = sorted(members, key=lambda item: str(item.get("item_id", "")))
        count = len(ordered)
        validation_count = max(1, count // 10) if count >= 3 else 0
        test_count = max(1, count // 10) if count >= 3 else 0
        train_end = count - validation_count - test_count
        for index, item in enumerate(ordered):
            split = (
                "train" if index < train_end
                else "validation" if index < count - test_count
                else "test"
            )
            assignments[str(item["item_id"])] = split
    return assignments


def consumer_contract(bundle_version: str = BUNDLE_VERSION) -> dict[str, Any]:
    """Return the exact portable handoff; consumers need no prompt conventions."""
    supports_splits = bundle_version == BUNDLE_VERSION
    record_fields = [
        "schema_version", "item_id", "task_sha256", "category",
        "episode_index", "episode_seed", "episode_success",
        "episode_termination", "episode_initial_reward", "episode_final_reward",
        "agent_model", "runtime_model", "agent_usage",
        *(["split"] if supports_splits else []), "transition",
    ]
    return {
        "version": "1.0",
        "kind": "agentic_rl_training_material_consumer_contract",
        "bundle_version": bundle_version,
        "records": {
            "path": TRANSITIONS_FILE,
            "media_type": "application/x-ndjson",
            "encoding": "utf-8",
            "ordering": ["item_id", "episode_index", "transition.step"],
            "identity": ["item_id", "episode_index", "transition.step"],
            **({"split_unit": "item_id"} if supports_splits else {}),
            "required_fields": record_fields,
            "policy_transition_fields": list(POLICY_TRANSITION_FIELDS),
            "json_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": record_fields,
                "properties": {
                    "schema_version": {"const": "2.0"},
                    "item_id": {"type": "string", "pattern": "^[0-9a-f]{16}-[0-9a-f]{16}$"},
                    "task_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "category": {"type": "string", "minLength": 1},
                    **(
                        {"split": {"enum": list(DATASET_SPLITS)}}
                        if supports_splits else {}
                    ),
                    "episode_index": {"type": "integer", "minimum": 0},
                    "episode_seed": {"type": "integer"},
                    "episode_success": {"type": "boolean"},
                    "episode_termination": {"type": "string", "minLength": 1},
                    "episode_initial_reward": {"type": "number"},
                    "episode_final_reward": {"type": "number"},
                    "agent_model": {"type": "string", "minLength": 1},
                    "runtime_model": {"type": "string", "minLength": 1},
                    "agent_usage": {"type": "object"},
                    "transition": {
                        "type": "object",
                        "required": list(POLICY_TRANSITION_FIELDS),
                        "properties": {
                            "step": {"type": "integer", "minimum": 0},
                            "agent_input": {
                                "type": "array", "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "required": ["role", "content"],
                                    "properties": {
                                        "role": {"enum": ["system", "user", "assistant"]},
                                        "content": {"type": "string"},
                                    },
                                },
                            },
                            "assistant_output": {"type": "string"},
                            "observation": {"type": "object"},
                            "action": {"type": ["object", "null"]},
                            "result": {"type": "object"},
                            "next_observation": {"type": "object"},
                            "reward": {"type": "number"},
                            "terminated": {"type": "boolean"},
                            "truncated": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
        },
        "environments": {
            "index": f"{BUNDLE_MANIFEST}#items",
            "root_template": "environments/{item_id}",
            "task_contract": "environments/{item_id}/task.json",
            "docker_build_context": "environments/{item_id}",
            "dockerfile": "environments/{item_id}/Dockerfile",
            "runtime_interface_source": "task.json#requirements.runtime_interface",
        },
        "visibility": {
            "policy_input": ["transition.agent_input", "transition.observation"],
            "policy_output": ["transition.assistant_output", "transition.action"],
            "environment_feedback": [
                "transition.result", "transition.next_observation", "transition.reward",
                "transition.terminated", "transition.truncated",
            ],
            "trainer_only": "environments/{item_id}/live_rollout.json",
            "forbid_policy_ingestion": [
                "trainer_metadata", "trajectory", "replay", "initial_state", "final_state",
            ],
        },
        "invariants": [
            "records_are_ordered_and_unique_by_identity",
            "assistant_output_parses_to_action",
            "next_observation_equals_following_observation_within_episode",
            "only_final_transition_is_terminal_or_truncated",
            "final_transition_reward_equals_episode_final_reward",
            "trainer_only_fields_are_not_policy_inputs",
        ],
    }
