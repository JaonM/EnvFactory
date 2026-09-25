"""Shared task similarity and deterministic near-duplicate family identities."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping


def normal_task_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\d+(?:\.\d+)?", "#", text)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def task_shingles(text: str, size: int = 3) -> set[str]:
    if len(text) <= size:
        return {text} if text else set()
    return {text[index:index + size] for index in range(len(text) - size + 1)}


def near_duplicate_rate(
    tasks: Iterable[Mapping[str, Any]], threshold: float = 0.9
) -> float:
    vectors: list[set[str]] = []
    duplicates = 0
    values = list(tasks)
    for task in values:
        vector = task_shingles(normal_task_text(task.get("task")))
        if vector and any(
            len(vector & previous) / len(vector | previous) >= threshold
            for previous in vectors if previous
        ):
            duplicates += 1
        vectors.append(vector)
    return duplicates / len(values) if values else 0.0


def task_family_ids(
    items: Iterable[Mapping[str, Any]], threshold: float = 0.9
) -> dict[str, str]:
    """Cluster the similarity graph, including transitive near-duplicate variants."""
    values = list(items)
    identities = [str(item.get("item_id", "")) for item in values]
    if any(not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("task family inputs need unique non-empty item_id values")
    texts = [normal_task_text(item.get("task")) for item in values]
    vectors = [task_shingles(text) for text in texts]
    parents = list(range(len(values)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(values)):
        if not vectors[left]:
            continue
        for right in range(left):
            if not vectors[right]:
                continue
            similarity = len(vectors[left] & vectors[right]) / len(
                vectors[left] | vectors[right]
            )
            if similarity >= threshold:
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(values)):
        components.setdefault(find(index), []).append(index)
    result = {}
    for members in components.values():
        canonical = "\0".join(sorted({texts[index] for index in members}))
        if not canonical:
            canonical = "\0".join(sorted(identities[index] for index in members))
        family_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        for index in members:
            result[identities[index]] = family_id
    return result
