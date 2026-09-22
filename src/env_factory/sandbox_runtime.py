"""Reusable, dependency-free runtime primitives for generated RL sandboxes.

The Code Agent receives a copy of this module as ``sandbox_runtime.py``.  It
owns transport/application wiring, while this module owns the invariants that
must not be reimplemented differently by every generated task:
authentication, episode isolation, idempotency, deterministic replay,
structured errors, runtime LLM calls, evaluator mocks, and trace hashes.
"""

from __future__ import annotations

import hashlib
import copy
import json
import os
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .runtime_llm import RuntimeLLMClient, RuntimeLLMConfig, RuntimeLLMError
except ImportError:  # copied as a flat module into a generated sandbox
    from runtime_llm import RuntimeLLMClient, RuntimeLLMConfig, RuntimeLLMError


class SandboxError(RuntimeError):
    def __init__(self, code: str, message: str, status: int = 400, details: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.status, self.details = code, message, status, details

    def body(self, request_id: str) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details, "request_id": request_id}}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def trainer_token_from_env() -> str:
    return os.getenv("SANDBOX_TRAINER_API_KEY", "")


def require_trainer(authorization: str | None) -> None:
    expected = trainer_token_from_env()
    if not expected:
        raise SandboxError("AUTH_NOT_CONFIGURED", "SANDBOX_TRAINER_API_KEY is not configured", 503)
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not supplied or not __import__("hmac").compare_digest(supplied, expected):
        raise SandboxError("UNAUTHORIZED", "Trainer authentication required", 401)


class JsonLog:
    def __init__(self, name: str = "sandbox") -> None:
        self.name = name

    def emit(self, event: str, **fields: Any) -> None:
        payload = {"event": event, "logger": self.name, "timestamp": time.time(), **fields}
        # Never include values that could contain credentials.
        for key in list(payload):
            if any(word in key.lower() for word in ("key", "secret", "token", "password")):
                payload[key] = "[REDACTED]"
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    seed: int
    data_hash: str
    schema_version: str


class EpisodeStore:
    """SQLite-backed isolated episodes, events, idempotency and replay."""

    def __init__(self, db_path: str | Path, *, schema_version: str = "1.0") -> None:
        self.db_path = str(db_path)
        self.schema_version = schema_version
        self._lock = threading.RLock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    seed INTEGER NOT NULL,
                    data_hash TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS events (
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (episode_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS idempotency (
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
                    idem_key TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (episode_id, idem_key)
                );
                CREATE TABLE IF NOT EXISTS episode_state (
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
                    state_key TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (episode_id, state_key)
                );
                CREATE TABLE IF NOT EXISTS migrations (
                    version TEXT PRIMARY KEY,
                    applied_at REAL NOT NULL
                );
            """)

    def reset(self, *, episode_id: str | None = None, seed: int | None = None, data_hash: str = "") -> Episode:
        with self._lock, self._connect() as db:
            episode_id = episode_id or f"ep-{uuid.uuid4().hex}"
            if seed is None:
                seed = random.SystemRandom().randrange(0, 2**63)
            db.execute("DELETE FROM events WHERE episode_id = ?", (episode_id,))
            db.execute("DELETE FROM idempotency WHERE episode_id = ?", (episode_id,))
            db.execute("DELETE FROM episode_state WHERE episode_id = ?", (episode_id,))
            db.execute("UPDATE episodes SET active=0 WHERE episode_id = ?", (episode_id,))
            db.execute("INSERT OR REPLACE INTO episodes(episode_id,seed,data_hash,schema_version,created_at,active) VALUES(?,?,?,?,?,1)", (episode_id, seed, data_hash, self.schema_version, time.time()))
            return Episode(episode_id, seed, data_hash, self.schema_version)

    def current(self) -> Episode:
        with self._connect() as db:
            row = db.execute("SELECT * FROM episodes WHERE active=1 ORDER BY created_at DESC LIMIT 1").fetchone()
        if row is None:
            return self.reset()
        return Episode(row["episode_id"], row["seed"], row["data_hash"], row["schema_version"])

    def event(self, event_type: str, payload: Any, result: Any = None, *, idem_key: str | None = None) -> Any:
        episode = self.current()
        with self._lock, self._connect() as db:
            if idem_key:
                existing = db.execute("SELECT response_json FROM idempotency WHERE episode_id=? AND idem_key=?", (episode.episode_id, idem_key)).fetchone()
                if existing:
                    return json.loads(existing[0])
            sequence = int(db.execute("SELECT COALESCE(MAX(sequence), 0) FROM events WHERE episode_id=?", (episode.episode_id,)).fetchone()[0]) + 1
            db.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", (episode.episode_id, sequence, event_type, canonical_json(payload), canonical_json(result), time.time()))
            if idem_key:
                db.execute("INSERT INTO idempotency VALUES(?,?,?,?)", (episode.episode_id, idem_key, canonical_json(result), time.time()))
        return result

    def set_state(self, state_key: str, value: Any) -> None:
        episode = self.current()
        with self._lock, self._connect() as db:
            db.execute("INSERT OR REPLACE INTO episode_state VALUES(?,?,?,?)", (episode.episode_id, state_key, canonical_json(value), time.time()))

    def get_state(self, state_key: str, default: Any = None) -> Any:
        episode = self.current()
        with self._connect() as db:
            row = db.execute("SELECT state_json FROM episode_state WHERE episode_id=? AND state_key=?", (episode.episode_id, state_key)).fetchone()
        return default if row is None else json.loads(row[0])

    def replay(self) -> dict[str, Any]:
        episode = self.current()
        with self._connect() as db:
            rows = db.execute("SELECT sequence,event_type,payload_json,result_json,created_at FROM events WHERE episode_id=? ORDER BY sequence", (episode.episode_id,)).fetchall()
        events = [{"sequence": r[0], "event": r[1], "payload": json.loads(r[2]), "result": json.loads(r[3]) if r[3] else None, "timestamp": r[4]} for r in rows]
        return {"episode_id": episode.episode_id, "seed": episode.seed, "schema_version": episode.schema_version, "data_hash": episode.data_hash, "events": events, "trace_hash": sha256_json(events)}


class DataManifestValidator:
    """Validate generated schema/rows before application-specific loading."""

    @staticmethod
    def validate(manifest: Mapping[str, Any], root: str | Path) -> dict[str, Any]:
        root = Path(root)
        tables = manifest.get("tables")
        if not isinstance(tables, list):
            raise SandboxError("DATA_MANIFEST_INVALID", "data manifest has no tables", 500)
        if not tables and manifest.get("environment_mode") in {"stateless", "external_capability"}:
            return {"tables": {}, "data_hash": sha256_json({})}
        if not tables:
            raise SandboxError("DATA_MANIFEST_INVALID", "data manifest has no tables", 500)
        loaded: dict[str, list[dict[str, Any]]] = {}
        foreign_keys: list[tuple[str, str, str, str]] = []
        for table in tables:
            name, schema_file, rows_file = table.get("table_name"), table.get("schema_file"), table.get("rows_file")
            if not all(isinstance(x, str) and x for x in (name, schema_file, rows_file)):
                raise SandboxError("DATA_MANIFEST_INVALID", "table manifest is incomplete", 500)
            schema = json.loads((root / schema_file).read_text(encoding="utf-8"))
            columns = {column["name"] for column in schema.get("columns", [])}
            rows = [json.loads(line) for line in (root / rows_file).read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows or any(set(row) != columns for row in rows):
                raise SandboxError("DATA_INVALID", f"table {name} rows do not match schema", 500)
            primary = schema.get("primary_key", [])
            if len({tuple(row[key] for key in primary) for row in rows}) != len(rows):
                raise SandboxError("DATA_INVALID", f"table {name} primary key is not unique", 500)
            loaded[name] = rows
            for foreign_key in schema.get("foreign_keys", []):
                if isinstance(foreign_key, dict):
                    foreign_keys.append((name, foreign_key.get("column"), foreign_key.get("ref_table"), foreign_key.get("ref_column")))
                elif isinstance(foreign_key, (list, tuple)) and len(foreign_key) == 4:
                    foreign_keys.append(tuple(foreign_key))
        for table, column, ref_table, ref_column in foreign_keys:
            if not all(isinstance(value, str) and value for value in (table, column, ref_table, ref_column)):
                raise SandboxError("DATA_INVALID", "foreign key declaration is invalid", 500)
            if ref_table not in loaded or any(row.get(ref_column) not in {item.get(ref_column) for item in loaded[ref_table]} for row in loaded[table]):
                raise SandboxError("DATA_INVALID", f"foreign key {table}.{column} is invalid", 500)
        return {"tables": loaded, "data_hash": sha256_json(loaded)}


class ManifestDataStore:
    """Episode-isolated business records loaded from a validated manifest."""

    STATE_KEY = "business_data"

    def __init__(
        self,
        manifest: Mapping[str, Any],
        root: str | Path,
        episode_store: EpisodeStore,
    ) -> None:
        validated = DataManifestValidator.validate(manifest, root)
        self.baseline: dict[str, list[dict[str, Any]]] = validated["tables"]
        self.data_hash: str = validated["data_hash"]
        self.episode_store = episode_store

    def reset(self, _episode: Episode | None = None) -> None:
        self.episode_store.set_state(self.STATE_KEY, copy.deepcopy(self.baseline))

    def _all(self) -> dict[str, list[dict[str, Any]]]:
        value = self.episode_store.get_state(self.STATE_KEY)
        if value is None:
            self.reset()
            value = self.episode_store.get_state(self.STATE_KEY)
        if not isinstance(value, dict):
            raise SandboxError("DATA_STATE_INVALID", "episode business data is invalid", 500)
        return value

    def table(self, name: str) -> list[dict[str, Any]]:
        tables = self._all()
        if name not in tables:
            raise SandboxError("NOT_FOUND", f"unknown business table: {name}", 404)
        return copy.deepcopy(tables[name])

    def select(self, name: str, **equals: Any) -> list[dict[str, Any]]:
        return [
            row for row in self.table(name)
            if all(row.get(field) == expected for field, expected in equals.items())
        ]

    def replace_table(self, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        tables = self._all()
        if name not in tables:
            raise SandboxError("NOT_FOUND", f"unknown business table: {name}", 404)
        original_columns = set(tables[name][0]) if tables[name] else set()
        replacement = [dict(row) for row in rows]
        if any(set(row) != original_columns for row in replacement):
            raise SandboxError("DATA_INVALID", f"replacement rows for {name} do not match schema", 400)
        tables[name] = replacement
        self.episode_store.set_state(self.STATE_KEY, tables)

    def insert(self, name: str, row: Mapping[str, Any]) -> dict[str, Any]:
        rows = self.table(name)
        candidate = dict(row)
        expected = set(rows[0]) if rows else set(candidate)
        if set(candidate) != expected:
            raise SandboxError("DATA_INVALID", f"insert row for {name} does not match schema", 400)
        rows.append(candidate)
        self.replace_table(name, rows)
        return copy.deepcopy(candidate)

    def update(self, name: str, selector: Mapping[str, Any], changes: Mapping[str, Any]) -> int:
        rows = self.table(name)
        if rows and set(changes) - set(rows[0]):
            raise SandboxError("DATA_INVALID", f"update fields for {name} do not match schema", 400)
        changed = 0
        for row in rows:
            if all(row.get(key) == value for key, value in selector.items()):
                row.update(changes)
                changed += 1
        self.replace_table(name, rows)
        return changed

    def delete(self, name: str, selector: Mapping[str, Any]) -> int:
        rows = self.table(name)
        kept = [row for row in rows if not all(row.get(key) == value for key, value in selector.items())]
        deleted = len(rows) - len(kept)
        self.replace_table(name, kept)
        return deleted

    def snapshot_hash(self) -> str:
        return sha256_json(self._all())


class DeclarativeToolCompiler:
    """Compile common read-only business tools from constrained specifications."""

    OPERATORS = {"eq", "in", "contains", "gte", "lte"}

    def __init__(self, data: ManifestDataStore) -> None:
        self.data = data

    @staticmethod
    def _matches(actual: Any, expected: Any, operator: str) -> bool:
        if operator == "eq":
            return actual == expected
        if operator == "in":
            return actual in expected if isinstance(expected, list) else False
        if operator == "contains":
            return expected in actual if isinstance(actual, (str, list)) else False
        if operator == "gte":
            return actual is not None and actual >= expected
        if operator == "lte":
            return actual is not None and actual <= expected
        raise SandboxError("TOOL_SPEC_INVALID", f"unsupported filter operator: {operator}", 500)

    def compile(self, spec: Mapping[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
        operation = spec.get("operation")
        if operation not in {"select", "aggregate_count", "insert", "update", "delete"}:
            raise SandboxError("TOOL_SPEC_INVALID", f"unsupported declarative operation: {operation}", 500)
        table = spec.get("table")
        result_field = spec.get("result_field", "records")
        filters = spec.get("filters", [])
        projection = spec.get("projection", [])
        order_by = spec.get("order_by", [])
        if not isinstance(table, str) or not table or not isinstance(result_field, str) or not result_field:
            raise SandboxError("TOOL_SPEC_INVALID", "select spec requires table and result_field", 500)
        if not isinstance(filters, list) or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("argument"), str)
            or not isinstance(item.get("column"), str)
            or item.get("operator") not in self.OPERATORS
            for item in filters
        ):
            raise SandboxError("TOOL_SPEC_INVALID", "select filters are invalid", 500)
        if not isinstance(projection, list) or not all(isinstance(item, str) for item in projection):
            raise SandboxError("TOOL_SPEC_INVALID", "select projection is invalid", 500)
        if not isinstance(order_by, list) or not all(isinstance(item, str) for item in order_by):
            raise SandboxError("TOOL_SPEC_INVALID", "select order_by is invalid", 500)

        def selected(arguments: dict[str, Any]) -> list[dict[str, Any]]:
            rows = self.data.table(table)
            for rule in filters:
                argument = rule["argument"]
                if argument not in arguments:
                    continue
                rows = [
                    row for row in rows
                    if self._matches(row.get(rule["column"]), arguments[argument], rule["operator"])
                ]
            if order_by:
                rows.sort(key=lambda row: tuple(str(row.get(field, "")) for field in order_by))
            if projection:
                rows = [{field: row.get(field) for field in projection} for row in rows]
            return rows

        def handler(arguments: dict[str, Any]) -> dict[str, Any]:
            if operation == "insert":
                mapping = spec.get("values", {})
                row = {column: arguments[argument] for argument, column in mapping.items()}
                return {"record": self.data.insert(table, row), "count": 1}
            selector_map = spec.get("selector", {})
            selector = {column: arguments[argument] for argument, column in selector_map.items()}
            if operation == "update":
                changes_map = spec.get("changes", {})
                changes = {column: arguments[argument] for argument, column in changes_map.items()}
                return {"updated_count": self.data.update(table, selector, changes)}
            if operation == "delete":
                return {"deleted_count": self.data.delete(table, selector)}
            rows = selected(arguments)
            if operation == "aggregate_count":
                return {result_field: len(rows)}
            return {result_field: rows, "count": len(rows)}

        return handler

    def compile_all(self, specs: Sequence[Mapping[str, Any]]) -> dict[str, Callable[[dict[str, Any]], Any]]:
        handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        for spec in specs:
            name = spec.get("tool_name")
            if not isinstance(name, str) or not name or name in handlers:
                raise SandboxError("TOOL_SPEC_INVALID", "tool implementation name is invalid", 500)
            handlers[name] = self.compile(spec)
        return handlers


class EvaluatorMock:
    """Deterministic evaluator seam; production and mock share one contract."""

    def __init__(self, handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"request_hash": sha256_json(request), "request": request})
        if self.handler is None:
            raise SandboxError("EVALUATOR_MOCK_NOT_CONFIGURED", "mock evaluator handler is not configured", 500)
        result = self.handler(request)
        if not isinstance(result, dict):
            raise SandboxError("EVALUATOR_INVALID_RESULT", "evaluator result must be an object", 500)
        return result


def validate_json_schema(schema: Mapping[str, Any], value: Any, path: str = "arguments") -> None:
    """Validate the JSON-Schema subset emitted for Function Tool arguments.

    Keeping this in the reviewed runtime prevents every generated sandbox from
    inventing a subtly different validator.  Unsupported schema constructs are
    rejected during generation; the runtime covers the emitted object, array,
    scalar, enum, required and additionalProperties constraints.
    """

    kind = schema.get("type")
    expected = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
    }.get(kind)
    if expected is not None:
        valid = isinstance(value, expected)
        if kind in {"integer", "number"} and isinstance(value, bool):
            valid = False
        if not valid:
            raise SandboxError("INVALID_ARGUMENT", f"{path} must be {kind}", 400)
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise SandboxError("INVALID_ARGUMENT", f"{path} is not an allowed value", 400)
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise SandboxError("INVALID_ARGUMENT", f"{path} is missing required properties", 400, missing)
        unknown = sorted(set(value) - set(properties))
        if schema.get("additionalProperties") is False and unknown:
            raise SandboxError("INVALID_ARGUMENT", f"{path} has unexpected properties", 400, unknown)
        for name, child in properties.items():
            if name in value:
                validate_json_schema(child, value[name], f"{path}.{name}")
    elif kind == "array":
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                validate_json_schema(item_schema, item, f"{path}[{index}]")


class ContractToolRegistry:
    """Generic Function Tool validation, dispatch, tracing and mutation seam."""

    def __init__(
        self,
        tools: Sequence[Mapping[str, Any]],
        handlers: Mapping[str, Callable[[dict[str, Any]], Any]],
        *,
        noise_tools: Sequence[Mapping[str, Any]] = (),
        event_recorder: Callable[..., Any] | None = None,
    ) -> None:
        self.tools = [dict(item) for item in tools]
        self.handlers = dict(handlers)
        self.event_recorder = event_recorder
        self.noise_tools = {
            item.get("name"): dict(item)
            for item in noise_tools
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
        self.schemas: dict[str, Mapping[str, Any]] = {}
        for item in self.tools:
            function = item.get("function")
            if item.get("type") != "function" or not isinstance(function, Mapping):
                raise SandboxError("TOOL_CONTRACT_INVALID", "invalid Function Tool declaration", 500)
            name, parameters = function.get("name"), function.get("parameters")
            if not isinstance(name, str) or not name or not isinstance(parameters, Mapping):
                raise SandboxError("TOOL_CONTRACT_INVALID", "tool name or parameters are invalid", 500)
            self.schemas[name] = parameters
        missing = sorted(set(self.schemas) - set(self.handlers) - set(self.noise_tools))
        if missing:
            raise SandboxError("TOOL_CONTRACT_INVALID", "declared tools have no handlers", 500, missing)

    def execute(self, name: str, arguments: Mapping[str, Any], *, idem_key: str | None = None) -> Any:
        if name not in self.schemas:
            raise SandboxError("NOT_FOUND", f"unknown tool: {name}", 404)
        if not isinstance(arguments, Mapping):
            raise SandboxError("INVALID_ARGUMENT", "tool arguments must be an object", 400)
        args = dict(arguments)
        validate_json_schema(self.schemas[name], args)
        mutation = os.getenv("SANDBOX_MUTATION_MODE", "disabled")
        if mutation == "constant_tool_result":
            result: Any = {"mutation": "constant_tool_result", "value": None}
        elif name in self.noise_tools:
            metadata = self.noise_tools[name]
            result = {
                "status": "ok",
                "tool": name,
                "category": metadata.get("category"),
                "message": "Tool executed without changing task-critical business state.",
            }
        else:
            result = self.handlers[name](args)
        if self.event_recorder is not None and mutation != "skip_business_write":
            self.event_recorder(
                "tool_call",
                {
                    "tool_name": name,
                    "arguments": args,
                    "noise": name in self.noise_tools,
                },
                result,
                idem_key=idem_key,
            )
        return result


class ContractRewardAggregator:
    """Apply the generated metric ranges, weights and clipping formula."""

    def __init__(self, metrics: Sequence[Mapping[str, Any]]) -> None:
        self.metrics = [dict(metric) for metric in metrics]
        if not self.metrics:
            raise SandboxError("REWARD_CONTRACT_INVALID", "metrics must not be empty", 500)

    def aggregate(self, scores: Mapping[str, Any]) -> dict[str, Any]:
        components: dict[str, float] = {}
        raw_reward = 0.0
        for metric in self.metrics:
            metric_id = metric.get("id")
            if not isinstance(metric_id, str) or not metric_id:
                raise SandboxError("REWARD_CONTRACT_INVALID", "metric id is invalid", 500)
            value = float(scores.get(metric_id, 0.0))
            score_range = metric.get("score_range")
            if not isinstance(score_range, list) or len(score_range) != 2:
                raise SandboxError("REWARD_CONTRACT_INVALID", f"metric {metric_id} range is invalid", 500)
            low, high = map(float, score_range)
            if not low <= value <= high:
                raise SandboxError("REWARD_SCORE_INVALID", f"metric {metric_id} score is outside its range", 500)
            weight = float(metric.get("weight", 0.0))
            components[metric_id] = value
            raw_reward += weight * value
        return {
            "reward": max(-1.0, min(1.0, raw_reward)),
            "raw_reward": raw_reward,
            "components": components,
        }


class DeclarativeMetricEvaluator:
    """Evaluate deterministic metric predicates from a constrained DSL."""

    OPERATORS = {"eq", "ne", "gte", "lte", "contains", "exists", "count_gte", "count_eq", "changed", "unchanged", "subset", "none_tool_calls"}

    @staticmethod
    def _resolve(value: Any, path: str) -> Any:
        current = value
        if path in {"", "$"}:
            return current
        normalized = path.removeprefix("$.")
        for part in normalized.split("."):
            if isinstance(current, Mapping):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                current = current[index] if index < len(current) else None
            else:
                return None
        return current

    @classmethod
    def _compare(cls, actual: Any, operator: str, expected: Any) -> bool:
        if operator == "eq":
            return actual == expected
        if operator == "ne":
            return actual != expected
        if operator == "gte":
            return actual is not None and actual >= expected
        if operator == "lte":
            return actual is not None and actual <= expected
        if operator == "contains":
            return expected in actual if isinstance(actual, (str, list, dict)) else False
        if operator == "exists":
            return (actual is not None) is bool(expected)
        if operator in {"count_gte", "count_eq"}:
            count = len(actual) if isinstance(actual, (str, list, dict)) else 0
            return count >= expected if operator == "count_gte" else count == expected
        if operator == "changed":
            return actual != expected
        if operator == "unchanged":
            return actual == expected
        if operator == "subset":
            return set(actual).issubset(set(expected)) if isinstance(actual, list) and isinstance(expected, list) else False
        if operator == "none_tool_calls":
            if not isinstance(actual, list) or not isinstance(expected, list):
                return False
            blocked = set(expected)
            return not any(
                isinstance(event, Mapping)
                and event.get("event") == "tool_call"
                and isinstance(event.get("payload"), Mapping)
                and event["payload"].get("tool_name") in blocked
                for event in actual
            )
        raise SandboxError("METRIC_SPEC_INVALID", f"unsupported metric operator: {operator}", 500)

    def evaluate(self, spec: Mapping[str, Any], context: Mapping[str, Any]) -> float:
        source = spec.get("source")
        path = spec.get("path", "$")
        operator = spec.get("operator")
        if source not in {"business_state", "trajectory", "final_agent_response", "observation"}:
            raise SandboxError("METRIC_SPEC_INVALID", "metric source is invalid", 500)
        if not isinstance(path, str) or operator not in self.OPERATORS:
            raise SandboxError("METRIC_SPEC_INVALID", "metric path or operator is invalid", 500)
        actual = self._resolve(context.get(source), path)
        passed = self._compare(actual, operator, spec.get("expected"))
        score_mapping = spec.get("score_mapping", {"pass": 1.0, "fail": 0.0})
        if not isinstance(score_mapping, Mapping):
            raise SandboxError("METRIC_SPEC_INVALID", "metric score_mapping is invalid", 500)
        value = score_mapping.get("pass" if passed else "fail")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SandboxError("METRIC_SPEC_INVALID", "metric mapped score is invalid", 500)
        return float(value)

    def evaluate_all(
        self, specs: Sequence[Mapping[str, Any]], context: Mapping[str, Any]
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for spec in specs:
            metric_id = spec.get("metric_id")
            if not isinstance(metric_id, str) or not metric_id or metric_id in scores:
                raise SandboxError("METRIC_SPEC_INVALID", "metric implementation id is invalid", 500)
            scores[metric_id] = self.evaluate(spec, context)
        return scores


class ContractUserSimulator:
    """Seeded, episode-isolated user-session controller with safe fallback."""

    STATE_KEY = "user_simulator"

    def __init__(
        self,
        episode_store: EpisodeStore,
        sessions: Sequence[Mapping[str, Any]],
        *,
        renderer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.episode_store = episode_store
        self.sessions = [dict(item) for item in sessions]
        self.renderer = renderer
        if not self.sessions:
            raise SandboxError("USER_SIMULATION_INVALID", "sessions must not be empty", 500)
        for session in self.sessions:
            if not isinstance(session.get("turns"), list) or not session["turns"]:
                raise SandboxError("USER_SIMULATION_INVALID", "each session requires turns", 500)
            user_turn_count = sum(1 for item in session["turns"] if isinstance(item, Mapping) and item.get("role") == "user")
            flags = session.get("user_end_flags")
            if not isinstance(flags, list) or len(flags) != user_turn_count or any(not isinstance(item, bool) for item in flags):
                raise SandboxError("USER_SIMULATION_INVALID", "user_end_flags must match user turns", 500)

    def reset(self, episode: Episode | None = None) -> None:
        episode = episode or self.episode_store.current()
        index = random.Random(episode.seed).randrange(len(self.sessions))
        self.episode_store.set_state(self.STATE_KEY, {"session_index": index, "turn_index": 0, "memory": []})

    def turn(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise SandboxError("INVALID_ARGUMENT", "messages must be an array", 400)
        if any(not isinstance(item, Mapping) or item.get("role") not in {"user", "assistant", "system"} or not isinstance(item.get("content"), str) for item in messages):
            raise SandboxError("INVALID_ARGUMENT", "messages contain invalid entries", 400)
        state = self.episode_store.get_state(self.STATE_KEY)
        if not isinstance(state, dict):
            self.reset()
            state = self.episode_store.get_state(self.STATE_KEY)
        session = self.sessions[int(state["session_index"])]
        user_turns = [item for item in session["turns"] if isinstance(item, Mapping) and item.get("role") == "user"]
        if not user_turns:
            raise SandboxError("USER_SIMULATION_INVALID", "session has no user turns", 500)
        index = min(int(state["turn_index"]), len(user_turns) - 1)
        fallback = {
            "user_query": str(user_turns[index].get("content", "")),
            "should_end": bool(session.get("user_end_flags", [False] * len(user_turns))[index]),
            "attachments": list(user_turns[index].get("attachments", [])),
        }
        result = fallback
        if self.renderer is not None:
            try:
                candidate = self.renderer({"messages": list(messages), "session": session, "turn_index": index, "fallback": fallback})
                if isinstance(candidate, Mapping) and isinstance(candidate.get("user_query"), str) and isinstance(candidate.get("should_end"), bool):
                    result = {"user_query": candidate["user_query"], "should_end": candidate["should_end"], "attachments": list(candidate.get("attachments", []))}
            except Exception:
                result = fallback
        state["turn_index"] = index + 1
        state["memory"] = [*state.get("memory", []), {"messages": list(messages), "result": result}]
        self.episode_store.set_state(self.STATE_KEY, state)
        self.episode_store.event("user_turn", {"messages": list(messages)}, result)
        return result


class SandboxApplication:
    """Dependency-free HTTP application boundary shared by generated tasks.

    Task code supplies business callbacks.  This class owns the stable Trainer
    protocol, authentication, error envelopes, reset/replay, tool discovery,
    idempotency forwarding and WSGI transport.
    """

    def __init__(
        self,
        *,
        episode_store: EpisodeStore,
        tool_registry: ContractToolRegistry,
        observation: Callable[[], Mapping[str, Any]],
        reward: Callable[[], Mapping[str, Any]],
        user_turn: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
        reset_hook: Callable[[Episode], None] | None = None,
        data_hash: str = "",
    ) -> None:
        self.episode_store = episode_store
        self.tool_registry = tool_registry
        self.observation_callback = observation
        self.reward_callback = reward
        self.user_turn_callback = user_turn
        self.reset_hook = reset_hook
        self.data_hash = data_hash

    @staticmethod
    def _header(headers: Mapping[str, str], name: str) -> str | None:
        wanted = name.casefold()
        return next((str(value) for key, value in headers.items() if key.casefold() == wanted), None)

    def handle(
        self,
        method: str,
        path: str,
        body: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        request = request_id()
        supplied = dict(headers or {})
        response_headers = {"Content-Type": "application/json", "X-Request-ID": request}
        try:
            payload = self._dispatch(method.upper(), path, body, supplied)
            return 200, dict(payload), response_headers
        except SandboxError as exc:
            return exc.status, exc.body(request), response_headers
        except Exception as exc:  # keep the public protocol stable
            error = SandboxError("INTERNAL_ERROR", "sandbox request failed", 500, {"type": type(exc).__name__})
            return error.status, error.body(request), response_headers

    def _dispatch(self, method: str, path: str, body: Any, headers: Mapping[str, str]) -> Mapping[str, Any]:
        if method == "GET" and path == "/health":
            return {"status": "ok"}
        if method == "GET" and path == "/v1/tools":
            return {"tools": self.tool_registry.tools}
        trainer_paths = {"/v1/reset", "/v1/observation", "/v1/user_simulator", "/v1/agent_response", "/v1/reward", "/v1/replay"}
        if path in trainer_paths:
            require_trainer(self._header(headers, "Authorization"))
        if method == "POST" and path == "/v1/reset":
            if body is None:
                body = {}
            if not isinstance(body, Mapping):
                raise SandboxError("INVALID_ARGUMENT", "reset body must be an object", 400)
            unknown = sorted(set(body) - {"episode_id", "seed"})
            if unknown:
                raise SandboxError("INVALID_ARGUMENT", "reset has unexpected properties", 400, unknown)
            episode = self.episode_store.reset(
                episode_id=body.get("episode_id"), seed=body.get("seed"), data_hash=self.data_hash
            )
            if self.reset_hook:
                self.reset_hook(episode)
            return {
                "episode_id": episode.episode_id,
                "seed": episode.seed,
                "data_hash": episode.data_hash,
                "schema_version": episode.schema_version,
            }
        if method == "GET" and path == "/v1/observation":
            return dict(self.observation_callback())
        if method == "POST" and path == "/v1/user_simulator":
            if not isinstance(body, Mapping) or not isinstance(body.get("messages"), list):
                raise SandboxError("INVALID_ARGUMENT", "messages must be an array", 400)
            return dict(self.user_turn_callback(body["messages"]))
        if method == "POST" and path == "/v1/agent_response":
            if not isinstance(body, Mapping) or not isinstance(body.get("content"), str) or not body["content"].strip():
                raise SandboxError("INVALID_ARGUMENT", "agent response requires non-empty content", 400)
            content = body["content"].strip()
            self.episode_store.set_state("final_agent_response", content)
            self.episode_store.event("agent_response", {"content": content}, {"accepted": True})
            return {"accepted": True}
        if method == "GET" and path == "/v1/reward":
            return dict(self.reward_callback())
        if method == "GET" and path == "/v1/replay":
            return self.episode_store.replay()
        prefix = "/v1/tools/"
        if method == "POST" and path.startswith(prefix) and len(path) > len(prefix):
            if body is None:
                body = {}
            if not isinstance(body, Mapping):
                raise SandboxError("INVALID_ARGUMENT", "tool body must be an object", 400)
            result = self.tool_registry.execute(
                path[len(prefix):], body, idem_key=self._header(headers, "Idempotency-Key")
            )
            if not isinstance(result, Mapping):
                return {"result": result}
            return dict(result)
        raise SandboxError("NOT_FOUND", f"unknown endpoint: {method} {path}", 404)

    def __call__(self, environ: Mapping[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
        method = str(environ.get("REQUEST_METHOD", "GET"))
        path = str(environ.get("PATH_INFO", "/"))
        length = int(environ.get("CONTENT_LENGTH") or 0)
        raw = environ["wsgi.input"].read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = None
            status, payload, headers = 400, SandboxError(
                "INVALID_JSON", "request body is not valid JSON", 400
            ).body(request_id()), {"Content-Type": "application/json"}
        else:
            request_headers = {
                key[5:].replace("_", "-"): value
                for key, value in environ.items() if key.startswith("HTTP_")
            }
            status, payload, headers = self.handle(method, path, body, request_headers)
        encoded = canonical_json(payload).encode("utf-8")
        response_headers = list(headers.items()) + [("Content-Length", str(len(encoded)))]
        reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 409: "Conflict", 500: "Internal Server Error", 503: "Service Unavailable"}.get(status, "Error")
        start_response(f"{status} {reason}", response_headers)
        return [encoded]


class AcceptanceScenarioRunner:
    """Execute EnvFactory-owned structured black-box scenarios."""

    def __init__(
        self,
        call: Callable[..., tuple[int, Any, Mapping[str, str]]],
        *,
        trainer_headers: Mapping[str, str] | None = None,
        business_snapshot: Callable[[], Any] | None = None,
        mutate_business_state: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.call = call
        self.trainer_headers = dict(trainer_headers or {})
        self.business_snapshot = business_snapshot
        self.mutate_business_state = mutate_business_state

    @staticmethod
    def _resolve(value: Any, variables: Mapping[str, Any]) -> Any:
        if isinstance(value, Mapping) and set(value) == {"$ref"}:
            return variables.get(str(value["$ref"]))
        if isinstance(value, Mapping):
            return {key: AcceptanceScenarioRunner._resolve(item, variables) for key, item in value.items()}
        if isinstance(value, list):
            return [AcceptanceScenarioRunner._resolve(item, variables) for item in value]
        return value

    def run(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        variables: dict[str, Any] = {}
        step_results: dict[str, Any] = {}
        history: list[dict[str, Any]] = []
        last_body: Any = None
        for index, step in enumerate(scenario.get("steps", [])):
            if not isinstance(step, Mapping):
                raise SandboxError("SCENARIO_INVALID", f"step {index} is not an object", 500)
            operation = step.get("operation")
            if operation == "reset":
                method, path, auth, body = "POST", "/v1/reset", True, self._resolve(step.get("body", {}), variables)
            elif operation == "tool_call":
                method, path, auth = "POST", f"/v1/tools/{step.get('tool_name', '')}", False
                body = self._resolve(step.get("arguments", {}), variables)
            elif operation == "agent_response":
                method, path, auth = "POST", "/v1/agent_response", True
                body = {"content": str(self._resolve(step.get("content", ""), variables))}
            elif operation in {"observation", "reward", "replay"}:
                method, path, auth, body = "GET", f"/v1/{operation}", True, None
            elif operation == "business_snapshot":
                if self.business_snapshot is None:
                    raise SandboxError("SCENARIO_INVALID", "business snapshot callback is unavailable", 500)
                status, last_body = 200, self.business_snapshot()
                method, path, auth, body = "INTERNAL", "business_snapshot", False, None
            elif operation == "mutate_business_state":
                if self.mutate_business_state is None:
                    raise SandboxError("SCENARIO_INVALID", "business mutation callback is unavailable", 500)
                mutation = self._resolve(step.get("mutation", {}), variables)
                status, last_body = 200, self.mutate_business_state(mutation)
                method, path, auth, body = "INTERNAL", "mutate_business_state", False, mutation
            else:
                raise SandboxError("SCENARIO_INVALID", f"unsupported operation: {operation}", 500)
            if method != "INTERNAL":
                status, last_body, _ = self.call(method, path, body, self.trainer_headers if auth else {})
            expected_status = step.get("expected_status", 200)
            if status != expected_status:
                raise SandboxError("SCENARIO_ASSERTION_FAILED", f"step {index} expected HTTP {expected_status}, got {status}", 500, {"body": last_body})
            capture = step.get("capture", {})
            if isinstance(capture, Mapping):
                for name, capture_path in capture.items():
                    variables[str(name)] = DeclarativeMetricEvaluator._resolve(last_body, str(capture_path))
            step_id = step.get("step_id")
            if isinstance(step_id, str) and step_id:
                step_results[step_id] = last_body
            history.append({"operation": operation, "status": status, "body": last_body})
        for assertion in scenario.get("assertions", []):
            if not isinstance(assertion, Mapping):
                raise SandboxError("SCENARIO_INVALID", "assertion must be an object", 500)
            source_name = assertion.get("source")
            if source_name == "variables":
                source = variables
            elif isinstance(source_name, str) and source_name.startswith("step:"):
                source = step_results.get(source_name.split(":", 1)[1])
            else:
                source = last_body
            actual = DeclarativeMetricEvaluator._resolve(source, str(assertion.get("path", "$")))
            expected = self._resolve(assertion.get("expected"), variables)
            if not DeclarativeMetricEvaluator._compare(actual, str(assertion.get("operator")), expected):
                raise SandboxError("SCENARIO_ASSERTION_FAILED", "scenario assertion failed", 500, {"actual": actual, "assertion": dict(assertion)})
        return {"scenario_id": scenario.get("scenario_id"), "history": history, "variables": variables, "step_results": step_results}


class MigrationRegistry:
    def __init__(self, store: EpisodeStore) -> None:
        self.store = store

    def apply(self, version: str, migration: Callable[[sqlite3.Connection], None]) -> None:
        with self.store._lock, self.store._connect() as db:
            if db.execute("SELECT 1 FROM migrations WHERE version=?", (version,)).fetchone():
                return
            migration(db)
            db.execute("INSERT INTO migrations VALUES(?,?)", (version, time.time()))


def request_id() -> str:
    return f"req-{uuid.uuid4().hex}"
