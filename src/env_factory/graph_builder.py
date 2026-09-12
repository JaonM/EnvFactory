"""Neo4j graph construction for the task knowledge graph."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import os
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

from .knowledge_graph import (
    SceneNode,
    SceneRelation,
    TaskType,
    TaskTypeNode,
    normalize_scene_name,
)


load_dotenv()


@dataclass(frozen=True)
class SceneEdge:
    """A directed relation between two scene nodes."""

    source: str
    target: str
    relation: SceneRelation


class Neo4jGraphStore:
    """Persist the task graph in Neo4j."""

    _RELATION_QUERIES = {
        SceneRelation.HIERARCHY: "HIERARCHY",
        SceneRelation.SAME_EVENT_ELEMENT: "SAME_EVENT_ELEMENT",
    }

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        *,
        database: str = "neo4j",
        driver: Driver | None = None,
    ) -> None:
        self.database = database
        self._owns_driver = driver is None
        self.driver = driver or GraphDatabase.driver(
            uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                user or os.getenv("NEO4J_USER", "neo4j"),
                password or os.getenv("NEO4J_PASSWORD", "password"),
            ),
        )

    def close(self) -> None:
        """Close the underlying driver when this store created it."""

        if self._owns_driver:
            self.driver.close()

    def __enter__(self) -> "Neo4jGraphStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def verify_connectivity(self) -> None:
        """Raise if Neo4j cannot be reached with the configured credentials."""

        self.driver.verify_connectivity()

    def ensure_schema(self) -> None:
        """Create uniqueness constraints required by graph upserts."""

        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT scene_id IF NOT EXISTS "
                "FOR (node:Scene) REQUIRE node.id IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT task_type_id IF NOT EXISTS "
                "FOR (node:TaskType) REQUIRE node.id IS UNIQUE"
            ).consume()

    def upsert_scene(self, node: SceneNode) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MERGE (scene:Scene {id: $id}) "
                "SET scene.name = $name, scene.words = $words, "
                "scene.expanded = coalesce(scene.expanded, false) OR $expanded, "
                "scene.expanded_words = CASE WHEN $expanded THEN "
                "reduce(result = coalesce(scene.expanded_words, []), word IN $expanded_words | "
                "CASE WHEN word IN result THEN result ELSE result + word END) "
                "ELSE coalesce(scene.expanded_words, []) END",
                id=normalize_scene_name(node.name),
                name=node.name,
                words=list(node.words),
                expanded=node.expanded,
                expanded_words=[normalize_scene_name(word) for word in node.expanded_words],
            ).consume()

    def get_expanded_scene_words(self) -> set[str]:
        """Return names and aliases of Scene nodes already expanded."""

        return {
            normalize_scene_name(word)
            for node in self.get_scene_nodes()
            if node.expanded
            for word in (node.name, *node.words)
        }

    def get_scene_nodes(self) -> tuple[SceneNode, ...]:
        """Load persisted Scene nodes for cross-run de-duplication."""

        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (scene:Scene) "
                "RETURN scene.name AS name, scene.words AS words, "
                "coalesce(scene.expanded, false) AS expanded, "
                "coalesce(scene.expanded_words, []) AS expanded_words"
            )
            return tuple(
                SceneNode(
                    name=str(record["name"]),
                    words=tuple(str(word) for word in (record["words"] or [])),
                    expanded=bool(record["expanded"]),
                    expanded_words=tuple(
                        str(word) for word in (record["expanded_words"] or [])
                    ),
                )
                for record in result
            )

    def upsert_task_type(self, node: TaskTypeNode) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MERGE (task_type:TaskType {id: $id}) "
                "SET task_type.name = $name, task_type.value = $value",
                id=node.task_type.value,
                name=node.task_type.value,
                value=node.task_type.value,
            ).consume()

    def add_scene_edge(self, edge: SceneEdge) -> None:
        relation = self._RELATION_QUERIES[edge.relation]
        query = (
            f"MATCH (source:Scene {{id: $source}}), (target:Scene {{id: $target}}) "
            f"MERGE (source)-[:{relation}]->(target)"
        )
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                source=normalize_scene_name(edge.source),
                target=normalize_scene_name(edge.target),
            ).consume()


class KnowledgeGraphBuilder:
    """Build and persist a graph while merging aliases into canonical scenes."""

    def __init__(
        self,
        *,
        canonicalizer: Callable[[str], str] = normalize_scene_name,
    ) -> None:
        self.canonicalizer = canonicalizer
        self._scenes: dict[str, SceneNode] = {}
        self._aliases: dict[str, str] = {}
        self._edges: set[SceneEdge] = set()
        self._relation_by_scene_pair: dict[frozenset[str], SceneRelation] = {}

    def add_scene(
        self,
        name: str,
        *,
        words: Iterable[str] = (),
        expanded: bool = False,
        expanded_words: Iterable[str] = (),
    ) -> SceneNode:
        """Add a scene or merge its words into an existing scene."""

        if not name.strip():
            raise ValueError("scene name must not be empty")
        words = tuple(words)
        expanded_words = tuple(expanded_words)
        raw_id = self.canonicalizer(name)
        scene_id = self._aliases.get(raw_id, raw_id)
        if scene_id not in self._scenes:
            word_scene_ids = {
                self._aliases[self.canonicalizer(word)]
                for word in words
                if self.canonicalizer(word) in self._aliases
            }
            if len(word_scene_ids) > 1:
                raise ValueError("scene words map to multiple existing scene nodes")
            if word_scene_ids:
                scene_id = word_scene_ids.pop()
        existing = self._scenes.get(scene_id)
        if existing is None:
            node = SceneNode(
                name=name.strip(), words=tuple(words), expanded=expanded,
                expanded_words=tuple(expanded_words)
            )
        else:
            node = SceneNode(
                name=existing.name,
                words=tuple(dict.fromkeys((*existing.words, *words))),
                expanded=existing.expanded or expanded,
                expanded_words=tuple(dict.fromkeys((*existing.expanded_words, *expanded_words))),
            )
        self._scenes[scene_id] = node
        self._aliases[raw_id] = scene_id
        for word in node.words:
            self._aliases[self.canonicalizer(word)] = scene_id
        return node

    def add_relation(self, source: str, target: str, relation: SceneRelation) -> None:
        """Add a relation, enforcing direction and one type per scene pair.

        Hierarchy is stored in the supplied direction. Same-event-element is
        symmetric and is therefore stored in both directions.
        """

        if not isinstance(relation, SceneRelation):
            raise ValueError(f"unsupported scene relation: {relation!r}")
        self.add_scene(source)
        self.add_scene(target)
        source_id = self._aliases[self.canonicalizer(source)]
        target_id = self._aliases[self.canonicalizer(target)]
        if source_id == target_id:
            return
        pair = frozenset((source_id, target_id))
        previous_relation = self._relation_by_scene_pair.get(pair)
        if previous_relation is not None and previous_relation != relation:
            raise ValueError(
                "a Scene node pair cannot have both hierarchy and same_event_element relations"
            )
        self._relation_by_scene_pair[pair] = relation
        self._edges.add(SceneEdge(source_id, target_id, relation))
        if relation is SceneRelation.SAME_EVENT_ELEMENT:
            self._edges.add(SceneEdge(target_id, source_id, relation))

    def scenes(self) -> tuple[SceneNode, ...]:
        return tuple(self._scenes.values())

    def edges(self) -> tuple[SceneEdge, ...]:
        return tuple(self._edges)

    def build(
        self,
        store: Neo4jGraphStore,
        *,
        task_types: Iterable[TaskType] = tuple(TaskType),
    ) -> None:
        """Write all collected nodes and relationships to Neo4j."""

        store.ensure_schema()
        for scene in self.scenes():
            store.upsert_scene(scene)
        for task_type in task_types:
            store.upsert_task_type(TaskTypeNode(task_type))
        for edge in self.edges():
            store.add_scene_edge(edge)
