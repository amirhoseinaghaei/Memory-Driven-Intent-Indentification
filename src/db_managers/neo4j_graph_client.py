from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol
from src.data_models.neo4j_conf import Neo4jConfig
from src.data_models.graph_schema import GraphSchema
from neo4j import GraphDatabase, Driver


class Neo4jGraphClient:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self.database = database
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def ensure_constraints(self) -> None:
        cypher_constraint = """
        CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.id IS UNIQUE
        """
        with self.driver.session(database=self.database) as session:
            session.run(cypher_constraint)

    @staticmethod
    def _chunks(rows: List[dict], chunk_size: int):
        for i in range(0, len(rows), chunk_size):
            yield rows[i:i + chunk_size]

    def upsert_nodes(self, nodes: List[NodeSpec], chunk_size: int = 5000) -> None:
        if not nodes:
            return

        rows = [
            {
                "id": n.id,
                "layer": n.layer,
                "label": n.label,
                "data": n.data,
                "embedding": n.embedding,
                "properties": n.properties,
            }
            for n in nodes
        ]

        cypher = """
        UNWIND $rows AS row
        MERGE (n:Entity {id: row.id})
        SET
          n.layer = row.layer,
          n.label = row.label,
          n.data = row.data,
          n.embedding = row.embedding
        SET n += row.properties
        """

        with self.driver.session(database=self.database) as session:
            for batch in self._chunks(rows, chunk_size):
                session.run(cypher, {"rows": batch})

    def upsert_edges(self, edges: List[EdgeSpec], chunk_size: int = 5000) -> None:
        if not edges:
            return

        grouped: Dict[str, List[dict]] = {}
        for e in edges:
            grouped.setdefault(e.rel_type, []).append(
                {
                    "from_id": e.from_id,
                    "to_id": e.to_id,
                    "properties": e.properties,
                }
            )

        with self.driver.session(database=self.database) as session:
            for rel_type, rows in grouped.items():
                cypher = f"""
                UNWIND $rows AS row
                MATCH (a:Entity {{id: row.from_id}})
                MATCH (b:Entity {{id: row.to_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += row.properties
                """
                for batch in self._chunks(rows, chunk_size):
                    session.run(cypher, {"rows": batch})