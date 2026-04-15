import json
import time
from pathlib import Path
from typing import Any, Dict

from src.db_managers.neo4j_graph_client import Neo4jGraphClient
from src.data_models.neo4j_conf import Neo4jConfig
from src.data_models.graph_schema import GraphSchema
from src.indexing_pipeline.mapper import Mapper 


class GraphIngestor:
    def __init__(
        self,
        schema: GraphSchema,
        graph_client: Neo4jGraphClient,
        normalizer: Mapper,
    ) -> None:
        self.schema = schema
        self.graph_client = graph_client
        self.normalizer = normalizer

    @classmethod
    def from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        embedder
    ) -> "GraphIngestor":
        schema = GraphSchema.from_dict(payload["schema"])
        graph_client = Neo4jGraphClient(
            uri=uri,
            user=user,
            password=password,
            database=database,
        )
        normalizer = Mapper(schema=schema, embedder=embedder)
        return cls(schema=schema, graph_client=graph_client, normalizer=normalizer)

    def ingest_payload(
        self,
        payload: Dict[str, Any],
        *,
        chunk_size: int = 5000,
        embed_nodes: bool = True,
    ) -> None:
        t0 = time.perf_counter()

        nodes, edges = self.normalizer.normalize(
            payload,
            embed_nodes=embed_nodes,
        )

        self.graph_client.ensure_constraints()
        self.graph_client.upsert_nodes(nodes, chunk_size=chunk_size)
        self.graph_client.upsert_edges(edges, chunk_size=chunk_size)

        elapsed = time.perf_counter() - t0
        print(f"Ingestion completed: {len(nodes)} nodes, {len(edges)} edges in {elapsed:.2f}s")

    def ingest_json_file(
        self,
        input_path: str | Path,
        *,
        chunk_size: int = 5000,
        embed_nodes: bool = True,
    ) -> None:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        self.ingest_payload(
            payload,
            chunk_size=chunk_size,
            embed_nodes=embed_nodes,
        )

    def close(self) -> None:
        self.graph_client.close()