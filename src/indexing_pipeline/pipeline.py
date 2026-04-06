from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.pipeline.graph_ingestor import GraphIngestor
from src.gen_ai_gateway.embedder import Embed
from src.pipeline.entity_extractor import SchemaDrivenExtractor, load_schema
from src.pipeline.parser import process_directory, write_output
from src.config.config import Settings, load_settings_from_json


class DummyEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


def load_settings(settings_path: str | None = None) -> Settings:
    if settings_path:
        return load_settings_from_json(Path(settings_path))
    return Settings()


def build_embedder(settings: Settings) -> Any:
    return Embed(settings)


def run_extraction(
    input_dir: Path,
    schema_path: Path,
    output_dir: Path,
    api_key: str,
    model: str,
    api_base: str,
) -> Path:
    schema = load_schema(schema_path)
    extractor = SchemaDrivenExtractor(
        api_key=api_key,
        model=model,
        base_url=api_base,
    )

    payload = process_directory(input_dir=input_dir, schema=schema, extractor=extractor)
    output_path = write_output(output_dir, payload)
    return output_path


def run_graph_ingestion(
    payload_path: Path,
    uri: str,
    user: str,
    password: str,
    database: str,
    embedder: Any,
    chunk_size: int,
    embed_nodes: bool,
) -> None:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    ingestor = GraphIngestor.from_payload(
        payload,
        uri=uri,
        user=user,
        password=password,
        database=database,
        embedder=embedder,
    )

    ingestor.ingest_payload(payload, chunk_size=chunk_size, embed_nodes=embed_nodes)
    ingestor.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extraction and ingestion pipeline using entity extractor, parser, and graph ingestor."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing .txt files to extract.")
    parser.add_argument("--schema_path", required=True, help="Path to the extraction schema JSON.")
    parser.add_argument("--output_dir", required=True, help="Directory to write extracted payload JSON.")
    parser.add_argument(
        "--api_key",
        default=None,
        help="Optional API key for extraction. If omitted, value is taken from config.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model name for extraction. If omitted, value is taken from config.",
    )
    parser.add_argument(
        "--api_base",
        default=None,
        help="Optional API base URL for extraction. If omitted, value is taken from config.",
    )
    parser.add_argument("--neo4j_database", default="neo4j", help="Neo4j database name.")
    parser.add_argument(
        "--config_path",
        default="src/config/config.json",
        help="Config JSON path for model and Neo4j credentials.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="Neo4j ingestion chunk size.",
    )
    parser.add_argument(
        "--embed_nodes",
        action="store_true",
        help="Enable node embeddings during ingestion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    schema_path = Path(args.schema_path)
    output_dir = Path(args.output_dir)
    settings = load_settings(args.config_path)

    api_key = args.api_key or settings.API_KEY
    model = args.model or settings.MODEL_NAME
    api_base = args.api_base or settings.API_BASE

    output_path = run_extraction(
        input_dir=input_dir,
        schema_path=schema_path,
        output_dir=output_dir,
        api_key=api_key,
        model=model,
        api_base=api_base,
    )

    embedder = build_embedder(settings)

    run_graph_ingestion(
        payload_path=output_path,
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=args.neo4j_database,
        embedder=embedder,
        chunk_size=args.chunk_size,
        embed_nodes=args.embed_nodes,
    )

    print(f"Pipeline completed. Payload written to: {output_path}")


if __name__ == "__main__":
    main()
