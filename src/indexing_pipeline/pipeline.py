from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
from tqdm import tqdm
from datetime import datetime

from src.indexing_pipeline.graph_ingestor import GraphIngestor
from src.gen_ai_gateway.embedder import Embed
from src.indexing_pipeline.entity_extractor import SchemaDrivenExtractor, load_schema
from src.indexing_pipeline.parser import process_directory, write_output
from src.config.config import Settings, load_settings_from_json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('indexing_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
    logger.info("=" * 80)
    logger.info("STEP 1: ENTITY EXTRACTION")
    logger.info("=" * 80)
    
    try:
        logger.info(f"🔍 Loading schema from: {schema_path}")
        schema = load_schema(schema_path)
        logger.info(f"✓ Schema loaded successfully")
        logger.info(f"  - Layers: {[layer.name for layer in schema.layers]}")
        logger.info(f"  - Relations: {len(schema.relations)}")
        
        logger.info(f"🔑 Initializing extractor")
        logger.info(f"  - Model: {model}")
        logger.info(f"  - API Base: {api_base}")
        extractor = SchemaDrivenExtractor(
            api_key=api_key,
            model=model,
            base_url=api_base,
        )
        logger.info(f"✓ Extractor initialized")
        
        # Count input files
        input_files = list(input_dir.glob("*.txt"))
        logger.info(f"📁 Input directory: {input_dir}")
        logger.info(f"📄 Found {len(input_files)} files to process")
        if len(input_files) == 0:
            logger.warning(f"⚠️  No .txt files found in {input_dir}")
        
        for file in input_files[:5]:
            logger.info(f"   - {file.name}")
        if len(input_files) > 5:
            logger.info(f"   ... and {len(input_files) - 5} more")
        
        logger.info(f"⚙️  Processing files...")
        payload = process_directory(input_dir=input_dir, schema=schema, extractor=extractor)
        logger.info(f"✓ Extraction completed successfully")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"💾 Writing output to: {output_dir}")
        output_path = write_output(output_dir, payload)
        logger.info(f"✓ Output written to: {output_path}")
        logger.info(f"  - File size: {output_path.stat().st_size / 1024:.2f} KB")
        
        return output_path
    
    except Exception as e:
        logger.error(f"❌ Extraction failed: {str(e)}", exc_info=True)
        raise


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
    logger.info("=" * 80)
    logger.info("STEP 2: GRAPH INGESTION")
    logger.info("=" * 80)
    
    try:
        logger.info(f"📖 Loading payload from: {payload_path}")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        logger.info(f"✓ Payload loaded successfully")
        
        # Log payload stats
        if "entities_by_layer" in payload:
            logger.info(f"📊 Payload statistics:")
            for layer, entities in payload.get("entities_by_layer", {}).items():
                logger.info(f"   - {layer}: {len(entities)} entities")
        
        if "relations" in payload:
            logger.info(f"   - relations: {len(payload['relations'])} relations")
        
        logger.info(f"🗄️  Initializing Neo4j ingestor")
        logger.info(f"   - URI: {uri}")
        logger.info(f"   - Database: {database}")
        logger.info(f"   - Chunk size: {chunk_size}")
        logger.info(f"   - Embed nodes: {embed_nodes}")
        
        ingestor = GraphIngestor.from_payload(
            payload,
            uri=uri,
            user=user,
            password=password,
            database=database,
            embedder=embedder,
        )
        logger.info(f"✓ Neo4j ingestor initialized")
        
        logger.info(f"📤 Ingesting payload into Neo4j...")
        ingestor.ingest_payload(payload, chunk_size=chunk_size, embed_nodes=embed_nodes)
        logger.info(f"✓ Ingestion completed successfully")
        
        ingestor.close()
        logger.info(f"✓ Neo4j connection closed")
    
    except Exception as e:
        logger.error(f"❌ Graph ingestion failed: {str(e)}", exc_info=True)
        raise


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
    logger.info("🚀 " + "=" * 78)
    logger.info(f"🚀  INDEXING PIPELINE STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("🚀 " + "=" * 78)
    
    try:
        args = parse_args()

        input_dir = Path(args.input_dir)
        schema_path = Path(args.schema_path)
        output_dir = Path(args.output_dir)
        
        logger.info("⚙️  PIPELINE CONFIGURATION:")
        logger.info(f"   - Input directory: {input_dir}")
        logger.info(f"   - Schema path: {schema_path}")
        logger.info(f"   - Output directory: {output_dir}")
        logger.info(f"   - Config path: {args.config_path}")
        logger.info(f"   - Chunk size: {args.chunk_size}")
        logger.info(f"   - Embed nodes: {args.embed_nodes}")
        
        logger.info("📁 Loading configuration...")
        settings = load_settings(args.config_path)
        logger.info(f"✓ Configuration loaded")

        api_key = args.api_key or settings.API_KEY
        model = args.model or settings.MODEL_NAME
        api_base = args.api_base or settings.API_BASE
        
        logger.info(f"🔐 Using model: {model}")

        logger.info("\n")
        output_path = run_extraction(
            input_dir=input_dir,
            schema_path=schema_path,
            output_dir=output_dir,
            api_key=api_key,
            model=model,
            api_base=api_base,
        )

        logger.info("\n")
        logger.info("📦 Building embedder...")
        embedder = build_embedder(settings)
        logger.info(f"✓ Embedder initialized")

        logger.info("\n")
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

        logger.info("\n")
        logger.info("✅ " + "=" * 78)
        logger.info(f"✅  PIPELINE COMPLETED SUCCESSFULLY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("✅ " + "=" * 78)
        logger.info(f"📊 Output file: {output_path}")
        
    except Exception as e:
        logger.error("❌ " + "=" * 78)
        logger.error(f"❌  PIPELINE FAILED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.error("❌ " + "=" * 78)
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
