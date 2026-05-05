from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

from src.indexing_pipeline.entity_extractor import SchemaDrivenExtractor, SchemaSpec, load_schema


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("json_builder")


def build_record_from_extraction(extracted: dict, schema: SchemaSpec) -> dict:
    entities_by_layer = extracted["entities_by_layer"]
    relations = extracted["relations"]
    root = extracted["document_root"]

    relation_lookup: Dict[tuple, List[str]] = {}
    for rel in relations:
        key = (rel["from_layer"], rel["type"], rel["from_id"])
        relation_lookup.setdefault(key, []).append(rel["to_id"])

    anatomy_by_id = {x["id"]: x for x in entities_by_layer.get("anatomy", [])}
    drug_by_id = {x["id"]: x for x in entities_by_layer.get("drug", [])}

    phenotypes_out = []
    for phenotype in entities_by_layer.get("phenotype", []):
        anatomy_ids = relation_lookup.get(("phenotype", "LOCATED_IN", phenotype["id"]), [])
        seen_anatomies = set()
        anatomies = []

        for anatomy_id in anatomy_ids:
            if anatomy_id in anatomy_by_id and anatomy_id not in seen_anatomies:
                anatomies.append({
                    "id": anatomy_by_id[anatomy_id]["id"],
                    "name": anatomy_by_id[anatomy_id]["name"],
                })
                seen_anatomies.add(anatomy_id)

        phenotypes_out.append({
            "id": phenotype["id"],
            "name": phenotype["name"],
            "anatomies": anatomies,
        })

    drug_ids = relation_lookup.get(("disease", "CURED_BY", root["id"]), [])
    seen_drugs = set()
    drugs_out = []

    for drug_id in drug_ids:
        if drug_id in drug_by_id and drug_id not in seen_drugs:
            drugs_out.append({
                "id": drug_by_id[drug_id]["id"],
                "name": drug_by_id[drug_id]["name"],
            })
            seen_drugs.add(drug_id)

    return {
        schema.root_layer: {
            "id": root["id"],
            "name": root["name"],
        },
        "phenotypes": phenotypes_out,
        "drugs": drugs_out,
    }


def build_final_output(records: List[dict], schema: SchemaSpec) -> Dict[str, Any]:
    return {
        "schema": {
            "layers": [{"index": x.index, "name": x.name} for x in schema.layers],
            "relations": [
                {"from": x.from_layer, "to": x.to_layer, "type": x.rel_type}
                for x in schema.relations
            ],
            "root_layer": schema.root_layer,
        },
        "records": records,
    }


def process_directory(
    input_dir: Path,
    schema: SchemaSpec,
    extractor: SchemaDrivenExtractor,
) -> Dict[str, Any]:
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {input_dir}")

    logger.info(f"Found {len(txt_files)} files to process")
    shaped_records: List[dict] = []
    failed_files: List[Dict[str, str]] = []

    for txt_file in tqdm(txt_files, desc="Processing files", unit="file"):
        logger.info(f"📄 Processing: {txt_file.name}")
        try:
            extracted = extractor.extract_file(schema=schema, txt_path=txt_file)
            shaped_record = build_record_from_extraction(extracted, schema)
            shaped_records.append(shaped_record)
            logger.info(f"✓ Successfully processed: {txt_file.name}")
        except Exception as exc:
            logger.error(f"❌ Failed processing {txt_file.name}: {str(exc)}", exc_info=True)
            failed_files.append({
                "file": txt_file.name,
                "error": str(exc),
            })

    logger.info(f"\n✅ Processed {len(shaped_records)} files successfully")
    if failed_files:
        logger.warning(f"⚠️  {len(failed_files)} files failed")
    
    if not shaped_records:
        raise RuntimeError("All files failed. No output was generated.")

    output = build_final_output(shaped_records, schema)

    if failed_files:
        output["failed_files"] = failed_files

    return output


def write_output(output_dir: Path, payload: Dict[str, Any], filename: str = "combined_graph_payload.json") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    logger.info(f"📝 Writing output to: {output_path}")
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    
    file_size_kb = output_path.stat().st_size / 1024
    logger.info(f"✓ Output written successfully")
    logger.info(f"   - File size: {file_size_kb:.2f} KB")
    logger.info(f"   - Records: {len(payload.get('records', []))}")
    if "failed_files" in payload:
        logger.info(f"   - Failed files: {len(payload['failed_files'])}")
    
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final schema + records JSON from text files using schema-driven LLM extraction."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing .txt files")
    parser.add_argument("--schema_path", required=True, help="Path to schema JSON")
    parser.add_argument("--output_dir", required=True, help="Directory for output JSON")
    parser.add_argument(
        "--api_key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key. If omitted, OPENAI_API_KEY env var is used.",
    )
    parser.add_argument("--model", default="gpt-5.2", help="Model name")
    parser.add_argument(
        "--api_base",
        default="https://api.forge.tensorblock.co/v1",
        help="API base URL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise ValueError("No API key provided. Use --api_key or set OPENAI_API_KEY.")

    input_dir = Path(args.input_dir)
    schema_path = Path(args.schema_path)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file does not exist: {schema_path}")

    schema = load_schema(schema_path)
    extractor = SchemaDrivenExtractor(
        api_key=args.api_key,
        model=args.model,
        base_url=args.api_base,
    )

    result = process_directory(
        input_dir=input_dir,
        schema=schema,
        extractor=extractor,
    )

    write_output(output_dir, result)
    logger.info("Finished successfully.")


if __name__ == "__main__":
    main()