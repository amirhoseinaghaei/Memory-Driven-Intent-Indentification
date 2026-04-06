


from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from openai import OpenAI


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LayerSpec:
    index: int
    name: str


@dataclass(frozen=True)
class RelationSpec:
    from_layer: str
    to_layer: str
    rel_type: str


@dataclass(frozen=True)
class SchemaSpec:
    layers: List[LayerSpec]
    relations: List[RelationSpec]
    root_layer: str


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def safe_json_loads(text: str) -> dict:
    raw = (text or "").strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    obj_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if obj_match:
        return json.loads(obj_match.group(0))

    raise ValueError("Could not parse model output as JSON.")


def normalize_id(raw_id: str, fallback_name: str) -> str:
    raw_id = (raw_id or "").strip()
    if raw_id:
        return raw_id
    return slugify(fallback_name)


# -----------------------------------------------------------------------------
# Schema loading
# -----------------------------------------------------------------------------
def load_schema(schema_path: Path) -> SchemaSpec:
    payload = json.loads(schema_path.read_text(encoding="utf-8"))

    if "layers" not in payload or not payload["layers"]:
        raise ValueError("Schema must contain a non-empty 'layers' field.")
    if "relations" not in payload:
        raise ValueError("Schema must contain a 'relations' field.")
    if "root_layer" not in payload or not payload["root_layer"]:
        raise ValueError("Schema must contain a 'root_layer' field.")

    layers = sorted(
        [LayerSpec(index=int(x["index"]), name=str(x["name"]).strip()) for x in payload["layers"]],
        key=lambda x: x.index,
    )

    relations = [
        RelationSpec(
            from_layer=str(x["from"]).strip(),
            to_layer=str(x["to"]).strip(),
            rel_type=str(x["type"]).strip(),
        )
        for x in payload["relations"]
    ]

    layer_names = {layer.name for layer in layers}

    if payload["root_layer"] not in layer_names:
        raise ValueError(f"root_layer '{payload['root_layer']}' is not present in schema layers.")

    for rel in relations:
        if rel.from_layer not in layer_names:
            raise ValueError(f"Relation from-layer '{rel.from_layer}' not found in schema.")
        if rel.to_layer not in layer_names:
            raise ValueError(f"Relation to-layer '{rel.to_layer}' not found in schema.")

    return SchemaSpec(
        layers=layers,
        relations=relations,
        root_layer=str(payload["root_layer"]).strip(),
    )


# -----------------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------------
def build_extraction_messages(schema: SchemaSpec, file_name: str, file_text: str) -> List[Dict[str, str]]:
    layers_desc = [{"index": layer.index, "name": layer.name} for layer in schema.layers]
    relations_desc = [
        {"from": rel.from_layer, "to": rel.to_layer, "type": rel.rel_type}
        for rel in schema.relations
    ]

    example_output = {
        "document_root": {
            "layer": schema.root_layer,
            "id": "<root_id>",
            "name": "<root_name>"
        },
        "entities_by_layer": {
            layer.name: [{"id": f"<{layer.name}_id>", "name": f"<{layer.name}_name>"}]
            for layer in schema.layers
        },
        "relations": [
            {
                "from_layer": rel.from_layer,
                "from_id": "<source_id>",
                "to_layer": rel.to_layer,
                "to_id": "<target_id>",
                "type": rel.rel_type
            }
            for rel in schema.relations
        ]
    }

    system_prompt = f"""
You are a schema-driven information extraction system.

Extract entities and relations from the document according to the schema below.

Schema layers:
{json.dumps(layers_desc, ensure_ascii=False, indent=2)}

Schema relations:
{json.dumps(relations_desc, ensure_ascii=False, indent=2)}

Root layer: "{schema.root_layer}"

Return STRICT JSON ONLY in exactly this structure:
{json.dumps(example_output, ensure_ascii=False, indent=2)}

Rules:
1. Extract only facts supported by the text.
2. Use only the given layers and relation types.
3. Every entity must appear under the correct layer in entities_by_layer.
4. Relations must only connect valid schema layer pairs.
5. Preserve ids from the text if they exist.
6. If no explicit id exists, generate a lowercase snake_case id.
7. Remove duplicates.
8. If a layer has no entities, return an empty list.
9. If the root is not explicit, infer the most likely root from the text or filename.
10. Return JSON only. No markdown. No explanation.
""".strip()

    user_prompt = f"""
File name: {file_name}

TEXT:
{file_text}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------
def normalize_extraction_output(payload: dict, schema: SchemaSpec, fallback_root_name: str) -> dict:
    layer_names = {layer.name for layer in schema.layers}
    allowed_relations = {(r.from_layer, r.to_layer, r.rel_type) for r in schema.relations}

    root_in = payload.get("document_root") or {}
    entities_in = payload.get("entities_by_layer") or {}
    relations_in = payload.get("relations") or []

    root_layer = root_in.get("layer") or schema.root_layer
    if root_layer not in layer_names:
        root_layer = schema.root_layer

    root_name = (root_in.get("name") or fallback_root_name).strip()
    root_id = normalize_id(root_in.get("id"), root_name)

    entities_out: Dict[str, List[Dict[str, str]]] = {layer.name: [] for layer in schema.layers}
    seen_entities: Dict[str, Set[str]] = {layer.name: set() for layer in schema.layers}

    for layer in schema.layers:
        for item in entities_in.get(layer.name, []) or []:
            name = (item.get("name") or "").strip()
            if not name:
                continue

            entity_id = normalize_id(item.get("id"), name)
            if entity_id in seen_entities[layer.name]:
                continue

            seen_entities[layer.name].add(entity_id)
            entities_out[layer.name].append({
                "id": entity_id,
                "name": name,
            })

    if root_id not in seen_entities[root_layer]:
        entities_out[root_layer].append({
            "id": root_id,
            "name": root_name,
        })
        seen_entities[root_layer].add(root_id)

    relations_out: List[Dict[str, str]] = []
    seen_relations: Set[Tuple[str, str, str, str, str]] = set()

    for rel in relations_in:
        from_layer = (rel.get("from_layer") or "").strip()
        to_layer = (rel.get("to_layer") or "").strip()
        rel_type = (rel.get("type") or "").strip()
        from_id = (rel.get("from_id") or "").strip()
        to_id = (rel.get("to_id") or "").strip()

        if (from_layer, to_layer, rel_type) not in allowed_relations:
            continue
        if not from_id or not to_id:
            continue
        if from_id not in seen_entities.get(from_layer, set()):
            continue
        if to_id not in seen_entities.get(to_layer, set()):
            continue

        key = (from_layer, from_id, to_layer, to_id, rel_type)
        if key in seen_relations:
            continue

        seen_relations.add(key)
        relations_out.append({
            "from_layer": from_layer,
            "from_id": from_id,
            "to_layer": to_layer,
            "to_id": to_id,
            "type": rel_type,
        })

    return {
        "document_root": {
            "layer": root_layer,
            "id": root_id,
            "name": root_name,
        },
        "entities_by_layer": entities_out,
        "relations": relations_out,
    }


# -----------------------------------------------------------------------------
# Extractor
# -----------------------------------------------------------------------------
class SchemaDrivenExtractor:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.2",
        base_url: str = "https://api.forge.tensorblock.co/v1",
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def extract_file(self, schema: SchemaSpec, txt_path: Path) -> dict:
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            raise ValueError(f"File is empty: {txt_path.name}")

        messages = build_extraction_messages(schema, txt_path.name, text)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )

        raw = (response.choices[0].message.content or "").strip()
        parsed = safe_json_loads(raw)

        return normalize_extraction_output(
            payload=parsed,
            schema=schema,
            fallback_root_name=txt_path.stem,
        ) 