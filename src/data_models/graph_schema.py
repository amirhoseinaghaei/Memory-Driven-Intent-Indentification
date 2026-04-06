from dataclasses import dataclass, field
from typing import Set, Any, Dict, List, Optional, Tuple
from src.data_models.layer_def import LayerDef
from src.data_models.relation_def import RelationDef

class GraphSchema:
    def __init__(
        self,
        layers: Dict[str, LayerDef],
        relations: Dict[tuple[str, str], RelationDef],
        root_layer: str,
    ) -> None:
        self.layers = layers
        self.relations = relations
        self.root_layer = root_layer

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GraphSchema":
        layers = {
            item["name"]: LayerDef(
                index=int(item["index"]),
                name=item["name"],
            )
            for item in payload.get("layers", [])
        }

        relations = {}
        for item in payload.get("relations", []):
            rel = RelationDef(
                source=item["from"],
                target=item["to"],
                rel_type=item["type"],
            )
            relations[(rel.source, rel.target)] = rel

        root_layer = payload.get("root_layer")
        if root_layer not in layers:
            raise ValueError(f"root_layer '{root_layer}' not found in schema layers")

        return cls(
            layers=layers,
            relations=relations,
            root_layer=root_layer,
        )

    def get_layer_index(self, label: str) -> int:
        if label not in self.layers:
            raise ValueError(f"Unknown label '{label}' in schema layers")
        return self.layers[label].index

    def get_relation_type(self, source: str, target: str) -> str:
        rel = self.relations.get((source, target))
        if rel is None:
            raise ValueError(f"No schema relation defined for {source} -> {target}")
        return rel.rel_type

    def has_layer(self, label: str) -> bool:
        return label in self.layers