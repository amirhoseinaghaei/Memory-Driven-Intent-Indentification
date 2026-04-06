from src.data_models.graph_schema import GraphSchema
from typing import Any, Dict, Iterable, List, Optional, Protocol
from src.data_models.node_spec import NodeSpec
from src.data_models.edge_spec import EdgeSpec

class Mapper:
    def __init__(
        self,
        schema: GraphSchema,
        embedder
    ) -> None:
        self.schema = schema
        self.embedder = embedder

    @staticmethod
    def normalize_typed_id(expected_prefix: str, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError(f"Empty ID for prefix '{expected_prefix}'")
        if value.startswith(f"{expected_prefix}:"):
            return value
        return f"{expected_prefix}:{value}"

    def _make_node(
        self,
        node_id: str,
        label: str,
        name: str,
        *,
        embed_nodes: bool = True,
        properties: Optional[Dict[str, Any]] = None,
    ) -> NodeSpec:
        print(name)
        return NodeSpec(
            id=node_id,
            layer=self.schema.get_layer_index(label),
            label=label,
            data=name or node_id,
            embedding=self.embedder.embed_query(name) if embed_nodes else None,
            properties=properties or {},
        )

    def normalize(
        self,
        payload: Dict[str, Any],
        *,
        embed_nodes: bool = True,
        include_genes: bool = False,
        gene_relation_type: str = "ASSOCIATED_WITH",
    ) -> tuple[List[NodeSpec], List[EdgeSpec]]:
        records = payload.get("records", [])

        nodes_by_id: Dict[str, NodeSpec] = {}
        edges: List[EdgeSpec] = []

        def add_node(node_id: str, label: str, name: str, properties: Optional[Dict[str, Any]] = None):
            if not self.schema.has_layer(label):
                raise ValueError(f"Label '{label}' not found in schema")
            if node_id not in nodes_by_id:
                nodes_by_id[node_id] = self._make_node(
                    node_id=node_id,
                    label=label,
                    name=name,
                    embed_nodes=embed_nodes,
                    properties=properties,
                )

        def add_edge(
            from_label: str,
            from_id: str,
            to_label: str,
            to_id: str,
            properties: Optional[Dict[str, Any]] = None,
        ):
            rel_type = self.schema.get_relation_type(from_label, to_label)
            edges.append(
                EdgeSpec(
                    from_id=from_id,
                    to_id=to_id,
                    rel_type=rel_type,
                    properties=properties or {},
                )
            )

        root_label = self.schema.root_layer

        for record in records:
            disease = record.get(root_label, {}) or {}
            disease_id = self.normalize_typed_id(root_label, disease.get("id", ""))
            disease_name = (disease.get("name") or disease_id).strip()

            add_node(disease_id, root_label, disease_name)

            # disease -> drug
            for drug in record.get("drugs", []) or []:
                drug_id = self.normalize_typed_id("drug", drug.get("id", ""))
                drug_name = (drug.get("name") or drug_id).strip()

                add_node(drug_id, "drug", drug_name)
                add_edge(
                    root_label,
                    disease_id,
                    "drug",
                    drug_id,
                    {"weight": 1.0},
                )

            # phenotype -> anatomy and anatomy -> disease
            anatomy_counts: Dict[str, int] = {}
            phenotypes_with_any = 0

            for phenotype in record.get("phenotypes", []) or []:
                phenotype_id = self.normalize_typed_id("phenotype", phenotype.get("id", ""))
                phenotype_name = (phenotype.get("name") or phenotype_id).strip()

                add_node(phenotype_id, "phenotype", phenotype_name)

                anatomies = phenotype.get("anatomies", []) or []
                if anatomies:
                    phenotypes_with_any += 1

                for anatomy in anatomies:
                    anatomy_id = self.normalize_typed_id("anatomy", anatomy.get("id", ""))
                    anatomy_name = (anatomy.get("name") or anatomy_id).strip()

                    add_node(anatomy_id, "anatomy", anatomy_name)

                    add_edge(
                        "phenotype",
                        phenotype_id,
                        "anatomy",
                        anatomy_id,
                        {
                            "weight": 1.0,
                            "disease_id": disease_id,
                        },
                    )

                    anatomy_counts[anatomy_id] = anatomy_counts.get(anatomy_id, 0) + 1

            denom = max(1, phenotypes_with_any)
            for anatomy_id, count in anatomy_counts.items():
                add_edge(
                    "anatomy",
                    anatomy_id,
                    root_label,
                    disease_id,
                    {"weight": float(count) / float(denom)},
                )

            # optional genes
            if include_genes:
                if not self.schema.has_layer("gene"):
                    raise ValueError("include_genes=True but 'gene' is not defined in schema layers")

                for gene in record.get("genes", []) or []:
                    gene_id = self.normalize_typed_id("gene", gene.get("id", ""))
                    gene_name = (gene.get("name") or gene_id).strip()

                    add_node(gene_id, "gene", gene_name)
                    edges.append(
                        EdgeSpec(
                            from_id=gene_id,
                            to_id=disease_id,
                            rel_type=gene_relation_type,
                            properties={"weight": 1.0},
                        )
                    )

        return list(nodes_by_id.values()), edges