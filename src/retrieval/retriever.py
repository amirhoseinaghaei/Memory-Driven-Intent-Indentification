from __future__ import annotations

import ast
import json
import logging
import math
import time
from math import e, log
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Literal

import hdbscan
import numpy as np
import umap
from neo4j import GraphDatabase, Driver
from sklearn.preprocessing import normalize
from src.config.config import settings
from src.data_models.neo4j_conf import Neo4jConfig
from src.gen_ai_gateway.embedder import Embed
from src.gen_ai_gateway.chat_completion import ChatCompletion
from src.graph_comparison.weighted_coverage import weighted_coverage
from src.utils.helpers import (
    safe_parse_llm_json,
    _strip_embeddings,
    adjacency_dense,
    save_partial_and_complete,
    _to_nx_partial_graph,
    _to_nx_complete_graph,
)


import json
import math
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── tuneable ──────────────────────────────────────────────────────────────────
_SCORE_WORKERS  = 8   # threads for flgw scoring  (CPU-bound but releases GIL via numpy)
_MATCH_WORKERS  = 8   # threads for anatomy matching
_SAVE_WORKERS   = 4   # threads for fire-and-forget disk saves
# ─────────────────────────────────────────────────────────────────────────────


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)



# ── helper: build NER prompt (extracted to reduce method size) ─────────────────
def _build_ner_messages(query: str, ranked: list) -> list:
    organ_list = (
        "['uterine cervix', 'islet of Langerhans', 'pituitary gland', 'zone of skin', 'lymph node', "
        "'tendon', 'dorsal root ganglion', 'large intestine', 'renal glomerulus', 'metanephros', "
        "'adult mammalian kidney', 'intestine', 'blood', 'colonic mucosa', 'prefrontal cortex', "
        "'material anatomical entity', 'anatomical system', 'multi-cellular organism', 'testis', "
        "'female reproductive system', 'embryo', 'stomach', 'aorta', 'heart', 'brain', "
        "'cerebral cortex', 'female gonad', 'uterus', 'vagina', 'mammalian vulva', 'adipose tissue', "
        "'central nervous system', 'esophagus', 'saliva-secreting gland', 'skeletal muscle tissue', "
        "'smooth muscle tissue', 'caecum', 'vermiform appendix', 'colon', 'sigmoid colon', "
        "'fundus of stomach', 'cortex of kidney', 'nephron tubule', 'adrenal cortex', "
        "'urinary bladder', 'pancreas', 'endometrium', 'myometrium', 'tibial nerve', "
        "'quadriceps femoris', 'vastus lateralis', 'muscle of leg', 'deltoid', 'biceps brachii', "
        "'coronary artery', 'tongue', 'palpebral conjunctiva', 'nasal cavity mucosa', 'gingiva', "
        "'frontal cortex', 'temporal lobe', 'parietal lobe', 'caudate nucleus', 'putamen', "
        "'globus pallidus', 'amygdala', 'nucleus accumbens', 'forebrain', 'midbrain', 'telencephalon', "
        "'medulla oblongata', 'dorsal plus ventral thalamus', 'hypothalamus', 'mammary gland', "
        "'neocortex', \"Ammon's horn\", 'epithelium of esophagus', 'placenta', 'occipital lobe', "
        "'epithelium of bronchus', 'cerebellum', 'substantia nigra', 'thyroid gland', 'lung', "
        "'hair follicle', 'cardiac atrium', 'cardiac ventricle', 'heart left ventricle', 'spleen', "
        "'liver', 'small intestine', 'kidney', 'duodenum', 'jejunum', 'cerebellar cortex', 'bronchus', "
        "'subcutaneous adipose tissue', 'spinal cord', 'cerebellar hemisphere', 'corpus callosum', "
        "'myocardium', 'peritoneum', 'prostate gland', 'adrenal gland', 'thymus', 'tonsil', "
        "'connective tissue', 'muscle tissue', 'primary visual cortex', 'decidua', 'esophagus mucosa', "
        "'superior frontal gyrus', 'entorhinal cortex', 'cingulate cortex', 'trachea', "
        "'epithelium of mammary gland', 'mouth mucosa', 'fallopian tube', 'metanephric glomerulus', "
        "'cervix epithelium', 'oviduct epithelium', 'kidney epithelium', 'thoracic mammary gland', "
        "'nasal cavity epithelium', 'Brodmann (1909) area 46', 'squamous epithelium', "
        "'layer of synovial tissue', 'adipose tissue of abdominal region', "
        "'dorsolateral prefrontal cortex', 'anterior cingulate cortex', 'omental fat pad', "
        "'Brodmann (1909) area 9', 'muscle organ', 'amniotic fluid', 'bone marrow', "
        "'medial globus pallidus', 'cerebellar vermis', 'cartilage tissue', 'frontal lobe', 'eye', "
        "'nasopharynx', 'synovial joint', 'skeletal muscle organ', 'nerve', "
        "'peripheral nervous system', 'retina', 'bone element', 'lacrimal gland', 'artery', "
        "'pericardium', 'breast', 'rectum', 'craniocervical region', 'oral cavity', 'pineal body', "
        "'epididymis', 'lens of camera-type eye', 'larynx', 'gall bladder', 'vein', 'optic choroid']"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a medical Named Entity Recognizer (NER).\n"
                "Task: Extract symptoms mentioned or strongly implied by the user query, "
                "and list exactly 3 organs most specifically associated with each symptom.\n\n"
                "Matching:\n"
                "- The user may describe symptoms directly, indirectly or in lay language.\n"
                "- You MUST semantically match to the closest symptom names from the Symptom List.\n"
                "- Use meaning-based matching, not exact keywords.\n\n"
                "Hard rules:\n"
                "1) Choose ONLY from the Symptom List (never invent symptoms).\n"
                "2) Return minimum 3 and AT MOST 15 symptoms total.\n"
                "3) For each symptom, list EXACTLY 3 organs from organ list "
                "that are implicated by that symptom.\n"
                f"Symptom List (flat list):\n{ranked}\n\n"
                f"Organ List:\n{organ_list}\n\n"
                "OUTPUT FORMAT (JSON ONLY):\n"
                "{\"symptom name from list\": [\"organ1\", \"organ2\", \"organ3\"], ...}\n"
            ),
        },
        {"role": "user", "content": query},
    ]

from dataclasses import dataclass, field

@dataclass
class TokenCounter:
    llm_input_tokens:  int = 0
    llm_output_tokens: int = 0
    embed_tokens:      int = 0

    def add_llm(self, usage) -> None:
        """Pass response.usage directly from create_response()."""
        if usage is None:
            return
        self.llm_input_tokens  += getattr(usage, "prompt_tokens",     0) or \
                                   getattr(usage, "input_tokens",      0) or 0
        self.llm_output_tokens += getattr(usage, "completion_tokens",  0) or \
                                   getattr(usage, "output_tokens",     0) or 0

    def reset(self) -> None:
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.embed_tokens = 0

    def add_embed(self, texts: list[str]) -> None:
        """Estimate tokens for a list of strings (cache misses only)."""
        # ~0.75 words per token is a reasonable approximation
        for t in texts:
            self.embed_tokens += max(1, round(len((t or "").split()) / 0.75))

    def summary(self) -> dict:
        total = self.llm_input_tokens + self.llm_output_tokens + self.embed_tokens
        return {
            "llm_input_tokens":  self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "embed_tokens":      self.embed_tokens,
            "total_tokens":      total,
        }


MatchMode = Literal["ANY", "ALL"]

_PHENOTYPE_Z_CYPHER = """
MATCH (p:Entity {label:"phenotype"})
      -[:LOCATED_IN]->(a:Entity {label:"anatomy"})
      -[:AFFECTS]->(d:Entity {label:"disease"})
RETURN p.id AS node_id, count(DISTINCT d) AS Z
"""

_ANATOMY_Z_CYPHER = """
MATCH (a:Entity {label:"anatomy"})
      -[:AFFECTS]->(d:Entity {label:"disease"})
RETURN a.id AS node_id, count(DISTINCT d) AS Z
"""

class Retriever:
    def __init__(self, settings, chat, database: str = "neo4j") -> None:
        self._config = Neo4jConfig(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=database,
  
        )
        self.token_counter = TokenCounter()   # ← add this line
        self._driver: Driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        self.chat = chat
        self.embedder = Embed(settings)

        self._emb_cache: Dict[Any, Any] = {}
        self._candidate_emb_cache: Dict[str, np.ndarray] = {}
        self._global_Z_cache: Dict[str, int] = {}
        self._global_p_cache: Dict[str, float] = {}

        self._ph_candidates_cache = {
            "t": time.time(),
            "v": self._fetch_layer_candidates(
                layer=1,
                allowed_label="phenotype",
                field="embedding",
            ),
        }

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    @staticmethod
    def typed_id(kind: str, raw_id: str) -> str:
        raw_id = "" if raw_id is None else str(raw_id).strip()
        return f"{kind}:{raw_id}"

    def embed_cached(self, text: str):
        key = (text or "").strip().lower()
        if not key:
            return None
        if key in self._emb_cache:
            return self._emb_cache[key]
        v = self.embedder.embed_query(text)
        self.token_counter.add_embed([text])   # ← add this line

        self._emb_cache[key] = v
        return v

    # -------------------------
    # Candidate fetching
    # -------------------------
    def _fetch_layer_candidates(
        self, layer: int, allowed_label: str, field: str = "embedding"
    ) -> List[Dict[str, Any]]:
        cypher = f"""
        MATCH (n:Entity)
        WHERE n.layer = $layer AND n.label = $lbl AND n.embedding IS NOT NULL
        RETURN n.id AS id, n.data AS data, n.{field} AS embedding
        """
        with self._driver.session(database=self._config.database) as s:
            return [dict(r) for r in s.run(cypher, {"layer": int(layer), "lbl": allowed_label})]

    def _fetch_anatomy_candidates_for_phenotype_ids(
        self, phenotype_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not phenotype_ids:
            return {}

        cypher = """
        MATCH (p:Entity {label:"phenotype"})-[:LOCATED_IN]->(a:Entity {label:"anatomy"})
        WHERE p.id IN $ph_ids AND a.embedding IS NOT NULL
        RETURN p.id AS ph_id, a.id AS id, a.data AS data, a.embedding AS embedding
        """
        with self._driver.session(database=self._config.database) as s:
            rows = [dict(r) for r in s.run(cypher, {"ph_ids": phenotype_ids})]

        out: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            ph_id = r.pop("ph_id")
            out.setdefault(ph_id, []).append(r)
        return out

    def _parse_embedding(self, vec: Any) -> Optional[np.ndarray]:
        if vec is None:
            return None
        if isinstance(vec, np.ndarray):
            a = vec.astype(np.float32, copy=False)
        else:
            if isinstance(vec, str):
                try:
                    vec = json.loads(vec)
                except Exception:
                    return None
            try:
                a = np.asarray(vec, dtype=np.float32)
            except Exception:
                return None
        a = np.squeeze(a)
        if a.ndim != 1 or a.size == 0:
            return None
        if not np.all(np.isfinite(a)):
            return None
        return a

    def _get_candidate_embedding(self, cid: str, embedding: Any) -> Optional[np.ndarray]:
        if not cid:
            return None
        cached = self._candidate_emb_cache.get(cid)
        if cached is not None:
            return cached
        emb = self._parse_embedding(embedding)
        if emb is not None:
            self._candidate_emb_cache[cid] = emb
        return emb

    def _fetch_node_degree_maps(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        if getattr(self, "_node_degree_cache", None) is not None:
            return self._node_degree_cache["ph"], self._node_degree_cache["an"]

        ph_out_deg: Dict[str, int] = {}
        an_out_deg: Dict[str, int] = {}

        cypher_ph = """
        MATCH (p:Entity {label:"phenotype"})-[:LOCATED_IN]->(a:Entity {label:"anatomy"})
        RETURN p.id AS ph_id, count(DISTINCT a) AS deg
        """

        cypher_an = """
        MATCH (a:Entity {label:"anatomy"})-[:AFFECTS]->(d:Entity {label:"disease"})
        RETURN a.id AS an_id, count(DISTINCT d) AS deg
        """

        with self._driver.session(database=self._config.database) as s:
            for r in s.run(cypher_ph):
                ph_out_deg[r["ph_id"]] = int(r["deg"])
            for r in s.run(cypher_an):
                an_out_deg[r["an_id"]] = int(r["deg"])

        self._node_degree_cache = {"ph": ph_out_deg, "an": an_out_deg}
        return ph_out_deg, an_out_deg

    def _fetch_candidate_disease_ids(
        self,
        ph_an_groups: List[Dict[str, Any]],
        candidate_limit: int,
    ) -> List[str]:
        cypher = """
        WITH $groups AS groups
        UNWIND groups AS g
        WITH g, coalesce(g.an_ids, []) AS an_ids

        MATCH (p:Entity {id: g.ph_id, label:"phenotype"})
                -[pa:LOCATED_IN]->(a:Entity {label:"anatomy"})
                -[:AFFECTS]->(d:Entity {label:"disease"})
        WHERE (size(an_ids) = 0 OR a.id IN an_ids)
            AND pa.disease_id = d.id

        RETURN DISTINCT d.id AS did
        LIMIT $candidate_limit
        """

        with self._driver.session(database=self._config.database) as s:
            return [r["did"] for r in s.run(cypher, {"groups": ph_an_groups, "candidate_limit": int(candidate_limit)})]

    def _fetch_graph_rows_for_diseases(
        self,
        disease_ids: List[str],
        ph_an_groups: List[Dict[str, Any]],
        include_parent_of: bool,
        max_full_pairs: int,
    ) -> List[Dict[str, Any]]:
        if not disease_ids:
            return []

        cypher = """
        WITH
        $disease_ids AS disease_ids,
        $groups AS groups,
        $include_parent_of AS include_po

        UNWIND disease_ids AS did
        MATCH (d:Entity {id: did, label:"disease"})

        OPTIONAL MATCH (d)-[:AFFECTS]-(a:Entity {label:"anatomy"})
        WITH d, include_po, groups, collect(DISTINCT a { .id, .label, .layer, .data }) AS full_anatomies

        CALL {
        WITH d, groups
        UNWIND groups AS g
        MATCH (p:Entity {id: g.ph_id, label:"phenotype"})
                -[:LOCATED_IN {disease_id: d.id}]->(a:Entity {label:"anatomy"})

        RETURN collect(DISTINCT {
            ph: p { .id, .label, .layer, .data },
            an: a { .id, .label, .layer, .data }
        }) AS partial_pairs
        }

        CALL {
        WITH d
        MATCH (p:Entity {label:"phenotype"})
                -[:LOCATED_IN {disease_id: d.id}]->(a:Entity {label:"anatomy"})

        WITH DISTINCT p, a
        LIMIT $max_full_pairs

        RETURN collect(DISTINCT {
            ph: p { .id, .label, .layer, .data },
            an: a { .id, .label, .layer, .data }
        }) AS full_pairs,
        collect(DISTINCT p { .id, .label, .layer, .data }) AS full_phenotypes
        }

        CALL {
        WITH include_po, d
        WITH d WHERE include_po = true
        OPTIONAL MATCH (p:Entity {label:"disease"})-[rp:PARENT_OF]->(d)
        RETURN collect(DISTINCT {
            parent: p { .id, .label, .layer, .data },
            rel: { weight: 1.0 }
        }) AS full_parents
        }

        CALL {
        WITH include_po, d
        WITH d WHERE include_po = true
        OPTIONAL MATCH (d)-[rc:PARENT_OF]->(c:Entity {label:"disease"})
        RETURN collect(DISTINCT {
            child: c { .id, .label, .layer, .data },
            rel: { weight: 1.0 }
        }) AS full_children
        }

        RETURN
        d { .id, .label, .layer, .data } AS disease,
        partial_pairs,
        full_pairs,
        full_phenotypes,
        full_anatomies,
        coalesce(full_parents, []) AS full_parents,
        coalesce(full_children, []) AS full_children
        """

        params = {
            "disease_ids": disease_ids,
            "groups": ph_an_groups,
            "include_parent_of": include_parent_of,
            "max_full_pairs": int(max_full_pairs),
        }
        with self._driver.session(database=self._config.database) as s:
            return [dict(r) for r in s.run(cypher, params)]

    # -------------------------
    # Matching
    # -------------------------
    def _match_inputs_to_candidates(
        self,
        inputs: List[str],
        embedder,
        candidates: List[Dict[str, Any]],
        sim_threshold: float = 0.9,
        top_k: int = 1,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
        from collections import Counter

        def _to_1d_float32(vec: Any) -> Optional[np.ndarray]:
            if vec is None:
                return None
            if isinstance(vec, np.ndarray):
                a = vec.astype(np.float32, copy=False)
            else:
                if isinstance(vec, str):
                    try:
                        vec = json.loads(vec)
                    except Exception:
                        return None
                try:
                    a = np.asarray(vec, dtype=np.float32)
                except Exception:
                    return None
            a = np.squeeze(a)
            if a.ndim != 1 or a.size == 0:
                return None
            if not np.all(np.isfinite(a)):
                return None
            return a

        uniq: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            cid = c.get("id")
            if not cid or c.get("embedding") is None:
                continue
            if cid not in uniq:
                uniq[cid] = c
        candidates = list(uniq.values())

        if not inputs:
            return {}, []
        if not candidates:
            return {x: [] for x in inputs}, list(inputs)

        seen: Set[str] = set()
        uniq_inputs: List[str] = []
        for x in inputs:
            if x and x not in seen:
                seen.add(x)
                uniq_inputs.append(x)

        if not hasattr(self, "_emb_cache") or self._emb_cache is None:
            self._emb_cache = {}

        q_vecs: Dict[str, Optional[np.ndarray]] = {}
        missing: List[str] = []
        for x in uniq_inputs:
            v = self._emb_cache.get(("q", x))
            if v is None:
                missing.append(x)
            else:
                vv = _to_1d_float32(v)
                q_vecs[x] = vv
                self._emb_cache[("q", x)] = vv

        if missing:
            new_vecs: List[Optional[np.ndarray]] = [None] * len(missing)
            self.token_counter.add_embed(missing)   # ← count only cache misses
            try:
                if hasattr(embedder, "encode"):
                    arr = np.asarray(embedder.encode(missing, normalize=True), dtype=np.float32)
                    for i in range(arr.shape[0]):
                        new_vecs[i] = _to_1d_float32(arr[i])
                elif hasattr(embedder, "embed_documents"):
                    arr = np.asarray(embedder.embed_documents(missing), dtype=np.float32)
                    for i in range(arr.shape[0]):
                        new_vecs[i] = _to_1d_float32(arr[i])
                else:
                    for i, x in enumerate(missing):
                        new_vecs[i] = _to_1d_float32(embedder.embed_query(x))
            except Exception:
                for i, x in enumerate(missing):
                    try:
                        new_vecs[i] = _to_1d_float32(embedder.embed_query(x))
                    except Exception:
                        new_vecs[i] = None

            for x, v in zip(missing, new_vecs):
                if v is None or v.size == 0:
                    q_vecs[x] = None
                    self._emb_cache[("q", x)] = None
                    continue
                v = v.astype(np.float32, copy=False)
                n = float(np.linalg.norm(v))
                if n <= 1e-12 or not np.isfinite(n):
                    q_vecs[x] = None
                    self._emb_cache[("q", x)] = None
                else:
                    v = v / n
                    q_vecs[x] = v
                    self._emb_cache[("q", x)] = v

        cand_rows: List[Tuple[str, Any, np.ndarray]] = []
        dims: List[int] = []
        for c in candidates:
            cid = c.get("id")
            if not cid:
                continue
            v = self._candidate_emb_cache.get(cid)
            if v is None:
                v = self._get_candidate_embedding(cid, c.get("embedding"))
            if v is None:
                continue
            cand_rows.append((cid, c.get("data"), v))
            dims.append(int(v.shape[0]))

        if not cand_rows:
            return {x: [] for x in inputs}, list(inputs)

        target_dim, _ = Counter(dims).most_common(1)[0]
        filtered_rows = [(cid, d, v) for (cid, d, v) in cand_rows if v.shape[0] == target_dim]
        if not filtered_rows:
            return {x: [] for x in inputs}, list(inputs)

        ids = [r[0] for r in filtered_rows]
        data = [r[1] for r in filtered_rows]
        E = np.vstack([r[2] for r in filtered_rows]).astype(np.float32, copy=False)
        E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)

        valid_qs: List[str] = []
        for q in uniq_inputs:
            v = q_vecs.get(q)
            if v is None:
                continue
            v = np.squeeze(np.asarray(v, dtype=np.float32))
            if v.ndim != 1 or v.shape[0] != E.shape[1]:
                q_vecs[q] = None
                self._emb_cache[("q", q)] = None
                continue
            valid_qs.append(q)

        matched: Dict[str, List[Dict[str, Any]]] = {x: [] for x in inputs}
        unmatched_set: Set[str] = set()

        if not valid_qs:
            return {x: [] for x in inputs}, list(inputs)

        V = np.stack([q_vecs[q] for q in valid_qs], axis=1).astype(np.float32, copy=False)
        scores_mat = E @ V

        for j, q in enumerate(valid_qs):
            scores = scores_mat[:, j]
            idx = np.flatnonzero(scores >= sim_threshold)
            if idx.size == 0:
                unmatched_set.add(q)
                continue
            if idx.size > top_k:
                top_local = np.argpartition(scores[idx], -top_k)[-top_k:]
                top_idx = idx[top_local]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            else:
                top_idx = idx[np.argsort(scores[idx])[::-1]]

            matched[q] = [
                {"id": ids[i], "data": data[i], "score": float(scores[i])}
                for i in top_idx[:top_k]
            ]

        for q in uniq_inputs:
            if q_vecs.get(q) is None:
                unmatched_set.add(q)

        out_matched: Dict[str, List[Dict[str, Any]]] = {}
        out_unmatched: List[str] = []
        for x in inputs:
            hits = matched.get(x, [])
            out_matched[x] = hits
            if x in unmatched_set and not hits:
                out_unmatched.append(x)

        return out_matched, out_unmatched



    def fetch_partial_and_complete_graphs_by_groups(
    self,
    ph_an_groups: List[Dict[str, Any]],
    phenotype_mode: MatchMode = "ANY",
    anatomy_mode: MatchMode = "ANY",
    include_parent_of: bool = False,
    limit: int = 50,
    max_full_pairs: int = 2000,
) -> List[Dict[str, Any]]:
        ph_an_groups = ph_an_groups or []
        if not ph_an_groups:
            return []

        ph_out_deg, anatomy_out_deg = self._fetch_node_degree_maps()
        disease_ids = self._fetch_candidate_disease_ids(ph_an_groups, limit)
        if not disease_ids:
            return []

        rows = self._fetch_graph_rows_for_diseases(
            disease_ids=disease_ids,
            ph_an_groups=ph_an_groups,
            include_parent_of=include_parent_of,
            max_full_pairs=max_full_pairs,
        )

        def _safe_weight(degree: int) -> float:
            return 1.0 / degree if degree > 0 else 0.0

        def _annotate_pair(pair: Dict[str, Any]) -> Dict[str, Any]:
            ph = pair.get("ph", {})
            an = pair.get("an", {})
            return {
                "ph": ph,
                "an": an,
                "rel_pa": {"weight": _safe_weight(ph_out_deg.get(ph.get("id", ""), 0))},
                "rel_ad": {"weight": _safe_weight(anatomy_out_deg.get(an.get("id", ""), 0))},
            }

        out: List[Dict[str, Any]] = []
        for r in rows:
            full_anatomies = [
                {
                    "an": an,
                    "rel": {"weight": _safe_weight(anatomy_out_deg.get(an.get("id", ""), 0))},
                }
                for an in (r.get("full_anatomies") or [])
            ]

            out.append({
                "disease": r.get("disease", {}),
                "partial_graph": {"pairs": [_annotate_pair(p) for p in (r.get("partial_pairs") or [])]},
                "complete_graph": {
                    "pairs": [_annotate_pair(p) for p in (r.get("full_pairs") or [])],
                    "phenotypes": r.get("full_phenotypes") or [],
                    "anatomies": full_anatomies,
                    "parents": r.get("full_parents") or [],
                    "children": r.get("full_children") or [],
                },
            })

        return out

    
    def build_clusters(
        self,
        *,
        n_neighbors: int = 30,
        n_components: int = 10,
        umap_metric: str = "cosine",
        random_state: int = 0,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        hdbscan_metric: str = "euclidean",
        skip_noise: bool = False,
        noise_sim_threshold: float = 0.55,
        export_html_path: str | None = None,
    ) -> Dict[str, Any]:
        ph_candidates = self._fetch_layer_candidates(
            layer=1,
            allowed_label="phenotype",
            field="embedding",
        )
        ph_candidates = sorted(ph_candidates, key=lambda p: p.get("id", ""))

        ids = []
        X_list = []
        for p in ph_candidates:
            emb = p.get("embedding", None)
            if emb:
                ids.append(p["id"])
                if str(len(emb)) != "3072":
                    emb = 3072 * [0]
                X_list.append(emb)

        if len(X_list) == 0:
            raise ValueError("No phenotype embeddings found.")

        X = normalize(np.asarray(X_list, dtype=np.float64), norm="l2")

        X2 = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=umap_metric,
            random_state=random_state,
        ).fit_transform(X)

        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=hdbscan_metric,
        ).fit_predict(X2)

        unique_labels = sorted(set(labels))
        non_noise_labels = [l for l in unique_labels if l != -1]

        if len(non_noise_labels) == 0:
            raise ValueError("No clusters found.")

        centroids = {}
        for lab in non_noise_labels:
            idx = np.where(labels == lab)[0]
            mean_emb = X[idx].mean(axis=0)
            centroids[lab] = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)

        noise_indices = np.where(labels == -1)[0]
        if len(noise_indices) > 0:
            centroid_matrix = np.stack([centroids[l] for l in non_noise_labels])
            centroid_labels = np.array(non_noise_labels)
            for i in noise_indices:
                sims = centroid_matrix @ X[i]
                best_idx = np.argmax(sims)
                if sims[best_idx] >= noise_sim_threshold or not skip_noise:
                    labels[i] = int(centroid_labels[best_idx])

        clusters = {}
        for lab in sorted(set(labels)):
            if skip_noise and lab == -1:
                continue
            idx = np.where(labels == lab)[0]
            if len(idx) == 0:
                continue
            mean_emb = X[idx].mean(axis=0)
            clusters[int(lab)] = {
                "size": int(len(idx)),
                "mean_embedding": mean_emb / (np.linalg.norm(mean_emb) + 1e-12),
                "member_indices": idx,
            }

        self.ids_filtered = ids
        self.X_normalized = X
        self.X2_umap = X2
        self.labels = labels
        self.clusters = clusters

        self.cluster_labels = np.asarray(sorted(clusters.keys()), dtype=np.int64)
        self.cluster_mean_matrix = np.vstack(
            [clusters[lab]["mean_embedding"] for lab in self.cluster_labels]
        ).astype(np.float32, copy=False)
        self.cluster_member_indices = [
            np.asarray(clusters[lab]["member_indices"], dtype=np.int64)
            for lab in self.cluster_labels
        ]

        return {
            "ids_filtered": ids,
            "X_normalized": X,
            "X2_umap": X2,
            "labels": labels,
            "clusters": clusters,
        }

# import numpy as np
# 
    def rank_clusters_and_get_top_symptoms(
        self,
        ids_filtered,
        X_normalized,
        clusters,
        query_text,
        embed_fn,
        ph_id_to_name=None,
        top_k_clusters=4,
        top_k_items_per_cluster=8,
    ):
        ph_id_to_name = ph_id_to_name or {}

        # query embedding
        q = np.asarray(embed_fn(query_text), dtype=np.float64).reshape(-1)
        q /= (np.linalg.norm(q) + 1e-12)

        if not clusters:
            return []

        # ---- score cluster means in a vectorized way ----
        labels = list(clusters.keys())
        mean_matrix = np.vstack([clusters[lab]["mean_embedding"] for lab in labels])  # [C, D]
        cluster_scores = mean_matrix @ q                                              # [C]

        k_clusters = min(top_k_clusters, len(labels))
        top_cluster_pos = np.argpartition(-cluster_scores, k_clusters - 1)[:k_clusters]
        top_cluster_pos = top_cluster_pos[np.argsort(-cluster_scores[top_cluster_pos])]
        top_labels = [labels[pos] for pos in top_cluster_pos]

        # ---- collect all member indices from top clusters ----
        selected_indices = np.concatenate(
            [np.asarray(clusters[lab]["member_indices"], dtype=np.int64) for lab in top_labels]
        )

        if selected_indices.size == 0:
            return []

        # ---- score all selected items at once ----
        item_scores = X_normalized[selected_indices] @ q

        k_items = min(top_k_clusters * top_k_items_per_cluster, selected_indices.size)
        top_item_pos = np.argpartition(-item_scores, k_items - 1)[:k_items]
        top_item_pos = top_item_pos[np.argsort(-item_scores[top_item_pos])]

        final_indices = selected_indices[top_item_pos]

        return [
            ph_id_to_name.get(ids_filtered[i], ids_filtered[i])
            for i in final_indices
        ]

    def _get_phenotype_catalog(self) -> dict:
        """
        Load phenotype_catalog.json exactly once and cache it on the instance.
        All subsequent calls return the cached dict instantly.
        """
        if getattr(self, "_phenotype_catalog", None) is None:
            path = "dataset/phenotype_catalog.json"
            with open(path, "r", encoding="utf-8") as fh:
                self._phenotype_catalog = json.load(fh)
            logger.debug("retrieve_partial_graphs: phenotype catalog loaded from disk")
        return self._phenotype_catalog

    def retrieve_partial_graphs(
        self,
        query: str,
        previous_groups: list,
        previous_diseases: list,
        sim_threshold_ph: float = 0.8,
        sim_threshold_an: float = 0.8,
        top_k_ph: int = 1,
        top_k_an_per_ph: int = 1,
        phenotype_mode="ANY",
        include_parent_of: bool = False,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:

        _EMPTY = [], [], {}, [], [], [], None, None

        try:
            t0 = time.perf_counter()
            # ── 1. catalog ─────────────────────────────────────────────
            catalog = self._get_phenotype_catalog()
            self.token_counter.reset()

            try:
                ranked = self.rank_clusters_and_get_top_symptoms(
                    ids_filtered=self.ids_filtered,
                    X_normalized=self.X_normalized,
                    clusters=self.clusters,
                    query_text=query,
                    embed_fn=self.embedder.embed_query,
                    ph_id_to_name=catalog,
                    top_k_clusters=10,
                )
            except Exception as e:
                print(e)
            # ── 3. LLM parsing ─────────────────────────────────────────
            messages = _build_ner_messages(query, ranked)
            response = self.chat.create_response(message=messages)
            self.token_counter.add_llm(getattr(response, "usage", None))
            raw = (response.choices[0].message.content or "").strip()
            ph_to_an_inputs = safe_parse_llm_json(raw) or {}
            phenotype_texts = [k for k in ph_to_an_inputs if k]

            if not phenotype_texts:
                return _EMPTY

            # ── 4. phenotype candidates (cached) ───────────────────────
            ttl_sec = 3600
            now_t = time.time()
            cache = getattr(self, "_ph_candidates_cache", None)

            if cache and (now_t - cache["t"] < ttl_sec):
                ph_candidates = cache["v"]
            else:
                ph_candidates = self._fetch_layer_candidates(
                    layer=1, allowed_label="phenotype", field="embedding"
                )
                self._ph_candidates_cache = {"t": now_t, "v": ph_candidates}

            # ── 5. phenotype matching ──────────────────────────────────
            matched_ph_top, _ = self._match_inputs_to_candidates(
                phenotype_texts,
                self.embedder,
                ph_candidates,
                sim_threshold=sim_threshold_ph,
                top_k=top_k_ph,
            )

            if not matched_ph_top:
                return _EMPTY

            # ── 6. anatomy candidates ──────────────────────────────────
            ph_ids = [
                hit["id"]
                for hits in matched_ph_top.values()
                for hit in (hits or [])[:2]
                if hit.get("id")
            ]

            an_cand_map = self._fetch_anatomy_candidates_for_phenotype_ids(ph_ids)

            # ── 7. anatomy matching (parallel) ─────────────────────────
            tasks = [
                (ph_text, ph_hit)
                for ph_text, ph_hits in matched_ph_top.items()
                for ph_hit in (ph_hits or [])
                if ph_hit.get("id")
            ]

            def _match_one(ph_text, ph_hit):
                ph_id = ph_hit["id"]
                anatomy_texts = [x for x in (ph_to_an_inputs.get(ph_text) or []) if x]
                an_candidates = an_cand_map.get(ph_id, [])

                if anatomy_texts and an_candidates:
                    matched_an, _ = self._match_inputs_to_candidates(
                        anatomy_texts,
                        self.embedder,
                        an_candidates,
                        sim_threshold=sim_threshold_an,
                        top_k=top_k_an_per_ph,
                    )
                else:
                    matched_an = {}

                an_ids = tuple(sorted(
                    h["id"]
                    for hits in matched_an.values()
                    for h in hits
                    if h.get("id")
                ))

                return ph_id, an_ids

            an_results = {}
            with ThreadPoolExecutor(max_workers=_MATCH_WORKERS) as pool:
                for ph_id, an_ids in pool.map(lambda x: _match_one(*x), tasks):
                    an_results.setdefault(ph_id, an_ids)

            # ── 8. assemble groups ─────────────────────────────────────
            groups = []
            seen = set()

            for ph_id, an_ids in an_results.items():
                key = (ph_id, an_ids)
                if key not in seen:
                    seen.add(key)
                    groups.append({"ph_id": ph_id, "an_ids": list(an_ids)})

            if not groups:
                return _EMPTY

            current_ph = [g["ph_id"] for g in groups]
            current_an = list({a for g in groups for a in g["an_ids"]})

            # ── 9. fetch subgraphs ─────────────────────────────────────
            all_groups = previous_groups + groups
            subgraphs = self.fetch_partial_and_complete_graphs_by_groups(
                ph_an_groups=all_groups,
                phenotype_mode=phenotype_mode,
                anatomy_mode="ANY",
                include_parent_of=include_parent_of,
                limit=limit,
            )

            # ── 10. filter ─────────────────────────────────────────────
            _set_ph = set(current_ph)
            ph_required = max(1, math.ceil(0.2 * len(current_ph)))

            nx_pairs = []
            for sg in subgraphs:
                pairs = sg["partial_graph"].get("pairs", [])
                sg_ph = {p["ph"]["id"] for p in pairs if p.get("ph") and p["ph"].get("id")}

                if len(sg_ph & _set_ph) >= ph_required:
                    nx_pairs.append({
                        "disease": sg.get("disease") or {},
                        "partial": _to_nx_partial_graph(sg),
                        "complete": _to_nx_complete_graph(sg),
                    })

            disease_in_nx_pairs = [item["disease"]["id"] for item in nx_pairs]
            _strip_embeddings(subgraphs)

            # ── 11. scoring (parallel) ─────────────────────────────────
            def _score_one(item):
                PN, _, PA_ = adjacency_dense(item["partial"])
                CN, _, CA_ = adjacency_dense(item["complete"])

                score = weighted_coverage(
                    CA_, PA_,
                    complete_node_ids=CN,
                    partial_node_ids=PN,
                )

                return {
                    "score": score,
                    "id": item["disease"]["id"],
                    "partial_graph": item["partial"],
                    "complete_graph": item["complete"],
                }

            with ThreadPoolExecutor(max_workers=_SCORE_WORKERS) as pool:
                unranked_result = list(pool.map(_score_one, nx_pairs))

            # ── 12. sort ───────────────────────────────────────────────
            sorted_list = sorted(unranked_result, key=lambda x: x["score"], reverse=True)
            sorted_diseases = [item["id"] for item in sorted_list]
            time_taken = time.perf_counter() - t0
            return (
                sorted_list,
                ranked,
                phenotype_texts,
                disease_in_nx_pairs,
                all_groups,
                sorted_diseases,
                self.token_counter.summary(),
                time_taken,
            )

        except Exception:
            return [], [], {}, [], [], [], None, None





