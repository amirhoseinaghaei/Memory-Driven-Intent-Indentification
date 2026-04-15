from __future__ import annotations
import hashlib
import math
from typing import Set, Any, Dict, List, Optional, Tuple
import re
import json
import os
import re
from pathlib import Path
import networkx as nx
import numpy as np
from pyvis.network import Network




def norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text





def safe_parse_llm_json(text: str) -> Dict[str, Any]:
    """
    Safely parse an LLM response that *should* be JSON.
    Handles cases where the model wraps JSON with extra text.
    Returns {} if parsing fails.
    """
    if not text:
        return {}

    s = text.strip()

    # 1) direct JSON parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # 2) try extracting first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        candidate = s[start : end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}


def _strip_embeddings(obj: Any) -> Any:
    """Recursively remove 'embedding' keys from nested dict/list structures."""
    if isinstance(obj, dict):
        return {k: _strip_embeddings(v) for k, v in obj.items() if k != "embedding"}
    if isinstance(obj, list):
        return [_strip_embeddings(x) for x in obj]
    return obj


def embed_partial_into_complete(
    PA: np.ndarray,
    PIX: Dict[str, int],   # partial node -> idx in PA
    CIX: Dict[str, int],   # complete node -> idx in full matrix
    dtype=np.float64
) -> np.ndarray:
    """Your original function - it's correct!"""
    nC = len(CIX)
    PA_big = np.zeros((nC, nC), dtype=dtype)

    # partial nodes that also exist in complete
    common_nodes = [n for n in PIX if n in CIX]

    # indices in partial and complete
    p = np.array([PIX[n] for n in common_nodes], dtype=int)
    c = np.array([CIX[n] for n in common_nodes], dtype=int)

    # embed partial adjacency into complete coordinate system
    PA_big[np.ix_(c, c)] = PA[np.ix_(p, p)]

    return PA_big



def adjacency_dense(G: nx.DiGraph) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    nodes = list(G.nodes())           # row/column order
    idx = {n: i for i, n in enumerate(nodes)}

    A = nx.to_numpy_array(
        G,
        nodelist=nodes,
        weight="weight",
        dtype=float
    )

    return nodes, idx, A


def save_graph_html(g: nx.DiGraph, title: str, out_path: str) -> None:
    """Save NetworkX graph as interactive HTML (zoomable, draggable)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color="black",
    )
    net.heading = title

    # nodes
    for node_id, attrs in g.nodes(data=True):
        label = attrs.get("label") or node_id
        kind = attrs.get("kind", "")

        if kind == "phenotype":
            color = "#AED6F1"
        elif kind == "anatomy":
            color = "#A9DFBF"
        elif kind == "disease":
            color = "#F9E79F"
        else:
            color = "#D7DBDD"

        net.add_node(
            node_id,
            label=label,
            title=f"{node_id}<br>{kind}",
            color=color,
        )

    # edges
    for src, dst, attrs in g.edges(data=True):
        rel = attrs.get("rel", "")
        weight = attrs.get("weight", 1.0)

        try:
            wf = float(weight)
        except Exception:
            wf = 1.0

        net.add_edge(
            src,
            dst,
            label=str(rel),
            title=f"{rel} (w={wf:.10f})",
            value=wf,
        )

    net.set_options(
        """
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -30000,
              "springLength": 120
            },
            "minVelocity": 0.75
          }
        }
        """
    )

    net.write_html(out_path)


def save_partial_and_complete(result: dict, out_dir: str = "graphs_test2") -> None:
    """
    result format:
    {
        "disease": {"id": "...", "data": "..."}   (optional)
        "partial": nx.DiGraph,
        "complete": nx.DiGraph
    }
    """
    disease = result.get("disease") or {}
    disease_id = disease.get("id", "unknown_disease")
    disease_name = disease.get("data", disease_id)

    safe_id = disease_id.replace(":", "_").replace("/", "_")

    partial_g: nx.DiGraph = result.get("partial")
    complete_g: nx.DiGraph = result.get("complete")

    if partial_g is None or complete_g is None:
        raise ValueError("Missing partial or complete graph in result")

    os.makedirs(out_dir, exist_ok=True)

    save_graph_html(
        partial_g,
        title=f"Partial Graph – {disease_name}",
        out_path=str(Path(out_dir) / f"{safe_id[0:40]}_partial.html"),
    )
    save_graph_html(
        complete_g,
        title=f"Complete Graph – {disease_name}",
        out_path=str(Path(out_dir) / f"{safe_id[0:40]}_complete.html"),
    )


def _to_nx_partial_graph(sg: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()

    d = sg.get("disease") or {}
    did = d.get("id") or "disease"
    dlabel = d.get("data") or did
    G.add_node(did, kind="disease", label=dlabel, layer=3)

    pairs = (sg.get("partial_graph") or {}).get("pairs", []) or []
    for item in pairs:
        ph = item.get("ph") or {}
        an = item.get("an") or {}

        ph_id = ph.get("id")
        an_id = an.get("id")
        if not ph_id or not an_id:
            continue

        G.add_node(ph_id, kind="phenotype", label=ph.get("data") or ph_id, layer=1)
        G.add_node(an_id, kind="anatomy", label=an.get("data") or an_id, layer=2)

        w_pa = float((item.get("rel_pa") or {}).get("weight", 1.0))
        w_ad = float((item.get("rel_ad") or {}).get("weight", 1.0))

        G.add_edge(ph_id, an_id, rel="LOCATED_IN", weight=w_pa)
        G.add_edge(an_id, did, rel="AFFECTS", weight=w_ad)

    return G


def _to_nx_complete_graph(sg: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()

    d = sg.get("disease") or {}
    did = d.get("id") or "disease"
    dlabel = d.get("data") or did
    G.add_node(did, kind="disease", label=dlabel, layer=3)

    cg = sg.get("complete_graph") or {}

    # 1) Pairs (best source of truth)
    for item in (cg.get("pairs") or []):
        ph = item.get("ph") or {}
        an = item.get("an") or {}

        ph_id = ph.get("id")
        an_id = an.get("id")
        if not ph_id or not an_id:
            continue

        G.add_node(ph_id, kind="phenotype", label=ph.get("data") or ph_id, layer=1)
        G.add_node(an_id, kind="anatomy", label=an.get("data") or an_id, layer=2)

        w_pa = float((item.get("rel_pa") or {}).get("weight", 1.0))
        w_ad = float((item.get("rel_ad") or {}).get("weight", 1.0))

        G.add_edge(ph_id, an_id, rel="LOCATED_IN", weight=w_pa)
        G.add_edge(an_id, did, rel="AFFECTS", weight=w_ad)

    # 2) Ensure all anatomies exist + AFFECTS edge
    for item in (cg.get("anatomies") or []):
        an = item.get("an") or {}
        an_id = an.get("id")
        if not an_id:
            continue

        G.add_node(an_id, kind="anatomy", label=an.get("data") or an_id, layer=2)

        w_ad = float((item.get("rel") or {}).get("weight", 1.0))
        if not G.has_edge(an_id, did):
            G.add_edge(an_id, did, rel="AFFECTS", weight=w_ad)

    # 3) Ensure all phenotypes exist
    for item in (cg.get("phenotypes") or []):
        ph = item.get("ph") or {}
        ph_id = ph.get("id")
        if not ph_id:
            continue
        G.add_node(ph_id, kind="phenotype", label=ph.get("data") or ph_id, layer=1)

    # 4) Optional disease relations
    for item in (cg.get("parents") or []):
        p = item.get("parent") or {}
        pid = p.get("id")
        if pid:
            G.add_node(pid, kind="disease", label=p.get("data") or pid, layer=3)
            w = float((item.get("rel") or {}).get("weight", 1.0))
            G.add_edge(pid, did, rel="PARENT_OF", weight=w)

    for item in (cg.get("children") or []):
        c = item.get("child") or {}
        cid = c.get("id")
        if cid:
            G.add_node(cid, kind="disease", label=c.get("data") or cid, layer=3)
            w = float((item.get("rel") or {}).get("weight", 1.0))
            G.add_edge(did, cid, rel="PARENT_OF", weight=w)

    return G


