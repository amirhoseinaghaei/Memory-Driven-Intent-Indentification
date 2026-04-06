import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, TypedDict, Tuple

from langgraph.graph import StateGraph, END
from src.medical_agent.tools import make_retrieve_tool
from src.retrieval.retriever import Retriever
from src.config.config import settings
from src.gen_ai_gateway.chat_completion import ChatCompletion

# ============================================================
# Logging / Tracing suppression
# ============================================================

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_DEBUG"] = "false"

LOGGERS_TO_SUPPRESS = [
    "langchain",
    "langchain_core",
    "langgraph",
    "neo4j",
    "neo4j.io",
    "neo4j.pool",
    "urllib3",
    "urllib3.connectionpool",
]

for logger_name in LOGGERS_TO_SUPPRESS:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logging.basicConfig(level=logging.WARNING)

# ============================================================
# Config
# ============================================================

THRESHOLD = 0.99  # (kept for fallback when target is not provided)
TOPK_FOR_CLARIFY = 3
MAX_SYMPTOMS_IN_QUESTION = 60
MAX_DISEASES_IN_ANSWER = 40

SYMPTOM_MAPPING_PATH = (
    "dataset/phenotype_catalog.json"
)

DEBUG = False  # Set to True to see extraction details

# ============================================================
# State schema
# ============================================================

class AgentState(TypedDict, total=False):
    user_query: str
    results: Optional[List[Dict[str, Any]]]
    final: Optional[str]
    clustering_result: Optional[List[str]]
    llm_entity_recognition_result: Optional[List[str]]
    retrieved_diseases: Optional[List[str]]
    need_clarification: bool
    followup_query: Optional[str]
    combined_query: Optional[str]
    symptom_name_to_value: Dict[str, str]
    _symptom_norm_index: Dict[str, str]
    previous_groups: List[Dict[str, Any]]
    previous_diseases: List[Dict[str, Any]]
    token_usage: Optional[Dict[str, Any]]
    retrieval_time: Optional[float]
    target_disease_id: Optional[str]

# ============================================================
# Utilities
# ============================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _normalize_key(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def load_symptom_mapping(path: str) -> Dict[str, str]:
    """
    Supports JSON formats:
      A) {"Nausea": "phenotype:123", ...}
      B) [{"name":"Nausea","value":"phenotype:123"}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return {str(k).strip(): str(v).strip() for k, v in data.items()}

    if isinstance(data, list):
        out: Dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            val = str(item.get("value", "")).strip()
            if name and val:
                out[name] = val
        return out

    raise ValueError("Unsupported mapping JSON format (must be dict or list[dict]).")

def build_mapping_state(mapping_path: str) -> Dict[str, Any]:
    """
    Load mapping ONCE and return the state fragment to merge into AgentState.
    """
    name_to_value = load_symptom_mapping(mapping_path)
    norm_index = {_normalize_key(k): v for k, v in name_to_value.items()}

    return {
        "symptom_name_to_value": name_to_value,
        "_symptom_norm_index": norm_index,
    }

def map_symptom_to_value(
    symptom_name: str,
    name_to_value: Dict[str, str],
    norm_index: Dict[str, str],
) -> Optional[str]:
    """
    Map symptom display name -> mapped value (phenotype id).
    """
    if not symptom_name:
        return None

    if symptom_name in name_to_value:
        return name_to_value[symptom_name]

    return norm_index.get(_normalize_key(symptom_name))

def split_symptom_phrases(text: str) -> List[str]:
    """
    Heuristic splitter for user symptom text.
    Splits by comma, ';', newline, and 'and' / '&'.
    """
    if not text:
        return []
    parts = re.split(r",|;|\n|\band\b|\b&\b", text, flags=re.IGNORECASE)
    out: List[str] = []
    for p in parts:
        p = " ".join(p.strip().split())
        if p:
            out.append(p)
    return out

def extract_user_symptom_values_from_text(
    text: str,
    name_to_value: Dict[str, str],
    norm_index: Dict[str, str],
) -> Tuple[Set[str], Set[str]]:
    """
    Returns:
      - user_values: phenotype IDs we can map from the user's text
      - user_norm_names: normalized symptom phrases (fallback filtering)
    """
    user_values: Set[str] = set()
    user_norm_names: Set[str] = set()

    for phrase in split_symptom_phrases(text):
        n = _normalize_key(phrase)
        user_norm_names.add(n)

        v = map_symptom_to_value(phrase, name_to_value, norm_index)
        if v:
            user_values.add(v)

    return user_values, user_norm_names

def extract_symptom_display_names(
    user_response: str,
    name_to_value: Dict[str, str],
    norm_index: Dict[str, str],
) -> List[str]:
    """
    Extract symptom display names from user's response.

    Input: "yes, I have cough and nausea"
    Output: ["Cough", "Nausea"]  (matching the original mapping names)

    Used to add extracted symptoms to user_query.
    """
    extracted_names: List[str] = []
    seen: Set[str] = set()

    phrases = split_symptom_phrases(user_response)

    for phrase in phrases:
        pheno_id = map_symptom_to_value(phrase, name_to_value, norm_index)

        if pheno_id:
            # Find canonical display name(s) corresponding to this pheno id
            for orig_name, val in name_to_value.items():
                if val == pheno_id and orig_name not in seen:
                    extracted_names.append(orig_name)
                    seen.add(orig_name)
                    break
        else:
            norm_phrase = _normalize_key(phrase)
            if norm_phrase and norm_phrase not in seen:
                extracted_names.append(phrase)
                seen.add(norm_phrase)

    return extracted_names

def extract_phenotype_names(complete_g: Any, max_items: int = 12) -> List[str]:
    """
    Extract phenotype names from a NetworkX graph, robust to different node schemas.
    """
    out: List[str] = []
    seen: Set[str] = set()

    def _add(val: Any) -> None:
        if val is None:
            return
        s = str(val).strip()
        if not s or s in seen:
            return
        seen.add(s)
        out.append(s)

    if complete_g is None:
        return out

    if not (hasattr(complete_g, "nodes") and callable(getattr(complete_g, "nodes", None))):
        return out

    try:
        for n, attrs in complete_g.nodes(data=True):
            if len(out) >= max_items:
                break

            attrs = attrs if isinstance(attrs, dict) else {}
            node_key = str(n) if n is not None else ""
            node_id = str(attrs.get("id") or attrs.get("node_id") or node_key)

            label = attrs.get("label") or attrs.get("type") or attrs.get("entity_label") or attrs.get("kind")
            layer = attrs.get("layer")

            is_ph = False
            if label == "phenotype":
                is_ph = True
            elif isinstance(layer, (int, float)) and int(layer) == 1:
                is_ph = True
            elif node_key.startswith("phenotype:") or node_id.startswith("phenotype:"):
                is_ph = True

            if not is_ph:
                continue

            name = (
                attrs.get("data")
                or attrs.get("name")
                or attrs.get("title")
                or attrs.get("text")
                or attrs.get("display")
            )
            _add(name or node_id or node_key)

    except Exception:
        pass

    return out

def extract_phenotype_values_from_graph(
    g: Any,
    name_to_value: Dict[str, str],
    norm_index: Dict[str, str],
) -> Set[str]:
    """
    Return phenotype IDs (phenotype:xxxx) found in a NetworkX graph.

    Works for:
      - node id like 'phenotype:123'
      - node attrs with label/kind/layer and a name in attrs['data']/['name'] that can be mapped
    """
    out: Set[str] = set()
    if g is None:
        return out
    if not (hasattr(g, "nodes") and callable(getattr(g, "nodes", None))):
        return out

    try:
        for n, attrs in g.nodes(data=True):
            attrs = attrs if isinstance(attrs, dict) else {}
            node_key = str(n) if n is not None else ""
            node_id = str(attrs.get("id") or attrs.get("node_id") or node_key)

            label = attrs.get("label") or attrs.get("type") or attrs.get("entity_label") or attrs.get("kind")
            layer = attrs.get("layer")

            is_ph = False
            if label == "phenotype":
                is_ph = True
            elif isinstance(layer, (int, float)) and int(layer) == 1:
                is_ph = True
            elif node_key.startswith("phenotype:") or node_id.startswith("phenotype:"):
                is_ph = True

            if not is_ph:
                continue

            if node_id.startswith("phenotype:"):
                out.add(node_id)
                continue

            name = (
                attrs.get("data")
                or attrs.get("name")
                or attrs.get("title")
                or attrs.get("text")
                or attrs.get("display")
            )
            if name:
                v = map_symptom_to_value(str(name), name_to_value, norm_index)
                if v:
                    out.add(v)
    except Exception:
        pass

    return out

# ============================================================
# Agent
# ============================================================

def build_graph_agent(gdb: Retriever) -> StateGraph[AgentState]:
    retrieve_tool = make_retrieve_tool(gdb)

    def _disease_keys(item: Dict[str, Any]) -> List[str]:
        s = item.get("id").strip()
        return [s] if s else []

    def _target_in_topk(res: List[Dict[str, Any]], target: Optional[str], k: int) -> bool:
        if not target:
            return False
        target = target.strip()
        for item in (res or [])[:k]:
            if target in _disease_keys(item):
                return True
        return False

    def _target_anywhere(res: List[Dict[str, Any]], target: Optional[str]) -> bool:
        if not target:
            return False
        target = target.strip()
        for item in (res or []):
            if target in _disease_keys(item):
                return True
        return False

    def rank(state: AgentState) -> AgentState:
        q = (state.get("combined_query") or state["user_query"]).strip()
        pg = state.get("previous_groups")
        pd = state.get("previous_diseases")

        res, flat_ranked, phenotype_texts, disease_in_nx_pairs, groups, previous_diseases, token_usage, time_taken = retrieve_tool.invoke({"query": q, "previous_groups": pg, "previous_diseases": pd})
        res = sorted(res, key=lambda x: _safe_float(x.get("score", 0.0)), reverse=True)

        return {
            **state,
            "results": res,
            "clustering_result": flat_ranked,
            "llm_entity_recognition_result": phenotype_texts,
            "retrieved_diseases": disease_in_nx_pairs,
            "previous_groups": groups,
            "previous_diseases": previous_diseases,
            "token_usage": token_usage,
            "retrieval_time": time_taken,
        }

    def clarify(state: AgentState) -> AgentState:
        res = state.get("results") or []
        target = state.get("target_disease_id")

        if not res:
            return {
                **state,
                "need_clarification": True,
                "final": (
                    "I couldn't find any matching graphs. "
                    "Can you add 1–2 more symptoms and (if possible) the affected body area?"
                ),
            }

        name_to_value = state.get("symptom_name_to_value") or {}
        norm_index = state.get("_symptom_norm_index") or {}

        top_n = res[:TOPK_FOR_CLARIFY]

        exclude_values: Set[str] = set()

        for item in top_n:
            pg = item.get("partial_graph")
            exclude_values |= extract_phenotype_values_from_graph(pg, name_to_value, norm_index)

        base_text = (state.get("combined_query") or state.get("user_query") or "").strip()
        user_values, _ = extract_user_symptom_values_from_text(base_text, name_to_value, norm_index)
        exclude_values |= user_values

        all_values: List[str] = []
        seen_values: Set[str] = set()

        for item in top_n:
            cg = item.get("complete_graph")
            symptom_names = extract_phenotype_names(cg, max_items=MAX_SYMPTOMS_IN_QUESTION)

            for s in symptom_names:
                v = map_symptom_to_value(s, name_to_value, norm_index)
                if not v:
                    continue
                if v in exclude_values:
                    continue
                if v in seen_values:
                    continue

                seen_values.add(v)
                all_values.append(v)

                if len(all_values) >= MAX_SYMPTOMS_IN_QUESTION:
                    break

            if len(all_values) >= MAX_SYMPTOMS_IN_QUESTION:
                break

        header: List[str] = []
        if target:
            if not _target_anywhere(res, target):
                header.append(f"(Target {target} is NOT in the candidate list yet.)")
            else:
                header.append(f"(Target {target} exists in candidates but NOT in top-{TOPK_FOR_CLARIFY}.)")

        if all_values:
            msg = "\n".join(
                header
                + [
                    f"I need one more detail to disambiguate between the top {TOPK_FOR_CLARIFY} candidates.",
                    "Do you have any of these additional symptoms (IDs)?",
                    *[f"- {v}" for v in all_values],
                    "",
                    "Reply with the ones you have (or say 'none'), and also where you feel the pain.",
                ]
            )
        else:
            msg = "\n".join(
                header
                + [
                    f"I need one more detail to disambiguate between the top {TOPK_FOR_CLARIFY} candidates.",
                    "I already used your current symptoms.",
                    "Can you share 1–2 additional symptoms (and where in the body you feel them)?",
                ]
            )

        return {**state, "need_clarification": True, "final": msg}

    def route_after_rank(state: AgentState) -> str:
        res = state.get("results") or []
        target = state.get("target_disease_id")

        if target:
            return "answer" if _target_in_topk(res, target, k=TOPK_FOR_CLARIFY) else "clarify"

        top_score = _safe_float(res[0].get("score", 0.0)) if res else 0.0
        return "clarify" if top_score < THRESHOLD else "answer"

    def answer(state: AgentState) -> AgentState:
        res = state.get("results") or []
        if not res:
            return {**state, "need_clarification": False, "final": "No results found."}

        target = state.get("target_disease_id")
        top_n = res[:TOPK_FOR_CLARIFY]

        lines: List[str] = []
        for i, item in enumerate(top_n, start=1):
            did = item.get("disease_id") or item.get("id") or "unknown"
            sc = _safe_float(item.get("score", 0.0))
            mark = " ✅ TARGET" if (target and target in _disease_keys(item)) else ""
            lines.append(f"{i}. {did} (score={sc:.6f}){mark}")

        confident = [x for x in res if _safe_float(x.get("score", 0.0)) >= THRESHOLD][:MAX_DISEASES_IN_ANSWER]
        extra = ""
        if confident:
            best = _safe_float(confident[0].get("score", 0.0))
            worst = _safe_float(confident[-1].get("score", 0.0))
            extra_lines: List[str] = []
            for i, item in enumerate(confident, start=1):
                did = item.get("disease_id") or item.get("id") or "unknown"
                sc = _safe_float(item.get("score", 0.0))
                extra_lines.append(f"{i}. {did} (score={sc:.2f})")
            extra = (
                "\n\n"
                f"Also {len(confident)} candidates with score ≥ {THRESHOLD} "
                f"(range: {worst:.2f} → {best:.2f}).\n"
                + "\n".join(extra_lines)
            )

        return {
            **state,
            "need_clarification": False,
            "final": "Top-3 candidates:\n" + "\n".join(lines) + extra,
        }

    g = StateGraph(AgentState)
    g.add_node("rank", rank)
    g.add_node("clarify", clarify)
    g.add_node("answer", answer)

    g.set_entry_point("rank")
    g.add_conditional_edges("rank", route_after_rank, {"clarify": "clarify", "answer": "answer"})
    g.add_edge("clarify", END)
    g.add_edge("answer", END)

    return g.compile()

# ============================================================
# CLI Runner
# ============================================================

def run_interactive(
    agent,
    first_query: str,
    *,
    target_disease_id: Optional[str] = None,
    max_rounds: int = 8
) -> None:
    """
    Updated runner that prints target disease position after each iteration.
    """

    def _disease_keys(item: Dict[str, Any]) -> List[str]:
        s = item.get("id").strip()
        return [s] if s else []

    def _target_rank(res: List[Dict[str, Any]], target: Optional[str]) -> Optional[int]:
        """
        Returns 1-based rank of target in res if found, else None.
        Matches inside composite ids like "disease:a_b_c".
        """
        if not target:
            return None
        target = target.strip()
        for i, item in enumerate(res or [], start=1):
            if target in _disease_keys(item):
                return i
        return None

    mapping_state = build_mapping_state(SYMPTOM_MAPPING_PATH)

    state: AgentState = {
        **mapping_state,
        "user_query": first_query,
        "previous_groups": [],
        "previous_diseases": [],
        "results": None,
        "clustering_result": [],
        "llm_entity_recognition_result": [],
        "retrieved_diseases": [],
        "final": None,
        "need_clarification": False,
        "followup_query": None,
        "combined_query": None,
        "token_usage": None,
        "target_disease_id": target_disease_id,
    }

    for round_num in range(max_rounds):
        out: AgentState = agent.invoke(state)

        print("\n" + (out.get("final") or ""))

        res = out.get("results") or []
        if target_disease_id:
            r = _target_rank(res, target_disease_id)
            top3 = [
                str((res[i].get("disease_id") or res[i].get("id") or "unknown"))
                for i in range(min(3, len(res)))
            ]
            if r is None:
                print(f"\n[Target rank] {target_disease_id}: NOT FOUND (list size={len(res)})")
            else:
                print(f"\n[Target rank] {target_disease_id}: {r}/{len(res)}")
            print("[Top-3]", " | ".join(top3))

        if not out.get("need_clarification"):
            return

        follow = input("\nYour reply: ").strip()

        new_user_query = follow

        if DEBUG:
            print(f"\n[DEBUG Round {round_num + 1}]")
            print(f"  User said: {follow}")
            print(f"  Updated user_query: {new_user_query}")

        state = {
            **state,
            "user_query": new_user_query,
            "followup_query": follow,
            "combined_query": None,
            "results": None,
            "clustering_result": [],
            "llm_entity_recognition_result": [],
            "retrieved_diseases": [],
            "final": None,
            "token_usage": None,
            "previous_groups": out.get("previous_groups") or [],
            "previous_diseases": out.get("previous_diseases") or [],
            "target_disease_id": target_disease_id,
        }

    print(
        "\nI still don't have enough signal. Please add more details "
        "(duration, severity, fever, vomiting, location)."
    )

if __name__ == "__main__":
    chat = ChatCompletion(settings)
    retriever = Retriever(settings, chat)
    retriever.build_clusters()
    mapping_state = build_mapping_state(SYMPTOM_MAPPING_PATH)

    agent = build_graph_agent(retriever)
    first_query = "My child has been peeing a lot and seems smaller than other kids the same age. Could polyuria from the kidney be related to short stature?"
    run_interactive(agent, first_query=first_query, max_rounds=5, target_disease_id="disease:8661")

