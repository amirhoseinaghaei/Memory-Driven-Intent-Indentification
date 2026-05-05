"""
Medical Diagnosis Agent Module

A sophisticated conversational AI agent for medical diagnosis that leverages graph-based
retrieval and iterative symptom clarification to identify potential diseases from patient
descriptions.

The agent employs a multi-step reasoning pipeline:
1. Receives user symptom descriptions
2. Retrieves candidate diseases from the knowledge graph
3. Clarifies symptoms if needed via targeted questions
4. Returns ranked disease candidates based on symptom matching

Author: Medical AI Team
Version: 1.0.0
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, TypedDict, Tuple

import numpy as np
from langgraph.graph import StateGraph, END

from src.medical_agent.tools import make_retrieve_tool
from src.retrieval.retriever2 import Retriever
from src.config.config import settings
from src.gen_ai_gateway.chat_completion import ChatCompletion
from src.graph_comparison.fpgw_dis import compute_node_importance


# ============================================================================
# LOGGER CONFIGURATION
# ============================================================================

def _configure_logging() -> None:
    """Suppress verbose logging from dependencies to reduce noise."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_DEBUG"] = "false"

    # Suppress verbose loggers
    verbose_loggers = [
        "langchain", "langchain_core", "langgraph",
        "neo4j", "neo4j.io", "neo4j.pool",
        "urllib3", "urllib3.connectionpool",
    ]
    for logger_name in verbose_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    logging.basicConfig(level=logging.WARNING)


_configure_logging()


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

CONFIDENCE_THRESHOLD = 0.99
"""Minimum score to consider a diagnosis confident without clarification."""

TOPK_FOR_CLARIFY = 3
"""Number of top candidates to consider for clarification questions."""

MAX_SYMPTOMS_IN_QUESTION = 60
"""Maximum symptoms to include in a clarification question."""

MAX_DISEASES_IN_ANSWER = 40
"""Maximum diseases to include in the final answer."""

SYMPTOM_MAPPING_PATH = "dataset/phenotype_catalog.json"
"""Path to symptom-to-phenotype-ID mapping file."""

DEBUG = False
"""Set to True to print detailed extraction and routing information."""


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class AgentState(TypedDict, total=False):
    """State schema for the medical diagnosis agent.
    
    Maintains conversation context and retrieval state across multiple rounds.
    """
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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_key(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    return " ".join((text or "").strip().split()).lower()


def load_symptom_mapping(path: str) -> Dict[str, str]:
    """Load symptom-to-phenotype mapping from JSON file.
    
    Supports two formats:
    - Dict: {"Symptom Name": "phenotype:123", ...}
    - List: [{"name": "Symptom Name", "value": "phenotype:123"}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return {str(k).strip(): str(v).strip() for k, v in data.items()}

    if isinstance(data, list):
        result: Dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            value = str(item.get("value", "")).strip()
            if name and value:
                result[name] = value
        return result

    raise ValueError("Unsupported symptom mapping format")


def build_mapping_state(mapping_path: str) -> Dict[str, Any]:
    """Build initial state with symptom mappings and normalized indices."""
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
    """Map symptom name to phenotype ID with exact and normalized matching."""
    if not symptom_name:
        return None
    if symptom_name in name_to_value:
        return name_to_value[symptom_name]
    return norm_index.get(_normalize_key(symptom_name))


def split_symptom_phrases(text: str) -> List[str]:
    """Split user text into symptom phrases by comma, semicolon, newline, 'and', '&'."""
    if not text:
        return []
    parts = re.split(r",|;|\n|\band\b|\b&\b", text, flags=re.IGNORECASE)
    result: List[str] = []
    for part in parts:
        cleaned = " ".join(part.strip().split())
        if cleaned:
            result.append(cleaned)
    return result


def extract_user_symptom_values_from_text(
    text: str,
    name_to_value: Dict[str, str],
    norm_index: Dict[str, str],
) -> Tuple[Set[str], Set[str]]:
    """Extract phenotype IDs from user text."""
    user_values: Set[str] = set()
    user_norm_names: Set[str] = set()

    for phrase in split_symptom_phrases(text):
        normalized = _normalize_key(phrase)
        user_norm_names.add(normalized)
        if symptom_id := map_symptom_to_value(phrase, name_to_value, norm_index):
            user_values.add(symptom_id)

    return user_values, user_norm_names


def extract_symptom_display_names(
    user_response: str,
    name_to_value: Dict[str, str],
    norm_index: Dict[str, str],
) -> List[str]:
    """Extract canonical symptom display names from user's response."""
    extracted_names: List[str] = []
    seen: Set[str] = set()

    for phrase in split_symptom_phrases(user_response):
        phenotype_id = map_symptom_to_value(phrase, name_to_value, norm_index)
        if phenotype_id:
            for original_name, value in name_to_value.items():
                if value == phenotype_id and original_name not in seen:
                    extracted_names.append(original_name)
                    seen.add(original_name)
                    break
        else:
            normalized = _normalize_key(phrase)
            if normalized and normalized not in seen:
                extracted_names.append(phrase)
                seen.add(normalized)

    return extracted_names


def extract_phenotype_names(graph: Any, max_items: int = 12) -> List[str]:
    """Extract phenotype names from NetworkX graph with robust schema handling."""
    results: List[str] = []
    seen: Set[str] = set()

    def _add(value: Any) -> None:
        if value is None:
            return
        s = str(value).strip()
        if not s or s in seen:
            return
        seen.add(s)
        results.append(s)

    if graph is None or not (hasattr(graph, "nodes") and callable(getattr(graph, "nodes", None))):
        return results

    try:
        for node_id, attrs in graph.nodes(data=True):
            if len(results) >= max_items:
                break
            attrs = attrs if isinstance(attrs, dict) else {}
            node_key = str(node_id) if node_id is not None else ""
            node_id_str = str(attrs.get("id") or attrs.get("node_id") or node_key)

            label = attrs.get("label") or attrs.get("type") or attrs.get("entity_label") or attrs.get("kind")
            layer = attrs.get("layer")

            is_phenotype = (
                label == "phenotype"
                or (isinstance(layer, (int, float)) and int(layer) == 1)
                or node_key.startswith("phenotype:")
                or node_id_str.startswith("phenotype:")
            )

            if not is_phenotype:
                continue

            name = (
                attrs.get("data") or attrs.get("name") or attrs.get("title") 
                or attrs.get("text") or attrs.get("display")
            )
            _add(name or node_id_str or node_key)
    except Exception:
        pass

    return results


def extract_phenotype_values_from_graph(
    graph: Any,
    name_to_value: Dict[str, str],
    norm_index: Dict[str, str],
) -> Set[str]:
    """Extract phenotype IDs from graph nodes and mapped names."""
    result: Set[str] = set()

    if graph is None or not (hasattr(graph, "nodes") and callable(getattr(graph, "nodes", None))):
        return result

    try:
        for node_id, attrs in graph.nodes(data=True):
            attrs = attrs if isinstance(attrs, dict) else {}
            node_key = str(node_id) if node_id is not None else ""
            node_id_str = str(attrs.get("id") or attrs.get("node_id") or node_key)

            label = attrs.get("label") or attrs.get("type") or attrs.get("entity_label") or attrs.get("kind")
            layer = attrs.get("layer")

            is_phenotype = (
                label == "phenotype"
                or (isinstance(layer, (int, float)) and int(layer) == 1)
                or node_key.startswith("phenotype:")
                or node_id_str.startswith("phenotype:")
            )

            if not is_phenotype:
                continue

            if node_id_str.startswith("phenotype:"):
                result.add(node_id_str)
            else:
                name = (
                    attrs.get("data") or attrs.get("name") or attrs.get("title") 
                    or attrs.get("text") or attrs.get("display")
                )
                if name and (phenotype_id := map_symptom_to_value(str(name), name_to_value, norm_index)):
                    result.add(phenotype_id)
    except Exception:
        pass

    return result


def extract_top_importance_symptoms(
    graph: Any,
    top_k: int = 5,
) -> List[Tuple[str, str, float]]:
    """Extract symptoms with highest node importance from graph.
    
    Args:
        graph: NetworkX graph with edges and node attributes
        top_k: Number of top symptoms to return
    
    Returns:
        List of tuples: (node_id, display_name, importance_score)
    """
    if graph is None or not (hasattr(graph, "nodes") and callable(getattr(graph, "nodes", None))):
        return []
    
    try:
        # Build adjacency matrix from graph
        node_list = list(graph.nodes())
        if not node_list:
            return []
        
        node_index = {node_id: idx for idx, node_id in enumerate(node_list)}
        n = len(node_list)
        A = np.zeros((n, n), dtype=np.float64)
        
        # Populate adjacency matrix from edges
        for src, tgt, data in graph.edges(data=True):
            if src in node_index and tgt in node_index:
                # Use edge weight if available, otherwise 1.0
                weight = 1.0
                if isinstance(data, dict):
                    weight = _safe_float(data.get("weight", 1.0))
                A[node_index[src], node_index[tgt]] = weight
        
        # Compute importance scores
        importance_scores = compute_node_importance(A)
        
        # Collect phenotype nodes with their importance
        phenotypes_with_importance: List[Tuple[str, str, float]] = []
        
        for idx, node_id in enumerate(node_list):
            attrs = graph.nodes[node_id] if hasattr(graph, "nodes") else {}
            attrs = attrs if isinstance(attrs, dict) else {}
            
            label = attrs.get("label") or attrs.get("type") or attrs.get("entity_label") or attrs.get("kind")
            layer = attrs.get("layer")
            node_key = str(node_id)
            
            is_phenotype = (
                label == "phenotype"
                or (isinstance(layer, (int, float)) and int(layer) == 1)
                or node_key.startswith("phenotype:")
            )
            
            if not is_phenotype:
                continue
            
            display_name = (
                attrs.get("data") or attrs.get("name") or attrs.get("title")
                or attrs.get("text") or attrs.get("display") or node_key
            )
            
            importance = _safe_float(importance_scores[idx], default=0.0)
            phenotypes_with_importance.append((node_key, str(display_name), importance))
        
        # Sort by importance (descending) and return top-k
        phenotypes_with_importance.sort(key=lambda x: x[2], reverse=True)
        return phenotypes_with_importance[:top_k]
    
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] Error extracting importance symptoms: {e}")
        return []



# ============================================================================
# AGENT BUILDER & STATE NODES
# ============================================================================

def build_graph_agent(retriever: Retriever) -> StateGraph[AgentState]:
    """Build LangGraph medical diagnosis agent.
    
    Agent flow: Rank → Route → (Clarify OR Answer) → END
    """
    retrieve_tool = make_retrieve_tool(retriever)

    def _disease_keys(item: Dict[str, Any]) -> List[str]:
        disease_id = item.get("id", "").strip()
        return [disease_id] if disease_id else []

    def _target_in_topk(results: List[Dict[str, Any]], target: Optional[str], k: int) -> bool:
        if not target:
            return False
        target = target.strip()
        return any(target in _disease_keys(item) for item in (results or [])[:k])

    def _target_anywhere(results: List[Dict[str, Any]], target: Optional[str]) -> bool:
        if not target:
            return False
        target = target.strip()
        return any(target in _disease_keys(item) for item in (results or []))

    def node_rank(state: AgentState) -> AgentState:
        """Stage 1: Retrieve disease candidates from knowledge graph."""
        query = (state.get("combined_query") or state["user_query"]).strip()
        previous_groups = state.get("previous_groups")
        previous_diseases = state.get("previous_diseases")

        (results, clustered, phenotypes, diseases, groups, prev_diseases, 
         tokens, time_taken) = retrieve_tool.invoke({
            "query": query,
            "previous_groups": previous_groups,
            "previous_diseases": previous_diseases,
        })

        results = sorted(results, key=lambda x: _safe_float(x.get("score", 0.0)), reverse=True)

        return {
            **state,
            "results": results,
            "clustering_result": clustered,
            "llm_entity_recognition_result": phenotypes,
            "retrieved_diseases": diseases,
            "previous_groups": groups,
            "previous_diseases": prev_diseases,
            "token_usage": tokens,
            "retrieval_time": time_taken,
        }

    def node_clarify(state: AgentState) -> AgentState:
        """Stage 2: Generate targeted clarification questions."""
        results = state.get("results") or []
        target = state.get("target_disease_id")

        if not results:
            return {
                **state,
                "need_clarification": True,
                "final": (
                    "I couldn't find any medical conditions matching your symptoms. "
                    "Please add 1-2 more specific symptoms and mention where you feel them."
                ),
            }

        name_to_value = state.get("symptom_name_to_value") or {}
        norm_index = state.get("_symptom_norm_index") or {}
        top_candidates = results[:TOPK_FOR_CLARIFY]

        # Collect already-extracted symptoms
        excluded: Set[str] = set()
        for item in top_candidates:
            excluded |= extract_phenotype_values_from_graph(
                item.get("partial_graph"), name_to_value, norm_index
            )

        current_query = (state.get("combined_query") or state.get("user_query") or "").strip()
        user_ids, _ = extract_user_symptom_values_from_text(current_query, name_to_value, norm_index)
        excluded |= user_ids

        # Collect candidate symptoms
        candidates: List[str] = []
        seen_ids: Set[str] = set()

        for item in top_candidates:
            if len(candidates) >= MAX_SYMPTOMS_IN_QUESTION:
                break
            complete_graph = item.get("complete_graph")
            for symptom_name in extract_phenotype_names(complete_graph, max_items=MAX_SYMPTOMS_IN_QUESTION):
                if len(candidates) >= MAX_SYMPTOMS_IN_QUESTION:
                    break
                symptom_id = map_symptom_to_value(symptom_name, name_to_value, norm_index)
                if symptom_id and symptom_id not in excluded and symptom_id not in seen_ids:
                    seen_ids.add(symptom_id)
                    candidates.append(symptom_id)

        # Build message
        lines: List[str] = []
        if target:
            if not _target_anywhere(results, target):
                lines.append(f"⚠️  Target {target} not found in candidates.")
            else:
                lines.append(f"⚠️  Target {target} exists but not in top-{TOPK_FOR_CLARIFY}.")

        if candidates:
            lines.extend([
                f"To narrow down the top {TOPK_FOR_CLARIFY} candidates, do you have any of these symptoms?",
                *[f"  • {s}" for s in candidates],
                "Please reply with the ones you have.",
            ])
        else:
            lines.extend([
                f"To narrow down the top {TOPK_FOR_CLARIFY} candidates, please provide 1-2 additional symptoms.",
                "Include information about the affected body area.",
            ])

        return {**state, "need_clarification": True, "final": "\n".join(lines)}

    def route_after_rank(state: AgentState) -> str:
        """Route to clarify or answer based on confidence and target presence."""
        results = state.get("results") or []
        target = state.get("target_disease_id")

        if target:
            return "answer" if _target_in_topk(results, target, k=TOPK_FOR_CLARIFY) else "clarify"

        top_score = _safe_float(results[0].get("score", 0.0)) if results else 0.0
        return "answer" if top_score >= CONFIDENCE_THRESHOLD else "clarify"

    def node_answer(state: AgentState) -> AgentState:
        """Stage 3: Format final ranked disease candidates with top-importance symptoms."""
        results = state.get("results") or []
        if not results:
            return {
                **state,
                "need_clarification": False,
                "final": "No medical conditions matched your symptoms.",
            }

        target = state.get("target_disease_id")
        top = results[:TOPK_FOR_CLARIFY]

        lines: List[str] = []
        for rank, item in enumerate(top, start=1):
            did = item.get("disease_id") or item.get("id") or "unknown"
            score = _safe_float(item.get("score", 0.0))
            marker = " ✅ TARGET" if (target and target in _disease_keys(item)) else ""
            
            # Extract high-importance symptoms from complete graph
            symptoms_text = ""
            complete_graph = item.get("complete_graph")
            if complete_graph:
                top_symptoms = extract_top_importance_symptoms(complete_graph, top_k=3)
                if top_symptoms:
                    symptom_names = [s[1] for s in top_symptoms]
                    symptoms_text = f" [Key symptoms: {', '.join(symptom_names)}]"
            
            lines.append(f"{rank}. {did} (score: {score:.6f}){marker}{symptoms_text}")

        main = "Top-3 disease candidates:\n" + "\n".join(lines)

        # Add high-confidence candidates
        high_conf = [
            x for x in results
            if _safe_float(x.get("score", 0.0)) >= CONFIDENCE_THRESHOLD
        ][:MAX_DISEASES_IN_ANSWER]

        extra = ""
        if high_conf:
            best = _safe_float(high_conf[0].get("score", 0.0))
            worst = _safe_float(high_conf[-1].get("score", 0.0))
            conf_lines: List[str] = []
            for rank, item in enumerate(high_conf, start=1):
                did = item.get("disease_id") or item.get("id") or "unknown"
                score = _safe_float(item.get("score", 0.0))
                
                # Extract high-importance symptoms from complete graph
                symptoms_text = ""
                complete_graph = item.get("complete_graph")
                if complete_graph:
                    top_symptoms = extract_top_importance_symptoms(complete_graph, top_k=2)
                    if top_symptoms:
                        symptom_names = [s[1] for s in top_symptoms]
                        symptoms_text = f" [Key: {', '.join(symptom_names)}]"
                
                conf_lines.append(f"{rank}. {did} (score: {score:.2f}){symptoms_text}")

            extra = (
                f"\n\n**Additional candidates (score ≥ {CONFIDENCE_THRESHOLD}):**\n"
                f"Found {len(high_conf)} conditions (range: {worst:.2f} → {best:.2f})\n"
                + "\n".join(conf_lines)
            )

        return {
            **state,
            "need_clarification": False,
            "final": main + extra,
        }

    # Build and compile graph
    graph = StateGraph(AgentState)
    graph.add_node("rank", node_rank)
    graph.add_node("clarify", node_clarify)
    graph.add_node("answer", node_answer)

    graph.set_entry_point("rank")
    graph.add_conditional_edges("rank", route_after_rank, {"clarify": "clarify", "answer": "answer"})
    graph.add_edge("clarify", END)
    graph.add_edge("answer", END)

    return graph.compile()


# ============================================================================
# INTERACTIVE CONVERSATION RUNNER
# ============================================================================

def run_interactive(
    agent: StateGraph[AgentState],
    first_query: str,
    *,
    target_disease_id: Optional[str] = None,
    max_rounds: int = 8
) -> None:
    """Run interactive multi-round diagnosis conversation.
    
    Args:
        agent: Compiled LangGraph agent
        first_query: Initial user symptom query
        target_disease_id: Expected disease ID (for evaluation)
        max_rounds: Maximum conversation rounds
    """

    def _disease_keys(item: Dict[str, Any]) -> List[str]:
        disease_id = item.get("id", "").strip()
        return [disease_id] if disease_id else []

    def _target_rank(results: List[Dict[str, Any]], target: Optional[str]) -> Optional[int]:
        if not target:
            return None
        target = target.strip()
        return next((i for i, item in enumerate(results or [], start=1) 
                    if target in _disease_keys(item)), None)

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
        output_state: AgentState = agent.invoke(state)
        print("\n" + (output_state.get("final") or ""))

        results = output_state.get("results") or []
        if target_disease_id:
            rank = _target_rank(results, target_disease_id)
            top_3 = [
                str(results[i].get("disease_id") or results[i].get("id") or "unknown")
                for i in range(min(3, len(results)))
            ]
            if rank is None:
                print(f"\n[Target Rank] {target_disease_id}: NOT FOUND (total: {len(results)})")
            else:
                print(f"\n[Target Rank] {target_disease_id}: {rank}/{len(results)}")
            print("[Top-3]", " | ".join(top_3))

        if not output_state.get("need_clarification"):
            return

        clarification = input("\nYour response: ").strip()
        state = {
            **state,
            "user_query": clarification,
            "followup_query": clarification,
            "combined_query": None,
            "results": None,
            "clustering_result": [],
            "llm_entity_recognition_result": [],
            "retrieved_diseases": [],
            "final": None,
            "token_usage": None,
            "previous_groups": output_state.get("previous_groups") or [],
            "previous_diseases": output_state.get("previous_diseases") or [],
            "target_disease_id": target_disease_id,
        }

        if DEBUG:
            print(f"\n[DEBUG Round {round_num + 1}] User input: {clarification}")

    print(
        "\nI need more specific information to provide an accurate diagnosis.\n"
        "Please describe:\n"
        "  • Duration (how long?)\n"
        "  • Severity (mild, moderate, severe)\n"
        "  • Body locations affected\n"
        "  • Associated symptoms (fever, vomiting, etc.)\n"
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    chat = ChatCompletion(settings)
    retriever = Retriever(settings, chat)
    retriever.build_clusters()

    agent = build_graph_agent(retriever)

    example_query = (
        "My child has been urinating very frequently and appears smaller than "
        "children of the same age. Could polyuria affecting the kidney be related to short stature?"
    )

    run_interactive(
        agent,
        first_query=example_query,
        max_rounds=5,
        target_disease_id="disease:10204"
    )


