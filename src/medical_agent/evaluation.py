# evaluation.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


def _disease_keys(item: Dict[str, Any]) -> List[str]:
    disease_id = str(item.get("id") or "").strip()
    return [disease_id] if disease_id else []


def target_in_topk(results: List[Dict[str, Any]], target: str, k: int = 3) -> bool:
    if not target:
        return False

    target = target.strip()
    for item in (results or [])[:k]:
        if target in _disease_keys(item):
            return True
    return False


def target_rank(results: List[Dict[str, Any]], target: str) -> Optional[int]:
    if not target:
        return None

    target = target.strip()
    for idx, item in enumerate(results or [], start=1):
        if target in _disease_keys(item):
            return idx
    return None


def _topk_ids(results: List[Dict[str, Any]], k: int) -> Optional[str]:
    if not results:
        return None

    top_ids = [
        str(results[i].get("disease_id") or results[i].get("id") or "unknown")
        for i in range(min(k, len(results)))
    ]
    return " | ".join(top_ids) if top_ids else None


def _split_symptoms(raw_value: Any) -> List[str]:
    if pd.isna(raw_value):
        return []
    return [x.strip() for x in str(raw_value).split(",") if x.strip()]


def _normalize_target_disease(raw_value: Any) -> Optional[str]:
    if pd.isna(raw_value):
        return None

    raw_str = str(raw_value).strip()
    if not raw_str or raw_str.lower() == "nan":
        return None

    return raw_str if raw_str.startswith("disease:") else f"disease:{raw_str}"


def _extract_token_usage_fields(token_usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    token_usage = token_usage or {}
    return {
        "llm_input_tokens": token_usage.get("llm_input_tokens"),
        "llm_output_tokens": token_usage.get("llm_output_tokens"),
        "embed_tokens": token_usage.get("embed_tokens"),
        "total_tokens": token_usage.get("total_tokens"),
    }


def _build_round_state(
    *,
    mapping_state: Dict[str, Any],
    query: str,
    target_disease_id: str,
    previous_groups: Optional[List[Dict[str, Any]]] = None,
    previous_diseases: Optional[List[Dict[str, Any]]] = None,
    followup_query: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        **mapping_state,
        "user_query": query,
        "previous_groups": previous_groups or [],
        "previous_diseases": previous_diseases or [],
        "results": None,
        "clustering_result": [],
        "llm_entity_recognition_result": [],
        "retrieved_diseases": [],
        "final": None,
        "token_usage": None,
        "need_clarification": False,
        "followup_query": followup_query,
        "combined_query": None,
        "target_disease_id": target_disease_id,
    }


def _update_row_with_round_outputs(
    row_out: Dict[str, Any],
    round_idx: int,
    out: Dict[str, Any],
    target: str,
    topk: int,
    expected_symptoms: List[str],
) -> Tuple[bool, Optional[int]]:
    results = out.get("results") or []
    hit = target_in_topk(results, target, k=topk)
    rank = target_rank(results, target)

    row_out[f"rank_r{round_idx}"] = rank
    row_out[f"top3_r{round_idx}"] = _topk_ids(results, topk)
    row_out[f"clustering_result{round_idx}"] = out.get("clustering_result")
    row_out[f"llm_entity_recognition_result{round_idx}"] = out.get("llm_entity_recognition_result")
    row_out[f"retrieved_diseases{round_idx}"] = out.get("retrieved_diseases")
    row_out[f"token_usage{round_idx}"] = out.get("token_usage")

    token_fields = _extract_token_usage_fields(out.get("token_usage"))
    row_out[f"token_usage{round_idx}_llm_input_tokens"] = token_fields["llm_input_tokens"]
    row_out[f"token_usage{round_idx}_llm_output_tokens"] = token_fields["llm_output_tokens"]
    row_out[f"token_usage{round_idx}_embed_tokens"] = token_fields["embed_tokens"]
    row_out[f"token_usage{round_idx}_total_tokens"] = token_fields["total_tokens"]

    row_out[f"time_taken{round_idx}"] = out.get("retrieval_time")
    row_out[f"is_clustering{round_idx}"] = all(
        x in (out.get("clustering_result") or []) for x in expected_symptoms
    )
    row_out[f"is_entity_recognition{round_idx}"] = all(
        x in (out.get("llm_entity_recognition_result") or []) for x in expected_symptoms
    )
    row_out[f"is_retrival{round_idx}"] = target in (out.get("retrieved_diseases") or [])

    return hit, rank


def evaluate_3round_excel(
    *,
    agent,
    excel_path: str,
    disease_id_col: str,
    q1_col: str,
    q2_col: str,
    q3_col: str,
    mapping_state: Dict[str, Any],
    topk: int = 3,
    combine_questions: bool = False,
    start_row: int = 0,
    print_every: int = 1,
    save_path: Optional[str] = None,
    save_every: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_excel(excel_path)
    if start_row > 0:
        df = df.iloc[start_row:].reset_index(drop=True)
    # print(df["rank_r1"])
    # df = df[df["rank_r1"].astype(str).str.strip().isin(["4.0", "5.0", "6.0"])]    
    # print(len(df))
    rows_out: List[Dict[str, Any]] = []

    all_cols = [
        "row_index", "target_disease_id", "solved_round",
        "round1_hit", "round2_hit", "round3_hit",
        "rank_r1", "rank_r2", "rank_r3",
        "top3_r1", "top3_r2", "top3_r3",
        "symptoms1", "symptoms2", "symptoms3",
        "q1", "q2", "q3",
        "query_r1", "query_r2", "query_r3",
        "clustering_result1", "clustering_result2", "clustering_result3",
        "llm_entity_recognition_result1", "llm_entity_recognition_result2", "llm_entity_recognition_result3",
        "retrieved_diseases1", "retrieved_diseases2", "retrieved_diseases3",
        "token_usage1", "token_usage1_llm_input_tokens", "token_usage1_llm_output_tokens", "token_usage1_embed_tokens", "token_usage1_total_tokens",
        "token_usage2", "token_usage2_llm_input_tokens", "token_usage2_llm_output_tokens", "token_usage2_embed_tokens", "token_usage2_total_tokens",
        "token_usage3", "token_usage3_llm_input_tokens", "token_usage3_llm_output_tokens", "token_usage3_embed_tokens", "token_usage3_total_tokens",
        "time_taken1", "time_taken2", "time_taken3",
        "is_retrival1", "is_retrival2", "is_retrival3",
        "is_entity_recognition1", "is_entity_recognition2", "is_entity_recognition3",
        "is_clustering1", "is_clustering2", "is_clustering3",
        "error",
    ]
    row_template = {col: None for col in all_cols}

    n = 0
    solved_r1_only = 0
    solved_r2_only = 0
    solved_r3_only = 0
    solved_by_r2 = 0
    solved_by_r3 = 0

    def print_running() -> None:
        if n <= 0:
            return

        print(f"\n[Running after {n} diseases]")
        print("  Exclusive accuracies:")
        print(f"    Round1: {100 * (solved_r1_only / n):.2f}%")
        print(f"    Round2: {100 * (solved_r2_only / n):.2f}%")
        print(f"    Round3: {100 * (solved_r3_only / n):.2f}%")
        print("  Cumulative accuracies:")
        print(f"    Solved by Round2: {100 * (solved_by_r2 / n):.2f}%")
        print(f"    Solved by Round3: {100 * (solved_by_r3 / n):.2f}%")
        print(
            f"  Counts: r1_only={solved_r1_only}, "
            f"r2_only={solved_r2_only}, "
            f"r3_only={solved_r3_only}, "
            f"unsolved={n - solved_by_r3}"
        )

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        target = _normalize_target_disease(row.get(disease_id_col))
        if not target:
            continue

        q1 = "" if pd.isna(row.get(q1_col)) else str(row.get(q1_col)).strip()
        q2 = "" if pd.isna(row.get(q2_col)) else str(row.get(q2_col)).strip()
        q3 = "" if pd.isna(row.get(q3_col)) else str(row.get(q3_col)).strip()

        if not q1:
            continue

        has_q2 = bool(q2)
        has_q3 = bool(q3)

        symptoms1 = _split_symptoms(row.get("symptoms_used_q1"))
        symptoms2 = _split_symptoms(row.get("symptoms_used_q2"))
        symptoms3 = _split_symptoms(row.get("symptoms_used_q3"))

        query_r1 = q1
        query_r2 = (" ".join([x for x in [q1, q2] if x]) if combine_questions else q2) if has_q2 else None
        query_r3 = (" ".join([x for x in [q1, q2, q3] if x]) if combine_questions else q3) if has_q3 else None

        n += 1

        row_out = {
            **row_template,
            "row_index": int(idx),
            "target_disease_id": target,
            "solved_round": 0,
            "round1_hit": False,
            "round2_hit": False,
            "round3_hit": False,
            "symptoms1": symptoms1,
            "symptoms2": symptoms2,
            "symptoms3": symptoms3,
            "q1": q1,
            "q2": q2 if has_q2 else None,
            "q3": q3 if has_q3 else None,
            "query_r1": query_r1,
            "query_r2": query_r2,
            "query_r3": query_r3,
            "error": None,
        }

        try:
            state1 = _build_round_state(
                mapping_state=mapping_state,
                query=query_r1,
                target_disease_id=target,
            )
            out1 = agent.invoke(state1)

            hit1, _ = _update_row_with_round_outputs(
                row_out, 1, out1, target, topk, symptoms1
            )

            if hit1:
                row_out["round1_hit"] = True
                row_out["solved_round"] = 1
                solved_r1_only += 1
                solved_by_r2 += 1
                solved_by_r3 += 1
                rows_out.append(row_out)
                if save_path and save_every and n % save_every == 0:
                    pd.DataFrame(rows_out).to_excel(save_path, index=False)
                if print_every > 0 and n % print_every == 0:
                    print_running()
                continue

            if not has_q2:
                rows_out.append(row_out)
                if save_path and save_every and n % save_every == 0:
                    pd.DataFrame(rows_out).to_excel(save_path, index=False)
                if print_every > 0 and n % print_every == 0:
                    print_running()
                continue

            state2 = _build_round_state(
                mapping_state=mapping_state,
                query=query_r2,
                target_disease_id=target,
                previous_groups=out1.get("previous_groups") or [],
                previous_diseases=out1.get("previous_diseases") or [],
                followup_query=q2,
            )
            out2 = agent.invoke(state2)

            hit2, _ = _update_row_with_round_outputs(
                row_out, 2, out2, target, topk, symptoms2
            )

            if hit2:
                row_out["round2_hit"] = True
                row_out["solved_round"] = 2
                solved_r2_only += 1
                solved_by_r2 += 1
                solved_by_r3 += 1
                rows_out.append(row_out)
                if save_path and save_every and n % save_every == 0:
                    pd.DataFrame(rows_out).to_excel(save_path, index=False)
                if print_every > 0 and n % print_every == 0:
                    print_running()
                continue

            if not has_q3:
                rows_out.append(row_out)
                if save_path and save_every and n % save_every == 0:
                    pd.DataFrame(rows_out).to_excel(save_path, index=False)
                if print_every > 0 and n % print_every == 0:
                    print_running()
                continue

            state3 = _build_round_state(
                mapping_state=mapping_state,
                query=query_r3,
                target_disease_id=target,
                previous_groups=out2.get("previous_groups") or [],
                previous_diseases=out2.get("previous_diseases") or [],
                followup_query=q3,
            )
            out3 = agent.invoke(state3)

            hit3, _ = _update_row_with_round_outputs(
                row_out, 3, out3, target, topk, symptoms3
            )

            if hit3:
                row_out["round3_hit"] = True
                row_out["solved_round"] = 3
                solved_r3_only += 1
                solved_by_r3 += 1

        except Exception as exc:
            row_out["error"] = str(exc)
            print(exc)
        rows_out.append(row_out)

        if save_path and save_every and n % save_every == 0:
            pd.DataFrame(rows_out).to_excel(save_path, index=False)

        if print_every > 0 and n % print_every == 0:
            print_running()

    out_df = pd.DataFrame(rows_out)

    if save_path:
        out_df.to_excel(save_path, index=False)

    summary = {
        "N": n,
        "topk": topk,
        "combine_questions": combine_questions,
        "exclusive": {
            "round1_accuracy": (solved_r1_only / n) if n else 0.0,
            "round2_accuracy": (solved_r2_only / n) if n else 0.0,
            "round3_accuracy": (solved_r3_only / n) if n else 0.0,
        },
        "cumulative": {
            "solved_by_round2_accuracy": (solved_by_r2 / n) if n else 0.0,
            "solved_by_round3_accuracy": (solved_by_r3 / n) if n else 0.0,
        },
        "counts": {
            "round1_solved_only": solved_r1_only,
            "round2_solved_only": solved_r2_only,
            "round3_solved_only": solved_r3_only,
            "solved_by_round2": solved_by_r2,
            "solved_by_round3": solved_by_r3,
            "unsolved_within_3": n - solved_by_r3,
        },
    }

    return out_df, summary