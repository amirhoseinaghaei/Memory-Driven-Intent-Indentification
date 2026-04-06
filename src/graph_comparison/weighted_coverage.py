import numpy as np


def compute_node_importance_from_adjacency(A):
    """
    Node importance = total incident weight (in + out).
    Returns normalized scores that sum to 1.
    Falls back to uniform if graph has no edges.
    """
    A = np.asarray(A, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.maximum(A, 0.0)

    importance = A.sum(axis=1) + A.sum(axis=0)
    total = importance.sum()

    if total > 1e-12:
        return importance / total

    n = len(importance)
    return np.full(n, 1.0 / max(n, 1), dtype=np.float64)


def weighted_node_coverage(A_ref, complete_node_ids, partial_node_ids):
    """
    Importance-weighted fraction of complete-graph nodes present
    in the partial graph.
    """
    node_importance = compute_node_importance_from_adjacency(A_ref)
    partial_set = set(partial_node_ids)

    numer = sum(
        float(node_importance[i])
        for i, nid in enumerate(complete_node_ids)
        if nid in partial_set
    )
    denom = float(node_importance.sum())

    return numer / denom if denom > 1e-12 else 0.0


def blended_containment_score(
    A_ref,
    A_tgt,
    complete_node_ids,
    partial_node_ids,
    w_node=1.0,
    w_edge=0.0,
    w_path=0.0,
):
    """
    Kept minimal for current usage.
    Only node coverage is used by flgw_partial_coverage.
    """
    node_cov = weighted_node_coverage(A_ref, complete_node_ids, partial_node_ids)
    containment = w_node * node_cov
    return containment, node_cov, 0.0, 0.0


def weighted_coverage(
    A_ref,
    A_tgt,
    complete_node_ids,
    partial_node_ids,
    alpha=0.5,
    numItermax=4000,
    structural_sharpness=5.0,
    w_node=1.0,
    w_edge=0.0,
    w_path=0.0,
):
    """
    Minimal version based on the current implementation logic.
    Right now the final score is exactly node coverage.
    """
    try:
        _, node_cov, _, _ = blended_containment_score(
            A_ref,
            A_tgt,
            complete_node_ids,
            partial_node_ids,
            w_node=w_node,
            w_edge=w_edge,
            w_path=w_path,
        )
        final_score = float(node_cov)
        return final_score
    except Exception:
        return 0.0