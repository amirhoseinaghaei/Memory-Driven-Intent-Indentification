"""
Fused Gromov-Wasserstein (FGW) for directed graphs.

Full (non-partial) variant: both marginals must sum to 1 and ALL mass
is transported.

q is inherited from p via shared node IDs (same semantics as the partial
version), then renormalized to sum to 1 because full FGW requires it.
"""

import traceback

import numpy as np
import ot
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def adj_to_directed_geodesic_cost(A, weight_as_similarity=False):
    """
    Directed shortest-path distances. A[i, j] = edge from i to j.
    """
    A = np.asarray(A, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.maximum(A, 0.0)
    np.fill_diagonal(A, 0.0)

    if weight_as_similarity:
        with np.errstate(divide="ignore"):
            A = np.where(
                A > 0,
                -np.log(np.clip(A, 1e-12, 1.0)),
                0.0,
            )

    D = dijkstra(csr_matrix(A), directed=True)

    finite_vals = D[np.isfinite(D)]
    max_val = np.max(finite_vals) if finite_vals.size > 0 else 1.0
    D = np.where(np.isinf(D), max_val * 2.0, D)

    np.fill_diagonal(D, 0.0)
    return D


def build_identity_feature_cost(row_ids, col_ids, mismatch_cost=1.0):
    """
    Feature cost M: 0 if same node id, mismatch_cost otherwise.
    """
    M = np.full((len(row_ids), len(col_ids)), mismatch_cost, dtype=np.float64)
    col_map = {nid: j for j, nid in enumerate(col_ids)}

    for i, nid in enumerate(row_ids):
        j = col_map.get(nid)
        if j is not None:
            M[i, j] = 0.0

    return M


# ─────────────────────────────────────────────
# Node importance
# ─────────────────────────────────────────────

def compute_node_importance(A):
    """
    IG(i) = log(1 + |N_in(i)| + |N_out(i)|), normalized to sum to 1.
    """
    A = np.asarray(A, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    A_bool = (A > 0).astype(np.float64)
    n_out = A_bool.sum(axis=1)
    n_in = A_bool.sum(axis=0)

    ig = np.log1p(n_in + n_out)
    total = ig.sum()

    if total > 1e-12:
        return ig / total

    n = A.shape[0]
    return np.full(n, 1.0 / max(n, 1), dtype=np.float64)


def compute_inherited_importance(
    complete_node_ids,
    partial_node_ids,
    complete_node_importance,
):
    """
    Build q by inheriting importance from the complete graph's p via
    shared node IDs, then renormalizing so q sums to 1.

    For full FGW, q MUST sum to 1 (unlike partial FGW). Any partial node
    not found in the complete graph gets 0 inherited mass and is then
    effectively dropped by the renormalization.

    Edge case: if no partial nodes match any complete nodes, falls back
    to a uniform distribution (otherwise q would be all zeros).
    """
    complete_map = {
        nid: complete_node_importance[i]
        for i, nid in enumerate(complete_node_ids)
    }
    q = np.array(
        [complete_map.get(nid, 0.0) for nid in partial_node_ids],
        dtype=np.float64,
    )

    total = q.sum()
    if total > 1e-12:
        return q / total

    # Fallback: no ID overlap at all -> uniform over partial nodes.
    n = len(partial_node_ids)
    return np.full(n, 1.0 / max(n, 1), dtype=np.float64)


# ─────────────────────────────────────────────
# MAIN: Directed Fused Gromov-Wasserstein (full)
# ─────────────────────────────────────────────

def fgw_directed_distance(
    A_ref,
    A_tgt,
    complete_node_ids,
    partial_node_ids,
    alpha=0.5,
    mismatch_cost=1.0,
    numItermax=2000,
):
    """
    Full Fused Gromov-Wasserstein distance between two directed graphs.

    q is inherited from p via shared node IDs and renormalized to 1.

    Returns
    -------
    dist  : float
    gamma : (n_ref, n_tgt) ndarray or None
    p, q  : ndarray or None   (both sum to 1)
    """
    try:
        A_ref = np.asarray(A_ref, dtype=np.float64)
        A_tgt = np.asarray(A_tgt, dtype=np.float64)

        n_complete = A_ref.shape[0]
        n_partial = A_tgt.shape[0]

        if n_complete == 0 or n_partial == 0:
            return 0.0, None, None, None

        if len(complete_node_ids) != n_complete or len(partial_node_ids) != n_partial:
            raise ValueError("Node id list length must match adjacency size.")

        # ── Directed structural cost matrices
        C_ref = adj_to_directed_geodesic_cost(A_ref)
        C_tgt = adj_to_directed_geodesic_cost(A_tgt)

        scale_C = max(C_ref.max(), C_tgt.max(), 1e-12)
        C_ref = C_ref / scale_C
        C_tgt = C_tgt / scale_C

        # ── Marginals
        p = compute_node_importance(A_ref)
        q = compute_inherited_importance(
            complete_node_ids=complete_node_ids,
            partial_node_ids=partial_node_ids,
            complete_node_importance=p,
        )

        # ── Feature cost (identity match)
        M = build_identity_feature_cost(
            complete_node_ids,
            partial_node_ids,
            mismatch_cost=mismatch_cost,
        )
        if M.max() > 0:
            M = M / M.max()

        # ── Full FGW optimization (no m, no partial)
        dist, log = ot.gromov.fused_gromov_wasserstein2(
            M,
            C_ref,
            C_tgt,
            p,
            q,
            alpha=alpha,
            loss_fun="square_loss",
            max_iter=numItermax,
            symmetric=False,
            log=True,
        )

        gamma = log["T"]

        return float(dist), gamma, p, q

    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return float("inf"), None, None, None