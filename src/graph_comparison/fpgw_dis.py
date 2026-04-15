


import numpy as np
import ot
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def adj_to_directed_geodesic_cost(A, weight_as_similarity=True):
    """
    Computes directed shortest-path distances.
    Assumes A represents directed edges A[i, j] (from i to j).
    """
    A = np.asarray(A, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.maximum(A, 0.0)
    np.fill_diagonal(A, 0.0)

    if weight_as_similarity:
        A = np.where(A > 0, 1.0 / (A + 1e-12), 0.0)

    # REMOVED symmetrization. Using directed=True for Dijkstra.
    D = dijkstra(csr_matrix(A), directed=True)

    # Cap infinite distances (unreachable directed nodes)
    finite_vals = D[np.isfinite(D)]
    max_val = np.max(finite_vals) if finite_vals.size > 0 else 1.0
    D = np.where(np.isinf(D), max_val * 2.0, D)

    np.fill_diagonal(D, 0.0)
    return D


def build_identity_feature_cost(row_ids, col_ids, mismatch_cost=1.0):
    """
    Feature cost M between complete nodes (rows) and partial nodes (cols).
    0 if same node id, mismatch_cost otherwise.
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

# def compute_node_importance(A):
#     A = np.asarray(A, dtype=np.float64)
#     A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
#     A = np.maximum(A, 0.0)

#     # Importance combines both outgoing and incoming edges
#     importance = A.sum(axis=1) + A.sum(axis=0)
#     total = importance.sum()

#     if total > 1e-12:
#         return importance / total

#     n = A.shape[0]
#     return np.full(n, 1.0 / max(n, 1), dtype=np.float64)

def compute_node_importance(A):
    A = np.asarray(A, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Paper uses cardinality |N_in| and |N_out|, so we count non-zero edges
    A_bool = (A > 0).astype(np.float64)
    
    n_out = A_bool.sum(axis=1) # Outgoing neighbors
    n_in = A_bool.sum(axis=0)  # Incoming neighbors
    
    # Apply IG(i) = log(1 + |N_in(i)| + |N_out(i)|)
    ig = np.log1p(n_in + n_out) 
    total = ig.sum()

    if total > 1e-12:
        return ig / total

    n = A.shape[0]
    return np.full(n, 1.0 / max(n, 1), dtype=np.float64)


def compute_partial_importance_compatible(
    complete_node_ids,
    partial_node_ids,
    complete_node_importance,
    A_tgt,
):
    target_prior = compute_node_importance(A_tgt)

    complete_map = {
        nid: complete_node_importance[i]
        for i, nid in enumerate(complete_node_ids)
    }

    inherited = np.array(
        [complete_map.get(nid, 0.0) for nid in partial_node_ids],
        dtype=np.float64,
    )

    inherited_sum = inherited.sum()
    if inherited_sum > 1e-12:
        inherited = inherited / inherited_sum
    return inherited
    


def compute_overlap_mass(
    complete_node_ids,
    partial_node_ids,
    p,
    q,
):
    complete_index = {nid: i for i, nid in enumerate(complete_node_ids)}
    partial_index = {nid: i for i, nid in enumerate(partial_node_ids)}

    overlap_ids = set(complete_index).intersection(partial_index)
    if not overlap_ids:
        return 0.0

    p_overlap = sum(p[complete_index[nid]] for nid in overlap_ids)
    q_overlap = sum(q[partial_index[nid]] for nid in overlap_ids)

    return float(min(p_overlap, q_overlap))


# ─────────────────────────────────────────────
# MAIN: Directed Partial Fused Gromov-Wasserstein
# ─────────────────────────────────────────────

def pflgw_directed_distance(
    A_ref,
    A_tgt,
    complete_node_ids,
    partial_node_ids,
    alpha=0.5,
    mismatch_cost=1.0,
    numItermax=1000,
):
    """
    Computes the Partial Fused Gromov-Wasserstein distance between two directed graphs.
    Returns:
        dist: The raw optimal transport cost (lower is better/more similar).
        gamma: The transport plan matrix.
        p, q: The marginal distributions of the graphs.
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

        # ── Directed Cost matrices
        C_ref = adj_to_directed_geodesic_cost(A_ref)
        C_tgt = adj_to_directed_geodesic_cost(A_tgt)

        scale = max(C_ref.max(), C_tgt.max(), 1e-12)
        C_ref = C_ref / scale
        C_tgt = C_tgt / scale

        # ── Distributions
        p = compute_node_importance(A_ref)


        q = compute_partial_importance_compatible(
            complete_node_ids=complete_node_ids,
            partial_node_ids=partial_node_ids,
            complete_node_importance=p,
            A_tgt=A_tgt        
            )

        # ── Feature cost
        M = build_identity_feature_cost(
            complete_node_ids,
            partial_node_ids,
            mismatch_cost=mismatch_cost,
        )
        
        overlap_mass = compute_overlap_mass(
            complete_node_ids=complete_node_ids,
            partial_node_ids=partial_node_ids,
            p=p,
            q=q,
        )
        
        m = max(1e-8, overlap_mass)
        
        # ── Optimization
        dist, log = ot.gromov.partial_fused_gromov_wasserstein2(
            M,
            C_ref,
            C_tgt,
            p,
            q,
            m=m,
            alpha=alpha,
            loss_fun="square_loss",
            numItermax=numItermax,
            log=True,
        )
        
        gamma = log["T"]
        
        # Return the raw calculated distance instead of coverage
        return float(dist), gamma, p, q

    except Exception as e:
        print("[ERROR]", e)
        return float('inf'), None, None, None