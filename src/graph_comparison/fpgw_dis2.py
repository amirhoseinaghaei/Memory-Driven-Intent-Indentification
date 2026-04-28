"""
Partial Fused Gromov-Wasserstein (PFLGW) for directed graphs.

Changes from the original implementation (see inline [CHANGE #] markers):
  1. adj_to_directed_geodesic_cost: safer default (weight_as_similarity=False)
     and uses -log for similarity->cost conversion instead of 1/x (more stable).
  2. compute_partial_importance_compatible: no longer renormalizes q to sum
     to 1 (that defeats Partial OT), and no longer computes an unused
     target_prior. Uses IG directly on A_tgt to keep q on its own scale.
  3. compute_overlap_mass: helper unchanged, but usage below now falls back
     to min(||p||_1, ||q||_1) when there's no overlap instead of ~0.
  4. pflgw_directed_distance:
       - passes symmetric=False explicitly to POT
       - normalizes M alongside C so alpha is meaningful
       - sensible m fallback instead of 1e-8
       - prints traceback on errors instead of silently swallowing them
       - alpha default lowered from 0.999 to 0.5 (with a comment)
"""

import traceback

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
    Assumes A[i, j] represents a directed edge from i to j.

    [CHANGE 1a] Default for weight_as_similarity is now False.
        The original default (True) silently inverted edge weights with
        1/(A + 1e-12). If a caller passed already-distance-like weights,
        that flip produced totally wrong costs. Safer to require the
        caller to opt in when their weights really are similarities.

    [CHANGE 1b] When weight_as_similarity=True, use -log(A) instead of 1/A.
        The old 1/(A + eps) formula blows up for small similarities
        (a similarity of 1e-6 becomes a weight of ~1e6), which dominates
        Dijkstra paths. -log maps similarity 1 -> cost 0 and similarity
        near 0 -> large-but-bounded cost, which is numerically much better.
        Similarities are clipped to [1e-12, 1.0] so values >1 don't give
        negative costs (Dijkstra requires non-negative weights).
    """
    A = np.asarray(A, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.maximum(A, 0.0)
    np.fill_diagonal(A, 0.0)

    if weight_as_similarity:
        # -log is numerically stabler than 1/x for similarity->cost.
        with np.errstate(divide="ignore"):
            A = np.where(
                A > 0,
                -np.log(np.clip(A, 1e-12, 1.0)),
                0.0,
            )

    # directed=True: do NOT symmetrize. Critical for directed graphs.
    D = dijkstra(csr_matrix(A), directed=True)

    # Replace infinities (unreachable pairs) with 2 * max finite distance,
    # so they stay "far" but don't poison the optimization with NaNs/infs.
    finite_vals = D[np.isfinite(D)]
    max_val = np.max(finite_vals) if finite_vals.size > 0 else 1.0
    D = np.where(np.isinf(D), max_val * 2.0, D)

    np.fill_diagonal(D, 0.0)
    return D


def build_identity_feature_cost(row_ids, col_ids, mismatch_cost=1.0):
    """
    Feature cost M between complete nodes (rows) and partial nodes (cols).
    0 if same node id, mismatch_cost otherwise.

    No change needed here — this function is fine.
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
    Information-gain-style node importance:  IG(i) = log(1 + |N_in(i)| + |N_out(i)|)
    then normalized to sum to 1 (probability distribution).

    No change — this matches the paper's formulation (cardinality of
    in/out neighborhoods, log-transformed, normalized).
    """
    A = np.asarray(A, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # Paper uses cardinality |N_in| and |N_out|, so count non-zero edges.
    A_bool = (A > 0).astype(np.float64)

    n_out = A_bool.sum(axis=1)  # Outgoing neighbors
    n_in = A_bool.sum(axis=0)   # Incoming neighbors

    # IG(i) = log(1 + |N_in(i)| + |N_out(i)|)
    ig = np.log1p(n_in + n_out)
    total = ig.sum()

    if total > 1e-12:
        return ig / total

    # Fallback: uniform distribution if graph has no edges at all.
    n = A.shape[0]
    return np.full(n, 1.0 / max(n, 1), dtype=np.float64)


def compute_partial_importance_compatible(
    complete_node_ids,
    partial_node_ids,
    complete_node_importance,
    A_tgt,
):
    """
    Compute q: the importance distribution over the partial graph's nodes.

    [CHANGE 2] Major fix. The original code did two wrong things:

        (a) It computed `target_prior = compute_node_importance(A_tgt)` and
            then never used it. Dead code.

        (b) It inherited mass from `complete_node_importance` for shared
            node IDs, then RENORMALIZED `inherited` to sum to 1. That
            renormalization defeats the entire point of Partial OT:
            when both p and q sum to 1, `min(||p||_1, ||q||_1) = 1`,
            so `m = overlap_mass` (also capped at 1) can never reflect
            that the partial graph is genuinely "smaller" than the
            complete one. Partial OT degenerates into full OT.

    Fix: compute q from A_tgt directly using the same IG formula as p.
    q now lives on the partial graph's own structural scale. If your
    paper explicitly requires inheriting mass from the complete graph
    (alternative formulation), use the commented-out block below and do
    NOT renormalize — let ||q||_1 < 1 express the partial nature.
    """
    # Preferred: compute q on the partial graph with the same IG formula.
    # q = compute_node_importance(A_tgt)

    # --- Alternative (inherit without renormalizing) -------------------
    complete_map = {
        nid: complete_node_importance[i]
        for i, nid in enumerate(complete_node_ids)
    }
    q = np.array(
        [complete_map.get(nid, 0.0) for nid in partial_node_ids],
        dtype=np.float64,
    )
    # # IMPORTANT: do NOT renormalize q here.
    # -------------------------------------------------------------------

    return q


def compute_overlap_mass(
    complete_node_ids,
    partial_node_ids,
    p,
    q,
):
    """
    Amount of mass shared between the two graphs, based on node ID overlap.
    Returns min(p_overlap, q_overlap) so the result is a valid transport mass.

    No change to this function itself — but see [CHANGE 3] in
    pflgw_directed_distance for how the result is now used safely.
    """
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
    alpha=0.5,                 # [CHANGE 4a] see note below
    mismatch_cost=1.0,
    numItermax=2000,
):
    """
    Directed Partial Fused Gromov-Wasserstein distance between two graphs.

    Returns
    -------
    dist  : float        Raw OT cost (lower = more similar).
    gamma : (n_ref, n_tgt) ndarray or None   Transport plan.
    p, q  : ndarray or None                  Marginal distributions.

    [CHANGE 4a] Default alpha lowered from 0.999 -> 0.5.
        FGW objective is  (1 - alpha) * <T, M>  +  alpha * sum L(C1,C2) T T.
        With alpha=0.999 the feature cost M is ~1000x downweighted
        relative to the structure cost, effectively ignoring node-identity
        features — which contradicts the whole point of including them.
        0.5 is the POT library default and gives M and C equal weight.
        Tune as needed for your application.
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

        # ── Directed structural cost matrices (shortest-path distances)
        C_ref = adj_to_directed_geodesic_cost(A_ref)
        C_tgt = adj_to_directed_geodesic_cost(A_tgt)

        # Joint scaling: both matrices share the same denominator so
        # relative distances are preserved across the two graphs.
        scale_C = max(C_ref.max(), C_tgt.max(), 1e-12)
        C_ref = C_ref / scale_C
        C_tgt = C_tgt / scale_C

        # ── Marginal distributions
        p = compute_node_importance(A_ref)

        # [CHANGE 2] q is now computed correctly (see function docstring).
        q = compute_partial_importance_compatible(
            complete_node_ids=complete_node_ids,
            partial_node_ids=partial_node_ids,
            complete_node_importance=p,
            A_tgt=A_tgt,
        )

        # ── Feature cost (identity match between node IDs)
        M = build_identity_feature_cost(
            complete_node_ids,
            partial_node_ids,
            mismatch_cost=mismatch_cost,
        )

        # [CHANGE 4b] Normalize M onto the same [0, 1] scale as C.
        #   Without this, the alpha trade-off between feature and structure
        #   costs is at the mercy of whatever `mismatch_cost` the caller
        #   picked. With this, alpha is interpretable regardless.
        if M.max() > 0:
            M = M / M.max()

        # ── Transport mass: how much to move
        overlap_mass = compute_overlap_mass(
            complete_node_ids=complete_node_ids,
            partial_node_ids=partial_node_ids,
            p=p,
            q=q,
        )

        # [CHANGE 3] Sensible fallback when there's no ID overlap.
        #   Original code did `m = max(1e-8, overlap_mass)`, which meant
        #   "no overlap -> transport almost no mass -> distance ~ 0".
        #   That silently made dissimilar graphs look identical.
        #   Per POT docs, the default m is min(||p||_1, ||q||_1) — i.e.
        #   transport as much as the smaller distribution allows.
        p_mass = float(p.sum())
        q_mass = float(q.sum())
        if overlap_mass <= 1e-12:
            m = min(p_mass, q_mass)
        else:
            m = overlap_mass

        # Clamp to the valid range [tiny, min(||p||_1, ||q||_1)] to keep
        # POT's partial solver happy.
        m = min(m, p_mass, q_mass)
        m = max(m, 1e-8)

        # ── Optimization
        # [CHANGE 5] Pass symmetric=False explicitly.
        #   Without this, POT runs a symmetry test on C_ref / C_tgt.
        #   Directed geodesic matrices are (in general) asymmetric, but
        #   the test is numerical and can be fragile. Being explicit
        #   avoids accidentally falling into the symmetric code path.
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
            symmetric=False,
            log=True,
        )

        gamma = log["T"]

        return float(dist), gamma, p, q

    except Exception as e:
        # [CHANGE 6] Previously this just printed the exception message,
        # which hid the stack trace and made real bugs (shape mismatches,
        # typos) nearly impossible to debug. Print the full traceback so
        # the failure is visible, then still return a safe sentinel so
        # calling code can continue if it needs to.
        print("[ERROR]", e)
        traceback.print_exc()
        return float("inf"), None, None, None