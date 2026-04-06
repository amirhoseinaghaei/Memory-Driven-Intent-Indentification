


# # import numpy as np
# # import ot
# # from scipy.sparse import csr_matrix
# # from scipy.sparse.csgraph import dijkstra

# # from Linearized_Partial_Gromov_Wasserstein.Linearized_Partial_Gromov_Wasserstein.lib.linear_gromov import LGW_embedding, LGW_dist


# # def adj_to_geodesic_cost(A, weight_as_similarity=False, disconnected_fill="max_times_2"):
# #     A = np.asarray(A, dtype=np.float64)
# #     A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
# #     A = np.maximum(A, 0.0)
# #     np.fill_diagonal(A, 0.0)

# #     if weight_as_similarity:
# #         A = np.where(A > 0, 1.0 / (A + 1e-12), 0.0)

# #     A_sym = 0.5 * (A + A.T)
# #     D = dijkstra(csr_matrix(A_sym), directed=False)

# #     finite_vals = D[np.isfinite(D) & (D > 0)]
# #     max_val = finite_vals.max() if finite_vals.size > 0 else 1.0

# #     if disconnected_fill == "max_times_2":
# #         fill = max_val * 2.0
# #     elif disconnected_fill == "max_times_1":
# #         fill = max_val
# #     else:
# #         fill = float(disconnected_fill)

# #     D = np.where(np.isinf(D), fill, D)
# #     np.fill_diagonal(D, 0.0)
# #     return D


# # def normalize_cost(C, eps=1e-12):
# #     C = np.asarray(C, dtype=np.float64)
# #     mx = C.max()
# #     return C / (mx + eps) if mx > eps else C


# # def exact_node_coverage(partial_node_ids, complete_node_ids):
# #     """
# #     Exact coverage when node identities are known.
# #     """
# #     partial_set = set(partial_node_ids)
# #     complete_set = set(complete_node_ids)
# #     if not complete_set:
# #         return 0.0
# #     return len(partial_set & complete_set) / len(complete_set)


# # def transport_coverage_soft(gamma, q, eps=1e-12):
# #     """
# #     Soft node coverage of complete graph by partial graph.

# #     gamma: shape (n_partial, n_complete)
# #     q:     target/complete graph node masses, shape (n_complete,)

# #     For each complete node j, look at incoming transported mass:
# #         m_j = sum_i gamma[i, j]

# #     Normalize by q_j, cap at 1, then average.
# #     """
# #     incoming = gamma.sum(axis=0)  # mass received by each complete-graph node
# #     ratio = incoming / (q + eps)
# #     covered = np.minimum(ratio, 1.0)
# #     return float(np.mean(covered))


# # def transport_coverage_hard(gamma, q, threshold_ratio=0.2, eps=1e-12):
# #     """
# #     Hard node coverage of complete graph by partial graph.

# #     A complete node is considered covered if received mass >= threshold_ratio * q_j.
# #     """
# #     incoming = gamma.sum(axis=0)
# #     covered = incoming >= (threshold_ratio * q + eps)
# #     return float(np.mean(covered.astype(np.float64)))


# # def structure_only_distance_two_graphs(
# #     A_ref,                  # partial graph
# #     A_tgt,                  # complete graph
# #     dim=20,
# #     use_unbalanced=False,
# #     ugw_eps=1.0,
# #     rho1=1.0, rho2=1.0,
# #     numItermax=5000,
# #     alpha=0,              # weight for structural closeness
# #     beta=1.0,               # exponent for multiplicative penalty
# #     combine_mode="weighted_sum",   # "weighted_sum" or "product"
# #     partial_node_ids=None,
# #     complete_node_ids=None,
# #     use_exact_coverage=True,
# # ):
# #     """
# #     Returns:
# #         dist
# #         structural_closeness
# #         coverage
# #         final_score
# #         gamma
# #     """
# #     n_ref, n_tgt = A_ref.shape[0], A_tgt.shape[0]

# #     C_ref = normalize_cost(adj_to_geodesic_cost(A_ref))
# #     C_tgt = normalize_cost(adj_to_geodesic_cost(A_tgt))

# #     p = np.full(n_ref, 1.0 / n_ref, dtype=np.float64)
# #     q = np.full(n_tgt, 1.0 / n_tgt, dtype=np.float64)

# #     gamma = ot.gromov.gromov_wasserstein(
# #         C_ref, C_tgt, p, q, log=False,
# #         max_iter=max(numItermax, 500 * n_ref),
# #     )

# #     if np.array_equal(A_ref, A_tgt):
# #         gamma = np.diag(p)

# #     pos_ref = C_ref
# #     pos_tgt = C_tgt

# #     emb_ref, _ = LGW_embedding(C_ref, pos_ref, p, p, gamma=np.diag(p), loss="square")
# #     emb_tgt, _ = LGW_embedding(C_ref, pos_tgt, p, q, gamma=gamma, loss="square")

# #     d2 = LGW_dist(emb_ref, emb_tgt, p)
# #     dist = float(np.sqrt(max(d2, 0.0)))

# #     structural_closeness = 1.0 / (1.0 + dist + 1e-12)

# #     # Coverage
# #     if use_exact_coverage:
# #         if partial_node_ids is None or complete_node_ids is None:
# #             raise ValueError("partial_node_ids and complete_node_ids are required when use_exact_coverage=True")
# #         coverage = exact_node_coverage(partial_node_ids, complete_node_ids)
# #     else:
# #         coverage = transport_coverage_soft(gamma, q)

# #     # Combine
# #     if combine_mode == "weighted_sum":
# #         final_score = alpha * structural_closeness + (1.0 - alpha) * coverage
# #     elif combine_mode == "product":
# #         final_score = structural_closeness * (coverage ** beta)
# #     else:
# #         raise ValueError("combine_mode must be 'weighted_sum' or 'product'")

# #     return dist, final_score



# import numpy as np
# import ot
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import dijkstra
# from sklearn.metrics import pairwise_distances

# from Linearized_Partial_Gromov_Wasserstein.Linearized_Partial_Gromov_Wasserstein.lib.linear_gromov import LGW_embedding, LGW_dist


# def adj_to_geodesic_cost(A, weight_as_similarity=False, disconnected_fill="max_times_2"):
#     A = np.asarray(A, dtype=np.float64)
#     A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
#     A = np.maximum(A, 0.0)
#     np.fill_diagonal(A, 0.0)

#     if weight_as_similarity:
#         A = np.where(A > 0, 1.0 / (A + 1e-12), 0.0)

#     A_sym = 0.5 * (A + A.T)
#     D = dijkstra(csr_matrix(A_sym), directed=False)

#     finite_vals = D[np.isfinite(D) & (D > 0)]
#     max_val = finite_vals.max() if finite_vals.size > 0 else 1.0

#     if disconnected_fill == "max_times_2":
#         fill = max_val * 2.0
#     elif disconnected_fill == "max_times_1":
#         fill = max_val
#     else:
#         fill = float(disconnected_fill)

#     D = np.where(np.isinf(D), fill, D)
#     np.fill_diagonal(D, 0.0)
#     return D


# def normalize_cost(C, eps=1e-12):
#     C = np.asarray(C, dtype=np.float64)
#     mx = C.max()
#     return C / (mx + eps) if mx > eps else C

# def build_identity_feature_cost(partial_node_ids, complete_node_ids, mismatch_cost=1.0):
#     """
#     M[i, j] = 0 if node IDs match, else mismatch_cost
#     """
#     M = np.full((len(partial_node_ids), len(complete_node_ids)),
#                 mismatch_cost, dtype=np.float64)

#     complete_index = {nid: j for j, nid in enumerate(complete_node_ids)}

#     for i, nid in enumerate(partial_node_ids):
#         j = complete_index.get(nid)
#         if j is not None:
#             M[i, j] = 0.0

#     return M



# def exact_node_coverage(partial_node_ids, complete_node_ids):
#     """
#     Exact coverage when node identities are known.
#     """
#     partial_set = set(partial_node_ids)
#     complete_set = set(complete_node_ids)
#     if not complete_set:
#         return 0.0
#     return len(partial_set & complete_set) / len(complete_set)


# def transport_coverage_soft(gamma, q, eps=1e-12):
#     """
#     Soft node coverage of complete graph by partial graph.

#     gamma: shape (n_partial, n_complete)
#     q:     target/complete graph node masses, shape (n_complete,)

#     For each complete node j, look at incoming transported mass:
#         m_j = sum_i gamma[i, j]

#     Normalize by q_j, cap at 1, then average.
#     """
#     incoming = gamma.sum(axis=0)  # mass received by each complete-graph node
#     ratio = incoming / (q + eps)
#     covered = np.minimum(ratio, 1.0)
#     return float(np.mean(covered))


# import numpy as np
# from sklearn.metrics import pairwise_distances


# def build_padded_onehot_feature_matrices(
#     partial_present_node_ids,
#     complete_node_ids,
#     metric="euclidean",
#     normalize=True,
# ):
#     """
#     Build feature matrices when A_ref has already been padded into the same
#     node order/space as A_tgt.

#     Assumption
#     ----------
#     - A_ref and A_tgt are both indexed by complete_node_ids
#     - partial_present_node_ids contains only the nodes that truly exist
#       in the partial graph
#     - nodes missing from partial graph get an all-zero feature row

#     Returns
#     -------
#     X_partial : (n, n)
#     X_complete: (n, n)
#     M         : (n, n)
#     """
#     n = len(complete_node_ids)
#     node_to_idx = {nid: i for i, nid in enumerate(complete_node_ids)}

#     # complete graph: one-hot identity matrix
#     X_complete = np.eye(n, dtype=np.float64)

#     # partial graph: zero rows for missing nodes
#     X_partial = np.zeros((n, n), dtype=np.float64)

#     partial_present_set = set(partial_present_node_ids)

#     for nid in partial_present_set:
#         if nid in node_to_idx:
#             i = node_to_idx[nid]
#             X_partial[i, i] = 1.0

#     M = pairwise_distances(X_partial, X_complete, metric=metric)

#     if normalize:
#         mx = M.max()
#         if mx > 1e-12:
#             M = M / mx

#     return X_partial, X_complete, M

# def build_feature_cost_from_node_ids(
#     partial_node_ids,
#     complete_node_ids,
#     metric="euclidean",
#     normalize=True,
# ):
#     """
#     Returns
#     -------
#     M          : (n_partial, n_complete) feature distance matrix
#     X_partial  : feature matrix for partial graph
#     X_complete : feature matrix for complete graph
#     """
#     X_partial, X_complete, all_ids, id_to_col = build_onehot_feature_matrices_from_node_ids(
#         partial_node_ids, complete_node_ids
#     )

#     M = pairwise_distances(X_partial, X_complete, metric=metric)

#     if normalize:
#         M = normalize_cost(M)

#     return M, X_partial, X_complete, all_ids, id_to_col

# def transport_coverage_hard(gamma, q, threshold_ratio=0.2, eps=1e-12):
#     """
#     Hard node coverage of complete graph by partial graph.

#     A complete node is considered covered if received mass >= threshold_ratio * q_j.
#     """
#     incoming = gamma.sum(axis=0)
#     covered = incoming >= (threshold_ratio * q + eps)
#     return float(np.mean(covered.astype(np.float64)))

# def build_identity_feature_cost(partial_node_ids, complete_node_ids, mismatch_cost=1.0):
#     """
#     Build M where:
#       M[i, j] = 0 if node IDs match
#       M[i, j] = mismatch_cost otherwise
#     """
#     M = np.full((len(partial_node_ids), len(complete_node_ids)),
#                 mismatch_cost, dtype=np.float64)

#     complete_index = {nid: j for j, nid in enumerate(complete_node_ids)}

#     for i, nid in enumerate(partial_node_ids):
#         j = complete_index.get(nid)
#         if j is not None:
#             M[i, j] = 0.0

#     return M


# # def structure_only_distance_two_graphs(
# #     A_ref,                  # partial graph
# #     A_tgt,                  # complete graph
# #     dim=20,
# #     use_unbalanced=False,
# #     ugw_eps=1.0,
# #     rho1=1.0, rho2=1.0,
# #     numItermax=5000,
# #     alpha=0,              # weight for structural closeness
# #     beta=1.0,               # exponent for multiplicative penalty
# #     combine_mode="weighted_sum",   # "weighted_sum" or "product"
# #     partial_node_ids=None,
# #     complete_node_ids=None,
# #     use_exact_coverage=False,
# # ):
# #     """
# #     Returns:
# #         dist
# #         structural_closeness
# #         coverage
# #         final_score
# #         gamma
# #     """
# #     n_ref, n_tgt = A_ref.shape[0], A_tgt.shape[0]

# #     C_ref = normalize_cost(adj_to_geodesic_cost(A_ref))
# #     C_tgt = normalize_cost(adj_to_geodesic_cost(A_tgt))

# #     p = np.full(n_ref, 1.0 / n_ref, dtype=np.float64)
# #     q = np.full(n_tgt, 1.0 / n_tgt, dtype=np.float64)

# #     gamma = ot.gromov.gromov_wasserstein(
# #         C_ref, C_tgt, p, q, log=False,
# #         max_iter=max(numItermax, 500 * n_ref),
# #     )

# #     if np.array_equal(A_ref, A_tgt):
# #         gamma = np.diag(p)

# #     pos_ref = C_ref
# #     pos_tgt = C_tgt

# #     emb_ref, _ = LGW_embedding(C_ref, pos_ref, p, p, gamma=np.diag(p), loss="square")
# #     emb_tgt, _ = LGW_embedding(C_ref, pos_tgt, p, q, gamma=gamma, loss="square")

# #     d2 = LGW_dist(emb_ref, emb_tgt, p)
# #     dist = float(np.sqrt(max(d2, 0.0)))

# #     structural_closeness = 1.0 / (1.0 + dist + 1e-12)

# #     # Coverage
# #     if use_exact_coverage:
# #         if partial_node_ids is None or complete_node_ids is None:
# #             raise ValueError("partial_node_ids and complete_node_ids are required when use_exact_coverage=True")
# #         coverage = exact_node_coverage(partial_node_ids, complete_node_ids)
# #     else:
# #         coverage = transport_coverage_soft(gamma, q)

# #     # Combine
# #     if combine_mode == "weighted_sum":
# #         final_score = alpha * structural_closeness + (1.0 - alpha) * coverage
# #     elif combine_mode == "product":
# #         final_score = structural_closeness * (coverage ** beta)
# #     else:
# #         raise ValueError("combine_mode must be 'weighted_sum' or 'product'")

# #     return dist, final_score



# def build_padded_identity_cost_matrix(partial_present_node_ids, complete_node_ids, mismatch_cost=1.0):
#     """
#     Build an (n, n) feature cost matrix M where n = len(complete_node_ids).

#     Interpretation:
#     - row i corresponds to padded partial node position i
#     - col j corresponds to complete node position j

#     If node i exists in the partial graph:
#         M[i, i] = 0
#         M[i, j!=i] = mismatch_cost

#     If node i is missing in the partial graph:
#         M[i, j] = mismatch_cost for all j
#     """
#     n = len(complete_node_ids)
#     M = np.full((n, n), mismatch_cost, dtype=np.float64)

#     node_to_idx = {nid: i for i, nid in enumerate(complete_node_ids)}
#     partial_present_set = set(partial_present_node_ids)

#     for nid in partial_present_set:
#         if nid in node_to_idx:
#             i = node_to_idx[nid]
#             M[i, i] = 0.0

#     return M

# def build_identity_feature_cost(partial_node_ids, complete_node_ids, mismatch_cost=1.0):
#     """
#     Build feature mismatch matrix M when partial adjacency is NOT padded.

#     M[i, j] = 0 if partial_node_ids[i] == complete_node_ids[j]
#             = mismatch_cost otherwise

#     Shape
#     -----
#     (n_partial, n_complete)
#     """
#     n_partial = len(partial_node_ids)
#     n_complete = len(complete_node_ids)

#     M = np.full((n_partial, n_complete), mismatch_cost, dtype=np.float64)

#     complete_index = {nid: j for j, nid in enumerate(complete_node_ids)}

#     for i, nid in enumerate(partial_node_ids):
#         j = complete_index.get(nid)
#         if j is not None:
#             M[i, j] = 0.0

#     return M


# def structure_linearized_distance_plus_feature_mismatch(
#     A_ref,                  # partial graph
#     A_tgt,                  # complete graph
#     dim=20,
#     numItermax=5000,
#     alpha=0.6,              # final score weight for closeness
#     beta=1.0,
#     combine_mode="weighted_sum",
#     partial_node_ids=None,
#     complete_node_ids=None,
#     use_exact_coverage=True,
#     feature_lambda=1,     # weight for feature mismatch term
#     use_identity_M=True,    # True: direct 0/1 identity cost, False: one-hot+pairwise_distances
# ):
#     """
#     1) Compute structural LGW distance exactly as before
#     2) Compute FGW-like feature mismatch term <gamma, M>
#     3) Add that term to the structural distance
#     """
#     n_ref, n_tgt = A_ref.shape[0], A_tgt.shape[0]



#     # ---------------------------------------------------------
#     # 1) Structural cost matrices
#     # ---------------------------------------------------------
#     C_ref = normalize_cost(adj_to_geodesic_cost(A_ref))
#     C_tgt = normalize_cost(adj_to_geodesic_cost(A_tgt))

#     # ---------------------------------------------------------
#     # 2) Uniform masses
#     # ---------------------------------------------------------
#     p = np.full(n_ref, 1.0 / n_ref, dtype=np.float64)
#     q = np.full(n_tgt, 1.0 / n_tgt, dtype=np.float64)

#     # ---------------------------------------------------------
#     # 3) Structural transport exactly as before
#     # ---------------------------------------------------------
#     gamma = ot.gromov.gromov_wasserstein(
#         C_ref, C_tgt, p, q, log=False,
#         max_iter=max(numItermax, 500 * n_ref),
#     )
#     M = build_identity_feature_cost(complete_node_ids, partial_node_ids)
#     feature_mismatch = float(np.sum(gamma * M))

#     if np.array_equal(A_ref, A_tgt) and n_ref == n_tgt:
#         gamma = np.diag(p)

#     # ---------------------------------------------------------
#     # 4) Structural LGW distance exactly as before
#     # ---------------------------------------------------------
#     pos_ref = C_ref
#     pos_tgt = C_tgt

#     emb_ref, _ = LGW_embedding(
#         C_ref, pos_ref, p, p,
#         gamma=np.diag(p),
#         loss="square"
#     )

#     emb_tgt, _ = LGW_embedding(
#         C_ref, pos_tgt, p, q,
#         gamma=gamma,
#         loss="square"
#     )

#     d2 = LGW_dist(emb_ref, emb_tgt, p)
#     dist_struct = float(np.sqrt(max(d2, 0.0)))


#     # ---------------------------------------------------------
#     # 6) Final distance
#     # ---------------------------------------------------------
#     try:


#         feature_closeness = 1.0 - feature_mismatch
#         structural_closeness = 1.0 / (1.0 + dist_struct + 1e-12)

#         closeness_final = (1- feature_lambda)* structural_closeness + feature_lambda * feature_closeness

#         distance_final = 1.0 / (1.0 + closeness_final)

        

#     except Exception as e: 
#         print(e)
#     return distance_final, float(closeness_final)


# def flgw_partial_coverage(
#     A_ref,                  # complete graph
#     A_tgt,                  # partial graph
#     complete_node_ids,
#     partial_node_ids,
#     alpha=1,              # 0 = pure structure, 1 = pure feature identity
#     numItermax=1000,
# ):
#     n_complete = A_ref.shape[0]
#     n_partial  = A_tgt.shape[0]

#     C_ref = normalize_cost(adj_to_geodesic_cost(A_ref))  # (n_complete, n_complete)
#     C_tgt = normalize_cost(adj_to_geodesic_cost(A_tgt))  # (n_partial,  n_partial)

#     p = np.full(n_complete, 1.0 / n_complete)
#     q = np.full(n_partial,  1.0 / n_partial)

#     # M[i, j] = 0 if complete_node_ids[i] == partial_node_ids[j], else 1
#     # shape: (n_complete, n_partial)  — matches gamma
#     M = build_identity_feature_cost(complete_node_ids, partial_node_ids)

#     # KEY: m encodes the expected coverage ratio.
#     # Partial FGW only transports this fraction of total mass,
#     # so complete nodes with no partial counterpart get zero outgoing mass.
#     m = n_partial / n_complete   # ∈ (0, 1]

#     # Partial FGW transport plan
#     # gamma shape: (n_complete, n_partial), sum(gamma) = m

#     gamma, log = ot.gromov.partial_gromov_wasserstein(
#         C_ref, C_tgt, p, q,
#         m=m,
#         loss_fun="square_loss",
#         numItermax=numItermax,
#         log=True,
#     )

#     # --- Coverage ---
#     # outgoing[i] = total mass sent from complete node i
#     # for covered nodes this ≈ 1/n_complete, for uncovered ≈ 0
#     outgoing = gamma.sum(axis=1)                          # (n_complete,)
#     ratio    = outgoing / (p + 1e-12)
#     coverage_soft = float(np.mean(np.minimum(ratio, 1.0)))

#     # --- Feature mismatch (normalized to [0,1]) ---
#     # divide by m so it's the average mismatch over *transported* mass only
#     feature_mismatch = float(np.sum(gamma * M)) / (m + 1e-12)

#     # --- Structural distance via LGW ---
#     # use partial plan for embedding — projects complete graph into partial space
#     emb_ref, _ = LGW_embedding(C_ref, C_ref, p, p, gamma=np.diag(p), loss="square")
#     emb_tgt, _ = LGW_embedding(C_ref, C_tgt, p, q, gamma=gamma,      loss="square")
#     d2 = LGW_dist(emb_ref, emb_tgt, p)
#     dist_struct = float(np.sqrt(max(d2, 0.0)))

#     # --- Final score ---
#     # structural closeness of the MATCHED subgraph only
#     structural_closeness = 1.0 / (1.0 + dist_struct + 1e-12)
#     feature_closeness    = 1.0 - feature_mismatch

#     # multiply by coverage: perfect structural match on 50% of graph → 0.5 score
#     matched_closeness = alpha * feature_closeness + (1.0 - alpha) * structural_closeness
#     final_score       =  matched_closeness
#     return final_score, coverage_soft, dist_struct, gamma




# import numpy as np
# import ot
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import dijkstra
# from sklearn.metrics import pairwise_distances

# from Linearized_Partial_Gromov_Wasserstein.lib.linear_gromov import LGW_embedding, LGW_dist


# def adj_to_geodesic_cost(A, weight_as_similarity=False, disconnected_fill="max_times_2"):
#     A = np.asarray(A, dtype=np.float64)
#     A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
#     A = np.maximum(A, 0.0)
#     np.fill_diagonal(A, 0.0)

#     if weight_as_similarity:
#         A = np.where(A > 0, 1.0 / (A + 1e-12), 0.0)

#     A_sym = 0.5 * (A + A.T)
#     D = dijkstra(csr_matrix(A_sym), directed=False)

#     finite_vals = D[np.isfinite(D) & (D > 0)]
#     max_val = finite_vals.max() if finite_vals.size > 0 else 1.0

#     if disconnected_fill == "max_times_2":
#         fill = max_val * 2.0
#     elif disconnected_fill == "max_times_1":
#         fill = max_val
#     else:
#         fill = float(disconnected_fill)

#     D = np.where(np.isinf(D), fill, D)
#     np.fill_diagonal(D, 0.0)
#     return D


# def normalize_cost(C, eps=1e-12):
#     C = np.asarray(C, dtype=np.float64)
#     mx = C.max()
#     return C / (mx + eps) if mx > eps else C

# def build_identity_feature_cost(partial_node_ids, complete_node_ids, mismatch_cost=1.0):
#     """
#     M[i, j] = 0 if node IDs match, else mismatch_cost
#     """
#     M = np.full((len(partial_node_ids), len(complete_node_ids)),
#                 mismatch_cost, dtype=np.float64)

#     complete_index = {nid: j for j, nid in enumerate(complete_node_ids)}

#     for i, nid in enumerate(partial_node_ids):
#         j = complete_index.get(nid)
#         if j is not None:
#             M[i, j] = 0.0

#     return M



# def exact_node_coverage(partial_node_ids, complete_node_ids):
#     """
#     Exact coverage when node identities are known.
#     """
#     partial_set = set(partial_node_ids)
#     complete_set = set(complete_node_ids)
#     if not complete_set:
#         return 0.0
#     return len(partial_set & complete_set) / len(complete_set)


# def transport_coverage_soft(gamma, q, eps=1e-12):
#     """
#     Soft node coverage of complete graph by partial graph.

#     gamma: shape (n_partial, n_complete)
#     q:     target/complete graph node masses, shape (n_complete,)

#     For each complete node j, look at incoming transported mass:
#         m_j = sum_i gamma[i, j]

#     Normalize by q_j, cap at 1, then average.
#     """
#     incoming = gamma.sum(axis=0)  # mass received by each complete-graph node
#     ratio = incoming / (q + eps)
#     covered = np.minimum(ratio, 1.0)
#     return float(np.mean(covered))


# import numpy as np
# from sklearn.metrics import pairwise_distances


# def build_padded_onehot_feature_matrices(
#     partial_present_node_ids,
#     complete_node_ids,
#     metric="euclidean",
#     normalize=True,
# ):
#     """
#     Build feature matrices when A_ref has already been padded into the same
#     node order/space as A_tgt.

#     Assumption
#     ----------
#     - A_ref and A_tgt are both indexed by complete_node_ids
#     - partial_present_node_ids contains only the nodes that truly exist
#       in the partial graph
#     - nodes missing from partial graph get an all-zero feature row

#     Returns
#     -------
#     X_partial : (n, n)
#     X_complete: (n, n)
#     M         : (n, n)
#     """
#     n = len(complete_node_ids)
#     node_to_idx = {nid: i for i, nid in enumerate(complete_node_ids)}

#     # complete graph: one-hot identity matrix
#     X_complete = np.eye(n, dtype=np.float64)

#     # partial graph: zero rows for missing nodes
#     X_partial = np.zeros((n, n), dtype=np.float64)

#     partial_present_set = set(partial_present_node_ids)

#     for nid in partial_present_set:
#         if nid in node_to_idx:
#             i = node_to_idx[nid]
#             X_partial[i, i] = 1.0

#     M = pairwise_distances(X_partial, X_complete, metric=metric)

#     if normalize:
#         mx = M.max()
#         if mx > 1e-12:
#             M = M / mx

#     return X_partial, X_complete, M

# def build_feature_cost_from_node_ids(
#     partial_node_ids,
#     complete_node_ids,
#     metric="euclidean",
#     normalize=True,
# ):
#     """
#     Returns
#     -------
#     M          : (n_partial, n_complete) feature distance matrix
#     X_partial  : feature matrix for partial graph
#     X_complete : feature matrix for complete graph
#     """
#     X_partial, X_complete, all_ids, id_to_col = build_onehot_feature_matrices_from_node_ids(
#         partial_node_ids, complete_node_ids
#     )

#     M = pairwise_distances(X_partial, X_complete, metric=metric)

#     if normalize:
#         M = normalize_cost(M)

#     return M, X_partial, X_complete, all_ids, id_to_col

# def transport_coverage_hard(gamma, q, threshold_ratio=0.2, eps=1e-12):
#     """
#     Hard node coverage of complete graph by partial graph.

#     A complete node is considered covered if received mass >= threshold_ratio * q_j.
#     """
#     incoming = gamma.sum(axis=0)
#     covered = incoming >= (threshold_ratio * q + eps)
#     return float(np.mean(covered.astype(np.float64)))

# def build_identity_feature_cost(partial_node_ids, complete_node_ids, mismatch_cost=1.0):
#     """
#     Build M where:
#       M[i, j] = 0 if node IDs match
#       M[i, j] = mismatch_cost otherwise
#     """
#     M = np.full((len(partial_node_ids), len(complete_node_ids)),
#                 mismatch_cost, dtype=np.float64)

#     complete_index = {nid: j for j, nid in enumerate(complete_node_ids)}

#     for i, nid in enumerate(partial_node_ids):
#         j = complete_index.get(nid)
#         if j is not None:
#             M[i, j] = 0.0

#     return M


# # def structure_only_distance_two_graphs(
# #     A_ref,                  # partial graph
# #     A_tgt,                  # complete graph
# #     dim=20,
# #     use_unbalanced=False,
# #     ugw_eps=1.0,
# #     rho1=1.0, rho2=1.0,
# #     numItermax=5000,
# #     alpha=0,              # weight for structural closeness
# #     beta=1.0,               # exponent for multiplicative penalty
# #     combine_mode="weighted_sum",   # "weighted_sum" or "product"
# #     partial_node_ids=None,
# #     complete_node_ids=None,
# #     use_exact_coverage=False,
# # ):
# #     """
# #     Returns:
# #         dist
# #         structural_closeness
# #         coverage
# #         final_score
# #         gamma
# #     """
# #     n_ref, n_tgt = A_ref.shape[0], A_tgt.shape[0]

# #     C_ref = normalize_cost(adj_to_geodesic_cost(A_ref))
# #     C_tgt = normalize_cost(adj_to_geodesic_cost(A_tgt))

# #     p = np.full(n_ref, 1.0 / n_ref, dtype=np.float64)
# #     q = np.full(n_tgt, 1.0 / n_tgt, dtype=np.float64)

# #     gamma = ot.gromov.gromov_wasserstein(
# #         C_ref, C_tgt, p, q, log=False,
# #         max_iter=max(numItermax, 500 * n_ref),
# #     )

# #     if np.array_equal(A_ref, A_tgt):
# #         gamma = np.diag(p)

# #     pos_ref = C_ref
# #     pos_tgt = C_tgt

# #     emb_ref, _ = LGW_embedding(C_ref, pos_ref, p, p, gamma=np.diag(p), loss="square")
# #     emb_tgt, _ = LGW_embedding(C_ref, pos_tgt, p, q, gamma=gamma, loss="square")

# #     d2 = LGW_dist(emb_ref, emb_tgt, p)
# #     dist = float(np.sqrt(max(d2, 0.0)))

# #     structural_closeness = 1.0 / (1.0 + dist + 1e-12)

# #     # Coverage
# #     if use_exact_coverage:
# #         if partial_node_ids is None or complete_node_ids is None:
# #             raise ValueError("partial_node_ids and complete_node_ids are required when use_exact_coverage=True")
# #         coverage = exact_node_coverage(partial_node_ids, complete_node_ids)
# #     else:
# #         coverage = transport_coverage_soft(gamma, q)

# #     # Combine
# #     if combine_mode == "weighted_sum":
# #         final_score = alpha * structural_closeness + (1.0 - alpha) * coverage
# #     elif combine_mode == "product":
# #         final_score = structural_closeness * (coverage ** beta)
# #     else:
# #         raise ValueError("combine_mode must be 'weighted_sum' or 'product'")

# #     return dist, final_score



# def build_padded_identity_cost_matrix(partial_present_node_ids, complete_node_ids, mismatch_cost=1.0):
#     """
#     Build an (n, n) feature cost matrix M where n = len(complete_node_ids).

#     Interpretation:
#     - row i corresponds to padded partial node position i
#     - col j corresponds to complete node position j

#     If node i exists in the partial graph:
#         M[i, i] = 0
#         M[i, j!=i] = mismatch_cost

#     If node i is missing in the partial graph:
#         M[i, j] = mismatch_cost for all j
#     """
#     n = len(complete_node_ids)
#     M = np.full((n, n), mismatch_cost, dtype=np.float64)

#     node_to_idx = {nid: i for i, nid in enumerate(complete_node_ids)}
#     partial_present_set = set(partial_present_node_ids)

#     for nid in partial_present_set:
#         if nid in node_to_idx:
#             i = node_to_idx[nid]
#             M[i, i] = 0.0

#     return M

# def build_identity_feature_cost(partial_node_ids, complete_node_ids, mismatch_cost=1.0):
#     """
#     Build feature mismatch matrix M when partial adjacency is NOT padded.

#     M[i, j] = 0 if partial_node_ids[i] == complete_node_ids[j]
#             = mismatch_cost otherwise

#     Shape
#     -----
#     (n_partial, n_complete)
#     """
#     n_partial = len(partial_node_ids)
#     n_complete = len(complete_node_ids)

#     M = np.full((n_partial, n_complete), mismatch_cost, dtype=np.float64)

#     complete_index = {nid: j for j, nid in enumerate(complete_node_ids)}

#     for i, nid in enumerate(partial_node_ids):
#         j = complete_index.get(nid)
#         if j is not None:
#             M[i, j] = 0.0

#     return M


# def structure_linearized_distance_plus_feature_mismatch(
#     A_ref,                  # partial graph
#     A_tgt,                  # complete graph
#     dim=20,
#     numItermax=5000,
#     alpha=0.6,              # final score weight for closeness
#     beta=1.0,
#     combine_mode="weighted_sum",
#     partial_node_ids=None,
#     complete_node_ids=None,
#     use_exact_coverage=True,
#     feature_lambda=1,     # weight for feature mismatch term
#     use_identity_M=True,    # True: direct 0/1 identity cost, False: one-hot+pairwise_distances
# ):
#     """
#     1) Compute structural LGW distance exactly as before
#     2) Compute FGW-like feature mismatch term <gamma, M>
#     3) Add that term to the structural distance
#     """
#     n_ref, n_tgt = A_ref.shape[0], A_tgt.shape[0]



#     # ---------------------------------------------------------
#     # 1) Structural cost matrices
#     # ---------------------------------------------------------
#     C_ref = normalize_cost(adj_to_geodesic_cost(A_ref))
#     C_tgt = normalize_cost(adj_to_geodesic_cost(A_tgt))

#     # ---------------------------------------------------------
#     # 2) Uniform masses
#     # ---------------------------------------------------------
#     p = np.full(n_ref, 1.0 / n_ref, dtype=np.float64)
#     q = np.full(n_tgt, 1.0 / n_tgt, dtype=np.float64)

#     # ---------------------------------------------------------
#     # 3) Structural transport exactly as before
#     # ---------------------------------------------------------
#     gamma = ot.gromov.gromov_wasserstein(
#         C_ref, C_tgt, p, q, log=False,
#         max_iter=max(numItermax, 500 * n_ref),
#     )
#     M = build_identity_feature_cost(complete_node_ids, partial_node_ids)
#     feature_mismatch = float(np.sum(gamma * M))

#     if np.array_equal(A_ref, A_tgt) and n_ref == n_tgt:
#         gamma = np.diag(p)

#     # ---------------------------------------------------------
#     # 4) Structural LGW distance exactly as before
#     # ---------------------------------------------------------
#     pos_ref = C_ref
#     pos_tgt = C_tgt

#     emb_ref, _ = LGW_embedding(
#         C_ref, pos_ref, p, p,
#         gamma=np.diag(p),
#         loss="square"
#     )

#     emb_tgt, _ = LGW_embedding(
#         C_ref, pos_tgt, p, q,
#         gamma=gamma,
#         loss="square"
#     )

#     d2 = LGW_dist(emb_ref, emb_tgt, p)
#     dist_struct = float(np.sqrt(max(d2, 0.0)))


#     # ---------------------------------------------------------
#     # 6) Final distance
#     # ---------------------------------------------------------
#     try:
#         print(f"Feature mismatch :{feature_mismatch}")
#         print(f"Distance mismatch :{dist_struct}")

#         feature_closeness = 1.0 - feature_mismatch
#         structural_closeness = 1.0 / (1.0 + dist_struct + 1e-12)

#         print(f"Feature closeness :{feature_closeness}")
#         print(f"Distance closeness :{structural_closeness}")
#         closeness_final = (1- feature_lambda)* structural_closeness + feature_lambda * feature_closeness

#         distance_final = 1.0 / (1.0 + closeness_final)

        
#         print(f"Final closeness :{closeness_final}")
#         print(f"Final distance :{distance_final}")
#     except Exception as e: 
#         print(e)
#     return distance_final, float(closeness_final)








import numpy as np
import ot
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import pairwise_distances
from Linearized_Partial_Gromov_Wasserstein.Linearized_Partial_Gromov_Wasserstein.lib.linear_gromov import (LGW_embedding, LGW_dist)
HAS_LGW = True


# ── Utility functions ──────────────────────────────────────────────────

def adj_to_geodesic_cost(A, weight_as_similarity=True, disconnected_fill="max_times_2"):
    A = np.asarray(A, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.maximum(A, 0.0)
    np.fill_diagonal(A, 0.0)

    if weight_as_similarity:
        A = np.where(A > 0, 1.0 / (A + 1e-12), 0.0)

    A_sym = 0.5 * (A + A.T)
    D = dijkstra(csr_matrix(A_sym), directed=False)

    finite_vals = D[np.isfinite(D) & (D > 0)]
    max_val = finite_vals.max() if finite_vals.size > 0 else 1.0

    if disconnected_fill == "max_times_2":
        fill = max_val * 2.0
    elif disconnected_fill == "max_times_1":
        fill = max_val
    else:
        fill = float(disconnected_fill)

    D = np.where(np.isinf(D), fill, D)
    np.fill_diagonal(D, 0.0)
    return D


def normalize_cost(C, eps=1e-12):
    C = np.asarray(C, dtype=np.float64)
    mx = C.max()
    return C / (mx + eps) if mx > eps else C


def build_identity_feature_cost(row_node_ids, col_node_ids, mismatch_cost=1.0):
    """
    M[i, j] = 0  if row_node_ids[i] == col_node_ids[j]
            = mismatch_cost otherwise

    Shape: (len(row_node_ids), len(col_node_ids))
    """
    n_row = len(row_node_ids)
    n_col = len(col_node_ids)
    M = np.full((n_row, n_col), mismatch_cost, dtype=np.float64)

    col_index = {nid: j for j, nid in enumerate(col_node_ids)}
    for i, nid in enumerate(row_node_ids):
        j = col_index.get(nid)
        if j is not None:
            M[i, j] = 0.0

    return M


def exact_node_coverage(partial_node_ids, complete_node_ids):
    partial_set = set(partial_node_ids)
    complete_set = set(complete_node_ids)
    if not complete_set:
        return 0.0
    return len(partial_set & complete_set) / len(complete_set)


# ── Fixed function ─────────────────────────────────────────────────────

def flgw_partial_coverage(
    A_ref,                  # complete graph adjacency
    A_tgt,                  # partial graph adjacency
    complete_node_ids,      # node IDs for complete graph (len = n_complete)
    partial_node_ids,       # node IDs for partial graph  (len = n_partial)
    p,
    q,         # FGW trade-off: 0 = pure structure, 1 = pure feature
    alpha=0.7,     

    coverage_weight=1.0,    # exponent on coverage penalty in final score
    numItermax=1000,
):
    try:
        n_complete = A_ref.shape[0]
        n_partial  = A_tgt.shape[0]
        C_ref_raw = adj_to_geodesic_cost(A_ref)  # (n_complete, n_complete)
        C_tgt_raw = adj_to_geodesic_cost(A_tgt)  # (n_partial,  n_partial)
        ref_max = C_ref_raw.max()
        shared_scale = ref_max if ref_max > 1e-12 else 1.0
        C_ref = C_ref_raw / shared_scale
        C_tgt = C_tgt_raw / shared_scale
        p = np.full(n_complete, 1.0 / n_complete, dtype=np.float64)
        q = np.full(n_partial,  1.0 / n_partial,  dtype=np.float64)
        M = build_identity_feature_cost(complete_node_ids, partial_node_ids)
        m = n_partial / n_complete

        if n_partial == n_complete:
            dist_struct, log = ot.gromov.fused_gromov_wasserstein2(
                M, C_ref, C_tgt, p, q,
                alpha=0.5,
                loss_fun="square_loss",
                max_iter=numItermax,
                log=True,
            )
        else:
            try:
                dist_struct, log = ot.gromov.partial_fused_gromov_wasserstein2(
                    M, C_ref, C_tgt, p, q,
                    m=m,
                    alpha=0.3,
                    loss_fun="square_loss",
                    numItermax=numItermax,
                    log=True,
                )
            except AttributeError:
                # Fallback for older POT versions without partial_fused_gromov_wasserstein
                print("[WARN] partial_fused_gromov_wasserstein not available, "
                    "falling back to partial_gromov_wasserstein (ignores features in optimization)")
                dist_struct, log = ot.gromov.partial_gromov_wasserstein2(
                    C_ref, C_tgt, p, q,
                    m=m,
                    loss_fun="square_loss",
                    numItermax=numItermax,
                    log=True,
                )
        coverage = exact_node_coverage(partial_node_ids, complete_node_ids)
        # outgoing = gamma.sum(axis=1)  # (n_complete,) mass sent per complete node
        # ratio = outgoing / (p + 1e-12)
        # coverage_soft = float(np.mean(np.minimum(ratio, 1.0)))
        # raw_mismatch = float(np.sum(gamma * M))
        # feature_mismatch = float(np.clip(raw_mismatch / (m + 1e-12), 0.0, 1.0))
        # feature_closeness = 1.0 - feature_mismatch
        # gamma_sum = gamma.sum()
        # if gamma_sum > 1e-12:
        #     gamma_rescaled = gamma / gamma_sum  # now sums to 1.0
        # else:
        #     gamma_rescaled = gamma
    
        # if HAS_LGW:
        #     emb_ref, _ = LGW_embedding(C_ref, C_ref, p, p, gamma=np.diag(p), loss="square")
        #     emb_tgt, _ = LGW_embedding(C_ref, C_tgt, p, q, gamma=gamma_rescaled, loss="square")

        #     d2 = LGW_dist(emb_ref, emb_tgt, p)
        #     dist_struct = float(np.sqrt(max(d2, 0.0)))
        # else:
        #     p_gamma = gamma_rescaled.sum(axis=1)
        #     q_gamma = gamma_rescaled.sum(axis=0)
        #     term1 = float(np.sum(C_ref ** 2 * np.outer(p_gamma, p_gamma)))
        #     term2 = float(np.sum(C_tgt ** 2 * np.outer(q_gamma, q_gamma)))
        #     term3 = float(np.trace(C_ref @ gamma_rescaled @ C_tgt.T @ gamma_rescaled.T))
        #     gw_cost = max(term1 + term2 - 2.0 * term3, 0.0)
        #     dist_struct = float(np.sqrt(gw_cost))
        # matched_quality = (
        #     alpha * feature_closeness
        #     + (1.0 - alpha) * structural_closeness
        # )
        structural_closeness = 1.0 / (1.0 + dist_struct + 1e-12)

        final_score = structural_closeness * (coverage ** coverage_weight)
    except Exception as e:
        print(e)
    return final_score, coverage, dist_struct, 0, 0

