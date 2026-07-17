import numpy as np
from numpy.typing import NDArray

from biotite.sequence.phylo import upgma, Tree

from natu.aligner import PairwiseAligner


def _build_guide_tree(norm_sims: NDArray[np.float32]) -> Tree:
    """

    :param norm_sims: Matrix of normalized similarity scores
    :return: Biotite UPGMA guide tree
    """
    distances = 1 - norm_sims
    tree = upgma(distances.astype(np.float64))
    return tree


def _column_score(col_a: NDArray[np.int32], col_b: NDArray[np.int32],
                   sub_matrix: NDArray, gap_repr: np.int32) -> float:
    """Sum-of-pairs score between two profile columns (gap rows excluded;
    gap costs are handled separately by the affine-gap DP, not here)."""
    a_vals = col_a[col_a != gap_repr]
    b_vals = col_b[col_b != gap_repr]
    if len(a_vals) == 0 or len(b_vals) == 0:
        return 0.0
    # vectorized outer-product average instead of a nested python loop
    return float(sub_matrix[np.ix_(a_vals, b_vals)].mean())


def _profile_profile_align(
    profile_a: NDArray[np.int32],
    profile_b: NDArray[np.int32],
    gap_repr: np.int32,
    aligner: PairwiseAligner,
) -> NDArray[np.int32]:
    """
    X[i,j]: alignment ending with a gap in profile_b (column i-1 of A unmatched).
            "End" region = j == 0 (leading) or j == m (trailing) -- both mean
            B hasn't contributed at this point, so use end-gap costs.
    Y[i,j]: ending with a gap in profile_a; "end" region = i == 0 or i == n.
    """
    sub_matrix = np.asarray(aligner.substitution_matrix)

    internal_gap_open = getattr(aligner, "open_internal_gap_score", -0.5)
    internal_gap_extend = getattr(aligner, "extend_internal_gap_score", -0.1)
    end_gap_open = getattr(aligner, "open_end_gap_score", -0.2)
    end_gap_extend = getattr(aligner, "extend_end_gap_score", -0.05)

    n, m = profile_a.shape[1], profile_b.shape[1]
    NEG_INF = -1e18

    M = np.full((n + 1, m + 1), NEG_INF)
    X = np.full((n + 1, m + 1), NEG_INF)
    Y = np.full((n + 1, m + 1), NEG_INF)
    M[0, 0] = 0.0

    for i in range(1, n + 1):
        X[i, 0] = end_gap_open + (i - 1) * end_gap_extend
    for j in range(1, m + 1):
        Y[0, j] = end_gap_open + (j - 1) * end_gap_extend

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = _column_score(profile_a[:, i - 1], profile_b[:, j - 1], sub_matrix, gap_repr)
            M[i, j] = max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1]) + s

            x_open, x_extend = (end_gap_open, end_gap_extend) if j in (0, m) else (internal_gap_open, internal_gap_extend)
            y_open, y_extend = (end_gap_open, end_gap_extend) if i in (0, n) else (internal_gap_open, internal_gap_extend)

            X[i, j] = max(M[i - 1, j] + x_open, X[i - 1, j] + x_extend)
            Y[i, j] = max(M[i, j - 1] + y_open, Y[i, j - 1] + y_extend)

    # Traceback re-derives the same open/extend choice at each step,
    # since it depends only on (i, j) position, not on path history.
    i, j = n, m
    state = int(np.argmax([M[i, j], X[i, j], Y[i, j]]))
    a_cols, b_cols = [], []

    while i > 0 or j > 0:
        if i == 0:
            a_cols.append(None); b_cols.append(j - 1); j -= 1; continue
        if j == 0:
            a_cols.append(i - 1); b_cols.append(None); i -= 1; continue

        if state == 0:
            a_cols.append(i - 1); b_cols.append(j - 1)
            state = int(np.argmax([M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1]]))
            i -= 1; j -= 1
        elif state == 1:
            a_cols.append(i - 1); b_cols.append(None)
            x_open, x_extend = (end_gap_open, end_gap_extend) if j in (0, m) else (internal_gap_open, internal_gap_extend)
            state = 0 if M[i - 1, j] + x_open >= X[i - 1, j] + x_extend else 1
            i -= 1
        else:
            a_cols.append(None); b_cols.append(j - 1)
            y_open, y_extend = (end_gap_open, end_gap_extend) if i in (0, n) else (internal_gap_open, internal_gap_extend)
            state = 0 if M[i, j - 1] + y_open >= Y[i, j - 1] + y_extend else 2
            j -= 1

    a_cols.reverse(); b_cols.reverse()
    n_a, n_b = profile_a.shape[0], profile_b.shape[0]
    merged = np.full((n_a + n_b, len(a_cols)), gap_repr, dtype=np.int32)
    for col, (ac, bc) in enumerate(zip(a_cols, b_cols)):
        if ac is not None:
            merged[:n_a, col] = profile_a[:, ac]
        if bc is not None:
            merged[n_a:, col] = profile_b[:, bc]
    return merged


def _align_along_tree(node, int_seqs, gap_repr, aligner):
    if node.is_leaf():
        return int_seqs[node.index].reshape(1, -1).astype(np.int32), [node.index]

    left_node, right_node = node.children
    left_profile, left_order = _align_along_tree(left_node, int_seqs, gap_repr, aligner)
    right_profile, right_order = _align_along_tree(right_node, int_seqs, gap_repr, aligner)
    merged = _profile_profile_align(left_profile, right_profile, gap_repr, aligner)
    return merged, left_order + right_order
