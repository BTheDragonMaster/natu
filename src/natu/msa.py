"""Multiple Sequence Alignment (MSA) module."""

import logging

import numpy as np
from numpy.typing import NDArray
from math import sqrt

from natu.aligner import PairwiseAligner
from natu.pairwise import T, Converter, _pairwise_alignment
from natu.progressive_msa import _align_along_tree, _build_guide_tree

log = logging.getLogger(__name__)


def _merge_center_alignment(
    msa: NDArray[np.int32],
    t_a: NDArray[np.int32],
    q_a: NDArray[np.int32],
    gap_repr: np.int32,
) -> NDArray[np.int32]:
    """
    Merge a new pairwise center-to-query alignment into an existing center-star MSA.

    :param msa: Existing MSA, with the center sequence in row 0.
    :param t_a: Newly aligned center sequence.
    :param q_a: Newly aligned query sequence.
    :param gap_repr: Integer representation of a gap.
    :return: Updated MSA with the new query sequence added.
    """
    center = msa[0]
    old_cols = []
    new_query = []

    col = 0

    for t_symbol, q_symbol in zip(t_a, q_a):
        if t_symbol == gap_repr:
            # New insertion relative to the center.
            old_cols.append(np.full(msa.shape[0], gap_repr, dtype=np.int32))
            new_query.append(q_symbol)
            continue

        # Copy existing gap columns in the current MSA.
        while col < msa.shape[1] and center[col] == gap_repr:
            old_cols.append(msa[:, col])
            new_query.append(gap_repr)
            col += 1

        if col >= msa.shape[1]:
            raise ValueError("new center alignment extends beyond existing MSA")

        if center[col] != t_symbol:
            raise ValueError(
                f"center alignment mismatch: expected {center[col]}, got {t_symbol}"
            )

        old_cols.append(msa[:, col])
        new_query.append(q_symbol)
        col += 1

    # Copy remaining existing MSA columns.
    while col < msa.shape[1]:
        old_cols.append(msa[:, col])
        new_query.append(gap_repr)
        col += 1

    updated_msa = np.column_stack(old_cols)
    updated_query = np.array(new_query, dtype=np.int32)

    return np.vstack([updated_msa, updated_query])


def _progressive_msa(
        similarity_matrix: NDArray[np.float32],
        converter: Converter,
        aligner: PairwiseAligner,
        int_seqs: list[NDArray[np.int32]],

) -> tuple[NDArray[np.int32], list[float], list[int]]:
    tree = _build_guide_tree(similarity_matrix)

    gap_repr = converter.gap_repr

    msa, row_order = _align_along_tree(tree.root, int_seqs, gap_repr, aligner)

    # No single "center" exists in a progressive alignment, unlike _star_msa.
    # Scoring each row against the first-visited leaf keeps the return shape
    # consistent, but reconsider whether this is actually the semantic you want.
    ref_idx = row_order[0]
    scores = [
        float(aligner.score(int_seqs[ref_idx], int_seqs[idx])) if idx != ref_idx
        else float(aligner.score(int_seqs[ref_idx], int_seqs[ref_idx]))
        for idx in row_order
    ]

    gap_columns = np.all(msa == gap_repr, axis=0)
    msa = msa[:, ~gap_columns]

    return msa, scores, row_order

def _star_msa(sims: NDArray[np.float32], center_star: int | None,
              converter: Converter,
              aligner: PairwiseAligner,
              int_seqs: list[NDArray[np.int32]]
              ) -> tuple[NDArray[np.int32], list[float], list[int]]:
    masked_sims = sims.copy()
    np.fill_diagonal(masked_sims, -np.inf)

    # Find center star sequence if not given: the one with the highest total
    # similarity to every other sequence. This must sum a version of the
    # matrix with the diagonal excluded as 0, NOT masked_sims (whose
    # diagonal is -inf) -- summing masked_sims made every single row/column
    # sum -inf (each one includes its own -inf diagonal entry), so
    # np.argmax always returned index 0 regardless of the real similarity
    # structure. masked_sims itself is still exactly what the descending
    # sort just below needs (it deliberately keeps the center's
    # self-similarity out of contention there).
    if center_star is None:
        self_excluded_sims = sims.copy()
        np.fill_diagonal(self_excluded_sims, 0.0)
        center_ind = int(np.argmax(self_excluded_sims.sum(axis=0)))
    else:
        center_ind = int(center_star)

    # Sort all by descending similarity to center star
    sorted_indices = np.argsort(masked_sims[center_ind])[::-1]
    other_inds = [int(i) for i in sorted_indices if i != center_ind]
    indices = [center_ind]
    indices.extend(other_inds)

    # Align every sequence to the center star sequence
    scores: list[float] = []

    center_seq = int_seqs[center_ind]
    msa: NDArray[np.int32] = np.array([center_seq], dtype=np.int32)

    for i, ind in enumerate(indices):
        # Align other sequence to the original ungapped center star sequence
        q = int_seqs[ind]

        # Always align the new sequence to the original ungapped center. Do not align against msa[0],
        # because msa[0] may already contain gap markers, which Biopython cannot accept as sequence items.
        s, t_a, q_a = _pairwise_alignment(aligner, center_seq, q, converter.gap_repr, trim=False)

        # Merge the new pairwise center-query alignment into the growing MSA.
        if i != 0:
            msa = _merge_center_alignment(msa=msa, t_a=t_a, q_a=q_a, gap_repr=converter.gap_repr)

        scores.append(float(s))

    # Double check if there are any gap-only columns, delete those
    gap_columns = np.all(msa == converter.gap_repr, axis=0)
    msa = msa[:, ~gap_columns]

    return msa, scores, indices

def calculate_msa(
    aligner: PairwiseAligner,
    to_align: list[list[T]],
    converter: Converter,
    center_star: int | None = None,
    progressive = False
) -> tuple[list[tuple[float, list[T | None]]], list[int]]:
    """
    Calculate a multiple sequence alignment (MSA) for the given sequences.

    :param aligner: PairwiseAligner object to use for pairwise alignments.
    :param to_align: List of sequences to align, where each sequence is a list of items.
    :param converter: Converter object to convert items in sequences to integers for alignment and back.
    :param center_star: Index of the sequence to use as the center for the star alignment (default: None, which means to
        use the first sequence).
    :param progressive: If true, use progressive alignment using UPGMA guide tree
    :return: Tuple containing a list of tuples, where each tuple contains the alignment score and the aligned sequence
        with None for gaps, and a list of indices representing the order of sequences in the MSA.
    """
    if not to_align:
        return [], []

    # Failsafe: set aligner mode to global
    alignment_mode = aligner.mode
    if alignment_mode != "global":
        log.warning(f"Aligner mode is '{alignment_mode}', but 'global' is required for MSA. Setting aligner mode to 'global'!")
        aligner.mode = "global"

    # Convert sequences into int arrays based on substitution matrix alphabet
    int_seqs: list[NDArray[np.int32]] = [converter.to_int_array(seq) for seq in to_align]

    # Create pairwise similarity matrix; use reverse orientation if it scores better
    sims = np.zeros((len(int_seqs), len(int_seqs)), dtype=np.float32)
    norm_sims = np.zeros((len(int_seqs), len(int_seqs)), dtype=np.float32)

    # Calculate diagonal first, as we need self-similarities to compute normalized scores
    for i, seq in enumerate(int_seqs):
        score = aligner.score(seq, seq)
        sims[i, i] = score
        norm_sims[i, i] = 1.0

    for i, int_seq1 in enumerate(int_seqs):
        for j, int_seq2 in enumerate(int_seqs):
            # We only calculate the lower triangle of the similarity matrix
            if i >= j:
                continue

            score = aligner.score(int_seq1, int_seq2)
            sims[i, j] = score
            sims[j, i] = score

            # Calculate normalised scores
            norm_score = score / sqrt(sims[i, i] * sims[j, j])
            norm_sims[i, j] = norm_score
            norm_sims[j, i] = norm_score

    # Mask similarity matrix to ignore self-similarity (diagonal)
    if not progressive:
        msa, scores, new_order = _star_msa(sims, center_star, converter, aligner, int_seqs)
    else:
        msa, scores, new_order = _progressive_msa(norm_sims, converter, aligner, int_seqs)

    # Convert back to original sequences with None for gaps
    aligned_seqs: list[tuple[float, list[T | None]]] = []
    for i, int_seq in enumerate(msa):
        aligned_seq = converter.from_int_array(int_seq)

        # Score is the pairwise alignment score to the center star sequence
        score = scores[i]

        aligned_seqs.append((score, aligned_seq))

    return aligned_seqs, new_order
