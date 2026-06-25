"""Multiple Sequence Alignment (MSA) module."""

import logging

import numpy as np
from numpy.typing import NDArray

from natu.aligner import PairwiseAligner
from natu.pairwise import T, Converter, _pairwise_alignment

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


def calculate_msa(
    aligner: PairwiseAligner,
    to_align: list[list[T]],
    converter: Converter,
    center_star: int | None = None,
) -> tuple[list[tuple[float, list[T | None]]], list[int]]:
    """
    Calculate a multiple sequence alignment (MSA) for the given sequences.

    :param aligner: PairwiseAligner object to use for pairwise alignments.
    :param to_align: List of sequences to align, where each sequence is a list of items.
    :param converter: Converter object to convert items in sequences to integers for alignment and back.
    :param center_star: Index of the sequence to use as the center for the star alignment (default: None, which means to
        use the first sequence).
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

    for i, int_seq1 in enumerate(int_seqs):
        for j, int_seq2 in enumerate(int_seqs[i + 1:], start=i + 1):
            score = aligner.score(int_seq1, int_seq2)

            # We only calculate the lower triangle of the similarity matrix
            if i >= j:
                continue

            sims[i, j] = score
            sims[j, i] = score

    # Mask similarity matrix to ignore self-similarity (diagonal)
    masked_sims = sims.copy()
    np.fill_diagonal(masked_sims, -np.inf)

    # Find center star sequence if not given, ignore the diagonal (self-similarity)
    if center_star is None:
        center_ind = int(np.argmax(masked_sims.sum(axis=0)))
    else:
        center_ind = int(center_star)

    # Sort all by descending similarity to center star
    sorted_indices = np.argsort(masked_sims[center_ind])[::-1]
    other_inds = [int(i) for i in sorted_indices if i != center_ind]

    # Align every sequence to the center star sequence
    scores: list[float] = []

    center_seq = int_seqs[center_ind]
    msa: NDArray[np.int32] = np.array([center_seq], dtype=np.int32)

    for other_ind in other_inds:
        # Align other sequence to the original ungapped center star sequence
        q = int_seqs[other_ind]

        # Always align the new sequence to the original ungapped center. Do not align against msa[0],
        # because msa[0] may already contain gap markers, which Biopython cannot accept as sequence items.
        s, t_a, q_a = _pairwise_alignment(aligner, center_seq, q, converter.gap_repr)

        # Merge the new pairwise center-query alignment into the growing MSA.
        msa = _merge_center_alignment(msa=msa, t_a=t_a, q_a=q_a, gap_repr=converter.gap_repr)

        scores.append(float(s))

    # Double check if there are any gap-only columns, delete those
    gap_columns = np.all(msa == converter.gap_repr, axis=0)
    msa = msa[:, ~gap_columns]

    # Convert back to original sequences with None for gaps
    aligned_seqs: list[tuple[float, list[T | None]]] = []
    for i, int_seq in enumerate(msa):
        aligned_seq = converter.from_int_array(int_seq)

        if i == 0:
            # Center star sequence, score is self-alignment score
            score = float(aligner.score(center_seq, center_seq))
        else:
            # Score is the pairwise alignment score to the center star sequence
            score = scores[i - 1]

        aligned_seqs.append((score, aligned_seq))

    new_order = [center_ind] + other_inds

    return aligned_seqs, new_order
