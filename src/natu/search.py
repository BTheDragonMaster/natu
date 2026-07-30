"""Pairwise search module."""

import logging

import numpy as np
from numpy.typing import NDArray

from natu.aligner import PairwiseAligner
from natu.pairwise import T, Converter, _pairwise_alignment

log = logging.getLogger(__name__)


def search(
    aligner: PairwiseAligner,
    query_sequences: list[list[T]],
    subject_sequences: list[list[T]],
    converter: Converter,
    threshold: float,
) -> list[list[tuple[float, list[T | None], list[T | None], int]]]:
    """
    Search for all subject sequences scoring above threshold, for each query sequence.

    :param aligner: PairwiseAligner object to use for pairwise alignments.
    :param query_sequences: List of query sequences to search with, where each sequence is a list of items.
    :param subject_sequences: List of subject sequences to search against, where each sequence is a list of items.
    :param converter: Converter object to convert items in sequences to integers for alignment and back.
    :param threshold: Minimum raw alignment score required for a query/subject pair to be considered a match.
    :return: List (one entry per query, same order as ``query_sequences``) of lists of
        ``(score, aligned_query_seq, aligned_subject_seq, subject_idx)`` tuples for every subject scoring above
        ``threshold``, sorted by score descending. Empty list for a query with no matches above threshold.
    """
    if not query_sequences:
        return []

    # Convert sequences into int arrays based on substitution matrix alphabet
    int_query_seqs: list[NDArray[np.int32]] = [converter.to_int_array(seq) for seq in query_sequences]
    int_subject_seqs: list[NDArray[np.int32]] = [converter.to_int_array(seq) for seq in subject_sequences]

    # Create pairwise similarity matrix
    sims = np.zeros((len(int_query_seqs), len(int_subject_seqs)), dtype=np.float32)

    for i, query in enumerate(int_query_seqs):
        for j, subject in enumerate(int_subject_seqs):
            sims[i, j] = aligner.score(subject, query)

    all_results: list[list[tuple[float, list[T | None], list[T | None], int]]] = []

    for i, query_seq in enumerate(query_sequences):
        int_query = int_query_seqs[i]
        row = sims[i]

        # Indices of every subject clearing threshold, best-scoring first
        candidate_js = [j for j in range(len(subject_sequences)) if row[j] >= threshold]
        candidate_js.sort(key=lambda j: row[j], reverse=True)

        query_matches: list[tuple[float, list[T | None], list[T | None], int]] = []

        for j in candidate_js:
            alignment = _pairwise_alignment(
                aligner, int_subject_seqs[j], int_query, gap_repr=converter.gap_repr
            )
            if alignment is not None:
                score, t_a, q_a = alignment
                gapped_subject = converter.from_int_array(t_a)
                gapped_query = converter.from_int_array(q_a)
                query_matches.append((float(score), gapped_query, gapped_subject, j))

        all_results.append(query_matches)

    return all_results