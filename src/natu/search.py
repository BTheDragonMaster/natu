"""Pairwise search module."""

import logging

import numpy as np
from numpy.typing import NDArray

from natu.aligner import PairwiseAligner
from natu.pairwise import (
    T,
    Converter,
    _pairwise_alignment,
    configure_glocal_end_gaps,
    strip_glocal_free_overhang,
)

log = logging.getLogger(__name__)


def search(
    aligner: PairwiseAligner,
    query_sequences: list[list[T]],
    subject_sequences: list[list[T]],
    converter: Converter,
    threshold: float,
    trim: bool = False,
    glocal: bool = False,
) -> list[list[tuple[float, list[T | None], list[T | None], int]]]:
    """
    Search for all subject sequences scoring above threshold, for each query sequence.

    :param aligner: PairwiseAligner object to use for pairwise alignments.
    :param query_sequences: List of query sequences to search with, where each sequence is a list of items.
    :param subject_sequences: List of subject sequences to search against, where each sequence is a list of items.
    :param converter: Converter object to convert items in sequences to integers for alignment and back.
    :param threshold: Minimum raw alignment score required for a query/subject pair to be considered a match.
    :param glocal: If True, switches to "glocal" alignment (as in BiG-SCAPE): for every query/subject
        pair, the longer sequence's end gaps are freed (0) and the shorter sequence's end gaps are
        penalized at ``aligner``'s own currently configured open/extend end-gap score (captured once,
        before any pair is processed), forcing the shorter sequence to be fully included while letting
        the longer sequence's non-matching overhang go unpenalized -- see
        ``natu.network.cluster_sequences`` for the same mechanism. ``aligner`` must already be set up
        with ``mode="glocal"`` (see ``natu.aligner.setup_aligner``) for this to have the intended
        effect. False (default) leaves the aligner's end-gap scores exactly as configured, unchanged
        for every pair.
    :return: List (one entry per query, same order as ``query_sequences``) of lists of
        ``(score, aligned_query_seq, aligned_subject_seq, subject_idx)`` tuples for every subject scoring above
        ``threshold``, sorted by score descending. Empty list for a query with no matches above threshold.
    """
    if not query_sequences:
        return []

    # Convert sequences into int arrays based on substitution matrix alphabet
    int_query_seqs: list[NDArray[np.int32]] = [converter.to_int_array(seq) for seq in query_sequences]
    int_subject_seqs: list[NDArray[np.int32]] = [converter.to_int_array(seq) for seq in subject_sequences]

    # Capture the aligner's own currently configured end-gap score once, before any pair
    # mutates it below -- see natu.network.cluster_sequences for the same mechanism.
    glocal_kept_scores = (
        (aligner.open_left_insertion_score, aligner.extend_left_insertion_score)
        if glocal else None
    )

    subject_lengths = np.array([len(s) for s in int_subject_seqs], dtype=np.int64)

    # Create pairwise similarity matrix
    sims = np.zeros((len(int_query_seqs), len(int_subject_seqs)), dtype=np.float32)

    for i, query in enumerate(int_query_seqs):
        if glocal:
            # configure_glocal_end_gaps only cares whether len_t is shorter than, longer
            # than, or equal to len_q -- never the actual magnitude (see its docstring).
            # So every subject on the same side of len(query) shares one configuration:
            # reconfigure the aligner up to 3 times for this whole query row, once per
            # side, instead of once per subject. Calling it once per pair here previously
            # made glocal searches dramatically slower than local/global, since each
            # reconfiguration is many attribute writes on the Biopython aligner -- cheap
            # individually, but ruinous multiplied by len(query) * len(subjects).
            for js, is_longer in (
                (np.flatnonzero(subject_lengths < len(query)), False),
                (np.flatnonzero(subject_lengths > len(query)), True),
                (np.flatnonzero(subject_lengths == len(query)), None),
            ):
                if js.size == 0:
                    continue
                len_t, len_q = (2, 1) if is_longer else (1, 2) if is_longer is False else (1, 1)
                configure_glocal_end_gaps(aligner, len_t, len_q, *glocal_kept_scores)
                for j in js:
                    sims[i, j] = aligner.score(int_subject_seqs[j], query)
        else:
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

        def _align_candidate(j: int) -> None:
            subject = int_subject_seqs[j]
            alignment = _pairwise_alignment(
                aligner, subject, int_query, gap_repr=converter.gap_repr, trim=trim
            )
            if alignment is not None:
                score, t_a, q_a = alignment
                if glocal:
                    t_a, q_a = strip_glocal_free_overhang(
                        t_a, q_a, converter.gap_repr, len(subject), len(int_query)
                    )
                gapped_subject = converter.from_int_array(t_a)
                gapped_query = converter.from_int_array(q_a)
                query_matches.append((float(score), gapped_query, gapped_subject, j))

        if glocal:
            # Group candidates by which side of len(query) they fall on so the aligner is
            # reconfigured once per bucket rather than once per candidate -- candidate_js
            # can be the whole subject list under a permissive threshold, and this loop
            # had the same per-pair reconfiguration cost problem as the sims matrix above.
            # query_matches is appended to out of score order this way, so re-sort by
            # score descending afterwards to restore candidate_js's original ordering.
            candidate_set = set(candidate_js)
            len_query = len(int_query)
            for js, is_longer in (
                (np.flatnonzero(subject_lengths < len_query), False),
                (np.flatnonzero(subject_lengths > len_query), True),
                (np.flatnonzero(subject_lengths == len_query), None),
            ):
                bucket_js = [int(j) for j in js if int(j) in candidate_set]
                if not bucket_js:
                    continue
                len_t, len_q = (2, 1) if is_longer else (1, 2) if is_longer is False else (1, 1)
                configure_glocal_end_gaps(aligner, len_t, len_q, *glocal_kept_scores)
                for j in bucket_js:
                    _align_candidate(j)
            query_matches.sort(key=lambda match: match[0], reverse=True)
        else:
            for j in candidate_js:
                _align_candidate(j)

        all_results.append(query_matches)

    return all_results