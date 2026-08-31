"""Sequence similarity network construction and clustering module."""

import csv
import logging
from pathlib import Path
from typing import Iterable, TypeVar

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from natu.aligner import PairwiseAligner
from natu.pairwise import Converter, _pairwise_alignment, configure_glocal_end_gaps, strip_glocal_free_overhang

log = logging.getLogger(__name__)

T = TypeVar("T")


def sequence_to_label(sequence: Iterable[T]) -> str:
    """
    Canonical string form of a monomer sequence, used both as a network node's "sequence"
    attribute (see ``cluster_sequences``) and to match --highlight targets in
    ``natu.network_viz`` / ``natu.cli``. Keeping this in one place means both sides of that
    matching always agree on how a sequence is stringified.

    :param sequence: Sequence to stringify, as a list of monomer items.
    :return: Pipe-joined string, e.g. "serine|leucine|glycine" -- the same format used in
        NATU's own FASTA files.
    """
    return "|".join(str(item) for item in sequence)


def compute_score_matrix(
    aligner: PairwiseAligner,
    sequences: list[list[T]],
    converter: Converter,
) -> NDArray[np.float32]:
    """
    Compute the all-vs-all raw alignment score ("bitscore") matrix for a set of sequences.

    Includes the diagonal (self-vs-self alignment scores), which is required to normalize
    raw scores into similarity scores afterwards.

    Materializes a dense (n x n) float32 matrix: ~4 bytes * n^2, e.g. ~4 GB already at
    n=32,000, before even normalizing. Fine for exploring small sequence sets (and used that
    way in this module's own tests), but do not use this for a large all-vs-all run -- use
    ``cluster_sequences`` instead, which never materializes a full matrix and prunes pairs
    that provably cannot pass the cutoff before aligning them.

    :param aligner: PairwiseAligner object to use for pairwise alignments.
    :param sequences: List of sequences to compare against each other, where each sequence is a list of items.
    :param converter: Converter object to convert items in sequences to integers for alignment.
    :return: Symmetric (n x n) matrix of raw alignment scores, where n is the number of sequences.
    """
    int_seqs: list[NDArray[np.int32]] = [converter.to_int_array(seq) for seq in sequences]
    n = len(int_seqs)
    scores = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        # Alignment score is symmetric for a symmetric substitution matrix and matched gap
        # penalties, so only the upper triangle (including the diagonal) needs computing.
        for j in range(i, n):
            score = aligner.score(int_seqs[i], int_seqs[j])
            scores[i, j] = score
            scores[j, i] = score

    return scores


def scores_to_similarities(score_matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Normalize a raw bitscore matrix into a similarity matrix.

    Each pairwise score is normalized against the weaker of the two sequences' own
    self-alignment score: ``sim(i, j) = score(i, j) / min(score(i, i), score(j, j))``.
    This puts similarities on a comparable scale regardless of sequence length or
    composition, since a sequence can never score higher against another sequence than
    it scores against itself. Values are not clipped, so a pair can exceed 1.0 in
    unusual cases (e.g. local alignment with favorable end-gap handling); that is left
    visible rather than hidden by clipping.

    :param score_matrix: Symmetric (n x n) matrix of raw alignment scores, as returned by
        ``compute_score_matrix``. The diagonal must hold each sequence's self-alignment score.
    :return: Symmetric (n x n) matrix of normalized similarity scores.
    :raises ValueError: If any self-alignment score (diagonal value) is zero or negative,
        since that makes normalization undefined or meaningless.
    """
    self_scores = np.diag(score_matrix)

    if np.any(self_scores <= 0):
        bad = np.where(self_scores <= 0)[0]
        raise ValueError(
            "self-alignment score must be positive to normalize similarities; "
            f"got non-positive self-scores at sequence indices {bad.tolist()}"
        )

    denom = np.minimum.outer(self_scores, self_scores)
    similarity_matrix = score_matrix / denom
    np.fill_diagonal(similarity_matrix, 1.0)

    return similarity_matrix.astype(np.float32)


def build_similarity_network(
    headers: list[str],
    similarity_matrix: NDArray[np.float32],
    cutoff: float,
) -> nx.Graph:
    """
    Build a sequence similarity network from a similarity matrix and a cutoff.

    Every input sequence becomes a node, including sequences with no edges above cutoff
    so they still show up as singleton clusters downstream instead of disappearing
    silently. An edge is added between two distinct sequences when their similarity is
    at or above ``cutoff``, weighted by that similarity.

    :param headers: Sequence identifiers, in the same order as the rows/columns of
        ``similarity_matrix``. Must be unique; used as node names.
    :param similarity_matrix: Symmetric (n x n) matrix of similarity scores, as returned by
        ``scores_to_similarities``.
    :param cutoff: Minimum similarity required to draw an edge between two sequences.
    :return: Undirected networkx Graph with sequence headers as nodes and
        similarity-weighted edges above cutoff.
    :raises ValueError: If the number of headers doesn't match the similarity matrix
        dimensions, or headers are not unique.
    """
    n = similarity_matrix.shape[0]

    if len(headers) != n:
        raise ValueError(f"got {len(headers)} headers for a {n}x{n} similarity matrix")

    if len(set(headers)) != len(headers):
        raise ValueError("headers must be unique to use as network node identifiers")

    graph = nx.Graph()
    graph.add_nodes_from(headers)

    for i in range(n):
        for j in range(i + 1, n):
            similarity = float(similarity_matrix[i, j])
            if similarity >= cutoff:
                graph.add_edge(headers[i], headers[j], weight=similarity)

    return graph


def get_clusters(graph: nx.Graph) -> list[set[str]]:
    """
    Extract clusters from a similarity network as its connected components.

    This is single-linkage clustering: any chain of edges above the network's cutoff
    transitively joins sequences into one cluster, even if two members of the same
    cluster never scored above cutoff against each other directly. A sequence with no
    edges above cutoff to anything else forms its own singleton cluster.

    :param graph: Similarity network, as returned by ``build_similarity_network``.
    :return: List of clusters, each a set of sequence headers. Sorted largest first.
    """
    components = [set(component) for component in nx.connected_components(graph)]
    components.sort(key=len, reverse=True)

    return components


def cluster_sequences(
    aligner: PairwiseAligner,
    headers: list[str],
    sequences: list[list[T]],
    converter: Converter,
    cutoff: float,
    min_alignment_length: int = 0,
    log_every: int = 2_000_000,
    glocal: bool = False,
) -> tuple[nx.Graph, list[set[str]]]:
    """
    Score, normalize, build the network, and cluster in one call -- without ever
    materializing a dense (n x n) matrix.

    For n sequences this still means up to n*(n-1)/2 candidate pairs, but each pair only
    costs O(1) memory: a self-score is computed once per sequence up front, and pairwise
    scores are streamed straight into the graph (as an edge, if the pair clears cutoff) and
    then discarded rather than accumulated into a matrix. Peak memory is O(n + edges)
    instead of O(n^2) -- at n=32,000 that is the difference between needing tens of GB and
    needing a few hundred MB at most.

    This does NOT reduce the number of *scoring* calls (still up to n*(n-1)/2 of them) --
    an earlier version of this function tried to additionally skip pairs using a length-based
    upper bound on similarity, but that bound turns out to always evaluate to ~1.0 whenever a
    sequence's self-alignment already realizes the substitution matrix's maximum value (the
    normal case), so it never actually pruned anything and has been removed. If runtime (not
    memory) becomes the bottleneck, look at parallelizing across pairs instead. Progress
    (pairs processed, edges found so far) is logged periodically via the standard ``logging``
    module so a long run doesn't look hung -- run with ``logging.basicConfig(level=logging.INFO)``
    to see it.

    ``min_alignment_length`` guards against a different failure mode than filtering short
    sequences out beforehand does: two sequences can each be individually long, and still
    only really share a short coincidental motif, which self-score normalization can still
    score as a high "similarity" (it only measures how much of the *shorter* sequence's own
    ceiling was reached, not how much of either sequence is actually covered by the match).
    When set above 0, an edge is only kept if the actual number of columns in its alignment
    (matches, mismatches, and internal gaps -- never the padding described below) also
    reaches ``min_alignment_length``.

    Checking this requires the full alignment (``aligner.align``, via
    ``natu.pairwise._pairwise_alignment``), not just its score (``aligner.score``) -- the
    score alone doesn't tell you how many columns produced it. So when
    ``min_alignment_length`` is set, every pair is aligned fully instead of just scored, and
    both the score (for the ``cutoff`` check) and the length (for this check) are read off
    that *one* alignment -- there is no separate, later re-alignment of pairs that already
    cleared cutoff. When ``min_alignment_length`` is 0 (the default), this doesn't apply at
    all and the main sweep stays on the cheaper score-only path.

    The alignment is always computed with ``trim=True``, regardless of whatever ``trim``
    setting is used elsewhere in a run. This is not optional: with ``trim=False``,
    ``_pairwise_alignment`` pads a local alignment's output with the unaligned prefix/suffix
    of both sequences so the gapped output lines up with their full original length -- useful
    for FASTA-style output, but it would make every pair look at least as long as its longer
    input sequence if used here, defeating the entire purpose of a minimum *alignment* length
    filter.

    :param aligner: PairwiseAligner object to use for pairwise alignments.
    :param headers: Sequence identifiers, in the same order as ``sequences``. Must be unique.
    :param sequences: Sequences to cluster, where each sequence is a list of items.
    :param converter: Converter object to convert items in sequences to integers for alignment.
    :param cutoff: Minimum similarity required to draw an edge between two sequences.
    :param min_alignment_length: Minimum number of aligned columns (matches, mismatches, and
        internal gaps) required to keep an edge, checked only for pairs that already clear
        ``cutoff``. 0 (default) disables this check.
    :param log_every: Emit a progress log line after this many candidate pairs have been
        considered (aligned or pruned). Set to 0 to disable progress logging.
    :param glocal: If True, switches to "glocal" alignment (as in BiG-SCAPE): for every
        pair, the longer sequence's end gaps are freed (0) and the shorter sequence's end
        gaps are penalized at ``aligner``'s own currently configured open/extend end-gap
        score (captured once, before any pair is processed, so it reflects whatever
        ``natu.aligner.setup_aligner`` was given), forcing the shorter sequence to be fully
        included while letting the longer sequence's non-matching overhang go unpenalized.
        Both sides being free would let the shorter sequence dangle unaligned too, losing
        exactly the "forced full inclusion" that distinguishes glocal from "local" -- so
        only the longer side is ever hardcoded to 0. ``aligner`` must already be set up with
        ``mode="glocal"`` (see ``natu.aligner.setup_aligner``) for this to have the intended
        effect -- this flag only triggers the per-pair reconfiguration, it does not change
        the aligner's base mode. False (default) leaves the aligner's end-gap scores exactly
        as configured, unchanged for every pair.
    :return: Tuple of (similarity network, list of clusters as sets of headers).
    :raises ValueError: If headers are not unique, or any sequence has a non-positive
        self-alignment score (which would make normalization undefined).
    """
    n = len(sequences)

    if len(headers) != n:
        raise ValueError(f"got {len(headers)} headers for {n} sequences")

    if len(set(headers)) != len(headers):
        raise ValueError("headers must be unique to use as network node identifiers")

    int_seqs: list[NDArray[np.int32]] = [converter.to_int_array(seq) for seq in sequences]
    lengths = np.array([len(seq) for seq in int_seqs], dtype=np.int64)

    # Capture the aligner's own currently configured end-gap score once, before any pair
    # mutates it below -- this is what a shorter sequence's own end gaps are penalized at
    # under glocal (the longer sequence's side is always freed to 0 instead, see
    # configure_glocal_end_gaps). All 8 end-gap attributes are set identically by
    # natu.aligner.setup_aligner, so any one of them reflects the configured value.
    glocal_kept_scores = (
        (aligner.open_left_insertion_score, aligner.extend_left_insertion_score)
        if glocal else None
    )

    self_scores = np.empty(n, dtype=np.float32)
    for i in range(n):
        if glocal:
            # Equal lengths (same sequence against itself), so this always resolves to the
            # "no side is longer" branch -- included for consistency, so the aligner is
            # never left in a state configured for some earlier, unrelated pair.
            configure_glocal_end_gaps(aligner, len(int_seqs[i]), len(int_seqs[i]), *glocal_kept_scores)
        self_scores[i] = aligner.score(int_seqs[i], int_seqs[i])

    non_positive = np.where(self_scores <= 0)[0]
    if non_positive.size:
        raise ValueError(
            "self-alignment score must be positive to normalize similarities; "
            f"got non-positive self-scores at sequence indices {non_positive.tolist()}"
        )

    check_length = min_alignment_length > 0

    graph = nx.Graph()
    graph.add_nodes_from(headers)
    # Attach each sequence as a node attribute (pipe-joined, matching FASTA monomer
    # formatting) so downstream visualization (natu.network_viz) can label nodes by
    # sequence content instead of by header alone. Written to GraphML by write_graphml
    # below and read back by network_viz.load_network.
    for header, sequence in zip(headers, sequences):
        graph.nodes[header]["sequence"] = sequence_to_label(sequence)

    total_pairs = n * (n - 1) // 2
    considered = 0
    length_filtered = 0

    for i in range(n):
        for j in range(i + 1, n):
            considered += 1

            denom = min(self_scores[i], self_scores[j])

            if glocal:
                configure_glocal_end_gaps(
                    aligner, len(int_seqs[i]), len(int_seqs[j]), *glocal_kept_scores
                )

            if check_length:
                # One alignment gives us both the score (for cutoff) and the column count
                # (for min_alignment_length) -- no separate aligner.score() call, and no
                # second alignment later for pairs that pass cutoff. trim=True is
                # unconditional here; see the docstring.
                #
                # _pairwise_alignment returns None when the aligner finds zero valid
                # alignments, which happens whenever the best possible local alignment
                # scores exactly 0 (Bio.Align.PairwiseAligner reports 0 alignments rather
                # than one with score 0 in that case) -- entirely plausible for two
                # genuinely unrelated sequences at real-world scale. That means no
                # meaningful match exists at all, so treat it as score 0 / length 0 rather
                # than crashing on the unpack (search.py guards the same call the same way).
                alignment_result = _pairwise_alignment(
                    aligner, int_seqs[i], int_seqs[j], gap_repr=converter.gap_repr, trim=True
                )
                if alignment_result is None:
                    score = 0.0
                    aligned_length = 0
                else:
                    score, t_a, q_a = alignment_result
                    if glocal:
                        t_a, q_a = strip_glocal_free_overhang(
                            t_a, q_a, converter.gap_repr, len(int_seqs[i]), len(int_seqs[j])
                        )
                    aligned_length = len(t_a)
            else:
                score = aligner.score(int_seqs[i], int_seqs[j])
                aligned_length = None

            similarity = float(score) / denom
            keep_edge = similarity >= cutoff

            if keep_edge and check_length and aligned_length < min_alignment_length:
                keep_edge = False
                length_filtered += 1

            if keep_edge:
                graph.add_edge(headers[i], headers[j], weight=similarity)

            if log_every and considered % log_every == 0:
                log.info(
                    "cluster_sequences: %d/%d pairs considered (%.1f%%), %d edges so far "
                    "(%d dropped by min_alignment_length)",
                    considered, total_pairs, 100 * considered / total_pairs,
                    graph.number_of_edges(), length_filtered,
                )

    if log_every:
        log.info(
            "cluster_sequences: done -- %d pairs considered, %d edges "
            "(%d candidate edges dropped by min_alignment_length)",
            considered, graph.number_of_edges(), length_filtered,
        )

    clusters = get_clusters(graph)

    return graph, clusters


def write_graphml(graph: nx.Graph, output: Path) -> None:
    """
    Write a similarity network to a GraphML file for use in Cytoscape or similar tools.

    :param graph: Similarity network to write.
    :param output: Output file path.
    """
    nx.write_graphml(graph, output)


def write_edge_list(graph: nx.Graph, output: Path) -> None:
    """
    Write a similarity network's edges to a TSV file (source, target, similarity).

    :param graph: Similarity network to write.
    :param output: Output file path.
    """
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source", "target", "similarity"])
        for source, target, data in graph.edges(data=True):
            writer.writerow([source, target, f"{data['weight']:.6f}"])


def write_clusters(clusters: list[set[str]], output: Path) -> None:
    """
    Write cluster membership to a TSV file (header, cluster_id), one row per sequence.

    Cluster ids are 0-indexed, assigned in the order returned by ``get_clusters`` (largest
    cluster first). They are only stable within a single clustering run, not across runs
    with different inputs or cutoffs.

    :param clusters: List of clusters, each a set of sequence headers, as returned by
        ``get_clusters``.
    :param output: Output file path.
    """
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["header", "cluster_id"])
        for cluster_id, cluster in enumerate(clusters):
            for header in sorted(cluster):
                writer.writerow([header, cluster_id])
