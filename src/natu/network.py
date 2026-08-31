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


def get_clusters(graph: nx.Graph) -> list[set[str]]:
    """
    Extract clusters from a similarity network as its connected components, sorted by size.

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
    Score, normalize, build the network, and cluster in one call

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

    # Capture the aligner's own currently configured end-gap score once, before any pair
    # mutates it below -- this is what a shorter sequence's own end gaps are penalized at
    # under glocal (the longer sequence's side is always freed to 0 instead, see
    # configure_glocal_end_gaps). All 8 end-gap attributes are set identically by
    # natu.aligner.setup_aligner, so any one of them reflects the configured value.
    glocal_kept_scores = (
        (aligner.open_left_insertion_score, aligner.extend_left_insertion_score)
        if glocal else None
    )

    if glocal:
        # Every self-alignment compares a sequence against itself (equal length), which
        # always resolves to configure_glocal_end_gaps's "no side is longer" branch --
        # that only depends on the *comparison* between len_t and len_q, never their
        # actual magnitudes (see its docstring), so one reconfiguration covers the whole
        # loop below instead of reconfiguring once per sequence for an identical result.
        configure_glocal_end_gaps(aligner, 1, 1, *glocal_kept_scores)

    self_scores = np.empty(n, dtype=np.float32)
    for i in range(n):
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

    lengths = np.array([len(s) for s in int_seqs], dtype=np.int64)

    def _process_pair(i: int, j: int) -> None:
        nonlocal considered, length_filtered

        considered += 1

        # Self-score to normalise by should be the smallest self-score of the two sequences

        denom = min(self_scores[i], self_scores[j])

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

    for i in range(n):
        if glocal:
            # configure_glocal_end_gaps only cares whether len_t is shorter than, longer
            # than, or equal to len_q -- never the actual magnitude.
            # So every j on the same side of len(int_seqs[i]) shares one configuration:
            # reconfigure the aligner up to 3 times for this whole row, once per bucket,
            # instead of once per pair.
            len_i = len(int_seqs[i])
            tail = np.arange(i + 1, n)
            tail_lengths = lengths[i + 1:]
            for js, len_t, len_q in (
                (tail[tail_lengths < len_i], 2, 1),
                (tail[tail_lengths > len_i], 1, 2),
                (tail[tail_lengths == len_i], 1, 1),
            ):
                if js.size == 0:
                    continue
                configure_glocal_end_gaps(aligner, len_t, len_q, *glocal_kept_scores)
                for j in js:
                    _process_pair(i, int(j))
        else:
            for j in range(i + 1, n):
                _process_pair(i, j)

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
