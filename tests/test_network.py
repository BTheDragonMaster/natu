"""Tests for natu.network: labeling, single-linkage clustering, and the
score/normalize/build/cluster pipeline in cluster_sequences.

Layout: one test class per function under test, named Test<FunctionName>,
with each case as a method on that class.
"""

from __future__ import annotations

from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest
from Bio.Align import substitution_matrices

import natu.network as network_mod
from natu.aligner import setup_aligner
from natu.constants import AlignmentMode
from natu.network import cluster_sequences, get_clusters, sequence_to_label


class TestSequenceToLabel:
    def test_pipe_joins_items(self):
        assert sequence_to_label(["ala", "gly", "ser"]) == "ala|gly|ser"

    def test_single_item_has_no_pipe(self):
        assert sequence_to_label(["ala"]) == "ala"

    def test_empty_sequence_is_empty_string(self):
        assert sequence_to_label([]) == ""


class TestGetClusters:
    """get_clusters is single-linkage via nx.connected_components."""

    def test_transitively_merges_a_chain_of_edges(self):
        """
        This is the mechanism the project explicitly documents as the reason
        natu cluster tends to produce one giant cluster on real data: a-b
        and b-c above cutoff merges a, b, AND c into one cluster even though
        a and c never scored above cutoff against each other directly.
        """
        graph = nx.Graph()
        graph.add_nodes_from(["a", "b", "c", "d", "e"])
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("d", "e")

        clusters = get_clusters(graph)

        assert {"a", "b", "c"} in clusters
        assert {"d", "e"} in clusters
        assert len(clusters) == 2

    def test_isolated_node_is_its_own_singleton(self):
        graph = nx.Graph()
        graph.add_nodes_from(["a", "b", "isolated"])
        graph.add_edge("a", "b")

        clusters = get_clusters(graph)

        assert {"isolated"} in clusters

    def test_sorted_largest_first(self):
        graph = nx.Graph()
        graph.add_nodes_from(["a", "b", "c", "d", "e", "f"])
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("d", "e")

        clusters = get_clusters(graph)

        assert [len(c) for c in clusters] == sorted((len(c) for c in clusters), reverse=True)


class TestClusterSequences:
    # --- input validation ---------------------------------------------

    def test_rejects_header_sequence_length_mismatch(self, make_aligner, converter):
        aligner = make_aligner(mode=AlignmentMode.GLOBAL)
        with pytest.raises(ValueError, match="got 2 headers for 3 sequences"):
            cluster_sequences(aligner, ["a", "b"], [["p"], ["p"], ["p"]], converter, cutoff=0.5)

    def test_rejects_duplicate_headers(self, make_aligner, converter):
        aligner = make_aligner(mode=AlignmentMode.GLOBAL)
        with pytest.raises(ValueError, match="headers must be unique"):
            cluster_sequences(aligner, ["a", "a"], [["p"], ["q"]], converter, cutoff=0.5)

    def test_rejects_non_positive_self_score(self, converter):
        # every diagonal entry (self-match) is negative -> self-score can't
        # be positive, so normalizing similarity against it would be
        # undefined.
        alphabet = ("p", "q", "r", "s")
        data = np.full((4, 4), -1.0)
        np.fill_diagonal(data, -2.0)
        bad_matrix = substitution_matrices.Array(alphabet, 2, data, np.float64)
        aligner = setup_aligner(bad_matrix, mode=AlignmentMode.GLOBAL)

        with pytest.raises(ValueError, match="non-positive self-scores at sequence indices \\[0, 1\\]"):
            cluster_sequences(aligner, ["a", "b"], [["p"], ["q"]], converter, cutoff=0.5)

    # --- similarity / cutoff / min_alignment_length --------------------

    def test_identical_sequences_get_a_full_similarity_edge(self, make_aligner, converter):
        aligner = make_aligner(mode=AlignmentMode.GLOBAL)
        graph, clusters = cluster_sequences(
            aligner, ["a", "b"], [["p", "p"], ["p", "p"]], converter, cutoff=0.99
        )

        assert graph.has_edge("a", "b")
        assert graph["a"]["b"]["weight"] == pytest.approx(1.0)
        assert clusters == [{"a", "b"}]

    def test_cutoff_excludes_low_similarity_pairs(self, make_aligner, converter):
        aligner = make_aligner(mode=AlignmentMode.GLOBAL)
        graph, clusters = cluster_sequences(
            aligner, ["a", "b"], [["p", "p"], ["q", "q"]], converter, cutoff=0.5
        )

        assert not graph.has_edge("a", "b")
        assert clusters == [{"a"}, {"b"}]

    def test_node_carries_its_sequence_label(self, make_aligner, converter):
        aligner = make_aligner(mode=AlignmentMode.GLOBAL)
        graph, _ = cluster_sequences(aligner, ["a"], [["p", "q"]], converter, cutoff=0.5)
        assert graph.nodes["a"]["sequence"] == "p|q"

    def test_min_alignment_length_drops_a_short_probe_spuriously_matching_a_longer_sequence(
        self, make_aligner, converter
    ):
        """
        Direct regression test for the mechanism the project reference
        calls out as reason #1 for spurious giant clusters: a short
        sequence fully embedded in an unrelated longer one can score ~1.0
        under local alignment (normalized against its own short
        self-score) while genuinely sharing only a couple of aligned
        columns. min_alignment_length is the mitigation -- this confirms
        it actually drops such an edge rather than being a no-op.
        """
        aligner = make_aligner(mode=AlignmentMode.LOCAL)
        short = ["r", "r"]
        long_ = ["p", "p", "r", "r", "p", "p"]

        graph_unfiltered, _ = cluster_sequences(
            aligner, ["short", "long"], [short, long_], converter, cutoff=0.9, min_alignment_length=0
        )
        assert graph_unfiltered.has_edge("short", "long")
        assert graph_unfiltered["short"]["long"]["weight"] == pytest.approx(1.0)

        graph_filtered, clusters_filtered = cluster_sequences(
            aligner, ["short", "long"], [short, long_], converter, cutoff=0.9, min_alignment_length=3
        )
        assert not graph_filtered.has_edge("short", "long")
        assert clusters_filtered == [{"short"}, {"long"}]

    # --- glocal per-row bucketing ---------------------------------------
    #
    # This is the exact mechanism from the code snippet originally asked
    # about: for each row i, cluster_sequences partitions the remaining
    # sequences into up to 3 length buckets (shorter/longer/equal-length
    # than sequence i) and calls configure_glocal_end_gaps once per
    # non-empty bucket instead of once per pair.

    def test_glocal_reconfigures_once_per_non_empty_length_bucket_per_row(self, make_aligner, converter):
        aligner = make_aligner(mode=AlignmentMode.GLOCAL)
        headers = ["s2", "s4a", "s4b", "s6"]
        # lengths: 2, 4, 4, 6
        sequences = [["p"] * 2, ["p"] * 4, ["q"] * 4, ["p"] * 6]

        calls: list[tuple[int, int]] = []
        real_configure = network_mod.configure_glocal_end_gaps

        def spy(aligner_, len_t, len_q, *rest):
            calls.append((len_t, len_q))
            return real_configure(aligner_, len_t, len_q, *rest)

        with patch.object(network_mod, "configure_glocal_end_gaps", side_effect=spy):
            cluster_sequences(aligner, headers, sequences, converter, cutoff=-100, glocal=True)

        # First call is the one-time self-alignment setup (len_t == len_q
        # == 1, since every self-comparison is trivially "equal length").
        assert calls[0] == (1, 1)

        # Row i=0 (len 2): tail lengths [4, 4, 6] are all > 2 -> one "longer" bucket call.
        # Row i=1 (len 4): tail lengths [4, 6] -> one "equal" (4==4) + one "longer" (6>4) bucket call.
        # Row i=2 (len 4): tail length [6] -> one "longer" bucket call.
        # Row i=3 (len 6): no tail left -> zero calls.
        assert calls[1:] == [(1, 2), (1, 2), (1, 1), (1, 2)]
        assert len(calls) == 5

    def test_glocal_false_never_calls_configure_glocal_end_gaps(self, make_aligner, converter):
        aligner = make_aligner(mode=AlignmentMode.GLOBAL)
        with patch.object(network_mod, "configure_glocal_end_gaps") as mocked:
            cluster_sequences(
                aligner, ["a", "b"], [["p", "p"], ["p", "p"]], converter, cutoff=0.5, glocal=False
            )
        mocked.assert_not_called()
